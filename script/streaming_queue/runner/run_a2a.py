# -*- coding: utf-8 -*-
"""
A2A 协议专用 Runner
  - 复用 RunnerBase 的通用流程
  - 通过 A2ACommBackend.spawn_local_agent 启动 A2A executor 的 HTTP 服务
  - NetworkBase 仅登记 (agent_id, address)
"""

from __future__ import annotations

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import httpx

# 路径设置
HERE = Path(__file__).resolve()
STREAMING_Q = HERE.parents[1]  # .../streaming_queue
A2A_DIR = STREAMING_Q / "protocol_backend" / "a2a"
sys.path.insert(0, str(STREAMING_Q))
sys.path.insert(0, str(HERE.parent))   # runner/
sys.path.insert(0, str(A2A_DIR))

from runner_base import RunnerBase, ColoredOutput  # type: ignore

# --------- NetworkBase / CommBackend 导入（带兜底）---------
try:
    from core.network_base import NetworkBase  # type: ignore
except Exception:
    try:
        from script.streaming_queue.core.network_base import NetworkBase  # type: ignore
    except Exception:
        from ..core.network_base import NetworkBase  # type: ignore

try:
    # A2A 后端（含 spawn_local_agent）
    from protocol_backend.a2a.comm import A2ACommBackend  # type: ignore
except Exception:
    # 兜底：直接从绝对路径导
    from script.streaming_queue.protocol_backend.a2a.comm import A2ACommBackend  # type: ignore

# 协议侧：A2A executors（已在 sys.path 注入 A2A_DIR）
from protocol_backend.a2a.coordinator import QACoordinatorExecutor    # type: ignore
from protocol_backend.a2a.worker import QAAgentExecutor            # type: ignore


class A2ARunner(RunnerBase):
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)
        # 复用一个全局 httpx client（也交给 backend 使用，避免重复连接池）
        self.httpx_client = httpx.AsyncClient(timeout=30.0)
        self._handles: List[Any] = []          # 本进程启动的 A2A Host 句柄（可选）
        self._backend: Optional[A2ACommBackend] = None  # 保存 backend 以便 spawn / close

    # ---------- 协议注入：创建网络 ----------
    async def create_network(self) -> NetworkBase:
        # 显式使用 A2ACommBackend，这样我们能用 spawn_local_agent
        self._backend = A2ACommBackend(httpx_client=self.httpx_client)
        return NetworkBase(comm_backend=self._backend)

    # ---------- 协议注入：创建/注册 agent ----------
    async def setup_agents(self) -> List[str]:
        out = self.output
        out.info("Initializing NetworkBase and A2A Agents...")

        qa_cfg = self._convert_config_for_qa_agent(self.config)
        assert self._backend is not None, "backend not initialized"

        # 1) Coordinator
        coord_port = int(self.config.get("qa", {}).get("coordinator", {}).get("port", 9998))
        coordinator_executor = QACoordinatorExecutor(self.config, out)
        coord_handle = await self._backend.spawn_local_agent(
            agent_id="Coordinator-1", host="localhost", port=coord_port, executor=coordinator_executor
        )
        self._handles.append(coord_handle)
        await self.network.register_agent("Coordinator-1", coord_handle.base_url)
        out.success(f"Coordinator-1 started @ {coord_handle.base_url}")

        # 2) Workers
        worker_count = int(self.config.get("qa", {}).get("worker", {}).get("count", 2))
        start_port = int(self.config.get("qa", {}).get("worker", {}).get("start_port", 10001))
        worker_ids: List[str] = []

        for i in range(worker_count):
            wid = f"Worker-{i+1}"
            port = start_port + i
            w_exec = QAAgentExecutor(qa_cfg)
            w_handle = await self._backend.spawn_local_agent(agent_id=wid, host="localhost", port=port, executor=w_exec)
            self._handles.append(w_handle)
            await self.network.register_agent(wid, w_handle.base_url)
            worker_ids.append(wid)
            out.success(f"{wid} started @ {w_handle.base_url}")

        # 告知协调者网络与 worker 集合（用于它的内部调度）
        if hasattr(coordinator_executor, "coordinator"):
            coordinator_executor.coordinator.set_network(self.network, worker_ids)

        return worker_ids

    # ---------- 协议注入：向协调者发指令 ----------
    async def send_command_to_coordinator(self, command: str) -> Optional[Dict[str, Any]]:
        coord_port = int(self.config.get("qa", {}).get("coordinator", {}).get("port", 9998))
        url = f"http://localhost:{coord_port}/message"

        payload = {
            "id": str(time.time_ns()),
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": command}],
                    "messageId": str(time.time_ns()),
                }
            },
        }

        try:
            resp = await self.httpx_client.post(url, json=payload, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()
            # 兼容两种事件格式
            if "events" in data and data["events"]:
                for ev in data["events"]:
                    if ev.get("type") == "agent_text_message":
                        return {"result": ev.get("data", ev.get("text", str(ev)))}
                    if ev.get("kind") == "message":
                        parts = ev.get("parts") or []
                        if parts and isinstance(parts[0], dict):
                            t = parts[0].get("text")
                            if t:
                                return {"result": t}
            return {"result": "Command processed"}
        except Exception as e:
            self.output.error(f"HTTP request to coordinator failed: {e}")
            return None

    # ---------- 工具：转换 QA 配置 ----------
    def _convert_config_for_qa_agent(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not config:
            return None
        core = config.get("core", {})
        if core.get("type") == "openai":
            return {
                "model": {
                    "type": "openai",
                    "name": core.get("name", "gpt-4o"),
                    "openai_api_key": core.get("openai_api_key"),
                    "openai_base_url": core.get("openai_base_url", "https://api.openai.com/v1"),
                    "temperature": core.get("temperature", 0.0),
                }
            }
        if core.get("type") == "local":
            return {
                "model": {
                    "type": "local",
                    "name": core.get("name", "Qwen2.5-VL-72B-Instruct"),
                    "temperature": core.get("temperature", 0.0),
                },
                "base_url": core.get("base_url", "http://localhost:8000/v1"),
                "port": core.get("port", 8000),
            }
        return None

    # ---------- 清理 ----------
    async def cleanup(self) -> None:
        try:
            # RunnerBase 会调用 self.network.close()，A2ACommBackend.close() 会停止本地 hosts
            await super().cleanup()
        finally:
            try:
                await self.httpx_client.aclose()
            except Exception:
                pass
            # 双保险（network.close 已处理，一般不会再剩）
            for h in self._handles:
                try:
                    await h.stop()
                except Exception:
                    pass


# 直接运行
async def _main():
    runner = A2ARunner()
    await runner.run()

if __name__ == "__main__":
    asyncio.run(_main())
