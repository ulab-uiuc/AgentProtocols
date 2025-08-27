# script/streaming_queue/runner/run_anp.py
"""
ANP (Agent Network Protocol) 协议专用 Runner
- 复用 RunnerBase 的通用流程
- 通过 ANPCommBackend.spawn_local_agent 启动 ANP executor 的 HTTP 服务
- NetworkBase 仅登记 (agent_id, address)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# 路径设置
HERE = Path(__file__).resolve()
STREAMING_Q = HERE.parents[1]  # .../streaming_queue
ANP_DIR = STREAMING_Q / "protocol_backend" / "anp"
sys.path.insert(0, str(STREAMING_Q))
sys.path.insert(0, str(HERE.parent))   # runner/
sys.path.insert(0, str(ANP_DIR))

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
    # ANP 后端（含 spawn_local_agent）
    from protocol_backend.anp.comm import ANPCommBackend  # type: ignore
except Exception:
    # 兜底：直接从绝对路径导
    from script.streaming_queue.protocol_backend.anp.comm import ANPCommBackend  # type: ignore

# 协议侧：ANP executors（已在 sys.path 注入 ANP_DIR）
from protocol_backend.anp.coordinator import ANPCoordinatorExecutor    # type: ignore
from protocol_backend.anp.worker import ANPWorkerExecutor            # type: ignore


class ANPRunner(RunnerBase):
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)
        self._backend: Optional[ANPCommBackend] = None
        self._handles: List[Any] = []  # 本进程启动的 ANP Host 句柄

    # ---------- 协议注入：创建网络 ----------
    async def create_network(self) -> NetworkBase:
        # 显式使用 ANPCommBackend，这样我们能用 spawn_local_agent
        self._backend = ANPCommBackend()
        return NetworkBase(comm_backend=self._backend)

    # ---------- 协议注入：创建/注册 agent ----------
    async def setup_agents(self) -> List[str]:
        out = self.output
        out.info("Initializing NetworkBase and ANP Agents...")

        qa_cfg = self._convert_config_for_qa_agent(self.config)
        assert self._backend is not None, "ANP backend not initialized"

        # 1) Coordinator
        coord_port = int(self.config.get("qa", {}).get("coordinator", {}).get("start_port", 9998))
        coordinator_executor = ANPCoordinatorExecutor(self.config, out)
        coord_handle = await self._backend.spawn_local_agent(
            agent_id="Coordinator-1", host="localhost", port=coord_port, executor=coordinator_executor
        )
        self._handles.append(coord_handle)
        await self.network.register_agent("Coordinator-1", coord_handle.base_url)
        out.success(f"ANP Coordinator-1 started @ {coord_handle.base_url}")

        # 2) Workers
        worker_count = int(self.config.get("qa", {}).get("worker", {}).get("count", 2))
        start_port = int(self.config.get("qa", {}).get("worker", {}).get("start_port", 10001))
        worker_ids: List[str] = []

        for i in range(worker_count):
            wid = f"Worker-{i+1}"
            port = start_port + i
            w_exec = ANPWorkerExecutor(qa_cfg)
            w_handle = await self._backend.spawn_local_agent(agent_id=wid, host="localhost", port=port, executor=w_exec)
            self._handles.append(w_handle)
            await self.network.register_agent(wid, w_handle.base_url)
            worker_ids.append(wid)
            out.success(f"ANP {wid} started @ {w_handle.base_url}")

        # 告知协调者网络与 worker 集合（用于它的内部调度）
        if hasattr(coordinator_executor, "coordinator"):
            coordinator_executor.coordinator.set_network(self.network, worker_ids)

        return worker_ids

    # ---------- 协议注入：向协调者发指令 ----------
    async def send_command_to_coordinator(self, command: str) -> Optional[Dict[str, Any]]:
        coord_port = int(self.config.get("qa", {}).get("coordinator", {}).get("start_port", 9998))
        url = f"http://localhost:{coord_port}/anp/message"

        # 构造ANP消息格式
        import uuid
        import time
        anp_message = {
            "protocol": "ANP",
            "version": "1.0",
            "message_id": str(uuid.uuid4()),
            "sender": "Runner",
            "receiver": "Coordinator-1",
            "message_type": "request",
            "payload": {
                "action": "execute",
                "data": command
            },
            "timestamp": time.time()
        }

        try:
            import httpx
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=anp_message)
                resp.raise_for_status()
                data = resp.json()
                
                # 从ANP响应中提取结果
                if isinstance(data, dict) and "payload" in data:
                    result_text = data["payload"].get("data", "Command processed")
                    return {"result": result_text}
                
                return {"result": "Command processed"}
        except Exception as e:
            self.output.error(f"ANP request to coordinator failed: {e}")
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
                    "name": core.get("name", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
                    "temperature": core.get("temperature", 0.0),
                },
                "base_url": core.get("base_url", "http://localhost:8000/v1"),
                "port": core.get("port", 8000),
            }
        return None

    # ---------- 清理 ----------
    async def cleanup(self) -> None:
        try:
            # RunnerBase 会调用 self.network.close()，ANPCommBackend.close() 会停止本地 hosts
            await super().cleanup()
        finally:
            # 双保险（network.close 已处理，一般不会再剩）
            for h in self._handles:
                try:
                    if hasattr(h, "stop"):
                        await h.stop()
                except Exception:
                    pass


# 直接运行
async def _main():
    runner = ANPRunner()
    await runner.run()

if __name__ == "__main__":
    asyncio.run(_main())
