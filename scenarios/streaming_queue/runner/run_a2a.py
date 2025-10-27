# -*- coding: utf-8 -*-
"""
A2A-specific Runner
    - Reuses RunnerBase's generic flow
    - Starts A2A executor HTTP services using A2ACommBackend.spawn_local_agent
    - NetworkBase only registers (agent_id, address)
"""

from __future__ import annotations

import os
import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import httpx

# PathSetup
HERE = Path(__file__).resolve()
STREAMING_Q = HERE.parents[1]  # .../streaming_queue
A2A_DIR = STREAMING_Q / "protocol_backend" / "a2a"
sys.path.insert(0, str(STREAMING_Q))
sys.path.insert(0, str(HERE.parent))   # runner/
sys.path.insert(0, str(A2A_DIR))

from runner_base import RunnerBase, ColoredOutput  # type: ignore

# --------- Setup paths for imports ---------
# Ensure streaming_queue is in sys.path for proper imports
if str(STREAMING_Q) not in sys.path:
    sys.path.insert(0, str(STREAMING_Q))

# --------- NetworkBase / CommBackend imports ---------
from core.network_base import NetworkBase  # type: ignore
from protocol_backend.a2a.comm import A2ACommBackend  # type: ignore
from protocol_backend.a2a.coordinator import QACoordinatorExecutor    # type: ignore
from protocol_backend.a2a.worker import QAAgentExecutor            # type: ignore


class A2ARunner(RunnerBase):
    def __init__(self, config_path: str = "config/a2a.yaml"):
        super().__init__(config_path)
        # Reuse a global httpx client (also provided to the backend to avoid duplicate connection pools)
        self.httpx_client = httpx.AsyncClient(timeout=30.0)
        self._handles: List[Any] = []          # Handles for A2A Hosts started in this process (optional)
        self._backend: Optional[A2ACommBackend] = None  # Keep a reference to the backend for spawn/close

    # ---------- Protocol injection: create network ----------
    async def create_network(self) -> NetworkBase:
        # Explicitly use A2ACommBackend so we can call spawn_local_agent
        self._backend = A2ACommBackend(httpx_client=self.httpx_client)
        return NetworkBase(comm_backend=self._backend)

    # ---------- Protocol injection: create/register agents ----------
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

        # Inform coordinator about the network and worker list (used for its internal scheduling)
        if hasattr(coordinator_executor, "coordinator"):
            coordinator_executor.coordinator.set_network(self.network, worker_ids, "a2a")
            # Set metrics collector to communication backend
            if hasattr(self._backend, 'set_metrics_collector') and hasattr(coordinator_executor.coordinator, 'metrics_collector'):
                self._backend.set_metrics_collector(coordinator_executor.coordinator.metrics_collector)

        return worker_ids

    # ---------- Protocol injection: send command to coordinator ----------
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
            # Support two possible event formats
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

    # ---------- Utilities: convert QA config ----------
    def _convert_config_for_qa_agent(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not config:
            return None
        core = config.get("core", {})
        if core.get("type") == "openai":
            # Prioritize environment variables
            api_key = os.getenv("OPENAI_API_KEY") or core.get("openai_api_key")
            base_url = os.getenv("OPENAI_BASE_URL") or core.get("openai_base_url", "https://api.openai.com/v1")
            return {
                "model": {
                    "type": "openai",
                    "name": core.get("name", "gpt-4o"),
                    "openai_api_key": api_key,
                    "openai_base_url": base_url,
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

    # ---------- Cleanup ----------
    async def cleanup(self) -> None:
        try:
            # RunnerBase will call self.network.close(); A2ACommBackend.close() will stop local hosts
            await super().cleanup()
        finally:
            try:
                await self.httpx_client.aclose()
            except Exception:
                pass
            # Extra safety: stop any remaining handles (usually none after network.close)
            for h in self._handles:
                try:
                    await h.stop()
                except Exception:
                    pass


# Direct execution
async def _main():
    runner = A2ARunner()
    await runner.run()

if __name__ == "__main__":
    asyncio.run(_main())
