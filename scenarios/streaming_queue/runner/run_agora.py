# script/streaming_queue/runner/run_agora.py
"""
Agora Protocol Runner
- Reuses the generic flow of RunnerBase
- Starts the HTTP service of the Agora executor through AgoraCommBackend.spawn_local_agent
- NetworkBase only registers (agent_id, address)
"""

from __future__ import annotations
import os
import asyncio
import sys
from pathlib import Path
import httpx
from typing import Any, Dict, List, Optional

# Setup paths (following the pattern from other runners)
HERE = Path(__file__).resolve()
STREAMING_Q = HERE.parents[1]  # .../streaming_queue
sys.path.insert(0, str(STREAMING_Q))
sys.path.insert(0, str(HERE.parent))   # runner/

from runner_base import RunnerBase
from core.network_base import NetworkBase
from protocol_backend.agora.comm import AgoraCommBackend
from protocol_backend.agora.coordinator import AgoraCoordinatorExecutor
from protocol_backend.agora.worker import AgoraWorkerExecutor


class AgoraRunner(RunnerBase):
    def __init__(self, config_path: str = "config/agora.yaml"):
        super().__init__(config_path)
        self._backend: Optional[AgoraCommBackend] = None
        self._handles: List[Any] = []  # Handles for Agora Hosts started by this process
        import httpx
        limits = httpx.Limits(max_connections=1000, max_keepalive_connections=200)
        self.httpx_client = httpx.AsyncClient(timeout=60.0, limits=limits)

    # ---------- Protocol injection: create network ----------
    async def create_network(self) -> NetworkBase:
        # Explicitly use AgoraCommBackend to use spawn_local_agent
        self._backend = AgoraCommBackend()
        return NetworkBase(comm_backend=self._backend)

    # ---------- Protocol injection: create/register agents ----------
    async def setup_agents(self) -> List[str]:
        out = self.output
        out.info("Initializing NetworkBase and Agora Agents...")

        qa_cfg = self._convert_config_for_qa_agent(self.config)
        assert self._backend is not None, "Agora backend not initialized"

        # 1) Coordinator
        coord_port = int(self.config.get("qa", {}).get("coordinator", {}).get("start_port", 9998))
        coordinator_executor = AgoraCoordinatorExecutor(self.config, out)
        coord_handle = await self._backend.spawn_local_agent(
            agent_id="Coordinator-1", host="localhost", port=coord_port, executor=coordinator_executor
        )
        self._handles.append(coord_handle)
        await self.network.register_agent("Coordinator-1", coord_handle.base_url)
        out.success(f"Agora Coordinator-1 started @ {coord_handle.base_url}")

        # 2) Workers
        worker_count = int(self.config.get("qa", {}).get("worker", {}).get("count", 2))
        start_port = int(self.config.get("qa", {}).get("worker", {}).get("start_port", 10001))
        worker_ids: List[str] = []

        for i in range(worker_count):
            wid = f"Worker-{i+1}"
            port = start_port + i
            w_exec = AgoraWorkerExecutor(qa_cfg)
            w_handle = await self._backend.spawn_local_agent(agent_id=wid, host="localhost", port=port, executor=w_exec)
            self._handles.append(w_handle)
            await self.network.register_agent(wid, w_handle.base_url)
            worker_ids.append(wid)
            out.success(f"Agora {wid} started @ {w_handle.base_url}")

        # Inform the coordinator about the network and worker set (for its internal scheduling)
        if hasattr(coordinator_executor, "coordinator"):
            coordinator_executor.coordinator.set_network(self.network, worker_ids, "agora")
            # Set metrics collector to communication backend
            if hasattr(self._backend, 'set_metrics_collector') and hasattr(coordinator_executor.coordinator, 'metrics_collector'):
                self._backend.set_metrics_collector(coordinator_executor.coordinator.metrics_collector)

        return worker_ids

    # ---------- Protocol injection: send command to coordinator ----------
    async def send_command_to_coordinator(self, command: str) -> Optional[Dict[str, Any]]:
        coord_port = int(self.config.get("qa", {}).get("coordinator", {}).get("start_port", 9998))
        coordinator_url = f"http://localhost:{coord_port}"
        
        # Construct Agora message format
        agora_payload = {
            "protocolHash": None,
            "body": command,
            "protocolSources": []
        }
        try:
            self.output.info(f"Sending '{command}' to {coordinator_url}...")
            # Use a timeout that allows long-running dispatch (no read timeout)
            import httpx
            timeout = httpx.Timeout(connect=60.0, read=None, write=60.0, pool=None)
            limits = httpx.Limits(max_connections=1000, max_keepalive_connections=200)
            async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:  # allow long server processing
                response = await client.post(
                    f"{coordinator_url}/",
                    json=agora_payload,
                    timeout=None,  # no per-request override; keep read=None
                    headers={"Content-Type": "application/json"}
                )
            self.output.info(f"Received response with status code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                self.output.info(f"Coordinator response body: {result.get('body')}")
                # ... (rest of the parsing logic is okay)
                raw_result = result.get("raw", {})
                status = raw_result.get("raw", {}).get("status", "failed")
                body = raw_result.get("raw", {}).get("body", "")

                if status == "success" and body:
                    answer = body
                    status = "success"
                else:
                    answer = "No answer received"
                    status = "failed"

                return {"result": result}
            else:
                self.output.error(f"Direct Agora call failed (status {response.status_code}). Response: {response.text}")
                return None
        except httpx.ConnectError as e:
            self.output.error(f"Connection to coordinator at {coordinator_url} failed: {e}")
            return None
        except Exception as e:
            self.output.error(f"An unexpected error occurred during command sending: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ---------- Utility: convert QA config ----------
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
                    "name": core.get("name", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
                    "temperature": core.get("temperature", 0.0),
                },
                "base_url": core.get("base_url", "http://localhost:8000/v1"),
                "port": core.get("port", 8000),
            }
        return None

    # ---------- Cleanup ----------
    async def cleanup(self) -> None:
        try:
            # RunnerBase will call self.network.close(), and AgoraCommBackend.close() will stop local hosts
            await super().cleanup()
        finally:
            # Double insurance (network.close already handles this, so this is unlikely to have remaining handles)
            for h in self._handles:
                try:
                    if hasattr(h, "stop"):
                        await h.stop()
                except Exception:
                    pass


# Direct execution
async def _main():
    runner = AgoraRunner()
    await runner.run()

if __name__ == "__main__":
    asyncio.run(_main())