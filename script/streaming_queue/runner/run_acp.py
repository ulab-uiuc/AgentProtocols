# script/streaming_queue/runner/run_acp.py
from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional
from .runner_base import RunnerBase
from ..core.network_base import NetworkBase
from ..protocol_backend.acp.comm import ACPCommBackend
from ..protocol_backend.acp.coordinator import ACPCoordinatorExecutor
from ..protocol_backend.acp.worker import ACPWorkerExecutor

class ACPRunner(RunnerBase):
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)
        self._backend: ACPCommBackend | None = None

    async def create_network(self) -> NetworkBase:
        self._backend = ACPCommBackend()
        return NetworkBase(comm_backend=self._backend)

    async def setup_agents(self) -> List[str]:
        assert self._backend is not None

        # Prepare the core config for QAWorkerBase (expects 'model' key)
        core_config = self._prepare_core_config()

        # Spawn local agents using ACP SDK (via spawn_local_agent)
        coord_executor = ACPCoordinatorExecutor(core_config, output=self.output)
        coord_handle = await self._backend.spawn_local_agent(
            "Coordinator-1", "127.0.0.1", 9900, coord_executor
        )
        await self.network.register_agent("Coordinator-1", coord_handle.base_url)

        worker_ids = []
        for i in range(int(self.config["qa"]["worker"]["count"])):
            wid = f"Worker-{i+1}"
            w_port = int(self.config["qa"]["worker"]["start_port"]) + i
            w_handle = await self._backend.spawn_local_agent(
                wid, "127.0.0.1", w_port, ACPWorkerExecutor(core_config, output=self.output)
            )
            await self.network.register_agent(wid, w_handle.base_url)
            worker_ids.append(wid)

        # Set star topology
        self.network.setup_star_topology("Coordinator-1")

        # Tell the coordinator about the network and workers (crucial step!)
        if hasattr(coord_executor, "coordinator"):
            coord_executor.coordinator.set_network(self.network, worker_ids)

        # Send network setup to coordinator via ACP protocol
        worker_list = ",".join(worker_ids)  # This will be "Worker-1,Worker-2,Worker-3,Worker-4"
        setup_command = f"setup_network {worker_list}"
        await self._send_setup_command(setup_command)

        # Give a small delay for setup to complete
        await asyncio.sleep(0.5)

        return worker_ids

    async def _send_setup_command(self, command: str) -> None:
        """Send setup command to coordinator"""
        coordinator_url = "http://127.0.0.1:9900"
        acp_payload = {
            "input": {
                "content": [{"type": "text", "text": command}]
            }
        }

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{coordinator_url}/runs", json=acp_payload)
                resp.raise_for_status()
                # We don't need to process the response for setup
        except Exception as e:
            print(f"Failed to send setup command: {e}")  # Use print since self.output may not be available

    def _prepare_core_config(self) -> dict:
        """Convert config.yaml format to QAWorkerBase expected format"""
        core_section = self.config.get("core", {})
        qa_section = self.config.get("qa", {})

        return {
            "model": {
                "type": core_section.get("type", "openai"),
                "name": core_section.get("name", "gpt-3.5-turbo"),
                "temperature": core_section.get("temperature", 0.3),
                "openai_api_key": core_section.get("openai_api_key"),
                "openai_base_url": core_section.get("openai_base_url", "https://api.openai.com/v1")
            },
            "base_url": core_section.get("base_url", "http://localhost:8000/v1"),
            "port": core_section.get("port", 8000),
            "qa": qa_section  # Add the full QA config section for coordinator
        }

    async def send_command_to_coordinator(self, command: str) -> Optional[Dict[str, Any]]:
        # Send direct HTTP request to coordinator (not via network routing)
        # since Runner is not part of the agent network topology
        coordinator_url = "http://127.0.0.1:9900"
        acp_payload = {
            "input": {
                "content": [{"type": "text", "text": command}]
            }
        }

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{coordinator_url}/runs", json=acp_payload, timeout=60.0)
                resp.raise_for_status()
                result = resp.json()
                return result
        except Exception as e:
            self.output.error(f"Failed to send command to coordinator: {e}")
            return None

async def _main():
    runner = ACPRunner()
    await runner.run()

if __name__ == "__main__":
    asyncio.run(_main())