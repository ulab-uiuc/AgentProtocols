# script/streaming_queue/runner/run_acp.py
"""
ACP Protocol Runner using ACP SDK 1.0.3

This runner uses the official ACP SDK to implement full Agent Communication Protocol support.
- Uses Sessions and Runs for structured communication
- Implements proper ACP message handling
- Supports native ACP tool calling and resource management
"""

from __future__ import annotations
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup paths (following the pattern from other runners)
HERE = Path(__file__).resolve()
STREAMING_Q = HERE.parents[1]  # .../streaming_queue
sys.path.insert(0, str(STREAMING_Q))
sys.path.insert(0, str(HERE.parent))   # runner/

from runner_base import RunnerBase
from core.network_base import NetworkBase
from protocol_backend.acp.comm import ACPCommBackend
from protocol_backend.acp.coordinator import ACPCoordinatorExecutor
from protocol_backend.acp.worker import ACPWorkerExecutor

# Import ACP SDK components
try:
    import acp_sdk
    from acp_sdk import RunCreateRequest, RunMode
    ACP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ACP SDK not available: {e}")
    ACP_AVAILABLE = False


class ACPRunner(RunnerBase):
    def __init__(self, config_path: str = "config.yaml"):
        if not ACP_AVAILABLE:
            raise RuntimeError("ACP SDK is required for ACP runner. Please install acp-sdk.")
        
        super().__init__(config_path)
        self._backend: Optional[ACPCommBackend] = None
        self._handles: List[Any] = []
        self._coordinator_handle = None

    # ---------- Protocol injection: create network ----------
    async def create_network(self) -> NetworkBase:
        """Create network with ACP backend."""
        self._backend = ACPCommBackend()
        return NetworkBase(comm_backend=self._backend)

    # ---------- Protocol injection: create/register agents ----------
    async def setup_agents(self) -> List[str]:
        out = self.output
        out.info("Initializing NetworkBase and ACP Agents...")

        qa_cfg = self._convert_config_for_qa_agent(self.config)
        assert self._backend is not None, "ACP backend not initialized"

        # 1) Coordinator
        coord_port = int(self.config.get("qa", {}).get("coordinator", {}).get("start_port", 9998))
        coordinator_executor = ACPCoordinatorExecutor(self.config, out)
        
        # Set backend for coordinator
        coordinator_executor.set_backend(self._backend)
        
        coord_handle = await self._backend.spawn_local_agent(
            agent_id="Coordinator-1", 
            host="localhost", 
            port=coord_port, 
            executor=coordinator_executor
        )
        self._coordinator_handle = coord_handle
        self._handles.append(coord_handle)
        await self.network.register_agent("Coordinator-1", coord_handle.base_url)
        out.success(f"ACP Coordinator-1 started @ {coord_handle.base_url}")

        # 2) Workers
        worker_count = int(self.config.get("qa", {}).get("worker", {}).get("count", 2))
        start_port = int(self.config.get("qa", {}).get("worker", {}).get("start_port", 10001))
        worker_ids: List[str] = []

        for i in range(worker_count):
            wid = f"Worker-{i+1}"
            port = start_port + i
            w_exec = ACPWorkerExecutor(qa_cfg)
            
            w_handle = await self._backend.spawn_local_agent(
                agent_id=wid, 
                host="localhost", 
                port=port, 
                executor=w_exec
            )
            self._handles.append(w_handle)
            await self.network.register_agent(wid, w_handle.base_url)
            worker_ids.append(wid)
            out.success(f"ACP {wid} started @ {w_handle.base_url}")

        # Inform the coordinator about the network and worker set
        if hasattr(coordinator_executor, "coordinator"):
            coordinator_executor.coordinator.set_network(self.network, worker_ids, "acp")

        return worker_ids

    # ---------- Protocol injection: send command to coordinator ----------
    async def send_command_to_coordinator(self, command: str) -> Optional[Dict[str, Any]]:
        """Send command using ACP messaging."""
        if not self._coordinator_handle:
            self.output.error("No coordinator handle available")
            return None

        try:
            self.output.info(f"Sending ACP command: '{command}'")
            
            # Create a run for this command
            run_request = RunCreateRequest(
                agent_name="Coordinator-1",
                input=[{"type": "text", "parts": [{"type": "text", "text": command}]}],
                mode=RunMode.ASYNC
            )
            run_id = await self._coordinator_handle.create_run(run_request)
            
            # Send message through ACP  
            response_msg = await self._coordinator_handle.send_message(run_id, command)
            
            if response_msg:
                # Extract response content
                response_text = ""
                if hasattr(response_msg, 'parts') and response_msg.parts:
                    for part in response_msg.parts:
                        if hasattr(part, 'type') and part.type == "text":
                            response_text += getattr(part, 'text', "")
                
                self.output.info(f"ACP Coordinator response: {response_text}")
                return {
                    "result": {
                        "run_id": run_id,
                        "message_id": response_msg.id if hasattr(response_msg, 'id') else None,
                        "content": response_text,
                        "status": "success"
                    }
                }
            else:
                self.output.error("No response from coordinator")
                return None
                
        except Exception as e:
            self.output.error(f"ACP command failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ---------- Utility: convert QA config ----------
    def _convert_config_for_qa_agent(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert config for QA agents."""
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
        elif core.get("type") == "local":
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
        """Clean up ACP resources."""
        try:
            await super().cleanup()
        finally:
            # Clean up ACP handles
            for h in self._handles:
                try:
                    if hasattr(h, "stop"):
                        await h.stop()
                except Exception:
                    pass


# Direct execution
async def _main():
    runner = ACPRunner()
    await runner.run()


if __name__ == "__main__":
    asyncio.run(_main())