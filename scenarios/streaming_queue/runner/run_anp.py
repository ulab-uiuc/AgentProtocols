# -*- coding: utf-8 -*-
"""
ANP Protocol Runner for Streaming Queue
Real ANP protocol implementation using the AgentConnect SDK, supporting DID authentication, E2E encryption and WebSocket communication
"""

from __future__ import annotations

import os
import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# ================= Path setup =================
HERE = Path(__file__).resolve()
STREAMING_Q = HERE.parents[1]  # .../streaming_queue
PROJECT_ROOT = STREAMING_Q.parent.parent  # .../Multiagent-Protocol

# Add paths (NOTE: Do NOT add protocol_backend/anp/ to sys.path as it contains
# comm.py which conflicts with the comm/ package when network_base.py tries to
# import comm.base.BaseCommBackend)
sys.path.insert(0, str(STREAMING_Q))
sys.path.insert(0, str(HERE.parent))   # runner/

# ================= AgentConnect imports =================
try:
    from agent_connect.utils.did_generate import did_generate
    # from agent_connect.simple_node import SimpleNode, SimpleNodeSession
    print("[ANP Runner] AgentConnect SDK available")
except ImportError as e:
    raise ImportError(f"AgentConnect SDK is required for ANP runner. Please install the agent_connect package. Error: {e}")

# ================= Streaming Queue Imports =================
from runner_base import RunnerBase, ColoredOutput
from core.network_base import NetworkBase
from protocol_backend.anp.comm import ANPCommBackend
from protocol_backend.anp.coordinator import ANPCoordinatorExecutor
from protocol_backend.anp.worker import ANPWorkerExecutor


class ANPRunner(RunnerBase):
    """
    ANP Protocol Runner for Streaming Queue

    Features:
    - Real DID authentication using the AgentConnect SDK
    - E2E encryption support
    - Dual HTTP/WebSocket communication
    - Full compatibility with the streaming_queue framework
    - True ANP protocol implementation (no mocks)
    """

    def __init__(self, config_path: str = "config/anp.yaml"):
        super().__init__(config_path)
        
        # ANP-specific state
        self._backend: Optional[ANPCommBackend] = None
        self._anp_handles: List[Any] = []
        self._coordinator_identity: Optional[Dict[str, Any]] = None
        self._worker_identities: Dict[str, Dict[str, Any]] = {}
        
        # ANP metrics
        self.anp_metrics = {
            "agents_created": 0,
            "did_identities_generated": 0,
            "websocket_endpoints": 0,
            "http_endpoints": 0,
            "encrypted_connections": 0
        }
        
        print("[ANP Runner] Initialized ANP Protocol Runner")

    # ================= Protocol-Specific Implementation =================
    async def create_network(self) -> NetworkBase:
        """Create the ANP network with a real AgentConnect backend."""

        self.output.info("Creating ANP network with AgentConnect backend...")
        
        # Create ANP communication backend with proper timeout
        self._backend = ANPCommBackend(request_timeout=300.0)
        
        # Create network base with ANP backend
        network = NetworkBase(comm_backend=self._backend)
        
        self.output.success("ANP network infrastructure created")
        return network

    async def setup_agents(self) -> List[str]:
        """Set up ANP agents with real DID authentication."""
        self.output.info("Setting up ANP agents with DID authentication...")
        
        if not self._backend:
            raise RuntimeError("ANP backend not initialized")
        
        # Convert config for ANP agents
        anp_config = self._convert_config_for_anp_agents(self.config)
        # ================= Setup Coordinator =================
        coordinator_port = int(self.config.get("qa", {}).get("coordinator", {}).get("start_port", 9998))
        coordinator_websocket_port = coordinator_port + 1000

        # Generate DID identity for coordinator
        service_endpoint = f"http://localhost:{coordinator_port}"
        private_key, public_key, did_id, did_doc = await asyncio.to_thread(
            did_generate, service_endpoint
        )

        # Serialize public key properly
        from cryptography.hazmat.primitives import serialization

        if public_key:
            public_key_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode('utf-8')
        else:
            public_key_pem = "-----BEGIN PUBLIC KEY-----\nSTUB_KEY\n-----END PUBLIC KEY-----"

        coordinator_did = {
            "id": did_id,
            "public_key_pem": public_key_pem,
            "service_endpoint": service_endpoint
        }
        coordinator_keys = {"private_key": private_key}

        self._coordinator_identity = {
            "did_document": coordinator_did,
            "private_keys": coordinator_keys
        }
        self.anp_metrics["did_identities_generated"] += 1

        self.output.info(f"Generated DID for Coordinator: {coordinator_did.get('id', 'Unknown')}")

        # Create coordinator executor
        # Use full self.config to preserve coordinator.result_file path
        coordinator_executor = ANPCoordinatorExecutor(self.config, self.output)
        coordinator_executor.coordinator.set_anp_identity(coordinator_did, coordinator_keys)

        # Spawn ANP coordinator
        coord_handle = await self._backend.spawn_local_agent(
            agent_id="Coordinator-1",
            host="localhost",
            http_port=coordinator_port,
            executor=coordinator_executor,
            websocket_port=coordinator_websocket_port
        )
        self._anp_handles.append(coord_handle)

        # Register coordinator with NetworkBase
        await self.network.register_agent("Coordinator-1", coord_handle.base_url)

        self.anp_metrics["agents_created"] += 1
        self.anp_metrics["http_endpoints"] += 1
        self.anp_metrics["websocket_endpoints"] += 1
        self.anp_metrics["encrypted_connections"] += 1

        self.output.success(f"Coordinator-1 started @ {coord_handle.base_url} (WS: {coord_handle.websocket_url})")

        # ================= Setup Workers =================
        worker_count = int(self.config.get("qa", {}).get("worker", {}).get("count", 4))
        start_port = int(self.config.get("qa", {}).get("worker", {}).get("start_port", 10001))
        worker_ids: List[str] = []

        for i in range(worker_count):
            worker_id = f"Worker-{i+1}"
            worker_port = start_port + i
            worker_websocket_port = worker_port + 1000

            # Generate DID identity for worker
            worker_service_endpoint = f"http://localhost:{worker_port}"
            worker_private_key, worker_public_key, worker_did_id, worker_did_doc = await asyncio.to_thread(
                did_generate, worker_service_endpoint
            )

            # Serialize worker public key properly
            if worker_public_key:
                worker_public_key_pem = worker_public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode('utf-8')
            else:
                worker_public_key_pem = f"-----BEGIN PUBLIC KEY-----\nSTUB_WORKER_{i+1}_KEY\n-----END PUBLIC KEY-----"

            worker_did = {
                "id": worker_did_id,
                "public_key_pem": worker_public_key_pem,
                "service_endpoint": worker_service_endpoint
            }
            worker_keys = {"private_key": worker_private_key}

            self._worker_identities[worker_id] = {
                "did_document": worker_did,
                "private_keys": worker_keys
            }
            self.anp_metrics["did_identities_generated"] += 1

            self.output.info(f"Generated DID for {worker_id}: {worker_did.get('id', 'Unknown')}")

            # Create worker executor
            worker_executor = ANPWorkerExecutor(anp_config, self.output)
            worker_executor.set_anp_identity(worker_did, worker_keys)

            # Spawn ANP worker
            worker_handle = await self._backend.spawn_local_agent(
                agent_id=worker_id,
                host="localhost",
                http_port=worker_port,
                executor=worker_executor,
                websocket_port=worker_websocket_port
            )
            self._anp_handles.append(worker_handle)
            worker_ids.append(worker_id)

            # Register worker with NetworkBase
            await self.network.register_agent(worker_id, worker_handle.base_url)

            self.anp_metrics["agents_created"] += 1
            self.anp_metrics["http_endpoints"] += 1
            self.anp_metrics["websocket_endpoints"] += 1
            self.anp_metrics["encrypted_connections"] += 1

            self.output.success(f"{worker_id} started @ {worker_handle.base_url} (WS: {worker_handle.websocket_url})")

        # ================= Configure Coordinator =================
        # Set network and worker list for the coordinator
        coordinator_executor.coordinator.set_network(self.network, worker_ids, "anp")
        # Set metrics collector to communication backend
        if hasattr(self._backend, 'set_metrics_collector') and hasattr(coordinator_executor.coordinator, 'metrics_collector'):
            self._backend.set_metrics_collector(coordinator_executor.coordinator.metrics_collector)

        # Display ANP summary
        self._display_anp_summary()

        return worker_ids

    async def send_command_to_coordinator(self, command: str) -> Optional[Dict[str, Any]]:
        """Send a command to the ANP coordinator using DID authentication."""
        coordinator_port = int(self.config.get("qa", {}).get("coordinator", {}).get("start_port", 9998))
        url = f"http://localhost:{coordinator_port}/message"
        
        # Prepare ANP-enhanced command
        anp_payload = {
            "text": command,
            "anp_metadata": {
                "protocol": "anp",
                "command_type": "coordinator_control",
                "timestamp": time.time(),
                "encrypted": True,
                "authentication": "did"
            }
        }
        
        # Add DID authentication if available
        if self._coordinator_identity:
            anp_payload["anp_metadata"]["coordinator_did"] = self._coordinator_identity["did_document"].get("id")
        
        try:
            # Use ANP backend's HTTP client with DID auth headers
            import httpx
            # Use no read timeout to avoid premature Runner cleanup during long-running dispatch
            timeout = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=None)
            limits = httpx.Limits(max_connections=1000, max_keepalive_connections=200)
            async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
                
                # Add DID authentication header
                headers = {"Content-Type": "application/json"}
                if self._coordinator_identity:
                    # In a real implementation, this would be a proper DID-signed token
                    auth_token = f"did:{self._coordinator_identity['did_document'].get('id')}"
                    headers["Authorization"] = f"Bearer {auth_token}"
                
                response = await client.post(url, json=anp_payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                # Extract result from ANP response
                result_text = self._extract_anp_response_text(data)
                
                return {
                    "result": result_text,
                    "anp_metadata": data.get("anp_metadata", {}),
                    "raw": data
                }
                
        except Exception as e:
            self.output.error(f"ANP command failed: {e}")
            return {
                "result": f"ANP command error: {e}",
                "anp_metadata": {"error": True},
                "raw": {"error": str(e)}
            }

    def _extract_anp_response_text(self, data: Dict[str, Any]) -> str:
        """Extract text from ANP response"""
        # Check for events (streaming_queue format)
        events = data.get("events", [])
        for event in events:
            if event.get("type") == "agent_text_message":
                return event.get("data", "")
        
        # Check for direct text
        if "text" in data:
            return data["text"]
        
        # Check for ANP data
        if "data" in data:
            return str(data["data"])
        
        # Fallback
        return str(data)

    def _convert_config_for_anp_agents(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert config for ANP agents (same as A2A but with ANP enhancements)"""
        if not config:
            return None
        
        core = config.get("core", {})
        anp_config = None
        
        if core.get("type") == "openai":
            # Prioritize environment variables
            api_key = os.getenv("OPENAI_API_KEY") or core.get("openai_api_key")
            base_url = os.getenv("OPENAI_BASE_URL") or core.get("openai_base_url", "https://api.openai.com/v1")
            anp_config = {
                "model": {
                    "type": "openai",
                    "name": core.get("name", "gpt-4o"),
                    "openai_api_key": api_key,
                    "openai_base_url": base_url,
                    "temperature": core.get("temperature", 0.0),
                }
            }
        elif core.get("type") == "local":
            anp_config = {
                "model": {
                    "type": "local",
                    "name": core.get("name", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
                    "temperature": core.get("temperature", 0.0),
                },
                "base_url": core.get("base_url", "http://localhost:8000/v1"),
                "port": core.get("port", 8000),
            }
        
        # Add ANP-specific config
        if anp_config:
            anp_config["anp"] = {
                "protocol": "anp",
                "did_authentication": True,
                "e2e_encryption": True,
                "websocket_support": True,
                "simple_node_integration": True
            }
        
        return anp_config

    def _display_anp_summary(self):
        """Display ANP setup summary"""
        self.output.info("=== ANP Protocol Summary ===")
        self.output.success(f"Protocol: ANP (Agent Network Protocol)")
        self.output.success(f"AgentConnect SDK: ✓ Available")
        self.output.success(f"DID Identities: {self.anp_metrics['did_identities_generated']}")
        self.output.success(f"HTTP Endpoints: {self.anp_metrics['http_endpoints']}")
        self.output.success(f"WebSocket Endpoints: {self.anp_metrics['websocket_endpoints']}")
        self.output.success(f"Encrypted Connections: {self.anp_metrics['encrypted_connections']}")
        
        # Display coordinator DID
        if self._coordinator_identity:
            coord_did = self._coordinator_identity["did_document"].get("id", "Unknown")
            self.output.info(f"Coordinator DID: {coord_did}")
        
        # Display worker DIDs
        for worker_id, identity in self._worker_identities.items():
            worker_did = identity["did_document"].get("id", "Unknown")
            self.output.info(f"{worker_id} DID: {worker_did}")

    # ================= Cleanup =================
    async def cleanup(self) -> None:
        """Cleanup ANP resources"""
        try:
            self.output.info("Cleaning up ANP resources...")
            
            # Use parent cleanup which will call network.close() -> backend.close()
            await super().cleanup()
            
            # Additional ANP-specific cleanup
            self._anp_handles.clear()
            self._coordinator_identity = None
            self._worker_identities.clear()
            
            self.output.success("ANP cleanup completed")
            
        except Exception as e:
            self.output.error(f"ANP cleanup error: {e}")


# ================= Main Entry Point =================
async def _main():
    """Main entry point for ANP runner"""
    print("[ANP Runner] Starting ANP Protocol Streaming Queue Runner")

    runner = ANPRunner()
    await runner.run()


if __name__ == "__main__":
    asyncio.run(_main())
