"""
Meta Protocol Runner - BaseAgent integration with streaming_queue protocols

Uses src/core/base_agent.py and src/core/network.py to create a unified
network with workers from different protocols (ACP, ANP, Agora, A2A).
"""

import asyncio
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

# Setup paths
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parents[1]  # script/streaming_queue/runner -> script/streaming_queue
project_root = streaming_queue_path.parent.parent
src_path = project_root / "src"

# Add paths - src first for priority
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))

# Import src components (must be from src/core)
from src.core.base_agent import BaseAgent
from src.core.network import AgentNetwork

# Import executor wrappers (adjust path after moving to runner folder)
from script.streaming_queue.protocol_backend.meta_protocol.executor_wrappers import create_protocol_worker, validate_executor_interface

from .runner_base import RunnerBase


class MetaProtocolRunner(RunnerBase):
    """
    Meta Protocol Runner using src/core architecture
    
    Creates BaseAgent instances with different protocol worker executors:
    - Each worker uses its native SDK (ACP 1.0.3, ANP AgentConnect, Agora, A2A)
    - All workers are managed through src/core/base_agent.py
    - Network topology managed by src/core/network.py
    """
    
    def __init__(self, config_path_or_dict = "config.yaml"):
        # Initialize RunnerBase first
        if isinstance(config_path_or_dict, str):
            super().__init__(config_path_or_dict)
            self.config = self._load_config(config_path_or_dict)
        else:
            # For dict config, create temporary file for RunnerBase
            import tempfile
            import yaml
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config_path_or_dict, f)
                temp_path = f.name
            super().__init__(temp_path)
            self.config = config_path_or_dict
        
        # Use src/core/network.py for BaseAgent management  
        self.agent_network = AgentNetwork()
        self.network = self.agent_network  # For RunnerBase compatibility
        
        # Track BaseAgent instances
        self.base_agents: Dict[str, BaseAgent] = {}
        self.protocol_map: Dict[str, str] = {}  # agent_id -> protocol
        self.meta_coordinator = None  # Meta protocol coordinator
        
        # Convert config for QA workers
        self.qa_config = self._convert_config_for_qa()
        
        print("[MetaProtocolRunner] Initialized with src/core architecture")
    
    # ---------- RunnerBase required methods ----------
    async def create_network(self):
        """Create network - already done in __init__"""
        return self.network
    
    async def setup_agents(self) -> List[str]:
        """Setup agents and return worker IDs"""
        return await self.create_protocol_workers()
    
    async def send_command_to_coordinator(self, command: str) -> Optional[Dict[str, Any]]:
        """Send command to meta protocol coordinator"""
        if not self.meta_coordinator:
            self.output.error("No meta coordinator available")
            return None
        
        try:
            self.output.info(f"Sending Meta-Protocol command: '{command}'")
            
            # Use meta coordinator to handle commands
            if command == "status":
                # Get network status
                metrics = self.agent_network.snapshot_metrics()
                return {
                    "result": {
                        "status": "healthy",
                        "agents": len(self.base_agents),
                        "protocols": len(set(self.protocol_map.values())),
                        "metrics": metrics
                    }
                }
            elif command == "dispatch":
                # Start QA dispatch process using standard QACoordinatorBase method
                await self.meta_coordinator.dispatch_round()
                return {
                    "result": {
                        "status": "dispatch_completed", 
                        "message": "Meta-protocol QA dispatch completed"
                    }
                }
            else:
                return {"result": {"status": "unknown_command", "command": command}}
                
        except Exception as e:
            self.output.error(f"Meta-Protocol command failed: {e}")
            return None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        config_file = Path(config_path)
        if not config_file.exists():
            config_file = streaming_queue_path / config_path
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def _convert_config_for_qa(self) -> Dict[str, Any]:
        """Convert config for QA workers"""
        core = self.config.get("core", {})
        
        # Check if type is explicitly set to "openai" or if openai_api_key is present
        model_type = core.get("type", "").strip().strip('"')
        has_openai_key = core.get("openai_api_key", "").strip()
        
        if model_type == "openai" or has_openai_key:
            # Use openai_base_url from config, fallback to default if empty
            base_url = core.get("openai_base_url", "").strip().strip('"')
            if not base_url:
                base_url = "https://api.openai.com/v1"
            
            return {
                "model": {
                    "type": "openai",
                    "name": core.get("name", "gpt-4o").strip().strip('"'),
                    "openai_api_key": core.get("openai_api_key", "").strip(),
                    "openai_base_url": base_url,
                    "temperature": core.get("temperature", 0.0),
                }
            }
        else:
            # Local model configuration
            return {
                "model": {
                    "type": "local",
                    "name": core.get("name", "Qwen2.5-VL-72B-Instruct"),
                    "temperature": core.get("temperature", 0.3),
                },
                "base_url": core.get("base_url", "http://localhost:8000/v1"),
                "port": core.get("port", 8000),
            }
    
    async def create_protocol_workers(self) -> List[str]:
        """
        Create protocol-specific BaseAgent servers using the stable factory methods
        and register them into AgentNetwork.
        """
        self.output.info("Creating BaseAgent instances with protocol-specific executors...")

        cfg = self._convert_config_for_qa()
        worker_start_port = self.config.get("qa", {}).get("worker", {}).get("start_port", 10001)  # Use original port range

        # protocol -> (agent_id, port)
        assignments = [
            ("acp",   "ACP-Worker-1",   worker_start_port + 0),
            ("anp",   "ANP-Worker-2",   worker_start_port + 1),
            ("agora", "Agora-Worker-3", worker_start_port + 2),
            ("a2a",   "A2A-Worker-4",   worker_start_port + 3),
        ]

        for proto, agent_id, port in assignments:
            try:
                if proto == "acp":
                    from script.streaming_queue.protocol_backend.meta_protocol.acp_agent import create_acp_meta_worker
                    meta = await create_acp_meta_worker(agent_id, {"core": cfg["model"]}, host="0.0.0.0", port=port, install_loopback=False)
                elif proto == "anp":
                    from script.streaming_queue.protocol_backend.meta_protocol.anp_agent import create_anp_meta_worker
                    meta = await create_anp_meta_worker(agent_id, {"core": cfg["model"]}, host="0.0.0.0", port=port, install_loopback=False)
                elif proto == "agora":
                    from script.streaming_queue.protocol_backend.meta_protocol.agora_agent import create_agora_meta_worker
                    meta = await create_agora_meta_worker(agent_id, {"core": cfg["model"]}, host="0.0.0.0", port=port, install_loopback=False)
                else:  # a2a
                    from script.streaming_queue.protocol_backend.meta_protocol.a2a_agent import create_a2a_meta_worker
                    meta = await create_a2a_meta_worker(agent_id, {"core": cfg["model"]}, host="0.0.0.0", port=port, install_loopback=False)

                # Keep references
                self.base_agents[agent_id] = meta.base_agent
                self.protocol_map[agent_id] = proto

                # Register into AgentNetwork (old AgentNetwork holds BaseAgent instances)
                await self.agent_network.register_agent(meta.base_agent)

                self.output.success(f"✅ {proto.upper()} worker created: {agent_id} @ {meta.base_agent.get_listening_address()}")

            except Exception as e:
                self.output.error(f"❌ Failed to create {proto} worker: {e}")
                continue
        
        # Create meta coordinator for QA dispatch
        await self.create_meta_coordinator()
        
        # Return worker IDs for RunnerBase compatibility
        return list(self.base_agents.keys())
    
    async def setup_topology(self) -> None:
        """Setup topology for RunnerBase compatibility"""
        await self.setup_network_topology()
    
    async def create_meta_coordinator(self):
        """Create meta protocol coordinator for QA testing"""
        try:
            # Import meta coordinator
            from script.streaming_queue.protocol_backend.meta_protocol.meta_coordinator import MetaProtocolCoordinator
            
            # Use A2A as router agent for coordination
            router_agent_id = "A2A-Worker-4"  # Default router
            if router_agent_id in self.base_agents:
                router_agent = self.base_agents[router_agent_id]
                
                self.meta_coordinator = MetaProtocolCoordinator(
                    config=self.qa_config,
                    output=self.output
                )
                
                # Set router agent reference
                self.meta_coordinator.router_ba = router_agent
                
                # Set meta agents and network info
                self.meta_coordinator.meta_agents = {}
                self.meta_coordinator.protocol_types = {}
                
                # Register all created agents (use BaseAgent directly)
                for agent_id, protocol in self.protocol_map.items():
                    # Create a simple wrapper for BaseAgent to match coordinator expectations
                    class BaseAgentWrapper:
                        def __init__(self, base_agent):
                            self.base_agent = base_agent
                    
                    wrapper = BaseAgentWrapper(self.base_agents[agent_id])
                    self.meta_coordinator.meta_agents[agent_id] = wrapper
                    self.meta_coordinator.protocol_types[agent_id] = protocol
                
                # Set router agent directly (not wrapped)
                router_agent_id = "A2A-Worker-4"
                self.meta_coordinator.router_ba = self.base_agents[router_agent_id]
                
                # Set network and worker list
                worker_ids = [aid for aid in self.base_agents.keys() if "worker" in aid.lower()]
                self.meta_coordinator.set_network(self.agent_network, worker_ids)
                
                # Initialize worker stats
                self.meta_coordinator._initialize_worker_stats(worker_ids)
                
                # Install outbound adapters for cross-protocol communication
                await self.meta_coordinator.install_outbound_adapters()
                
                self.output.info(f"Meta Protocol Coordinator created with router: {router_agent_id}")
                self.output.info(f"Registered {len(worker_ids)} workers: {worker_ids}")
            else:
                self.output.error(f"Router agent {router_agent_id} not found")
            
        except Exception as e:
            self.output.error(f"Failed to create meta coordinator: {e}")
            raise
    
    async def setup_network_topology(self) -> None:
        """Setup network topology using src/core/network.py"""
        self.output.info("Setting up meta protocol network topology...")
        
        # Get all worker IDs
        worker_ids = [aid for aid in self.base_agents.keys() if "worker" in aid.lower()]
        
        if len(worker_ids) < 2:
            self.output.error("Not enough workers for topology setup")
            return
        
        # Skip mesh topology setup due to import issues, go directly to cross-protocol test
        self.output.info("Skipping mesh topology setup - testing direct cross-protocol communication")
        
        self.output.success("Network topology ready for cross-protocol communication")
    
    async def test_cross_protocol_communication(self) -> None:
        """
        Cross-protocol test strictly over the network via BaseAgent.send(dst_id, payload).
        """
        self.output.info("=== Cross-Protocol Communication Test (network path) ===")

        # Install outbound adapters from a chosen router (prefer A2A)
        async def _install_from_router():
            # choose router
            router_id = None
            for aid, proto in self.protocol_map.items():
                if proto == "a2a":
                    router_id = aid; break
            if not router_id:
                for aid, proto in self.protocol_map.items():
                    if proto != "anp":
                        router_id = aid; break
            if not router_id:
                self.output.error("No suitable router found"); return None

            router_ba = self.base_agents[router_id]
            # directory
            directory = {}
            for aid, ba in self.base_agents.items():
                url = ba.get_listening_address()
                if "0.0.0.0" in url: url = url.replace("0.0.0.0", "127.0.0.1")
                directory[aid] = (self.protocol_map[aid], url)

            # adapters
            try:
                from src.agent_adapters.a2a_adapter import A2AAdapter
            except Exception: A2AAdapter = None
            try:
                from src.agent_adapters.acp_adapter import ACPAdapter
            except Exception: ACPAdapter = None
            try:
                from src.agent_adapters.agora_adapter import AgoraClientAdapter
            except Exception: AgoraClientAdapter = None

            for dst_id, (proto, url) in directory.items():
                if dst_id == router_id: continue
                if proto == "anp":  # DID-based adapter
                    try:
                        await self._install_anp_outbound_adapter(router_ba, dst_id)
                        self.output.success(f"Installed ANP DID adapter: {router_id} -> {dst_id}")
                    except Exception as e:
                        self.output.error(f"ANP DID adapter failed: {e}")
                    continue
                try:
                    if proto == "a2a" and A2AAdapter:
                        adp = A2AAdapter(httpx_client=router_ba._httpx_client, base_url=url)
                        await adp.initialize()
                        router_ba.add_outbound_adapter(dst_id, adp)
                    elif proto == "acp" and ACPAdapter:
                        adp = ACPAdapter(httpx_client=router_ba._httpx_client, base_url=url, agent_id=dst_id)
                        await adp.initialize()
                        router_ba.add_outbound_adapter(dst_id, adp)
                    elif proto == "agora" and AgoraClientAdapter:
                        # Force no toolformer to avoid JSON schema building errors
                        toolformer = None
                        adp = AgoraClientAdapter(
                            httpx_client=router_ba._httpx_client,
                            toolformer=toolformer,
                            target_url=url,
                            agent_id=dst_id
                        )
                        await adp.initialize()
                        router_ba.add_outbound_adapter(dst_id, adp)
                except Exception as e:
                    self.output.error(f"Install adapter {router_id}->{dst_id} failed: {e}")
            return router_id

        async def _install_anp_outbound_adapter(router_ba, dst_id: str) -> None:
            """Install DID-based ANP outbound adapter on router BaseAgent."""
            import socket
            
            def _find_free_port() -> int:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', 0))
                    return s.getsockname()[1]
            
            # Get server DID from ANP server card
            anp_agent = self.base_agents[dst_id]
            server_card = anp_agent.get_card()
            
            server_did = server_card.get("id") \
                         or (server_card.get("authentication") or {}).get("did")
            if not server_did:
                raise RuntimeError("ANP server card does not contain DID")
            
            # Store server_card for fallback connection
            self._anp_server_card = server_card

            # Create local DID for client
            from agent_connect.python.utils.did_generate import did_generate
            from agent_connect.python.utils.crypto_tool import get_pem_from_private_key
            
            local_ws_port = _find_free_port()
            local_ws_endpoint = f"ws://127.0.0.1:{local_ws_port}/ws"
            private_key, _, local_did, did_document_json = did_generate(local_ws_endpoint)
            local_did_info = {
                "private_key_pem": get_pem_from_private_key(private_key),
                "did": local_did,
                "did_document_json": did_document_json
            }

            # Install ANP adapter with DID service parameters and server card for fallback
            from src.agent_adapters.anp_adapter import ANPAdapter
            import os
            anp_adp = ANPAdapter(
                httpx_client=router_ba._httpx_client,
                target_did=server_did,
                local_did_info=local_did_info,
                host_domain="127.0.0.1",
                host_port=str(local_ws_port),
                host_ws_path="/ws",
                did_service_url=os.getenv("ANP_DID_SERVICE_URL"),   # for did:wba support
                did_api_key=os.getenv("ANP_DID_API_KEY"),           # for did:wba support
                enable_protocol_negotiation=False,
                enable_e2e_encryption=True
            )
            # Store server card for fallback connection
            anp_adp._server_card = server_card
            await anp_adp.initialize()
            router_ba.add_outbound_adapter(dst_id, anp_adp)

        # Make _install_anp_outbound_adapter available to _install_from_router
        self._install_anp_outbound_adapter = _install_anp_outbound_adapter

        router_id = await _install_from_router()
        if not router_id:
            return

        cases = [
            (router_id, "ACP-Worker-1",   {"text": "Hello ACP from router"}),
            (router_id, "Agora-Worker-3", {"text": "Hello Agora from router"}),
            (router_id, "ANP-Worker-2",   {"text": "Hello ANP from router"})  # now enabled via DID
            # Removed A2A->A2A self-send (no self-loop adapter installed)
        ]
        for src_id, dst_id, payload in cases:
            src_ba = self.base_agents.get(src_id)
            if not src_ba:
                self.output.error(f"Missing source: {src_id}"); continue
            try:
                content = await src_ba.send(dst_id, payload)
                self.output.success(f"{src_id} -> {dst_id}: {str(content)[:150]}")
            except Exception as e:
                self.output.error(f"{src_id} -> {dst_id} failed: {e}")
    
    async def display_network_status(self) -> None:
        """Display network status"""
        self.output.info("=== Meta Protocol Network Status ===")
        
        # BaseAgent health checks
        for agent_id, agent in self.base_agents.items():
            protocol = self.protocol_map[agent_id]
            try:
                # Use BaseAgent's health check
                is_healthy = await agent.health_check() if hasattr(agent, 'health_check') else True
                status_icon = "✅" if is_healthy else "❌"
                self.output.info(f"{status_icon} {protocol.upper()}: {agent_id} @ {agent.get_listening_address()}")
            except Exception as e:
                self.output.error(f"❌ {protocol.upper()}: {agent_id} - Error: {e}")
        
        # Network metrics from src/core/network.py
        try:
            metrics = self.agent_network.snapshot_metrics()
            self.output.info("=== Network Metrics (src/core) ===")
            self.output.info(f"Total agents: {metrics.get('agent_count', len(self.base_agents))}")
            self.output.info(f"Total edges: {metrics.get('edge_count', 0)}")
            self.output.info(f"Topology: {metrics.get('topology', 'unknown')}")
            self.output.info(f"Protocol diversity: {len(set(self.protocol_map.values()))} protocols")
        except Exception as e:
            self.output.error(f"Error getting network metrics: {e}")
    
    async def run(self) -> None:
        """Run meta protocol streaming queue QA test"""
        try:
            self.output.info("🚀 Meta Protocol QA Performance Test")
            self.output.info("=" * 60)
            
            # Use RunnerBase standard flow for QA testing
            await super().run()
            
            self.output.success("Meta Protocol QA test completed!")
            
        except Exception as e:
            self.output.error(f"Meta Protocol test failed: {e}")
            raise
        finally:
            # Cleanup
            await self.cleanup_agents()
    
    async def cleanup_agents(self):
        """Cleanup all BaseAgent instances"""
        self.output.info("Cleaning up BaseAgent instances...")
        for agent_id, agent in self.base_agents.items():
            try:
                await agent.stop()  # BaseAgent uses stop() not close()
                protocol = self.protocol_map.get(agent_id, 'unknown')
                self.output.info(f"Stopped {protocol} BaseAgent: {agent_id}")
            except Exception as e:
                self.output.error(f"Error stopping {agent_id}: {e}")
        
        self.output.success("Meta protocol cleanup completed")


# Main execution
async def main():
    # Use the actual config.yaml file from streaming_queue
    config_path = "config.yaml"  # This will load script/streaming_queue/config.yaml
    
    runner = MetaProtocolRunner(config_path)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
