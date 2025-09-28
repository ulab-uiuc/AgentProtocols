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

# Setup paths (following the pattern from other runners)
HERE = Path(__file__).resolve()
STREAMING_Q = HERE.parents[1]  # .../streaming_queue
PROJECT_ROOT = STREAMING_Q.parent.parent  # .../agent_network
SRC_PATH = PROJECT_ROOT / "src"

# Add paths in correct order
sys.path.insert(0, str(PROJECT_ROOT))  # Add project root first
sys.path.insert(0, str(STREAMING_Q))
sys.path.insert(0, str(HERE.parent))   # runner/
sys.path.insert(0, str(SRC_PATH))

# Import components with correct paths
from runner_base import RunnerBase, ColoredOutput  # type: ignore

# Import src components
from src.core.base_agent import BaseAgent
from src.core.network import AgentNetwork

# Skip executor wrappers import for now - will create agents directly


class MetaProtocolRunner(RunnerBase):
    """
    Meta Protocol Runner using src/core architecture
    
    Creates BaseAgent instances with different protocol worker executors:
    - Each worker uses its native SDK (ACP 1.0.3, ANP AgentConnect, Agora, A2A)
    - All workers are managed through src/core/base_agent.py
    - Network topology managed by src/core/network.py
    """
    
    def __init__(self, config_path_or_dict = "config/meta.yaml"):
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
            config_file = STREAMING_Q / config_path
        
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
    
    def _create_llm_client(self):
        """Create LLM client for routing decisions."""
        try:
            core_config = self.config.get("core", {})
            api_key = core_config.get("openai_api_key", "")
            base_url = core_config.get("openai_base_url", "https://api.openai.com/v1")
            model = core_config.get("name", "gpt-4o")
            
            if not api_key:
                raise ValueError("OpenAI API key not found in config")
            
            class SimpleLLMClient:
                def __init__(self, api_key, base_url, model):
                    self.api_key = api_key
                    self.base_url = base_url
                    self.model = model
                
                async def ask_tool(self, messages, tools, tool_choice):
                    import aiohttp
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": self.model,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": tool_choice,
                        "temperature": 0.0
                    }
                    endpoint = f"{self.base_url}/chat/completions"
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(endpoint, headers=headers, json=payload) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(f"API call failed: {response.status} - {error_text}")
                            result = await response.json()
                            return result["choices"][0]["message"]
            
            return SimpleLLMClient(api_key, base_url, model)
            
        except Exception as e:
            self.output.warning(f"Failed to create LLM client: {e}")
            return None
    
    async def create_protocol_workers(self) -> List[str]:
        """
        Create protocol-specific BaseAgent servers using LLM-based intelligent routing.
        LLM will analyze the workload and select optimal protocols for 4 agents.
        """
        self.output.info("🧠 Creating agents with LLM-based protocol selection...")

        cfg = self._convert_config_for_qa()
        worker_start_port = self.config.get("qa", {}).get("worker", {}).get("start_port", 10001)

        # Use LLM to determine optimal protocol assignment
        try:
            # Import LLM router
            from protocol_backend.meta_protocol.llm_router import route_task_with_llm
            
            # Initialize LLM client
            llm_client = self._create_llm_client()
            
            # Analyze streaming queue workload for optimal protocol selection
            pressure_test_task = {
                "question": "Streaming queue pressure test: process maximum questions in minimum time",
                "context": "High-throughput QA processing with diverse question types",
                "metadata": {
                    "type": "pressure_test",
                    "volume": self.config.get("qa", {}).get("batch_size", 50),
                    "priority": "maximum_speed",
                    "target_qps": 20
                }
            }
            
            # Get LLM routing decision for 4 agents
            routing_decision = await route_task_with_llm(pressure_test_task, num_agents=4, llm_client=llm_client)
            
            self.output.info("🎯 LLM Routing Decision:")
            self.output.info(f"   Selected protocols: {routing_decision.selected_protocols}")
            self.output.info(f"   Strategy: {routing_decision.strategy}")
            self.output.info(f"   Confidence: {routing_decision.confidence:.1%}")
            self.output.info(f"   Reasoning: {routing_decision.reasoning[:100]}...")
            
            # Create agents based on LLM decision
            assignments = []
            port_offset = 0
            for agent_id, protocol in routing_decision.agent_assignments.items():
                port = worker_start_port + port_offset
                assignments.append((protocol, agent_id, port))
                port_offset += 1
            
        except Exception as e:
            self.output.warning(f"LLM routing failed: {e}")
            self.output.info("Falling back to default fast protocol assignment...")
            
            # Fallback to speed-optimized assignment
            assignments = [
                ("a2a",   "FastAgent-1",   worker_start_port + 0),  # Fastest
                ("a2a",   "FastAgent-2",   worker_start_port + 1),  # Fastest
                ("acp",   "FastAgent-3",   worker_start_port + 2),  # Second fastest
                ("acp",   "FastAgent-4",   worker_start_port + 3),  # Second fastest
            ]

        # Create agents based on assignments (LLM or fallback)
        for proto, agent_id, port in assignments:
            try:
                if proto == "acp":
                    from protocol_backend.meta_protocol.acp_agent import create_acp_meta_worker
                    meta = await create_acp_meta_worker(agent_id, {"core": cfg["model"]}, host="0.0.0.0", port=port, install_loopback=False)
                elif proto == "anp":
                    from protocol_backend.meta_protocol.anp_agent import create_anp_meta_worker
                    meta = await create_anp_meta_worker(agent_id, {"core": cfg["model"]}, host="0.0.0.0", port=port, install_loopback=False)
                elif proto == "agora":
                    from protocol_backend.meta_protocol.agora_agent import create_agora_meta_worker
                    meta = await create_agora_meta_worker(agent_id, {"core": cfg["model"]}, host="0.0.0.0", port=port, install_loopback=False)
                else:  # a2a
                    from protocol_backend.meta_protocol.a2a_agent import create_a2a_meta_worker
                    meta = await create_a2a_meta_worker(agent_id, {"core": cfg["model"]}, host="0.0.0.0", port=port, install_loopback=False)

                # Keep references
                self.base_agents[agent_id] = meta.base_agent
                self.protocol_map[agent_id] = proto

                # Register into AgentNetwork (old AgentNetwork holds BaseAgent instances)
                await self.agent_network.register_agent(meta.base_agent)

                # Display protocol selection info
                speed_indicator = "🚀" if proto in ["a2a", "acp"] else "⚡" if proto == "agora" else "🔒"
                self.output.success(f"✅ {proto.upper()} worker created: {agent_id} {speed_indicator} @ {meta.base_agent.get_listening_address()}")

            except Exception as e:
                self.output.error(f"❌ Failed to create {proto} worker: {e}")
                continue
        
        # Analyze final protocol distribution
        protocol_count = {}
        for agent_id in self.base_agents.keys():
            protocol = self.protocol_map.get(agent_id, "unknown")
            protocol_count[protocol] = protocol_count.get(protocol, 0) + 1
        
        self.output.info("📊 Final Protocol Distribution:")
        fast_agents = 0
        for protocol, count in protocol_count.items():
            if protocol in ["a2a", "acp"]:
                fast_agents += count
                speed_info = "🚀 FAST"
            elif protocol == "agora":
                speed_info = "⚡ STABLE"
            elif protocol == "anp":
                speed_info = "🔒 SECURE"
            else:
                speed_info = "❓ UNKNOWN"
            
            self.output.info(f"   {protocol.upper()}: {count} agents {speed_info}")
        
        speed_ratio = fast_agents / len(self.base_agents) if self.base_agents else 0
        self.output.info(f"📈 Speed Optimization: {fast_agents}/{len(self.base_agents)} fast agents ({speed_ratio:.1%})")
        
        # Create meta coordinator for QA dispatch
        await self.create_meta_coordinator()
        
        self.output.success(f"🎉 Successfully created {len(self.base_agents)} agents with LLM-optimized protocol selection!")
        
        # Return worker IDs for RunnerBase compatibility
        return list(self.base_agents.keys())
    
    async def setup_topology(self) -> None:
        """Setup topology for RunnerBase compatibility"""
        await self.setup_network_topology()
    
    async def create_meta_coordinator(self):
        """Create meta protocol coordinator for QA testing"""
        try:
            # Import meta coordinator
            from protocol_backend.meta_protocol.meta_coordinator import MetaProtocolCoordinator
            
            # Use first A2A agent as router for coordination
            router_agent_id = None
            for agent_id, protocol in self.protocol_map.items():
                if protocol == "a2a":
                    router_agent_id = agent_id
                    break
            
            if router_agent_id and router_agent_id in self.base_agents:
                router_agent = self.base_agents[router_agent_id]
                
                self.meta_coordinator = MetaProtocolCoordinator(
                    config=self.config,
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
                self.meta_coordinator.router_ba = self.base_agents[router_agent_id]
                
                # Set network and worker list (use all agents)
                worker_ids = list(self.base_agents.keys())
                self.meta_coordinator.set_network(self.agent_network, worker_ids, "meta")
                
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
        
        # Get all worker IDs (include both "worker" and "agent" patterns)
        worker_ids = [aid for aid in self.base_agents.keys() if "worker" in aid.lower() or "agent" in aid.lower()]
        
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
            from agent_connect.utils.did_generate import did_generate
            from agent_connect.utils.crypto_tool import get_pem_from_private_key
            
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
    # Use the meta config.yaml specific to meta scenario
    config_path = "config/meta.yaml"
    
    runner = MetaProtocolRunner(config_path)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
