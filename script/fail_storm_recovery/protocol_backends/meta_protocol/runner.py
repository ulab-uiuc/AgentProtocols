#!/usr/bin/env python3
"""
Meta Protocol runner for Fail-Storm Recovery scenario.

This module implements the Meta Protocol specific functionality using
LLM-based intelligent routing to select the best underlying protocol
for each agent, while inheriting all core logic from the base runner.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import time
import asyncio
import json
import random
import uuid

# Add paths for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.simple_base_agent import SimpleBaseAgent as BaseAgent
from protocol_backends.base_runner import FailStormRunnerBase

# Import LLM router
from .llm_router import LLMIntelligentRouter, RoutingDecision

# Import src/core network for cross-protocol communication
project_root = Path(__file__).parent.parent.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.core.intelligent_network_manager import IntelligentNetworkManager, create_intelligent_network
from src.core.router_interface import RouterType
from src.core.base_agent import BaseAgent as MetaBaseAgent
from src.core.protocol_converter import ENCODE_TABLE, DECODE_TABLE

# Import underlying protocol agents
from protocol_backends.a2a.agent import create_a2a_agent, A2AAgent
from protocol_backends.acp.agent import create_acp_agent, ACPAgent
from protocol_backends.agora.agent import create_agora_agent, AgoraAgent
from protocol_backends.anp.agent import create_anp_agent, ANPAgent

# Import shard_qa components dynamically to avoid circular imports
shard_qa_path = Path(__file__).parent.parent.parent / "shard_qa"
sys.path.insert(0, str(shard_qa_path))
import importlib.util
spec = importlib.util.spec_from_file_location("agent_executor", shard_qa_path / "shard_worker" / "agent_executor.py")
agent_executor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_executor_module)

# Create Meta-specific implementations
class MetaAgentExecutor(agent_executor_module.BaseAgentExecutor):
    """Meta-specific agent executor that delegates to underlying protocol executors"""
    def __init__(self, protocol_type: str, config: Dict[str, Any]):
        self.protocol_type = protocol_type
        self.config = config
    
    async def execute(self, context, event_queue):
        # Meta protocol delegates to the actual protocol executor
        pass
    
    async def cancel(self, context, event_queue):
        # Meta protocol cancellation
        pass

class MetaRequestContext(agent_executor_module.BaseRequestContext):
    """Meta-specific request context"""
    def __init__(self, input_data):
        self.input_data = input_data
    
    def get_user_input(self):
        return self.input_data

class MetaEventQueue(agent_executor_module.BaseEventQueue):
    """Meta-specific event queue"""
    def __init__(self):
        self.events = []
    
    async def enqueue_event(self, event):
        self.events.append(event)
        return event

def meta_new_agent_text_message(text, role="user"):
    """Meta-specific text message creation"""
    return {"type": "text", "content": text, "role": str(role), "protocol": "meta"}

# Inject Meta implementations into the agent_executor module
agent_executor_module.AgentExecutor = MetaAgentExecutor
agent_executor_module.RequestContext = MetaRequestContext
agent_executor_module.EventQueue = MetaEventQueue
agent_executor_module.new_agent_text_message = meta_new_agent_text_message

ShardWorkerExecutor = agent_executor_module.ShardWorkerExecutor


class MetaProtocolRunner(FailStormRunnerBase):
    """
    Meta Protocol runner with LLM-based intelligent routing.
    
    Implements protocol-specific agent creation and management for Meta Protocol
    while inheriting all core Fail-Storm functionality from FailStormRunnerBase.
    This runner uses LLM to intelligently select the best underlying protocol
    for each agent based on actual fail_storm_recovery performance data.
    """

    def __init__(self, config_path: str = "config.yaml"):
        # If using default config, use configs/config_meta.yaml
        if config_path == "config.yaml":
            configs_dir = Path(__file__).parent.parent.parent / "configs"
            protocol_config = configs_dir / "config_meta.yaml"
            if protocol_config.exists():
                config_path = str(protocol_config)
                print(f"📋 Using Meta Protocol config from: {config_path}")
            else:
                # Fallback to protocol-specific config
                protocol_config = Path(__file__).parent / "config.yaml"
                if protocol_config.exists():
                    config_path = str(protocol_config)
        
        super().__init__(config_path)
        
        # Ensure protocol is set correctly in config
        if "scenario" not in self.config:
            self.config["scenario"] = {}
        self.config["scenario"]["protocol"] = "meta"
        
        # Meta Protocol specific attributes
        self.llm_router = LLMIntelligentRouter()
        self.protocol_agents: Dict[str, Any] = {}  # agent_id -> underlying protocol agent
        self.protocol_types: Dict[str, str] = {}   # agent_id -> protocol_type
        self.agent_sessions: Dict[str, Any] = {}   # agent_id -> session info
        self.routing_decision: Optional[RoutingDecision] = None  # Store LLM routing decision
        
        # Cross-protocol communication network using src/core IntelligentNetworkManager
        self.intelligent_network = IntelligentNetworkManager(
            router_type=RouterType.LLM_BASED,
            router_config=self.config.get("llm", {})
        )
        self.meta_agents: Dict[str, MetaBaseAgent] = {}  # agent_id -> MetaBaseAgent for cross-protocol
        
        # Initialize LLM client
        self._initialize_llm_client()
        
        # Register protocol factories with intelligent network
        self._register_protocol_factories()
        
        # Pre-calculate routing for all agents
        self._precalculate_routing()
        
        print(f"🚀 Meta Protocol Runner initialized with LLM routing")
        print(f"📊 Based on actual fail_storm_recovery performance data:")
        print(f"   • ANP: 61.0% success, 22.0% answer rate (Best accuracy)")
        print(f"   • Agora: 60.0% success, 20.0% answer rate (Best throughput)")  
        print(f"   • A2A: 59.6% success, 19.1% answer rate (High volume)")
        print(f"   • ACP: 59.0% success, 17.9% answer rate (Best recovery)")

    def _precalculate_routing(self):
        """Pre-calculate LLM routing for all 8 agents to avoid fallback."""
        try:
            if not self.llm_router.llm_client:
                print("⚠️  LLM client not available - routing will be calculated when first agent is created")
                return
            
            print(f"🧠 LLM client available - routing will be calculated when needed")
            
        except Exception as e:
            print(f"❌ Pre-calculation setup failed: {e}")
            raise RuntimeError(f"Cannot initialize Meta Protocol: {e}")

    def _register_protocol_factories(self):
        """Register protocol factories with intelligent network manager."""
        try:
            from src.core.router_interface import ProtocolCapability
            
            # Create factory functions that return MetaBaseAgent instances
            async def create_meta_a2a_factory(agent_id: str, config: Dict[str, Any]) -> MetaBaseAgent:
                """Factory for A2A Meta agents"""
                from .a2a_agent import create_a2a_meta_worker
                meta_agent = await create_a2a_meta_worker(agent_id, config)
                return meta_agent.base_agent
            
            async def create_meta_acp_factory(agent_id: str, config: Dict[str, Any]) -> MetaBaseAgent:
                """Factory for ACP Meta agents"""
                from .acp_agent import create_acp_meta_worker
                meta_agent = await create_acp_meta_worker(agent_id, config)
                return meta_agent.base_agent
            
            async def create_meta_agora_factory(agent_id: str, config: Dict[str, Any]) -> MetaBaseAgent:
                """Factory for Agora Meta agents"""
                from .agora_agent import create_agora_meta_worker
                meta_agent = await create_agora_meta_worker(agent_id, config)
                return meta_agent.base_agent
            
            async def create_meta_anp_factory(agent_id: str, config: Dict[str, Any]) -> MetaBaseAgent:
                """Factory for ANP Meta agents"""
                from .anp_agent import create_anp_meta_worker
                meta_agent = await create_anp_meta_worker(agent_id, config)
                return meta_agent.base_agent
            
            # Register factories with capabilities based on actual performance data
            self.intelligent_network.register_protocol_factory(
                "a2a", 
                create_meta_a2a_factory,
                ProtocolCapability(
                    name="a2a",
                    agent_id="A2A-Meta-Worker",
                    strengths=["High volume (178 tasks)", "Fast recovery (6.0s)", "Good throughput"],
                    best_for=["High-volume processing", "Quick recovery", "Load balancing"],
                    performance_metrics={
                        "avg_response_time": 7.39,
                        "success_rate": 0.596,
                        "answer_discovery_rate": 0.191,
                        "recovery_time": 6.0
                    },
                    current_load=0,
                    max_capacity=100
                )
            )
            
            self.intelligent_network.register_protocol_factory(
                "acp",
                create_meta_acp_factory, 
                ProtocolCapability(
                    name="acp",
                    agent_id="ACP-Meta-Worker",
                    strengths=["Best recovery (0.70s avg)", "Fault tolerance", "Most recovery tasks (22)"],
                    best_for=["Fault recovery", "Critical operations", "Recovery scenarios"],
                    performance_metrics={
                        "avg_response_time": 7.83,
                        "success_rate": 0.590,
                        "answer_discovery_rate": 0.179,
                        "recovery_time": 8.0
                    },
                    current_load=0,
                    max_capacity=100
                )
            )
            
            self.intelligent_network.register_protocol_factory(
                "agora",
                create_meta_agora_factory,
                ProtocolCapability(
                    name="agora",
                    agent_id="Agora-Meta-Worker",
                    strengths=["Highest throughput (180 tasks)", "Good success rate (60.0%)", "Balanced performance"],
                    best_for=["Maximum throughput", "Balanced workloads", "Comprehensive coverage"],
                    performance_metrics={
                        "avg_response_time": 7.10,
                        "success_rate": 0.600,
                        "answer_discovery_rate": 0.200,
                        "recovery_time": 6.1
                    },
                    current_load=0,
                    max_capacity=100
                )
            )
            
            self.intelligent_network.register_protocol_factory(
                "anp",
                create_meta_anp_factory,
                ProtocolCapability(
                    name="anp",
                    agent_id="ANP-Meta-Worker",
                    strengths=["Highest success rate (61.0%)", "Best answer rate (22.0%)", "Best response time (6.76s)"],
                    best_for=["High accuracy", "Best answer discovery", "Quality scenarios"],
                    performance_metrics={
                        "avg_response_time": 6.76,
                        "success_rate": 0.610,
                        "answer_discovery_rate": 0.220,
                        "recovery_time": 10.0
                    },
                    current_load=0,
                    max_capacity=100
                )
            )
            
            print("🔧 Registered all protocol factories with IntelligentNetworkManager")
            
        except Exception as e:
            print(f"❌ Failed to register protocol factories: {e}")
            raise

    async def _calculate_routing_for_all_agents(self):
        """Calculate LLM routing for all agents at once."""
        try:
            agent_count = self.config.get("scenario", {}).get("agent_count", 8)
            
            # Create comprehensive task for all agents
            sample_task = {
                "question": "Fail-storm recovery test with diverse question types and fault injection",
                "context": f"Routing decision for {agent_count} agents in fail storm recovery scenario",
                "metadata": {
                    "type": "fail_storm_recovery",
                    "agent_count": agent_count,
                    "priority": ["answer_discovery_rate", "recovery_time"],
                    "scenario": "fault_injection_with_recovery",
                    "critical_metrics": {
                        "answer_discovery_rate": "22.0% (ANP) > 20.0% (Agora) > 19.1% (A2A) > 17.9% (ACP)",
                        "recovery_time": "6.0s (A2A) < 6.1s (Agora) < 8.0s (ACP) < 10.0s (ANP)"
                    }
                }
            }
            
            print(f"🧠 Calculating LLM routing for all {agent_count} agents...")
            print(f"🎯 Optimizing for: Answer discovery rate + Recovery time")
            
            # Get LLM routing decision for all agents at once
            self.routing_decision = await self.llm_router.route_task_with_llm(sample_task, num_agents=agent_count)
            
            print(f"✅ LLM routing completed:")
            print(f"   Selected protocols: {self.routing_decision.selected_protocols}")
            print(f"   Strategy: {self.routing_decision.strategy}")
            print(f"   Confidence: {self.routing_decision.confidence:.2%}")
            print(f"   Agent assignments: {len(self.routing_decision.agent_assignments)} agents")
            
            # Log protocol distribution
            protocol_counts = {}
            for agent_id, protocol in self.routing_decision.agent_assignments.items():
                protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
            
            print(f"📊 Planned Protocol Distribution:")
            for protocol, count in sorted(protocol_counts.items()):
                print(f"   {protocol.upper()}: {count} agents")
            
        except Exception as e:
            print(f"❌ LLM routing calculation failed: {e}")
            raise RuntimeError(f"Cannot proceed without LLM routing: {e}")

    async def _create_meta_agent_wrapper(self, protocol: str, agent_id: str, 
                                       config: Dict[str, Any], host: str, port: int):
        """Create Meta Protocol agent wrapper using the appropriate meta agent."""
        if protocol == "a2a":
            from .a2a_agent import create_a2a_meta_worker
            return await create_a2a_meta_worker(agent_id, config, host, port, install_loopback=False)
        elif protocol == "acp":
            from .acp_agent import create_acp_meta_worker
            return await create_acp_meta_worker(agent_id, config, host, port, install_loopback=False)
        elif protocol == "agora":
            from .agora_agent import create_agora_meta_worker
            return await create_agora_meta_worker(agent_id, config, host, port, install_loopback=False)
        elif protocol == "anp":
            from .anp_agent import create_anp_meta_worker
            return await create_anp_meta_worker(agent_id, config, host, port, install_loopback=False)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

    def _create_compatible_agent(self, base_agent: MetaBaseAgent, protocol: str):
        """Create a compatible agent wrapper for fail_storm_recovery."""
        class CompatibleAgent:
            def __init__(self, base_agent, protocol):
                self.base_agent = base_agent
                self.agent_id = base_agent.agent_id
                self.protocol = protocol
                self.host = base_agent.host
                self.port = base_agent.port
                
            async def send_message(self, target_agent_id: str, message: Dict[str, Any]):
                """Send message via BaseAgent (with protocol conversion)."""
                return await self.base_agent.send(target_agent_id, message)
                
            def get_status(self):
                """Get agent status."""
                return {
                    "agent_id": self.agent_id,
                    "protocol": self.protocol,
                    "host": self.host,
                    "port": self.port,
                    "meta_enabled": True
                }
                
            async def stop(self):
                """Stop the agent."""
                await self.base_agent.stop()
        
        return CompatibleAgent(base_agent, protocol)

    async def _install_cross_protocol_adapters(self):
        """Install outbound adapters for cross-protocol communication."""
        try:
            print("🔗 Installing cross-protocol outbound adapters...")
            
            # Import all adapters
            from src.agent_adapters.a2a_adapter import A2AAdapter
            from src.agent_adapters.acp_adapter import ACPAdapter
            from src.agent_adapters.agora_adapter import AgoraClientAdapter
            from src.agent_adapters.anp_adapter import ANPAdapter
            
            # For each BaseAgent, install adapters to all other BaseAgents
            for src_id, src_base_agent in self.meta_agents.items():
                src_protocol = self.protocol_types[src_id]
                
                for dst_id, dst_base_agent in self.meta_agents.items():
                    if src_id == dst_id:
                        continue  # Skip self
                    
                    dst_protocol = self.protocol_types[dst_id]
                    dst_url = dst_base_agent.get_listening_address()
                    
                    # Fix URL for local connections
                    if "0.0.0.0" in dst_url:
                        dst_url = dst_url.replace("0.0.0.0", "127.0.0.1")
                    
                    try:
                        # Create appropriate adapter based on destination protocol
                        if dst_protocol == "a2a":
                            adapter = A2AAdapter(
                                httpx_client=src_base_agent._httpx_client,
                                base_url=dst_url
                            )
                        elif dst_protocol == "acp":
                            adapter = ACPAdapter(
                                httpx_client=src_base_agent._httpx_client,
                                base_url=dst_url,
                                agent_id=dst_id
                            )
                        elif dst_protocol == "agora":
                            adapter = AgoraClientAdapter(
                                httpx_client=src_base_agent._httpx_client,
                                toolformer=None,  # Use minimal toolformer
                                target_url=dst_url,
                                agent_id=dst_id
                            )
                        elif dst_protocol == "anp":
                            # Try DID-based ANP adapter first
                            import os
                            if os.getenv("ANP_DID_SERVICE_URL") and os.getenv("ANP_DID_API_KEY"):
                                # ANP requires DID-based connection
                                adapter = await self._create_anp_adapter(src_base_agent, dst_base_agent, dst_id)
                            else:
                                # Fallback to simple HTTP adapter for development/testing (DID missing)
                                print(f"⚠️  ANP DID missing; falling back to simple HTTP adapter for {dst_id} (dev-only)")
                                adapter = self._create_simple_http_adapter(src_base_agent, dst_url, dst_id)
                        else:
                            continue
                        
                        # 初始化统一加超时（所有协议都建议加）
                        try:
                            await asyncio.wait_for(adapter.initialize(), timeout=10.0)
                        except asyncio.TimeoutError:
                            print(f"⚠️  Adapter init timeout: {src_id} -> {dst_id}, skip this link")
                            continue
                        
                        src_base_agent.add_outbound_adapter(dst_id, adapter)
                        
                        print(f"🔗 Installed {src_protocol.upper()} → {dst_protocol.upper()} adapter: {src_id} → {dst_id}")
                        
                    except Exception as e:
                        print(f"⚠️  Failed to install adapter {src_id} → {dst_id}: {e}")
                        continue
            
            print("✅ Cross-protocol adapters installation completed")
            
        except Exception as e:
            print(f"❌ Failed to install cross-protocol adapters: {e}")
            raise

    def _create_simple_http_adapter(self, src_base_agent, dst_url: str, dst_id: str):
        """Create an ANP-HTTP adapter for ANP fallback."""
        
        class ANPHTTPAdapter:
            """ANP-HTTP adapter that maintains ANP protocol semantics over HTTP transport."""
            
            def __init__(self, httpx_client, base_url: str, agent_id: str):
                self.httpx_client = httpx_client
                self.base_url = base_url.rstrip('/')
                self.agent_id = agent_id
                self.protocol_name = "ANP"  # ANP protocol over HTTP transport
                
            async def initialize(self):
                """Initialize adapter - check if target ANP agent is reachable."""
                try:
                    response = await self.httpx_client.get(f"{self.base_url}/health", timeout=5.0)
                    if response.status_code == 200:
                        return
                except:
                    pass
                # If /health fails, that's ok for ANP-HTTP adapter
                
            async def send_message(self, dst_id: str, payload):
                """Send message using ANP-compatible format over HTTP."""
                import json
                import time
                import uuid
                
                # Convert to ANP agent expected format (simple content + meta)
                anp_message = {
                    "content": payload,  # ANP agent expects 'content' field
                    "sender": self.agent_id,
                    "meta": {
                        "id": str(uuid.uuid4()),
                        "timestamp": time.time(),
                        "transport": "anp_http_fallback"
                    }
                }
                
                # Try ANP-specific endpoints first, then fallback
                endpoints = ["/anp/message", "/message", "/"]
                
                for endpoint in endpoints:
                    try:
                        headers = {
                            "Content-Type": "application/json",
                            "X-Protocol": "ANP-HTTP",  # Indicate ANP protocol over HTTP
                            "X-Agent-ID": self.agent_id
                        }
                        
                        response = await self.httpx_client.post(
                            f"{self.base_url}{endpoint}",
                            json=anp_message,
                            headers=headers,
                            timeout=30.0
                        )
                        response.raise_for_status()
                        return response.json()
                    except Exception as e:
                        if endpoint != endpoints[-1]:  # Not the last endpoint
                            continue  # Try next endpoint
                        else:
                            raise e  # All endpoints failed
                
                raise ConnectionError(f"All ANP-HTTP endpoints failed for {self.base_url}")
            
            async def receive_message(self):
                """Receive message (not used in outbound adapter)."""
                return {}
            
            def get_agent_card(self):
                """Get agent card (not used in outbound adapter)."""
                return {"agent_id": self.agent_id, "protocol": "anp_http"}
        
        return ANPHTTPAdapter(src_base_agent._httpx_client, dst_url, dst_id)

    async def _create_anp_adapter(self, src_agent: MetaBaseAgent, dst_agent: MetaBaseAgent, dst_id: str):
        """Create ANP adapter with DID-based connection."""
        # Get server DID from destination agent card
        server_card = dst_agent.get_card()
        server_did = server_card.get("id") or server_card.get("did")
        
        if not server_did:
            raise RuntimeError(f"No DID found for ANP agent {dst_id}")
        
        # Create local DID for client
        from agent_connect.python.utils.did_generate import did_generate
        from agent_connect.python.utils.crypto_tool import get_pem_from_private_key
        import socket
        
        def _find_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                return s.getsockname()[1]
        
        local_ws_port = _find_free_port()
        local_ws_endpoint = f"ws://127.0.0.1:{local_ws_port}/ws"
        private_key, _, local_did, did_document_json = did_generate(local_ws_endpoint)
        
        local_did_info = {
            "private_key_pem": get_pem_from_private_key(private_key),
            "did": local_did,
            "did_document_json": did_document_json
        }
        
        from src.agent_adapters.anp_adapter import ANPAdapter
        import os
        
        return ANPAdapter(
            httpx_client=src_agent._httpx_client,
            target_did=server_did,
            local_did_info=local_did_info,
            host_domain="127.0.0.1",
            host_port=str(local_ws_port),
            host_ws_path="/ws",
            did_service_url=os.getenv("ANP_DID_SERVICE_URL"),
            did_api_key=os.getenv("ANP_DID_API_KEY"),
            enable_protocol_negotiation=False,
            enable_e2e_encryption=True
        )

    def get_protocol_name(self) -> str:
        """Return the protocol name for this runner."""
        return "meta"

    def _initialize_llm_client(self):
        """Initialize LLM client for routing decisions."""
        try:
            # Get LLM config (following fail_storm_recovery pattern)
            llm_config = self.config.get("llm", {})
            api_key = llm_config.get("openai_api_key", "")
            base_url = llm_config.get("openai_base_url", "https://api.openai.com/v1")
            model = llm_config.get("model", "gpt-4o")
            
            if not api_key:
                print("⚠️  No OpenAI API key found, LLM routing disabled - using performance-based fallback")
                return
            
            # Create simple LLM client
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
            
            llm_client = SimpleLLMClient(api_key, base_url, model)
            self.llm_router.set_llm_client(llm_client)
            
            print(f"🧠 LLM client initialized: {model}")
            
        except Exception as e:
            print(f"⚠️  Failed to initialize LLM client: {e}")

    async def create_agent(self, agent_id: str, host: str, port: int, 
                          executor: Any, is_coordinator: bool = False) -> BaseAgent:
        """
        Create a Meta Protocol agent using pre-calculated LLM routing.
        
        This method uses the pre-calculated LLM routing decision to select
        the optimal protocol for each agent.
        """
        try:
            # Get protocol assignment from pre-calculated routing
            if not self.routing_decision:
                # If not pre-calculated, do it now for all agents
                await self._calculate_routing_for_all_agents()
            
            # Get the protocol assignment for this specific agent
            # agent ID should match directly (agent0, agent1, agent2, ...)
            selected_protocol = self.routing_decision.agent_assignments.get(agent_id)
            if not selected_protocol:
                raise RuntimeError(f"No protocol assignment found for agent {agent_id} in LLM routing decision")
            
            # Create Meta Protocol agent (already BaseAgent wrapped)
            meta_agent_wrapper = await self._create_meta_agent_wrapper(
                protocol=selected_protocol,
                agent_id=agent_id,
                config=self.config,
                host=host,
                port=port
            )
            
            # Get the BaseAgent from the wrapper
            base_agent = meta_agent_wrapper.base_agent
            
            # Register with intelligent network for cross-protocol communication
            await self.intelligent_network.network.register_agent(base_agent)
            
            # Store protocol mapping and session info
            self.protocol_types[agent_id] = selected_protocol
            self.protocol_agents[agent_id] = meta_agent_wrapper  # Store the wrapper
            self.meta_agents[agent_id] = base_agent  # Store the BaseAgent
            self.agent_sessions[agent_id] = {
                "base_url": f"http://{host}:{port}",
                "session_id": f"session_{agent_id}_{int(time.time())}",
                "protocol": selected_protocol
            }
            
            print(f"✅ Created {selected_protocol.upper()} Meta agent: {agent_id} @ http://{host}:{port}")
            print(f"🔗 Cross-protocol communication enabled via BaseAgent")
            
            # Return a compatible agent for fail_storm_recovery
            # Create a simple wrapper that mimics the original protocol agent interface
            return self._create_compatible_agent(base_agent, selected_protocol)
            
        except Exception as e:
            print(f"❌ Failed to create Meta Protocol agent {agent_id}: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def _create_underlying_agent(self, protocol: str, agent_id: str, 
                                     host: str, port: int, executor: Any) -> BaseAgent:
        """Create the underlying protocol agent."""
        if protocol == "a2a":
            return await create_a2a_agent(agent_id, host, port, executor)
        elif protocol == "acp":
            return await create_acp_agent(agent_id, host, port, executor)
        elif protocol == "agora":
            return await create_agora_agent(agent_id, host, port, executor)
        elif protocol == "anp":
            return await create_anp_agent(agent_id, host, port, executor)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

    def get_protocol_info(self, agent_id: str, port: int, data_file: str) -> str:
        """Get Meta protocol display information."""
        protocol = self.protocol_types.get(agent_id, "unknown")
        return f"🧠 [META-{protocol.upper()}] Created {agent_id} - HTTP: {port}, Data: {data_file}"

    def get_reconnection_info(self, agent_id: str, port: int) -> List[str]:
        """Get Meta protocol reconnection information."""
        protocol = self.protocol_types.get(agent_id, "unknown")
        return [
            f"🧠 [META-{protocol.upper()}] Agent {agent_id} RECONNECTED on port {port}",
            f"📡 [META] LLM-selected {protocol.upper()} protocol active",
            f"🌐 [META] HTTP REST API: http://127.0.0.1:{port}"
        ]

    async def _setup_mesh_topology(self) -> None:
        """Setup mesh topology between Meta Protocol agents with cross-protocol communication."""
        self.output.progress("🧠 [META] Setting up LLM-routed mesh topology with cross-protocol adapters...")
        
        # Setup basic mesh topology using intelligent network
        await self.intelligent_network.network.setup_mesh_topology()
        
        # Only install cross-protocol adapters if we have multiple protocols
        used_protocols = set(self.protocol_types.values())
        if len(used_protocols) > 1:
            print(f"🔗 Installing cross-protocol adapters for {len(used_protocols)} protocols...")
            await self._install_cross_protocol_adapters()
        else:
            print(f"🔗 Single protocol deployment ({list(used_protocols)[0]}), no cross-protocol adapters needed")
        
        # Verify connectivity
        topology = self.intelligent_network.network.get_topology()
        expected_connections = len(self.meta_agents) * (len(self.meta_agents) - 1)
        actual_connections = sum(len(edges) for edges in topology.values())
        
        self.output.success(f"🧠 [META] Cross-protocol mesh topology established: {actual_connections}/{expected_connections} connections")
        
        # Log protocol distribution
        protocol_groups = {}
        for agent_id, protocol in self.protocol_types.items():
            if protocol not in protocol_groups:
                protocol_groups[protocol] = []
            protocol_groups[protocol].append(agent_id)
        
        self.output.progress("📊 [META] Protocol Distribution:")
        for protocol, agent_ids in protocol_groups.items():
            self.output.progress(f"   {protocol.upper()}: {len(agent_ids)} agents")

    async def _broadcast_document(self) -> None:
        """Broadcast the document to all Meta Protocol agents (use intelligent network)."""
        if not self.meta_agents:
            raise RuntimeError("No Meta Protocol agents available for broadcast")

        self.output.progress("📡 [META] Broadcasting document using LLM-routed protocols...")

        # 用在 intelligent_network 上已注册的任意一个 BaseAgent 作为 broadcaster
        broadcaster_id = next(iter(self.meta_agents.keys()))

        # ✅ 统一走 intelligent_network（与注册/拓扑一致）
        results = await self.intelligent_network.network.broadcast_message(broadcaster_id, self.document)

        successful_deliveries = sum(1 for r in results.values() if "error" not in str(r))
        total_targets = len(results)
        self.output.success(f"📡 [META] Document broadcast: {successful_deliveries}/{total_targets} deliveries successful")

    async def setup_agent_connections(self) -> None:
        """Set up connections between Meta Protocol agents."""
        try:
            print("🔗 Setting up Meta Protocol agent connections...")
            
            # Group agents by protocol for efficient connection setup
            protocol_groups = {}
            for agent_id, protocol in self.protocol_types.items():
                if protocol not in protocol_groups:
                    protocol_groups[protocol] = []
                protocol_groups[protocol].append(agent_id)
            
            # Set up connections within each protocol group and cross-protocol
            for protocol, agent_ids in protocol_groups.items():
                print(f"🔗 Setting up {protocol.upper()} protocol connections for {len(agent_ids)} agents...")
                
                # Protocol-specific connection setup
                if protocol == "acp":
                    await self._setup_acp_connections(agent_ids)
                elif protocol == "agora":
                    await self._setup_agora_connections(agent_ids)
                # A2A and ANP handle connections internally
            
            print(f"✅ Meta Protocol connections established")
            
            # Log protocol distribution
            print("📊 Final Protocol Distribution:")
            for protocol, agent_ids in protocol_groups.items():
                print(f"   {protocol.upper()}: {len(agent_ids)} agents")
            
        except Exception as e:
            print(f"❌ Failed to setup Meta Protocol connections: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def _setup_acp_connections(self, agent_ids: List[str]) -> None:
        # 在使用统一 outbound adapters 的情况下，可以直接 no-op
        return

    async def _setup_agora_connections(self, agent_ids: List[str]) -> None:
        # 同上
        return

    async def send_message_to_agent(self, src_agent: BaseAgent, dst_agent_id: str, message: Dict[str, Any]):
        """Send message between Meta Protocol agents using BaseAgent router."""
        try:
            if dst_agent_id not in self.meta_agents:
                return {"status": "error", "error": f"Agent {dst_agent_id} not found"}

            # 选择一个"路由器"BaseAgent来发（优先 A2A，否则任意一个）
            router_id = None
            for aid, proto in self.protocol_types.items():
                if proto == "a2a":
                    router_id = aid
                    break
            if router_id is None:
                router_id = next(iter(self.meta_agents.keys()))

            router_ba = self.meta_agents[router_id]
            resp = await router_ba.send(dst_agent_id, message)

            return {"status": "success", "response": resp, "protocol": self.protocol_types.get(dst_agent_id, "unknown")}
        except Exception as e:
            return {"status": "error", "error": str(e), "protocol": self.protocol_types.get(dst_agent_id, "meta")}

    async def health_check_agent(self, agent: BaseAgent, agent_id: str) -> bool:
        try:
            if agent_id not in self.meta_agents:
                return False
            ba = self.meta_agents[agent_id]
            if hasattr(ba, 'health_check'):
                return await ba.health_check()
            # 回退：能拿到监听地址就认为活着
            return bool(ba.get_listening_address())
        except Exception:
            return False

    async def cleanup_agents(self) -> None:
        """Clean up all Meta Protocol agents."""
        try:
            print("🧹 Cleaning up Meta Protocol agents...")
            
            for agent_id, agent in self.protocol_agents.items():
                try:
                    if hasattr(agent, 'stop'):
                        await agent.stop()
                    elif hasattr(agent, 'close'):
                        await agent.close()
                except Exception as e:
                    print(f"⚠️  Error cleaning up agent {agent_id}: {e}")
            
            self.protocol_agents.clear()
            self.protocol_types.clear()
            self.agent_sessions.clear()
            
            print("✅ Meta Protocol cleanup completed")
            
        except Exception as e:
            print(f"⚠️  Meta Protocol cleanup error: {e}")

    def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a Meta Protocol agent."""
        if agent_id not in self.protocol_agents:
            return {"error": f"Agent {agent_id} not found"}
        
        try:
            protocol = self.protocol_types.get(agent_id, "unknown")
            session = self.agent_sessions.get(agent_id, {})
            
            return {
                "agent_id": agent_id,
                "protocol": protocol,
                "meta_protocol": "llm_routed",
                "status": "active",
                "session": session
            }
            
        except Exception as e:
            return {"error": str(e)}

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics from Meta Protocol."""
        try:
            # Count agents by protocol
            protocol_distribution = {}
            for protocol in self.protocol_types.values():
                protocol_distribution[protocol] = protocol_distribution.get(protocol, 0) + 1
            
            # Get health status of all agents
            health_status = {}
            for agent_id, agent in self.protocol_agents.items():
                health_status[agent_id] = await self.health_check_agent(agent, agent_id)
            
            return {
                "protocol_distribution": protocol_distribution,
                "health_status": health_status,
                "total_agents": len(self.protocol_agents),
                "llm_routing_enabled": self.llm_router.llm_client is not None
            }
            
        except Exception as e:
            return {"error": str(e)}

    def get_results_paths(self) -> Dict[str, str]:
        """Get result file paths for Meta Protocol."""
        base_path = Path(self.config.get("results", {}).get("base_path", "results"))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        return {
            "results_file": str(base_path / f"meta_protocol_results_{timestamp}.json"),
            "detailed_results_file": str(base_path / f"meta_protocol_detailed_{timestamp}.json"),
            "metrics_file": str(base_path / f"meta_protocol_metrics_{timestamp}.json")
        }

    def _get_current_phase(self) -> str:
        """Get current phase for proper task classification."""
        if not self.metrics_collector:
            return "normal"
            
        if not hasattr(self.metrics_collector, 'fault_injection_time') or self.metrics_collector.fault_injection_time is None:
            return "normal"
        elif not hasattr(self.metrics_collector, 'steady_state_time') or self.metrics_collector.steady_state_time is None:
            return "recovery"
        else:
            return "post_fault"

    async def _reconnect_agent(self, agent_id: str) -> None:
        """Reconnect a killed Meta Protocol agent with full cross-protocol restoration."""
        if agent_id not in self.killed_agents:
            return

        try:
            print(f"🔄 [META] Reconnecting agent {agent_id} with cross-protocol support...")

            # CRITICAL: 先从 IntelligentNetwork 中注销被kill的agent
            if agent_id in self.meta_agents:
                try:
                    await self.intelligent_network.network.unregister_agent(agent_id)
                    print(f"🗑️  Unregistered killed agent {agent_id} from IntelligentNetwork")
                except Exception as e:
                    print(f"⚠️  Failed to unregister {agent_id}: {e}")

            # 重新创建（create_agent 内部会把 BaseAgent 注册回 intelligent_network）
            import socket
            def _free_port():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', 0))
                    return s.getsockname()[1]

            port = _free_port()
            # 注意：Meta 跑法的 create_agent 并不依赖 executor，这里传 None
            new_agent = await self.create_agent(agent_id, "127.0.0.1", port, executor=None)

            # 维护本地字典
            self.agents[agent_id] = new_agent
            self.killed_agents.discard(agent_id)

            # 🔗 把重连的 BaseAgent 与其它 BaseAgent 在 intelligent_network 上建立双向连边
            for other_id in self.meta_agents.keys():
                if other_id == agent_id:
                    continue
                await self.intelligent_network.network.connect_agents(agent_id, other_id)
                await self.intelligent_network.network.connect_agents(other_id, agent_id)

            # ♻️ 重装跨协议适配器（双向）
            await self._reinstall_outbound_adapters_for_agent(agent_id)

            print(f"✅ [META] Agent {agent_id} successfully reconnected & reattached to routing graph!")

        except Exception as e:
            print(f"❌ [META] Failed to reconnect agent {agent_id}: {e}")
            import traceback; traceback.print_exc()

    async def _reinstall_outbound_adapters_for_agent(self, reconnected_agent_id: str):
        """Reinstall outbound adapters for a reconnected agent."""
        try:
            print(f"🔗 Reinstalling outbound adapters for reconnected agent {reconnected_agent_id}...")
            
            if reconnected_agent_id not in self.meta_agents:
                print(f"⚠️  Agent {reconnected_agent_id} not found in meta_agents")
                return
            
            # Import all adapters
            from src.agent_adapters.a2a_adapter import A2AAdapter
            from src.agent_adapters.acp_adapter import ACPAdapter
            from src.agent_adapters.agora_adapter import AgoraClientAdapter
            from src.agent_adapters.anp_adapter import ANPAdapter
            
            src_base_agent = self.meta_agents[reconnected_agent_id]
            src_protocol = self.protocol_types[reconnected_agent_id]
            
            # Install adapters to all OTHER agents
            for dst_id, dst_base_agent in self.meta_agents.items():
                if dst_id == reconnected_agent_id:
                    continue  # Skip self
                
                # Skip if destination agent is also killed
                if dst_id in self.killed_agents:
                    continue
                
                dst_protocol = self.protocol_types[dst_id]
                dst_url = dst_base_agent.get_listening_address()
                
                # Fix URL for local connections
                if "0.0.0.0" in dst_url:
                    dst_url = dst_url.replace("0.0.0.0", "127.0.0.1")
                
                try:
                    # Create appropriate adapter based on destination protocol
                    if dst_protocol == "a2a":
                        adapter = A2AAdapter(
                            httpx_client=src_base_agent._httpx_client,
                            base_url=dst_url
                        )
                    elif dst_protocol == "acp":
                        adapter = ACPAdapter(
                            httpx_client=src_base_agent._httpx_client,
                            base_url=dst_url,
                            agent_id=dst_id
                        )
                    elif dst_protocol == "agora":
                        adapter = AgoraClientAdapter(
                            httpx_client=src_base_agent._httpx_client,
                            toolformer=None,
                            target_url=dst_url,
                            agent_id=dst_id
                        )
                    elif dst_protocol == "anp":
                        # Try DID-based ANP adapter first
                        import os
                        if os.getenv("ANP_DID_SERVICE_URL") and os.getenv("ANP_DID_API_KEY"):
                            adapter = await self._create_anp_adapter(src_base_agent, dst_base_agent, dst_id)
                        else:
                            # Skip ANP agents without DID (avoid protocol mismatch)
                            print(f"⚠️  ANP DID missing for {dst_id}; skipping outbound adapter (dev-only)")
                            continue
                    else:
                        continue
                    
                    # Initialize and install adapter
                    await adapter.initialize()
                    src_base_agent.add_outbound_adapter(dst_id, adapter)
                    
                    print(f"🔗 Restored {src_protocol.upper()} → {dst_protocol.upper()} adapter: {reconnected_agent_id} → {dst_id}")
                    
                except Exception as e:
                    print(f"⚠️  Failed to restore adapter {reconnected_agent_id} → {dst_id}: {e}")
                    continue
            
            # Also install adapters FROM other agents TO this reconnected agent
            for src_id, src_base_agent in self.meta_agents.items():
                if src_id == reconnected_agent_id or src_id in self.killed_agents:
                    continue
                
                dst_base_agent = self.meta_agents[reconnected_agent_id]
                dst_protocol = self.protocol_types[reconnected_agent_id]
                dst_url = dst_base_agent.get_listening_address()
                
                if "0.0.0.0" in dst_url:
                    dst_url = dst_url.replace("0.0.0.0", "127.0.0.1")
                
                try:
                    src_protocol = self.protocol_types[src_id]
                    
                    # Create appropriate adapter
                    if dst_protocol == "a2a":
                        adapter = A2AAdapter(
                            httpx_client=src_base_agent._httpx_client,
                            base_url=dst_url
                        )
                    elif dst_protocol == "acp":
                        adapter = ACPAdapter(
                            httpx_client=src_base_agent._httpx_client,
                            base_url=dst_url,
                            agent_id=reconnected_agent_id
                        )
                    elif dst_protocol == "agora":
                        adapter = AgoraClientAdapter(
                            httpx_client=src_base_agent._httpx_client,
                            toolformer=None,
                            target_url=dst_url,
                            agent_id=reconnected_agent_id
                        )
                    elif dst_protocol == "anp":
                        adapter = await self._create_anp_adapter(src_base_agent, dst_base_agent, reconnected_agent_id)
                    else:
                        continue
                    
                    # Initialize and install adapter
                    await adapter.initialize()
                    src_base_agent.add_outbound_adapter(reconnected_agent_id, adapter)
                    
                    print(f"🔗 Restored {src_protocol.upper()} → {dst_protocol.upper()} adapter: {src_id} → {reconnected_agent_id}")
                    
                except Exception as e:
                    print(f"⚠️  Failed to restore adapter {src_id} → {reconnected_agent_id}: {e}")
                    continue
            
            print(f"✅ Cross-protocol adapters restored for agent {reconnected_agent_id}")
            
        except Exception as e:
            print(f"❌ Failed to reinstall adapters for {reconnected_agent_id}: {e}")