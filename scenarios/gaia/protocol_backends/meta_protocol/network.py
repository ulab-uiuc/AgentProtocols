"""
Meta Protocol Network implementation for GAIA framework.
Provides intelligent protocol selection and cross-protocol communication management.
"""

import asyncio
import json
import time
import os
import logging
from typing import Any, Dict, Optional, List
import sys
from pathlib import Path

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.network import MeshNetwork
from core.schema import Message, ExecutionStatus, Colors
from protocol_backends.meta_protocol.agent import MetaProtocolAgent

# Import src/core for intelligent network management
# Use absolute path approach to avoid relative import issues
current_file = Path(__file__).resolve()

# Find agent_network root by looking for src directory
search_path = current_file.parent
while search_path.parent != search_path:  # Not at filesystem root
    if (search_path / "src" / "core" / "base_agent.py").exists():
        agent_network_root = search_path
        break
    search_path = search_path.parent
else:
    raise RuntimeError("Cannot find agent_network root directory")

src_path = agent_network_root / "src"

# Ensure paths are in sys.path
if str(agent_network_root) not in sys.path:
    sys.path.insert(0, str(agent_network_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Change working directory to agent_network root to fix relative imports
original_cwd = os.getcwd()
os.chdir(str(agent_network_root))

try:
    from src.core.intelligent_network_manager import IntelligentNetworkManager, RouterType
    from src.core.network import AgentNetwork
    print(f"[MetaProtocolNetwork] Meta core imported successfully from {src_path}")
except ImportError as e:
    print(f"[MetaProtocolNetwork] Meta core import failed: {e}")
    print(f"[MetaProtocolNetwork] Agent network root: {agent_network_root}")
    print(f"[MetaProtocolNetwork] Current working directory: {os.getcwd()}")
    raise ImportError(f"[META_PROTOCOL]: Failed to import IntelligentNetworkManager from src.core - {e}")
finally:
    # Restore original working directory
    os.chdir(original_cwd)

# Setup logger
logger = logging.getLogger(__name__)


class MetaProtocolCommBackend:
    """Meta protocol communication backend for GAIA framework."""

    def __init__(self) -> None:
        self._endpoints: Dict[str, str] = {}  # agent_id -> endpoint uri
        self._protocol_networks: Dict[str, Any] = {}  # protocol -> network instance
        self._agent_protocols: Dict[str, str] = {}  # agent_id -> selected protocol
        
        # Initialize intelligent network manager
        self._intelligent_network = IntelligentNetworkManager(
            router_type=RouterType.LLM_BASED
        )

        # Initialize simple transmission metrics to avoid AttributeError during deliver()
        self.pkt_cnt = 0
        self.bytes_tx = 0
        self.bytes_rx = 0
        self.header_overhead = 0

    async def register_endpoint(self, agent_id: str, address: str) -> None:
        """Register meta protocol agent endpoint."""
        self._endpoints[agent_id] = address
        logger.info(f"{Colors.CYAN}[MetaProtocol] Registered agent: {agent_id} @ {address}{Colors.RESET}")

    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send message via meta protocol with intelligent routing."""
        try:
            # Get destination endpoint
            endpoint = self._endpoints.get(dst_id)
            if not endpoint:
                raise RuntimeError(f"Unknown destination agent: {dst_id}")
            
            # Determine optimal protocol for this communication
            optimal_protocol = await self._select_communication_protocol(src_id, dst_id, payload)
            
            # Convert payload to meta protocol message format
            meta_message = self._to_meta_message(payload, optimal_protocol)
            
            # Send HTTP request to agent endpoint (following GAIA pattern)
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{endpoint}/message",
                    json=meta_message,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                raw_response = response.json()
                text_content = self._extract_text_from_response(raw_response)
                
                return {
                    "raw": raw_response,
                    "text": text_content,
                    "protocol_used": optimal_protocol,
                    "meta_routing": True
                }
            
        except Exception as e:
            error_msg = f"[MetaProtocol] Send failed {src_id} -> {dst_id}: {e}"
            logger.error(error_msg)
            return {
                "error": str(e),
                "protocol_used": "error",
                "meta_routing": False,
                "raw": None,
                "text": ""
            }
    
    def _to_meta_message(self, payload: Dict[str, Any], protocol: str) -> Dict[str, Any]:
        """Convert payload to meta protocol message format."""
        return {
            "type": payload.get("type", "meta_message"),
            "content": payload.get("content") or json.dumps(payload),
            "protocol": protocol,
            "meta_routing": True,
            "timestamp": time.time()
        }
    
    def _extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """Extract text content from meta protocol response."""
        # Try various response formats
        if "text" in response:
            return response["text"]
        elif "content" in response:
            return response["content"]
        elif "result" in response:
            return str(response["result"])
        else:
            return json.dumps(response)

    async def _select_communication_protocol(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> str:
        """Select optimal protocol for communication based on message characteristics."""
        # Analyze message characteristics
        message_size = len(json.dumps(payload))
        message_type = payload.get("type", "unknown")
        priority = payload.get("priority", "normal")
        
        # For GAIA integration, use simple heuristics
        # In fail_storm_recovery, this would use LLM-based selection
        
        if message_size > 10000:  # Large messages
            return "a2a"  # Fast for large data
        elif priority == "high":
            return "agora"  # Reliable for high priority
        elif message_type == "task":
            return "anp"  # Good for task distribution
        else:
            return "acp"  # Default for general communication

    async def _send_via_intelligent_network(self, src_id: str, dst_id: str, 
                                          payload: Dict[str, Any], protocol: str) -> str:
        """Send message using BaseAgent routing (following fail_storm_recovery pattern)."""
        try:
            # Ensure router_agent is resolved from instance attribute if available
            router_agent = getattr(self, '_router_agent', None)
            # Use BaseAgent.send() method like fail_storm_recovery does
            print(f"[MetaProtocol] Routing {src_id} -> {dst_id} via {protocol}")
            
            # For GAIA, use direct communication instead of BaseAgent routing
            # The routing is handled at the network level, not at the backend level
            print(f"[MetaProtocol] Using direct communication for {src_id} -> {dst_id} via {protocol}")
            
            if router_agent and hasattr(router_agent, '_base_agent') and router_agent._base_agent:
                # Use BaseAgent.send() for routing
                response = await router_agent._base_agent.send(dst_id, payload)
                return f"Routed via {protocol}: {response}"
            else:
                # No BaseAgent available, use direct communication
                return await self._send_direct(src_id, dst_id, payload, protocol)
            
        except Exception as e:
            print(f"[MetaProtocol] BaseAgent routing failed: {e}")
            return await self._send_direct(src_id, dst_id, payload, protocol)

    async def _send_direct(self, src_id: str, dst_id: str, 
                          payload: Dict[str, Any], protocol: str) -> str:
        """Direct communication fallback."""
        # For GAIA, this would typically delegate to the underlying protocol network
        # For now, return a mock response indicating the protocol used
        return f"[MetaProtocol] Message sent via {protocol} from {src_id} to {dst_id}: {payload.get('content', str(payload))}"

    async def health_check(self, agent_id: str) -> bool:
        """Check health of meta protocol agent."""
        try:
            endpoint = self._endpoints.get(agent_id)
            if not endpoint:
                return False
            
            # For GAIA integration, we'll assume agents are healthy if registered
            # In a full implementation, this would check all underlying protocols
            return True
            
        except Exception:
            return False

    async def close(self) -> None:
        """Close meta protocol communication backend."""
        try:
            if self._intelligent_network:
                await self._intelligent_network.cleanup()
            
            # Cleanup protocol networks
            for protocol, network in self._protocol_networks.items():
                try:
                    if hasattr(network, 'close'):
                        await network.close()
                except Exception as e:
                    print(f"[MetaProtocol] Error closing {protocol} network: {e}")
            
            print("[MetaProtocol] Communication backend closed")
            
        except Exception as e:
            print(f"[MetaProtocol] Error closing backend: {e}")


class MetaProtocolNetwork(MeshNetwork):
    """
    Meta Protocol Network for GAIA framework.
    
    Provides intelligent protocol selection and cross-protocol communication
    management similar to fail_storm_recovery and streaming_queue implementations.
    """
    
    def __init__(self, config: Dict[str, Any], task_id: str):
        # Ensure config includes a protocol field so parent MeshNetwork will
        # construct workspace/log paths correctly (avoid defaulting to 'general').
        enhanced_config = {**config, "task_id": task_id, "protocol": config.get("protocol", "meta_protocol")}
        super().__init__(enhanced_config)
        
        self._comm_backend = MetaProtocolCommBackend()
        self._meta_config = config.get("meta_protocol", {})
        self._available_protocols = self._meta_config.get("available_protocols", ["a2a", "acp", "agora", "anp"])
        self._router_config = self._meta_config.get("router", {})
        
        # Performance tracking
        self._protocol_metrics: Dict[str, Dict[str, Any]] = {}
        self._network_start_time: Optional[float] = None
        
        # Store for LLM routing during start()
        self._pending_agents_config: List[Dict[str, Any]] = []
        
        print(f"[MetaProtocolNetwork] Initialized with protocols: {self._available_protocols}")

    def _get_agent_by_id(self, dst: int):
        """Resolve agent by numeric or string ID when self.agents may be a list or a dict.
        More robust than the previous placeholder implementation so downstream logic
        (deliver -> result_callback) can find the target agent instance.
        """
        # If agents is a dict-like container
        if hasattr(self, "agents") and isinstance(self.agents, dict):
            # direct key match
            if dst in self.agents:
                return self.agents[dst]
            str_key = str(dst)
            if str_key in self.agents:
                return self.agents[str_key]
            # scan values
            for a in self.agents.values():
                try:
                    if getattr(a, "id", None) == dst or str(getattr(a, "id", "")) == str_key:
                        return a
                except Exception:
                    continue
        # If agents is a list-like container
        if hasattr(self, "agents") and isinstance(self.agents, list):
            for a in self.agents:
                try:
                    if getattr(a, "id", None) == dst or str(getattr(a, "id", "")) == str(dst):
                        return a
                except Exception:
                    continue
        # Final fallback: attempt to match common agent_id/agent_id field names
        for a in getattr(self, "agents", []) or []:
            try:
                if getattr(a, "agent_id", None) == dst or str(getattr(a, "agent_id", "")) == str(dst):
                    return a
            except Exception:
                continue
        return None

    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """
        Deliver message via BaseAgent.send(...) using a source agent (not the destination itself).
        Ensure that the destination agent's memory and result_callback are updated so
        the MeshNetwork can capture messages into NetworkMemoryPool.step_executions.
        """
        try:
            # Resolve destination agent
            dst_agent = self._get_agent_by_id(dst)
            if not dst_agent:
                raise RuntimeError(f"Destination agent {dst} not found")
            dst_base = getattr(dst_agent, "_base_agent", None)
            if dst_base is None:
                raise RuntimeError(f"Agent {dst_agent.name} has no BaseAgent bound")
            dst_base_id = getattr(dst_base, "agent_id", str(dst))

            # Attempt to resolve a source agent; prefer explicit src info in message
            src_agent = None
            src_hint = None
            if isinstance(msg, dict):
                src_hint = (msg.get("src_id") or (msg.get("meta") or {}).get("src_id") or msg.get("from") or msg.get("src"))
            if src_hint is not None:
                # Try match by BaseAgent agent_id or by numeric id
                for a in (self.agents if hasattr(self, 'agents') else []):
                    try:
                        if getattr(a, '_base_agent', None) and getattr(a, '_base_agent').agent_id == src_hint:
                            src_agent = a
                            break
                        if getattr(a, 'id', None) == src_hint or str(getattr(a, 'id', '')) == str(src_hint):
                            src_agent = a
                            break
                    except Exception:
                        continue

            # If still unknown, pick the first agent that is not the destination
            if src_agent is None:
                for a in (self.agents if hasattr(self, 'agents') else []):
                    try:
                        if a is not dst_agent:
                            src_agent = a
                            break
                    except Exception:
                        continue

            # Fallback: use destination itself only if no better choice
            if src_agent is None:
                src_agent = dst_agent

            src_base = getattr(src_agent, "_base_agent", None)
            if src_base is None:
                raise RuntimeError(f"Agent {src_agent.name} has no BaseAgent bound")

            # Build a robust payload envelope
            content = msg.get("content")
            if content is None:
                content = (msg.get("text") or msg.get("result") or json.dumps(msg, ensure_ascii=False))

            payload = {
                "message": {
                    "type": msg.get("type", "meta_message"),
                    "content": content
                },
                "meta": {
                    "src_id": getattr(src_base, "agent_id", f"agent:{getattr(src_agent,'id', '')}"),
                    "task_id": getattr(self, "task_id", None),
                    "assigned_protocol": getattr(getattr(dst_agent, "config", {}), "get", lambda *_: None)("assigned_protocol")
                }
            }

            # Send from SOURCE BaseAgent to DESTINATION BaseAgent (id must match adapter registration)
            raw_response = await src_base.send(dst_base_id, payload)

            # Normalize response and extract text
            text_content = self._extract_text_from_response(raw_response) if raw_response is not None else ""

            # Ensure messages are appended to destination agent's memory so network can capture them
            try:
                # Add original user message
                if hasattr(dst_agent, 'memory') and getattr(dst_agent, 'memory') is not None:
                    try:
                        dst_agent.memory.add_message(Message.user_message(content))
                    except Exception:
                        pass
                    try:
                        dst_agent.memory.add_message(Message.assistant_message(text_content))
                    except Exception:
                        pass
            except Exception:
                pass

            # Trigger destination agent's result_callback with the response so MeshNetwork.capture_result is called
            complete_message = {
                "agent_id": getattr(dst_agent, 'id', None),
                "agent_name": getattr(dst_agent, 'name', None),
                "sender_id": getattr(src_agent, 'id', None),
                "message_type": payload.get('message', {}).get('type'),
                "original_content": payload.get('message', {}).get('content'),
                "assistant_response": text_content,
                "assistant_msg": Message.assistant_message(text_content),
                "processing_steps": getattr(dst_agent, 'current_step', None),
                "status": "completed"
            }

            if hasattr(dst_agent, 'result_callback') and dst_agent.result_callback:
                try:
                    if asyncio.iscoroutinefunction(dst_agent.result_callback):
                        asyncio.create_task(dst_agent.result_callback(complete_message))
                    else:
                        dst_agent.result_callback(complete_message)
                except Exception as e:
                    print(f"[MetaProtocol] Error calling result_callback for {dst_agent}: {e}")

            # Metrics / logging
            self.pkt_cnt += 1
            try:
                self.bytes_tx += len(json.dumps(msg).encode("utf-8"))
            except Exception:
                pass
            logger.info(f"[MetaProtocol] Delivered {getattr(src_agent,'name','?')} -> {dst_agent.name} via BaseAgent.send(...)")

            return {
                "raw": raw_response,
                "text": text_content,
                "protocol_used": payload.get('meta', {}).get('assigned_protocol', 'unknown'),
                "meta_routing": True
            }

        except Exception as e:
            error_msg = f"[MetaProtocol] Delivery failed to {dst}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def register_agents_from_config(self) -> None:
        """Register meta protocol agents from configuration with LLM-based protocol assignment."""
        print(f"[MetaProtocolNetwork] Registering agents for task {self.task_id}")
        
        # Debug: print full config structure
        print(f"[MetaProtocolNetwork] DEBUG: Config keys: {list(self.config.keys())}")
        agents_config = self.config.get("agents", [])
        print(f"[MetaProtocolNetwork] DEBUG: Found {len(agents_config)} agents in config")
        
        if not agents_config:
            print(f"[MetaProtocolNetwork] WARNING: No agents found in config!")
            print(f"[MetaProtocolNetwork] Config structure: {self.config}")
            return
        
        # Store agents config for later LLM routing during start()
        self._pending_agents_config = agents_config
        print(f"[MetaProtocolNetwork] Stored {len(agents_config)} agents for LLM protocol assignment during start()")
    
    async def _assign_protocols_with_llm_router(self, agents_config: List[Dict[str, Any]]) -> Dict[str, str]:
        """Use GAIA-specific LLM router to assign protocols for each agent based on task characteristics."""
        try:
            # Import GAIA-specific LLM router
            from .llm_router import GAIALLMRouter
            
            # Initialize GAIA LLM router
            llm_router = GAIALLMRouter()
            
            # Extract task information for LLM analysis
            task_analysis = self.config.get("task_analysis", {})
            task_type = task_analysis.get("task_type", "unknown")
            complexity = task_analysis.get("complexity", "medium")
            required_tools = task_analysis.get("required_tools", [])
            domain_areas = task_analysis.get("domain_areas", [])
            
            # Create task description for LLM router
            task_description = {
                "question": f"GAIA task {self.task_id}",
                "context": f"Task type: {task_type}, Complexity: {complexity}, Tools: {required_tools}, Domains: {domain_areas}",
                "metadata": {
                    "type": "gaia_task",
                    "agents_count": len(agents_config),
                    "tools_required": required_tools,
                    "complexity": complexity,
                    "domain_areas": domain_areas
                }
            }
            
            print(f"[MetaProtocolNetwork] 🧠 Calling LLM router for {len(agents_config)} agents")
            print(f"[MetaProtocolNetwork] 📋 Task analysis: {task_analysis}")
            
            # Call GAIA LLM router to get protocol assignments
            routing_decision = await llm_router.route_gaia_task(
                task_analysis=task_analysis,
                agents_config=agents_config
            )
            
            print(f"[MetaProtocolNetwork] 🎯 LLM routing decision:")
            print(f"[MetaProtocolNetwork]   Selected protocols: {routing_decision.selected_protocols}")
            print(f"[MetaProtocolNetwork]   Agent assignments: {routing_decision.agent_assignments}")
            print(f"[MetaProtocolNetwork]   Reasoning: {routing_decision.reasoning}")
            print(f"[MetaProtocolNetwork]   Confidence: {routing_decision.confidence}")
            
            # Convert agent assignments to match our agent IDs
            protocol_assignments = {}
            for i, agent_config in enumerate(agents_config):
                agent_id = str(agent_config["id"])
                
                # Try to get assignment from LLM decision
                assigned_protocol = None
                
                # Check direct agent ID mapping
                if agent_id in routing_decision.agent_assignments:
                    assigned_protocol = routing_decision.agent_assignments[agent_id]
                # Check agent_{i} format
                elif f"agent_{i}" in routing_decision.agent_assignments:
                    assigned_protocol = routing_decision.agent_assignments[f"agent_{i}"]
                # Check agent{i} format (no underscore)
                elif f"agent{i}" in routing_decision.agent_assignments:
                    assigned_protocol = routing_decision.agent_assignments[f"agent{i}"]
                
                # Fallback to round-robin from selected protocols
                if not assigned_protocol and routing_decision.selected_protocols:
                    assigned_protocol = routing_decision.selected_protocols[i % len(routing_decision.selected_protocols)]
                
                # Ultimate fallback
                if not assigned_protocol:
                    assigned_protocol = "a2a"
                
                protocol_assignments[agent_id] = assigned_protocol
                print(f"[MetaProtocolNetwork]   Agent {agent_config['name']} (ID {agent_id}) → {assigned_protocol}")
            
            
            return protocol_assignments
            
        except Exception as e:
            print(f"[MetaProtocolNetwork] ❌ LLM router failed: {e}")
            print(f"[MetaProtocolNetwork] 🔄 Using fallback protocol assignment")
            
            # Fallback: assign protocols based on fail_storm_recovery performance data
            fallback_protocols = ["a2a", "agora", "anp", "acp"]  # Ordered by recovery performance
            protocol_assignments = {}
            
            for i, agent_config in enumerate(agents_config):
                agent_id = str(agent_config["id"])
                assigned_protocol = fallback_protocols[i % len(fallback_protocols)]
                protocol_assignments[agent_id] = assigned_protocol
                print(f"[MetaProtocolNetwork]   Fallback: Agent {agent_config['name']} (ID {agent_id}) → {assigned_protocol}")
            
            
            return protocol_assignments
    
    async def _create_protocol_agent(self, agent_config: Dict[str, Any], assigned_protocol: str):
        """Create actual protocol agent based on LLM assignment."""
        try:
            agent_id = f"{assigned_protocol}_{agent_config['name']}_{agent_config['id']}"
            host = "0.0.0.0"
            port = agent_config["port"]
            
            if assigned_protocol == "a2a":
                from .a2a_agent import create_a2a_meta_worker
                meta_agent = await create_a2a_meta_worker(
                    agent_id=agent_id,
                    config=agent_config,
                    host=host,
                    port=port
                )
                return meta_agent.base_agent

            elif assigned_protocol == "acp":
                from .acp_agent import create_acp_meta_worker
                meta_agent = await create_acp_meta_worker(
                    agent_id=agent_id,
                    config=agent_config,
                    host=host,
                    port=port
                )
                return meta_agent.base_agent

            elif assigned_protocol == "agora":
                from .agora_agent import create_agora_meta_worker
                meta_agent = await create_agora_meta_worker(
                    agent_id=agent_id,
                    config=agent_config,
                    host=host,
                    port=port
                )
                return meta_agent.base_agent

            elif assigned_protocol == "anp":
                from .anp_agent import create_anp_meta_worker
                meta_agent = await create_anp_meta_worker(
                    agent_id=agent_id,
                    config=agent_config,
                    host=host,
                    port=port
                )
                return meta_agent.base_agent

            else:
                raise ValueError(f"Unsupported protocol: {assigned_protocol}")

        except Exception as e:
            print(f"[MetaProtocolNetwork] Failed to create {assigned_protocol} agent: {e}")
            raise
    
    async def _install_outbound_adapters(self) -> None:
        """Install outbound adapters for cross-protocol communication."""
        try:
            print(f"[MetaProtocolNetwork] Installing outbound adapters for {len(self.agents)} agents")
            
            # Install adapters for each agent to communicate with all agents (including self)
            for src_agent in self.agents:
                if not hasattr(src_agent, '_base_agent'):
                    print(f"[MetaProtocolNetwork] Agent {src_agent.name} has no BaseAgent")
                    continue
                    
                src_base_agent = src_agent._base_agent
                
                # Install adapters to all agents (including self for loopback)
                for dst_agent in self.agents:
                    dst_protocol = getattr(dst_agent, 'config', {}).get('assigned_protocol', 'a2a')
                    dst_base_agent = getattr(dst_agent, '_base_agent', None)
                    
                    if not dst_base_agent:
                        continue
                    
                    dst_url = dst_base_agent.get_listening_address()
                    if "0.0.0.0" in dst_url:
                        dst_url = dst_url.replace("0.0.0.0", "127.0.0.1")
                    
                    # Use BaseAgent's actual agent_id for adapter registration
                    dst_base_agent_id = getattr(dst_base_agent, "agent_id", str(dst_agent.id))
                    dst_id = dst_base_agent_id
                    
                    try:
                        # Create appropriate adapter based on destination protocol
                        if dst_protocol == "a2a":
                            from src.agent_adapters.a2a_adapter import A2AAdapter
                            adapter = A2AAdapter(
                                httpx_client=src_base_agent._httpx_client,
                                base_url=dst_url
                            )
                        elif dst_protocol == "acp":
                            from src.agent_adapters.acp_adapter import ACPAdapter
                            adapter = ACPAdapter(
                                httpx_client=src_base_agent._httpx_client,
                                base_url=dst_url,
                                agent_id=dst_id
                            )
                        elif dst_protocol == "agora":
                            from src.agent_adapters.agora_adapter import AgoraClientAdapter
                            adapter = AgoraClientAdapter(
                                httpx_client=src_base_agent._httpx_client,
                                toolformer=None,
                                target_url=dst_url,
                                agent_id=dst_id
                            )
                        elif dst_protocol == "anp":
                            from src.agent_adapters.anp_adapter import ANPAdapter
                            # ANP requires DID configuration
                            local_did_info = {
                                "did": f"did:gaia:{src_agent.name}",
                                "private_key_pem": "dummy_key_for_gaia"
                            }
                            adapter = ANPAdapter(
                                httpx_client=src_base_agent._httpx_client,
                                target_did=f"did:gaia:{dst_agent.name}",
                                local_did_info=local_did_info,
                                host_domain="127.0.0.1",
                                host_port=str(dst_agent.port),
                                host_ws_path="/ws"
                            )
                        else:
                            print(f"[MetaProtocolNetwork] Unknown protocol: {dst_protocol}")
                            continue
                        
                        # Initialize and install adapter
                        await adapter.initialize()
                        src_base_agent.add_outbound_adapter(dst_id, adapter)
                        
                        if src_agent.id == dst_agent.id:
                            print(f"[MetaProtocolNetwork] ✅ Installed {dst_protocol} loopback adapter: {src_agent.name} -> self")
                        else:
                            print(f"[MetaProtocolNetwork] ✅ Installed {dst_protocol} adapter: {src_agent.name} -> {dst_agent.name}")
                        
                    except Exception as e:
                        print(f"[MetaProtocolNetwork] ❌ Failed to install adapter from {src_agent.name} to {dst_agent.name}: {e}")
            
            print(f"[MetaProtocolNetwork] Outbound adapters installation completed")
            
        except Exception as e:
            print(f"[MetaProtocolNetwork] Failed to install outbound adapters: {e}")

    async def start(self) -> None:
        """Start meta protocol network and all agents."""
        self._network_start_time = time.time()
        
        print(f"[MetaProtocolNetwork] Starting meta protocol network for task {self.task_id}")
        
        # Perform LLM-based protocol assignment and create agents
        if hasattr(self, '_pending_agents_config') and self._pending_agents_config:
            await self._create_agents_with_llm_routing()
        
        # Install outbound adapters for cross-protocol communication
        await self._install_outbound_adapters()
        
        # Start all agents
        await super().start()
        
        # Initialize protocol-specific resources
        await self._initialize_protocol_resources()
        
        print(f"[MetaProtocolNetwork] Meta protocol network started with {len(self.agents)} agents")
    
    async def _create_agents_with_llm_routing(self):
        """Create agents with LLM-based protocol assignment."""
        agents_config = self._pending_agents_config
        
        # Use LLM router to assign protocols for this specific task
        protocol_assignments = await self._assign_protocols_with_llm_router(agents_config)

        # Persist assignments on the network instance for later inspection (e.g., runner output)
        try:
            self.protocol_assignments = protocol_assignments
        except Exception:
            # Defensive: if attribute cannot be set for some reason, continue without failing
            pass
        
        # Resolve global agent prompts map if any
        prompts_map = self.config.get("agent_prompts", {}) or {}

        for i, agent_config in enumerate(agents_config):
            try:
                agent_id = str(agent_config["id"])
                assigned_protocol = protocol_assignments.get(agent_id, "a2a")  # Default fallback

                print(f"[MetaProtocolNetwork] Creating agent {i+1}/{len(agents_config)}: {agent_config}")
                print(f"[MetaProtocolNetwork] 🎯 LLM assigned protocol '{assigned_protocol}' to agent {agent_config['name']}")
                
                # Inject agent_prompt into per-agent config
                agent_prompt = prompts_map.get(agent_config["name"]) or agent_config.get("role") or ""
                cfg = {
                    **agent_config,
                    "agent_prompt": agent_prompt,
                    "assigned_protocol": assigned_protocol,
                    "task_id": self.task_id
                }
                
                # Create actual protocol agent based on LLM assignment
                protocol_agent = await self._create_protocol_agent(
                    agent_config=cfg,          # use cfg here
                    assigned_protocol=assigned_protocol
                )
                
                # Wrap with MetaProtocolAgent for GAIA compatibility
                meta_agent = MetaProtocolAgent(
                    node_id=agent_config["id"],
                    name=agent_config["name"],
                    tool=agent_config["tool"],
                    port=agent_config["port"],
                    config={
                        **cfg,  # use cfg so MetaProtocolAgent sees agent_prompt too
                        **self._meta_config
                    },
                    task_id=self.task_id
                )
                
                # Bind the actual protocol agent's BaseAgent
                meta_agent._base_agent = protocol_agent
                
                # Register meta agent
                self.register_agent(meta_agent)
                
                # Register endpoint with communication backend (for health check only)
                endpoint = f"http://localhost:{agent_config['port']}"
                asyncio.create_task(self._comm_backend.register_endpoint(str(agent_config['id']), endpoint))
                
                print(f"[MetaProtocolNetwork] ✅ Registered {meta_agent.name} (ID {meta_agent.id}) with protocol {assigned_protocol}")
                
            except Exception as e:
                print(f"[MetaProtocolNetwork] ❌ Failed to register agent {agent_config}: {e}")
                import traceback
                traceback.print_exc()

    async def _initialize_protocol_resources(self):
        """Initialize resources for all available protocols."""
        for protocol in self._available_protocols:
            try:
                # Initialize protocol-specific resources
                # This could include setting up protocol networks, connections, etc.
                self._protocol_metrics[protocol] = {
                    "messages_sent": 0,
                    "messages_received": 0,
                    "errors": 0,
                    "avg_latency": 0.0
                }
                
                print(f"[MetaProtocolNetwork] Initialized {protocol} resources")
                
            except Exception as e:
                print(f"[MetaProtocolNetwork] Failed to initialize {protocol}: {e}")

    async def stop(self) -> None:
        """Stop meta protocol network and cleanup resources."""
        try:
            # Generate final metrics report
            if self._network_start_time:
                total_runtime = time.time() - self._network_start_time
                print(f"[MetaProtocolNetwork] Total runtime: {total_runtime:.2f}s")
            
            # Cleanup communication backend
            await self._comm_backend.close()
            
            # Stop all agents
            await super().stop()
            
            print(f"[MetaProtocolNetwork] Meta protocol network stopped")
            
        except Exception as e:
            print(f"[MetaProtocolNetwork] Error stopping network: {e}")

    def _update_delivery_metrics(self, src_id: str, dst_id: str, result: Dict[str, Any]):
        """Update delivery metrics for performance tracking."""
        protocol_used = result.get("protocol_used", "unknown")
        
        if protocol_used in self._protocol_metrics:
            metrics = self._protocol_metrics[protocol_used]
            metrics["messages_sent"] += 1
            
            if "error" in result:
                metrics["errors"] += 1

    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status including meta protocol metrics."""
        base_status = super().get_network_status() if hasattr(super(), 'get_network_status') else {}
        
        meta_status = {
            "protocol_type": "meta_protocol",
            "available_protocols": self._available_protocols,
            "protocol_metrics": self._protocol_metrics,
            "agents_count": len(self.agents),
            "meta_core_available": True,
            "network_runtime": (time.time() - self._network_start_time) if self._network_start_time else 0.0
        }
        
        return {**base_status, "meta_protocol": meta_status}

    async def health_check(self) -> bool:
        """Check health of meta protocol network."""
        try:
            # Check communication backend
            if not self._comm_backend:
                return False
            
            # Check all agents
            # self.agents is a list in MeshNetwork; iterate directly
            for agent in (self.agents if isinstance(self.agents, list) else getattr(self.agents, 'values', lambda: [])()):
                try:
                    if not await self._comm_backend.health_check(str(agent.id)):
                        return False
                except Exception:
                    return False
            
            return True
            
        except Exception as e:
            print(f"[MetaProtocolNetwork] Health check failed: {e}")
            return False

    def _extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """Delegate extraction of text from a raw response to the comm backend when possible.
        This ensures MetaProtocolNetwork.deliver can normalize responses without duplicating logic.
        """
        try:
            if hasattr(self, '_comm_backend') and self._comm_backend:
                if hasattr(self._comm_backend, '_extract_text_from_response'):
                    return self._comm_backend._extract_text_from_response(response)
            # Fallback: inspect common keys
            if isinstance(response, dict):
                if 'text' in response:
                    return response['text']
                if 'content' in response:
                    return response['content']
                if 'result' in response:
                    return str(response['result'])
            return str(response)
        except Exception:
            try:
                return json.dumps(response)
            except Exception:
                return str(response)
