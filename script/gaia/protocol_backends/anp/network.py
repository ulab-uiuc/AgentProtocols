"""
ANP Network implementation for GAIA framework.
Implements broadcast and communication management for ANP protocol using original ANP-SDK.
"""

import asyncio
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add the parent directory to sys.path for imports  
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.network import MeshNetwork
from core.schema import Message, ExecutionStatus
from protocol_backends.anp.agent import ANPAgent
from core.schema import Colors

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Check ANP-SDK availability
try:
    # Import AgentConnect components to check availability
    from ....agentconnect_src.agent_connect.python.simple_node.simple_node import SimpleNode
    from ....agentconnect_src.agent_connect.python.authentication import DIDAllClient
    ANP_SDK_AVAILABLE = True
    print("✅ ANP-SDK components available")
except ImportError:
    ANP_SDK_AVAILABLE = False
    print("⚠️ ANP-SDK components not available")


class ANPCommBackend:
    """ANP protocol communication backend for GAIA framework."""

    def __init__(self, **kwargs):
        self._endpoints = {}  # agent_id -> DID mapping
        self._agent_sessions = {}  # agent_id -> ANPAgent mapping
        
    async def register_endpoint(self, agent_id: str, agent: 'ANPAgent') -> None:
        """Register ANP agent and its DID endpoint."""
        did = agent.local_did_info.get('did', f'did:all:agent_{agent_id}')
        self._endpoints[agent_id] = did
        self._agent_sessions[agent_id] = agent
        logger.info(f"{Colors.CYAN}Registered ANP agent: {agent_id} @ {did}{Colors.RESET}")
        
    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """Send message via ANP protocol using native ANP-SDK."""
        dst_agent = self._agent_sessions.get(dst_id)
        if not dst_agent:
            raise RuntimeError(f"Unknown destination agent: {dst_id}")
            
        try:
            # Send message using ANP agent's native send_msg method
            src_agent = self._agent_sessions.get(src_id)
            if src_agent:
                await src_agent.send_msg(int(dst_id), payload)
                
                # Wait for response (simplified for testing)
                response = await dst_agent.recv_msg(timeout=5.0)
                
                if response:
                    return {
                        "raw": response,
                        "text": self._extract_text_from_anp_response(response)
                    }
                else:
                    return {"raw": None, "text": "No response"}
                    
        except Exception as e:
            print(f"[ANPCommBackend] Send failed {src_id} -> {dst_id}: {e}")
            return {"raw": None, "text": f"ANP Error: {e}"}
            
    async def health_check(self, agent_id: str) -> bool:
        """Check ANP agent health."""
        agent = self._agent_sessions.get(agent_id)
        if not agent:
            return False
            
        try:
            return agent._connected and agent.anp_initialized
        except Exception:
            return False
            
    async def close(self) -> None:
        """Close ANP communication backend."""
        for agent in list(self._agent_sessions.values()):
            try:
                await agent.stop()
            except Exception:
                pass
        self._agent_sessions.clear()
        self._endpoints.clear()
        
    def _extract_text_from_anp_response(self, response: Dict[str, Any]) -> str:
        """Extract text content from ANP response."""
        try:
            if isinstance(response, dict):
                if "content" in response:
                    return str(response["content"])
                elif "body" in response:
                    return str(response["body"])
                elif "text" in response:
                    return str(response["text"])
                else:
                    return str(response)
            return str(response) if response else ""
        except Exception:
            return ""


class ANPNetwork(MeshNetwork):
    """
    ANP Network implementation with broadcast and enhanced communication capabilities.
    Uses native ANP-SDK components for authentic Agent Network Protocol communication.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Ensure protocol is set for downstream workspace pathing
        if isinstance(config, dict):
            config = {**config, "protocol": config.get("protocol", "anp")}
        super().__init__(config=config)
        
        # Store original runner config if available (for model config)
        self._runner_config = getattr(self, '_original_config', None)
        
        self.comm_backend = ANPCommBackend()
        
        # ANP-specific configuration
        self.base_port = config.get("network", {}).get("port_range", {}).get("start", 9000)
        self.host_domain = config.get("network", {}).get("host", "127.0.0.1")
        
        # Check ANP-SDK availability
        if not ANP_SDK_AVAILABLE:
            print("⚠️  ANP-SDK not fully available, using fallback implementation")
        
        self.register_agents_from_config()
        
        print("🌐 ANP Network initialized")

    def create_anp_agent(self,
        agent_config: Dict[str, Any], task_id: str, 
        agent_prompts: Optional[Dict[str, Any]] = None,
        general_config: Optional[Dict[str, Any]] = None
    ) -> ANPAgent:
        """Create an ANPAgent from configuration."""
        agent_id = agent_config['id']
        
        # Get system prompt from agent_prompts[agent_id] if available
        system_prompt = None
        if agent_prompts and str(agent_id) in agent_prompts:
            system_prompt = agent_prompts[str(agent_id)].get('system_prompt')
        
        # Ensure protocol name in config for workspace pathing
        proto_name = "anp"

        # Get global model config for LLM settings
        model_config = {}
        if general_config and 'model' in general_config:
            model_config = general_config['model']
        elif hasattr(self, 'config') and 'model' in self.config:
            model_config = self.config['model']
            
        # Calculate port for this agent
        port = self.base_port + agent_id
        
        # ANP specific configuration
        anp_config = {
            "host_domain": self.host_domain,
            "host_port": port,
            "enable_encryption": True,
            "enable_negotiation": False,  # Simplified for GAIA
        }
        
        return ANPAgent(
            node_id=agent_id,
            name=agent_config['name'],
            tool=agent_config['tool'],
            port=port,
            config={
                'max_tokens': agent_config.get('max_tokens', 500),
                'role': agent_config.get('role', 'agent'),
                'priority': agent_config.get('priority', 1),
                'system_prompt': system_prompt,
                'model_name': model_config.get('name', 'gpt-4o'),
                'temperature': model_config.get('temperature', 0.0),
                'openai_api_key': model_config.get('api_key'),
                'openai_base_url': model_config.get('base_url', 'https://api.openai.com/v1'),
                'protocol': proto_name,
                'anp': anp_config
            },
            task_id=task_id
        )

    def register_agents_from_config(self) -> None:
        """
        Create and register multiple ANP agents from configuration.
        """     
        # Update task_id in workflow
        if "workflow" not in self.config:
            raise ValueError("Full configuration must contain 'workflow' key")
        
        # Extract agent configurations and prompts
        agent_configs = self.config.get('agents', [])
        agent_prompts = self.config.get('agent_prompts', {})
        
        print(f"📝 准备创建 {len(agent_configs)} 个ANP Agent")
        
        for agent_info in agent_configs:
            try:
                # Create ANP agent from configuration with proper system prompt
                agent = self.create_anp_agent(
                    agent_config=agent_info, 
                    task_id=self.task_id, 
                    agent_prompts=agent_prompts,
                    general_config=self.config
                )
                
                # Register agent to network
                self.register_agent(agent)
                
                print(f"✅ ANP Agent {agent_info['name']} (ID: {agent_info['id']}) 已创建并注册")
                
            except Exception as e:
                print(f"❌ 创建和注册ANP Agent {agent_info.get('name', 'unknown')} 失败: {e}")
                raise
        
        print(f"🎉 总共成功注册了 {len(agent_configs)} 个ANP Agent")
    
    # ==================== Communication Methods ====================

    
    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """
        Deliver message to specific agent using ANP protocol.
        
        Args:
            dst: Destination agent ID
            msg: Message payload
        """
        try:
            # Send via ANP native communication
            body = msg.get("content") or json.dumps(msg, ensure_ascii=False)
            payload = {"body": body, "type": msg.get("type", "message")}
            resp = await self.comm_backend.send(src_id="network", dst_id=str(dst), payload=payload)
            text = resp.get("text") if isinstance(resp, dict) else None

            # Metrics
            self.pkt_cnt += 1
            self.bytes_tx += len(json.dumps(msg).encode('utf-8'))
            print(f"📤 ANPNetwork -> {dst}: {msg.get('type','unknown')} | resp: {str(text)[:80] if text else 'ok'}")
        except Exception as e:
            print(f"❌ Failed to deliver ANP message to agent {dst}: {e}")

    async def start(self):
        """Start ANP network with enhanced initialization."""
        print("🌐 Starting ANP multi-agent network...")
        try:
            if not ANP_SDK_AVAILABLE:
                print("⚠️ ANP-SDK not available, using fallback implementation")
            
            # Start all agents (which will start their ANP nodes)
            for agent in self.agents:
                # Register agent endpoint
                await self.comm_backend.register_endpoint(str(agent.id), agent)
                
                # Start the agent (which will start its ANP SimpleNode)
                print(f"🚀 Starting ANP agent {agent.name} on port {agent.port}...")
                await agent.start()
                print(f"✅ ANP agent {agent.name} started")
                
            print("🔗 ANP communication backend initialized and nodes started")
            
            # Give nodes time to fully start and establish connections
            print("⏳ Waiting for ANP nodes to fully initialize...")
            await asyncio.sleep(3)
            
        except Exception as e:
            print(f"❌ Failed to initialize ANP communication backend: {e}")
            raise

        await super().start()
        logger.info(f"{Colors.MAGENTA}🚀 ANP network started successfully{Colors.RESET}")
        
        # Optional: report agent health after nodes start (non-fatal)
        try:
            ok = await self.monitor_agent_health()
            if not ok:
                print(f"{Colors.YELLOW}⚠️  Post-start health check: some agents may be unreachable yet{Colors.RESET}")
        except Exception as _e:
            print(f"[ANPNetwork] Post-start health check failed: {_e}")

    async def stop(self):
        """Stop ANP network with proper cleanup."""
        print("🛑 Stopping ANP network...")
        
        # Stop all agents first
        for agent in self.agents:
            try:
                await agent.stop()
            except Exception as e:
                print(f"⚠️ Error stopping agent {agent.id}: {e}")
        
        # Stop communication backend
        try:
            await self.comm_backend.close()
            print("✅ ANP communication backend closed")
        except Exception as e:
            print(f"⚠️ Error closing ANP communication backend: {e}")
        
        # Call parent stop method
        await super().stop()
        
        print("✅ ANP network stopped")

    def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive ANP network statistics."""
        stats = {
            "network_type": "anp",
            "total_agents": len(self.agents),
            "anp_sdk_available": ANP_SDK_AVAILABLE,
            "performance_metrics": {
                "bytes_tx": self.bytes_tx,
                "bytes_rx": self.bytes_rx,
                "pkt_cnt": self.pkt_cnt,
                "header_overhead": self.header_overhead,
                "token_sum": self.token_sum
            },
            "agent_stats": []
        }
        
        # Collect individual agent statistics
        for agent in self.agents:
            try:
                agent_stats = agent.get_agent_card()
                stats["agent_stats"].append(agent_stats)
            except Exception as e:
                print(f"⚠️ Failed to get stats for agent {agent.id}: {e}")
                stats["agent_stats"].append({
                    "agent_id": agent.id,
                    "agent_name": agent.name,
                    "error": str(e)
                })
        
        return stats

    # Override to use ANP native communication while maintaining step-based memory
    async def _execute_agent_step(self, agent_id: int, context_message: str, step_idx: int) -> str:
        """Execute one step and record messages into NetworkMemoryPool for later summary."""
        agent = self.get_agent_by_id(agent_id)
        if not agent:
            raise Exception(f"Agent {agent_id} not found")

        # 1) Create step execution record and mark processing
        self.network_memory.add_step_execution(
            step=step_idx,
            agent_id=str(agent_id),
            agent_name=agent.name,
            task_id=self.task_id,
            user_message=context_message,
        )
        self.network_memory.update_step_status(step_idx, ExecutionStatus.PROCESSING)

        # 2) Execute directly via agent's execute method (not ANP protocol)
        try:
            # Call agent's execute method directly for better integration
            text = await agent.execute(context_message)
            
            # Construct conversation for summary
            messages = [
                Message.user_message(context_message),
                Message.assistant_message(text or "")
            ]

            # 3) Update step status and messages
            if not text or not str(text).strip():
                self.network_memory.update_step_status(
                    step_idx,
                    ExecutionStatus.ERROR,
                    error_message="Empty result from agent",
                )
                return "No meaningful result generated by agent"
            else:
                self.network_memory.update_step_status(
                    step_idx,
                    ExecutionStatus.SUCCESS,
                    messages=messages,
                )
                return text

        except Exception as e:
            # Record the error in step execution
            self.network_memory.update_step_status(
                step_idx,
                ExecutionStatus.ERROR,
                error_message=str(e),
            )
            return f"Error: {e}"

    # ==================== ANP Protocol Methods ====================

    async def broadcast(self, msg: Dict[str, Any], exclude_sender: Optional[int] = None) -> None:
        """
        Broadcast message to all agents using ANP protocol.
        
        Args:
            msg: Message payload
            exclude_sender: Exclude sender ID
        """
        print(f"📡 ANP Broadcasting: {msg.get('type','unknown')}")
        delivered = 0
        
        for agent in self.agents:
            if exclude_sender is not None and agent.id == exclude_sender:
                continue
            
            try:
                await self.deliver(agent.id, msg)
                delivered += 1
            except Exception as e:
                print(f"❌ ANP broadcast to {agent.id} failed: {e}")
        
        print(f"📡 ANP broadcast done, {delivered} agents")

    async def broadcast_init(self, doc: str, chunk_size: int = 400) -> None:
        """Broadcast initial document using ANP protocol."""
        chunks = [doc[i:i+chunk_size] for i in range(0, len(doc), chunk_size)]
        pkt = {"type": "doc_init", "chunks": chunks}
        await self.broadcast(pkt, exclude_sender=None)

    async def poll(self) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Poll all agents for messages using ANP protocol.
        
        Returns:
            List of (sender_id, message) tuples
        """
        messages = []
        
        for agent in self.agents:
            if isinstance(agent, ANPAgent):
                try:
                    # Use non-blocking receive
                    msg = await agent.recv_msg(timeout=0.0)
                    if msg:
                        self.pkt_cnt += 1
                        self.bytes_rx += len(json.dumps(msg).encode("utf-8"))
                        messages.append((agent.id, msg))
                        
                except Exception as e:
                    # Non-blocking operation, ignore errors
                    pass
        
        return messages

    def register_agent(self, agent: ANPAgent) -> None:
        """Register ANP agent to network."""
        if not isinstance(agent, ANPAgent):
            raise TypeError(f"Expected ANPAgent, got {type(agent)}")
        
        super().register_agent(agent)
        
        # Set network reference for agent
        agent.network_ref = self
        
        print(f"📝 ANP registered agent {agent.id} ({agent.name})")
        
        # If network is already running, register with backend
        if self.running and ANP_SDK_AVAILABLE:
            asyncio.create_task(self.comm_backend.register_endpoint(str(agent.id), agent))

    def get_network_status(self) -> Dict[str, Any]:
        """Get ANP network status."""
        return {
            "protocol": "ANP",
            "anp_sdk_available": ANP_SDK_AVAILABLE,
            "agents_count": len(self.agents),
            "running": self.running,
            "base_port": self.base_port,
            "host_domain": self.host_domain
        }

    def get_agent_cards(self) -> Dict[int, Dict[str, Any]]:
        """Get all agent cards."""
        cards = {}
        for agent in self.agents:
            if isinstance(agent, ANPAgent):
                cards[agent.id] = agent.get_agent_card()
        return cards

    async def health_check(self) -> Dict[str, Any]:
        """ANP network health check."""
        health_status = {
            "network_running": self.running,
            "anp_available": ANP_SDK_AVAILABLE,
            "agents_healthy": 0,
            "agents_total": len(self.agents)
        }
        
        # Check agent health
        for agent in self.agents:
            if isinstance(agent, ANPAgent):
                try:
                    if hasattr(agent, '_connected') and agent._connected:
                        health_status["agents_healthy"] += 1
                except Exception:
                    pass
        
        health_status["overall_healthy"] = (
            health_status["network_running"] and
            health_status["agents_healthy"] > 0
        )
        
        return health_status