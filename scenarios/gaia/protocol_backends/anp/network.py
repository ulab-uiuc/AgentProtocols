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

# Add AgentConnect SDK to sys.path by searching upwards for 'agentconnect_src'
from pathlib import Path as _Path
_cur = _Path(__file__).resolve()
for _p in _cur.parents:
    _cand = _p / "agentconnect_src"
    if _cand.exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from core.network import MeshNetwork
from core.schema import Message, ExecutionStatus
from protocol_backends.anp.agent import ANPAgent
from core.schema import Colors

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Import ANP-SDK components
try:
    from agent_connect.simple_node import SimpleNode, SimpleNodeSession
    print("✅ ANP-SDK components available")
except ImportError as e:
    raise ImportError(f"ANP-SDK components required but not available: {e}")


class ANPCommBackend:
    """ANP protocol communication backend for GAIA framework (ANP-SDK based)."""

    def __init__(self, **kwargs):
        # agent_id -> DID mapping
        self._endpoints: Dict[str, str] = {}
        # agent_id -> ANPAgent instance
        self._agents: Dict[str, 'ANPAgent'] = {}
        # (src_id, dst_id) -> SimpleNodeSession
        self._sessions: Dict[tuple, 'SimpleNodeSession'] = {}

    async def register_endpoint(self, agent_id: str, agent: 'ANPAgent') -> None:
        """Register ANP agent and its DID endpoint using info from the agent."""
        did = agent.local_did_info.get('did', f'did:all:agent_{agent_id}')
        self._endpoints[str(agent_id)] = did
        self._agents[str(agent_id)] = agent
        logger.info(f"{Colors.CYAN}Registered ANP agent: {agent_id} @ {did}{Colors.RESET}")

    async def connect(self, src_id: str, dst_id: str) -> None:
        """Establish ANP SimpleNode session from src to dst using ANP-SDK."""
        key = (str(src_id), str(dst_id))
        if key in self._sessions:
            return
        src_agent = self._agents.get(str(src_id))
        dst_did = self._endpoints.get(str(dst_id))
        # If src agent is not registered (e.g., network-originated), skip session creation
        if not src_agent or not dst_did:
            return
        if not getattr(src_agent, 'simple_node', None):
            return
        try:
            session = await src_agent.simple_node.connect_to_did(dst_did)
            if session:
                self._sessions[key] = session
                logger.info(f"ANP session established {src_id} -> {dst_id} ({dst_did})")
        except Exception as e:
            logger.warning(f"ANP connect failed {src_id}->{dst_id}: {e}")

    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """Send message via ANP SimpleNode session (native ANP-SDK)."""
        # If src is not a registered agent (e.g., 'network'), fallback to local delivery
        if str(src_id) not in self._agents:
            dst_agent = self._agents.get(str(dst_id))
            if dst_agent and hasattr(dst_agent, 'message_queue'):
                # Map to MeshAgent expected schema
                await dst_agent.message_queue.put({
                    "sender_id": 0,
                    "type": payload.get("type", "task_execution"),
                    "content": payload.get("body") or payload.get("text") or json.dumps(payload, ensure_ascii=False)
                })
                return {"raw": {"status": "queued"}, "text": "ok"}
        
        # Ensure session (best-effort)
        await self.connect(src_id, dst_id)
        key = (str(src_id), str(dst_id))
        session = self._sessions.get(key)
        src_agent = self._agents.get(str(src_id))
        dst_did = self._endpoints.get(str(dst_id))

        # If no session, fallback to local queue as last resort
        if not session or not src_agent or not dst_did:
            dst_agent = self._agents.get(str(dst_id))
            if dst_agent and hasattr(dst_agent, 'message_queue'):
                await dst_agent.message_queue.put({
                    "sender_id": int(dst_id) if str(dst_id).isdigit() else 0,
                    "type": payload.get("type", "task_execution"),
                    "content": payload.get("body") or payload.get("text") or json.dumps(payload, ensure_ascii=False)
                })
                return {"raw": {"status": "queued"}, "text": "ok"}
            raise RuntimeError(f"No ANP route for {src_id}->{dst_id}")

        # Build ANP message envelope (protocol-native)
        anp_message = {
            "type": payload.get("type", "message"),
            "request_id": f"{int(time.time()*1000)}_{src_id}",
            "source_did": src_agent.local_did_info.get('did'),
            "target_did": dst_did,
            "timestamp": time.time(),
            "payload": payload,
        }
        try:
            message_json = json.dumps(anp_message, separators=(',', ':'))
            ok = await session.send_message(message_json)
            if not ok:
                raise RuntimeError("ANP session send_message returned False")
            return {"raw": {"status": "sent"}, "text": "ok"}
        except Exception as e:
            logger.error(f"ANP send failed {src_id}->{dst_id}: {e}")
            return {"raw": None, "text": f"ANP Error: {e}"}

    async def health_check(self, agent_id: str) -> bool:
        """Check ANP agent health (SimpleNode initialized and connected)."""
        agent = self._agents.get(str(agent_id))
        if not agent:
            return False
        try:
            return bool(agent.anp_initialized and agent._connected and agent.simple_node)
        except Exception:
            return False

    async def close(self) -> None:
        """Close all ANP sessions (does not stop agents; network.stop() will stop agents)."""
        for key, session in list(self._sessions.items()):
            try:
                if hasattr(session, 'close') and callable(session.close):
                    await session.close()
            except Exception:
                pass
        self._sessions.clear()
        
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
                # Prioritize environment variables
                'openai_api_key': os.getenv("OPENAI_API_KEY") or model_config.get('api_key'),
                'openai_base_url': os.getenv("OPENAI_BASE_URL") or model_config.get('base_url', 'https://api.openai.com/v1'),
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
        
        print(f"📝 Preparing to create {len(agent_configs)} ANP agents")
        
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
                
                print(f"✅ ANP Agent {agent_info['name']} (ID: {agent_info['id']}) created and registered")
                
            except Exception as e:
                print(f"❌ Failed to create and register ANP Agent {agent_info.get('name', 'unknown')}: {e}")
                raise
        
        print(f"🎉 Successfully registered {len(agent_configs)} ANP agents in total")
    
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
            # Register agent endpoints first (DID might be available after node starts; we register placeholder)
            for agent in self.agents:
                await self.comm_backend.register_endpoint(str(agent.id), agent)
            
        except Exception as e:
            print(f"❌ Failed to initialize ANP communication backend: {e}")
            raise

        # Use base class to concurrently start agents (non-blocking)
        await super().start()
        logger.info(f"{Colors.MAGENTA}🚀 ANP network started successfully{Colors.RESET}")
        
        # Give SimpleNode some time to boot and DID to be generated, then refresh endpoints
        try:
            await asyncio.sleep(1.0)
            for agent in self.agents:
                await self.comm_backend.register_endpoint(str(agent.id), agent)
        except Exception:
            pass
        
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
        
        # Stop communication backend sessions first
        try:
            await self.comm_backend.close()
            print("✅ ANP communication backend closed")
        except Exception as e:
            print(f"⚠️ Error closing ANP communication backend: {e}")
        
        # Call parent stop method to stop agents
        await super().stop()
        
        print("✅ ANP network stopped")

    def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive ANP network statistics."""
        stats = {
            "network_type": "anp",
            "total_agents": len(self.agents),
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
        """Delegate to base MeshNetwork implementation to leverage tool-calling workflow."""
        return await super()._execute_agent_step(agent_id, context_message, step_idx)

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
        if self.running:
            asyncio.create_task(self.comm_backend.register_endpoint(str(agent.id), agent))

    def get_network_status(self) -> Dict[str, Any]:
        """Get ANP network status."""
        return {
            "protocol": "ANP",
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