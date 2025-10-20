"""
A2A Network implementation for GAIA framework.
Implements broadcast and communication management for A2A protocol.
"""

import asyncio
import json
import time
import httpx
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import sys

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.network import MeshNetwork
from core.schema import Message, ExecutionStatus
from protocol_backends.a2a.agent import A2AAgent
from core.schema import Colors

# A2A SDK imports
try:
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.utils import new_agent_text_message
    from a2a.types import MessageSendParams
except ImportError as e:
    raise ImportError(f"A2A SDK components required but not available: {e}")

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class A2ACommBackend:
    """A2A protocol communication backend for GAIA framework."""

    def __init__(self, **kwargs):
        self._endpoints = {}  # agent_id -> endpoint uri
        # Per-agent HTTP clients (connection pooling & keep-alive per host)
        self._http_clients = {}

    async def register_endpoint(self, agent_id: str, address: str) -> None:
        """Register A2A agent endpoint."""
        self._endpoints[agent_id] = address
        logger.info(f"{Colors.CYAN}Registered agent: {agent_id} @ {address}{Colors.RESET}")

    async def connect(self, src_id: str, dst_id: str) -> None:
        """A2A doesn't require explicit connection setup."""
        return None

    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """Send message via A2A protocol."""
        endpoint = self._endpoints.get(dst_id)
        if not endpoint:
            raise RuntimeError(f"Unknown destination agent: {dst_id}")

        # Convert payload to A2A message format
        a2a_message = self._to_a2a_message(payload)
        
        try:
            # Send HTTP request to A2A agent endpoint
            client = self._get_http_client(dst_id)
            response = await client.post(
                f"{endpoint}/message",
                json=a2a_message,
                headers={"Content-Type": "application/json"},
                timeout=25.0  # Reduce timeout to prevent hanging
            )
            response.raise_for_status()
            
            raw_response = response.json()
            text_content = self._extract_text_from_a2a_response(raw_response)
            
            return {
                "raw": raw_response,
                "text": text_content
            }
            
        except Exception as e:
            print(f"[A2ACommBackend] Send failed {src_id} -> {dst_id}: {e}")
            print(f"[A2ACommBackend] Endpoint: {endpoint}")
            print(f"[A2ACommBackend] Payload: {a2a_message}")
            import traceback
            traceback.print_exc()
            return {"raw": None, "text": ""}

    async def health_check(self, agent_id: str) -> bool:
        """Check A2A agent health."""
        endpoint = self._endpoints.get(agent_id)
        if not endpoint:
            return False
            
        try:
            # For real HTTP endpoints, do actual health check
            client = self._get_http_client(agent_id)
            response = await client.get(f"{endpoint}/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            print(f"[A2ACommBackend] Health check failed for {agent_id}: {e}")
            return False

    async def close(self) -> None:
        """Close A2A communication backend."""
        # Close all per-agent HTTP clients
        for aid, client in list(self._http_clients.items()):
            try:
                await client.aclose()
            except Exception:
                pass
        self._http_clients.clear()

    # ---------------------- Helper Methods ----------------------
    
    def _to_a2a_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convert standard payload to A2A message format."""
        # Extract message content
        message_content = ""
        if "body" in payload:
            message_content = payload["body"]
        elif "content" in payload:
            message_content = payload["content"]
        else:
            message_content = str(payload)
        
        # Create A2A message format
        return {
            "id": str(int(time.time() * 1000)),
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": message_content}],
                    "messageId": str(int(time.time() * 1000))
                }
            }
        }

    def _extract_text_from_a2a_response(self, response: Dict[str, Any]) -> str:
        """Extract text content from A2A response."""
        try:
            # A2A response format: {"events": [...]}
            if isinstance(response, dict) and "events" in response:
                events = response["events"]
                if isinstance(events, list):
                    for event in events:
                        if isinstance(event, dict):
                            # Check for message event with text parts
                            if event.get("kind") == "message" and "parts" in event:
                                parts = event.get("parts", [])
                                if parts and isinstance(parts, list):
                                    for part in parts:
                                        if isinstance(part, dict) and part.get("kind") == "text":
                                            return part.get("text", "")
                            # Fallback: look for text field directly
                            elif "text" in event:
                                return event["text"]
            
            # Fallback: convert entire response to string
            return str(response) if response else ""
        except Exception:
            return ""

    # ---------------------- Client Cache Helpers ----------------------
    def _get_http_client(self, agent_id: str) -> httpx.AsyncClient:
        """Get or create a per-agent HTTP client with keep-alive pooling and reasonable timeout."""
        client = self._http_clients.get(agent_id)
        if client is None:
            # Use more conservative timeout and connection limits to prevent hanging
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            client = httpx.AsyncClient(timeout=25.0, limits=limits)  # Reduce from 30s to 25s
            self._http_clients[agent_id] = client
        return client
    

class A2ANetwork(MeshNetwork):
    """
    A2A Network implementation with broadcast and enhanced communication capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Ensure protocol is set for downstream workspace pathing
        if isinstance(config, dict):
            config = {**config, "protocol": config.get("protocol", "a2a")}
        super().__init__(config=config)
        
        # Store original runner config if available (for model config)
        self._runner_config = getattr(self, '_original_config', None)
        
        self.comm_backend = A2ACommBackend()
        self.register_agents_from_config()
    
    def create_a2a_agent(self,
        agent_config: Dict[str, Any], task_id: str, 
        agent_prompts: Optional[Dict[str, Any]] = None,
        general_config: Optional[Dict[str, Any]] = None
    ) -> A2AAgent:
        """Create an A2AAgent from configuration."""
        agent_id = agent_config['id']

        # Get system prompt from agent_prompts[agent_id] if available
        system_prompt = None
        if agent_prompts and str(agent_id) in agent_prompts:
            system_prompt = agent_prompts[str(agent_id)].get('system_prompt')
        
        # Ensure protocol name in config for workspace pathing
        proto_name = "a2a"

        # Get global model config for LLM settings
        model_config = {}
        if general_config and 'model' in general_config:
            model_config = general_config['model']
# Debug info (can be enabled for troubleshooting)
            # print(f"[DEBUG] Using model config from general_config: {model_config.get('name', 'unknown')}")
        elif hasattr(self, 'config') and 'model' in self.config:
            model_config = self.config['model']
            # print(f"[DEBUG] Using model config from self.config: {model_config.get('name', 'unknown')}")
        # else:
            # print(f"[DEBUG] No model config found. general_config keys: {list(general_config.keys()) if general_config else 'None'}")
            # print(f"[DEBUG] self.config keys: {list(self.config.keys()) if hasattr(self, 'config') else 'No self.config'}")
            
        return A2AAgent(
            node_id=agent_id,
            name=agent_config['name'],
            tool=agent_config['tool'],
            port=agent_config['port'],
            config={
                'max_tokens': agent_config.get('max_tokens', 500),
                'role': agent_config.get('role', 'agent'),
                'priority': agent_config.get('priority', 1),
                'system_prompt': system_prompt,
                'model_name': model_config.get('name', 'gpt-4o'),
                'temperature': model_config.get('temperature', 0.0),
                'openai_api_key': model_config.get('api_key'),
                'openai_base_url': model_config.get('base_url', 'https://api.openai.com/v1'),
                'protocol': proto_name
            },
            task_id=task_id
        )

    def register_agents_from_config(self) -> Dict[str, Any]:
        """
        Create and register multiple A2A agents from configuration.
        """     
        # Update task_id in workflow
        if "workflow" not in self.config:
            raise ValueError("Full configuration must contain 'workflow' key")
        
        # Extract agent configurations and prompts
        agent_configs = self.config.get('agents', [])
        agent_prompts = self.config.get('agent_prompts', {})
        
        print(f"📝 Preparing to create {len(agent_configs)} A2A Agents")
        
        for agent_info in agent_configs:
            try:
                # Create A2A agent from configuration with proper system prompt
                agent = self.create_a2a_agent(
                    agent_config=agent_info, 
                    task_id=self.task_id, 
                    agent_prompts=agent_prompts,
                    general_config=self.config
                )
                
                # Register agent to network
                self.register_agent(agent)
                
                # Register HTTP endpoint (defer to start() method for async operations)
                endpoint = f"http://localhost:{agent_info['port']}"
                # Store endpoint info for later registration in start()
                agent._endpoint = endpoint
                
                print(f"✅ A2A Agent {agent_info['name']} (ID: {agent_info['id']}) created and registered")
                
            except Exception as e:
                print(f"❌ Failed to create and register A2A Agent {agent_info.get('name', 'unknown')}: {e}")
                raise
        
        print(f"🎉 Successfully registered a total of {len(agent_configs)} A2A Agents")

    # ==================== Communication Methods ====================
    
    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """
        Deliver message to specific agent using A2A protocol.
        
        Args:
            dst: Destination agent ID
            msg: Message payload
        """
        try:
            # Send via HTTP endpoint
            body = msg.get("content") or json.dumps(msg, ensure_ascii=False)
            payload = {"body": body}
            resp = await self.comm_backend.send(src_id="network", dst_id=str(dst), payload=payload)
            text = resp.get("text") if isinstance(resp, dict) else None

            # Metrics
            self.pkt_cnt += 1
            self.bytes_tx += len(json.dumps(msg).encode('utf-8'))
            print(f"📤 A2ANetwork -> {dst}: {msg.get('type','unknown')} | resp: {str(text)[:80] if text else 'ok'}")
        except Exception as e:
            print(f"❌ Failed to deliver A2A message to agent {dst}: {e}")

    async def start(self):
        """Start A2A network with enhanced initialization."""
        print("🌐 Starting A2A multi-agent network...")
        try:
            # Start all agents (which will start their A2A servers)
            for agent in self.agents:
                # Register endpoints again for safety
                endpoint = f"http://localhost:{agent.port}"
                await self.comm_backend.register_endpoint(str(agent.id), endpoint)
                # Start the agent (which will start its A2A server)
                print(f"🚀 Starting A2A agent {agent.name} on port {agent.port}...")
                await agent.connect()
                print(f"✅ A2A agent {agent.name} started")
                
            print("🔗 A2A communication backend initialized and servers started")
            
            # Give servers more time to fully start
            print("⏳ Waiting for servers to fully initialize...")
            await asyncio.sleep(3)
        except Exception as e:
            print(f"❌ Failed to initialize A2A communication backend: {e}")
            raise

        await super().start()
        logger.info(f"{Colors.MAGENTA}🚀 A2A network started successfully{Colors.RESET}")
        
        # Optional: report agent health after servers start (non-fatal)
        try:
            ok = await self.monitor_agent_health()
            if not ok:
                print(f"{Colors.YELLOW}⚠️  Post-start health check: some agents may be unreachable yet{Colors.RESET}")
        except Exception as _e:
            print(f"[A2ANetwork] Post-start health check failed: {_e}")

    async def stop(self):
        """Stop A2A network with proper cleanup."""
        print("🛑 Stopping A2A network...")
        
        # Stop all agents first
        for agent in self.agents:
            try:
                await agent.disconnect()
            except Exception as e:
                print(f"⚠️ Error stopping agent {agent.id}: {e}")
        
        # Stop communication backend
        try:
            await self.comm_backend.close()
            print("✅ A2A communication backend closed")
        except Exception as e:
            print(f"⚠️ Error closing A2A communication backend: {e}")
        
        # Call parent stop method
        await super().stop()
        
        print("✅ A2A network stopped")

    def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive A2A network statistics."""
        stats = {
            "network_type": "a2a",
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
                agent_stats = agent.get_connection_status()
                stats["agent_stats"].append(agent_stats)
            except Exception as e:
                print(f"⚠️ Failed to get stats for agent {agent.id}: {e}")
                stats["agent_stats"].append({
                    "agent_id": agent.id,
                    "agent_name": agent.name,
                    "error": str(e)
                })
        
        return stats

    # Override to call A2A backend while maintaining step-based memory for summarization
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

        # 2) Execute directly via agent's execute method (not HTTP)
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
