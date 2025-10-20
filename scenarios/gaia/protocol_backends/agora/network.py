"""
Agora Network implementation for GAIA framework.
Implements communication management for Agora protocol using agora SDK.
"""

import asyncio
import json
import time
import httpx
import os
import logging
from typing import Any, Dict, Optional
import sys

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.network import MeshNetwork
from core.schema import Message, ExecutionStatus, Colors
from protocol_backends.agora.agent import AgoraAgent

# Agora Protocol imports
try:
    import agora
    from langchain_openai import ChatOpenAI
except ImportError as e:
    raise ImportError(f"Agora Protocol SDK required but not available: {e}")

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AgoraCommBackend:
    """Agora protocol communication backend for GAIA framework."""

    def __init__(self) -> None:
        self._endpoints: Dict[str, str] = {}  # agent_id -> endpoint uri
        self._http_clients: Dict[str, httpx.AsyncClient] = {}

    async def register_endpoint(self, agent_id: str, address: str) -> None:
        """Register Agora agent endpoint."""
        self._endpoints[agent_id] = address
        logger.info(f"{Colors.CYAN}Registered agent: {agent_id} @ {address}{Colors.RESET}")

    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send message via Agora protocol."""
        endpoint = self._endpoints.get(dst_id)
        logger.info(f">>> {Colors.BLUE}[AgoraCommBackend] Sending from {src_id} to {dst_id} via {endpoint}{Colors.RESET}")
        if not endpoint:
            raise RuntimeError(f"Unknown destination agent: {dst_id}")

        # Convert payload to Agora message format
        agora_message = self._to_agora_message(payload)
        
        try:
            # Send HTTP request to Agora agent endpoint
            client = self._get_http_client(dst_id)
            response = await client.post(
                f"{endpoint}/",
                json=agora_message,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            raw_response = response.json()
            text_content = self._extract_text_from_agora_response(raw_response)
            
            return {
                "raw": raw_response,
                "text": text_content
            }
            
        except Exception as e:
            print(f"[AgoraCommBackend] Send failed {src_id} -> {dst_id}: {e}")
            return {"raw": None, "text": ""}

    async def health_check(self, agent_id: str) -> bool:
        """Check Agora agent health."""
        endpoint = self._endpoints.get(agent_id)
        if not endpoint:
            return False
            
        try:
            client = self._get_http_client(agent_id)
            response = await client.get(f"{endpoint}/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            print(f"[AgoraCommBackend] Health check failed for {agent_id}: {e}")
            return False

    async def close(self) -> None:
        """Close Agora communication backend."""
        for client in self._http_clients.values():
            try:
                await client.aclose()
            except Exception:
                pass
        self._http_clients.clear()

    def _to_agora_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convert standard payload to Agora message format."""
        if "body" in payload:
            return {
                "protocolHash": payload.get("protocolHash"),
                "body": payload["body"],
                "protocolSources": payload.get("protocolSources", [])
            }
        elif "content" in payload:
            return {
                "protocolHash": None,
                "body": payload["content"],
                "protocolSources": []
            }
        else:
            return {
                "protocolHash": None,
                "body": str(payload),
                "protocolSources": []
            }

    def _extract_text_from_agora_response(self, response: Dict[str, Any]) -> str:
        """Extract text content from Agora response."""
        try:
            if isinstance(response, str):
                return response
            elif isinstance(response, dict):
                if "body" in response:
                    return response["body"]
                elif "text" in response:
                    return response["text"]
                elif "result" in response:
                    result = response["result"]
                    if isinstance(result, str):
                        return result
                    elif isinstance(result, dict) and "body" in result:
                        return result["body"]
                elif "data" in response:
                    return str(response["data"])
                else:
                    return str(response)
            return ""
        except Exception:
            return ""

    def _get_http_client(self, agent_id: str) -> httpx.AsyncClient:
        """Get or create a per-agent HTTP client with keep-alive pooling."""
        client = self._http_clients.get(agent_id)
        if client is None:
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            client = httpx.AsyncClient(timeout=30.0, limits=limits)
            self._http_clients[agent_id] = client
        return client

class AgoraNetwork(MeshNetwork):
    """Agora Network implementation with communication management."""
    
    def __init__(self, config: Dict[str, Any]):
        if isinstance(config, dict):
            config = {**config, "protocol": config.get("protocol", "agora")}
        super().__init__(config=config)
        self.comm_backend = AgoraCommBackend()
        self.register_agents_from_config()
    
    def create_agora_agent(self,
        agent_config: Dict[str, Any], task_id: str, 
        agent_prompts: Optional[Dict[str, Any]] = None
    ) -> AgoraAgent:
        """Create an AgoraAgent from configuration."""
        agent_id = agent_config['id']

        # Get system prompt from agent_prompts[agent_id] if available
        system_prompt = None
        if agent_prompts and str(agent_id) in agent_prompts:
            system_prompt = agent_prompts[str(agent_id)].get('system_prompt')

        return AgoraAgent(
            node_id=agent_id,
            name=agent_config['name'],
            tool=agent_config['tool'],
            port=agent_config['port'],
            config={
                'max_tokens': agent_config.get('max_tokens', 500),
                'role': agent_config.get('role', 'agent'),
                'priority': agent_config.get('priority', 1),
                'system_prompt': system_prompt,
                'openai_model': agent_config.get('openai_model', 'gpt-4o-mini'),
                'openai_temperature': agent_config.get('openai_temperature', 0.1),
                'protocol': 'agora'
            },
            task_id=task_id
        )

    def register_agents_from_config(self) -> Dict[str, Any]:
        """Create and register multiple Agora agents from configuration."""     
        if "workflow" not in self.config:
            raise ValueError("Full configuration must contain 'workflow' key")
        
        agent_configs = self.config.get('agents', [])
        agent_prompts = self.config.get('agent_prompts', {})
        
        print(f"📝 Preparing to create {len(agent_configs)} Agora agents")
        
        for agent_info in agent_configs:
            try:
                agent = self.create_agora_agent(
                    agent_config=agent_info, 
                    task_id=self.task_id, 
                    agent_prompts=agent_prompts
                )
                
                self.register_agent(agent)
                
                endpoint = f"http://localhost:{agent_info['port']}"
                # Store endpoint info for later registration
                agent._endpoint = endpoint
                
                print(f"✅ Agora Agent {agent_info['name']} (ID: {agent_info['id']}) created and registered")
                
            except Exception as e:
                print(f"❌ Failed to create and register Agora Agent {agent_info.get('name', 'unknown')}: {e}")
                raise
        
        print(f"🎉 Successfully registered a total of {len(agent_configs)} Agora agents")

    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """Deliver message to specific agent using Agora protocol."""
        try:
            body = msg.get("content") or json.dumps(msg, ensure_ascii=False)
            payload = {"body": body}
            resp = await self.comm_backend.send(src_id="network", dst_id=str(dst), payload=payload)
            text = resp.get("text") if isinstance(resp, dict) else None

            self.pkt_cnt += 1
            self.bytes_tx += len(json.dumps(msg).encode('utf-8'))
            print(f"📤 AgoraNetwork -> {dst}: {msg.get('type','unknown')} | resp: {str(text)[:80] if text else 'ok'}")
        except Exception as e:
            print(f"❌ Failed to deliver Agora message to agent {dst}: {e}")

    async def start(self):
        """Start Agora network with enhanced initialization."""
        print("🌐 Starting Agora multi-agent network...")
        try:
            # Register endpoints and start agents
            for agent in self.agents:
                endpoint = f"http://localhost:{agent.port}"
                await self.comm_backend.register_endpoint(str(agent.id), endpoint)
                agent._endpoint = endpoint
                # Start the agent's Agora server
                await agent.connect()
                print(f"🚀 Started Agora agent {agent.name} on port {agent.port}")
                
            print("🔗 Agora communication backend initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Agora communication backend: {e}")
            raise

        await super().start()
        logger.info(f"{Colors.MAGENTA}🚀 Agora network started successfully{Colors.RESET}")
        
        try:
            ok = await self.monitor_agent_health()
            if not ok:
                print(f"{Colors.YELLOW}⚠️  Post-start health check: some agents may be unreachable yet{Colors.RESET}")
        except Exception as _e:
            print(f"[AgoraNetwork] Post-start health check failed: {_e}")

    async def stop(self):
        """Stop Agora network with proper cleanup."""
        print("🛑 Stopping Agora network...")
        
        # Stop all agents first
        for agent in self.agents:
            try:
                await agent.disconnect()
            except Exception as e:
                print(f"⚠️ Error stopping agent {agent.id}: {e}")
        
        try:
            await self.comm_backend.close()
            print("✅ Agora communication backend closed")
        except Exception as e:
            print(f"⚠️ Error closing Agora communication backend: {e}")
        
        await super().stop()
        print("✅ Agora network stopped")

    def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive Agora network statistics."""
        stats = {
            "network_type": "agora",
            "total_agents": len(self.agents),
            "agora_available": True,
            "performance_metrics": {
                "bytes_tx": self.bytes_tx,
                "bytes_rx": self.bytes_rx,
                "pkt_cnt": self.pkt_cnt,
                "header_overhead": self.header_overhead,
                "token_sum": self.token_sum
            },
            "agent_stats": []
        }
        
        for agent in self.agents:
            try:
                agent_stats = agent.get_connection_status()
                stats["agent_stats"].append(agent_stats)
            except Exception as e:
                stats["agent_stats"].append({
                    "agent_id": agent.id,
                    "agent_name": agent.name,
                    "error": str(e)
                })
        
        return stats

    async def _execute_agent_step(self, agent_id: int, context_message: str, step_idx: int) -> str:
        """Execute one step and record messages into NetworkMemoryPool for later summary."""
        agent = self.get_agent_by_id(agent_id)
        if not agent:
            raise Exception(f"Agent {agent_id} not found")

        self.network_memory.add_step_execution(
            step=step_idx,
            agent_id=str(agent_id),
            agent_name=agent.name,
            task_id=self.task_id,
            user_message=context_message,
        )
        self.network_memory.update_step_status(step_idx, ExecutionStatus.PROCESSING)

        try:
            # Call agent's execute method directly for better integration
            text = await agent.execute(context_message)
            
            messages = [
                Message.user_message(context_message),
                Message.assistant_message(text or "")
            ]

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
            self.network_memory.update_step_status(
                step_idx,
                ExecutionStatus.ERROR,
                error_message=str(e),
            )
            return f"Error: {e}"
