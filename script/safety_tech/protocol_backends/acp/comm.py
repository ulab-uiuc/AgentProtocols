# -*- coding: utf-8 -*-
"""
ACP Communication Backend for Privacy Testing
Implements ACP protocol communication for the privacy testing framework.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

# Import base communication interface
try:
    from ...comm.base import BaseCommBackend
except ImportError:
    from comm.base import BaseCommBackend

# ACP SDK imports (optional dependencies)
try:
    from acp_sdk.server import Server, Context, RunYield
    from acp_sdk import Message, MessagePart
    import uvicorn
    ACP_AVAILABLE = True
except ImportError:
    ACP_AVAILABLE = False
    
    # Minimal stubs for when ACP SDK is not available
    class Server: pass
    class Context: pass
    class RunYield: pass
    class Message: pass
    class MessagePart: pass


@dataclass
class ACPAgentHandle:
    """Handle for locally spawned ACP agent."""
    agent_id: str
    host: str
    port: int
    base_url: str
    server_task: Optional[asyncio.Task] = None


class ACPCommBackend(BaseCommBackend):
    """ACP protocol communication backend for privacy testing."""

    def __init__(self, **kwargs):
        self._endpoints: Dict[str, str] = {}  # agent_id -> endpoint uri
        self._client = httpx.AsyncClient(timeout=30.0)
        self._local_agents: Dict[str, ACPAgentHandle] = {}  # For locally spawned agents

    async def register_endpoint(self, agent_id: str, address: str) -> None:
        """Register ACP agent endpoint."""
        self._endpoints[agent_id] = address
        print(f"[ACPCommBackend] Registered {agent_id} @ {address}")

    async def connect(self, src_id: str, dst_id: str) -> None:
        """ACP doesn't require explicit connection setup."""
        return None

    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """Send message via ACP protocol."""
        endpoint = self._endpoints.get(dst_id)
        if not endpoint:
            raise RuntimeError(f"Unknown destination agent: {dst_id}")

        # Convert payload to ACP message format
        acp_message = self._to_acp_message(payload)
        
        try:
            # Send HTTP request to ACP agent endpoint
            response = await self._client.post(
                f"{endpoint}/message",
                json=acp_message,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            raw_response = response.json()
            text_content = self._extract_text_from_acp_response(raw_response)
            
            return {
                "raw": raw_response,
                "text": text_content
            }
            
        except Exception as e:
            print(f"[ACPCommBackend] Send failed {src_id} -> {dst_id}: {e}")
            return {"raw": None, "text": ""}

    async def health_check(self, agent_id: str) -> bool:
        """Check ACP agent health."""
        endpoint = self._endpoints.get(agent_id)
        if not endpoint:
            return False
            
        # For simulation mode with mock endpoints, consider agents always healthy
        if endpoint.startswith("acp://"):
            return True
            
        try:
            # For real HTTP endpoints, do actual health check
            response = await self._client.get(f"{endpoint}/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            print(f"[ACPCommBackend] Health check failed for {agent_id}: {e}")
            return False

    async def close(self) -> None:
        """Close ACP communication backend."""
        await self._client.aclose()
        
        # Stop any locally spawned agents
        for handle in self._local_agents.values():
            if handle.server_task and not handle.server_task.done():
                handle.server_task.cancel()
                try:
                    await handle.server_task
                except asyncio.CancelledError:
                    pass

    async def spawn_local_agent(self, agent_id: str, host: str, port: int, executor: Any) -> ACPAgentHandle:
        """Spawn local ACP agent server."""
        if not ACP_AVAILABLE:
            raise RuntimeError("ACP SDK not available. Cannot spawn local agents.")
        
        base_url = f"http://{host}:{port}"
        
        # Create ACP server with executor
        server = Server()
        
        @server.message()
        async def handle_message(context: Context) -> RunYield:
            try:
                # Extract user input from ACP context
                user_input = self._extract_user_input_from_context(context)
                
                # Create mock event queue for executor
                events = []
                
                class MockEventQueue:
                    async def enqueue_event(self, event):
                        events.append(event)
                
                mock_queue = MockEventQueue()
                
                # Execute agent logic
                await executor.execute(context, mock_queue)
                
                # Return events as ACP responses
                for event in events:
                    if event.get("type") == "agent_text_message":
                        yield Message(parts=[MessagePart(text=event.get("data", ""))])
                
            except Exception as e:
                print(f"[ACPCommBackend] Agent execution error: {e}")
                yield Message(parts=[MessagePart(text="Internal server error")])

        # Start server in background task
        server_task = asyncio.create_task(
            self._run_acp_server(server, host, port)
        )
        
        handle = ACPAgentHandle(
            agent_id=agent_id,
            host=host,
            port=port,
            base_url=base_url,
            server_task=server_task
        )
        
        self._local_agents[agent_id] = handle
        
        # Wait a bit for server to start
        await asyncio.sleep(1.0)
        
        return handle

    # ---------------------- Helper Methods ----------------------
    
    def _to_acp_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convert standard payload to ACP message format."""
        if "text" in payload:
            return {
                "messageId": str(int(time.time() * 1000)),
                "role": "user",
                "parts": [{"kind": "text", "text": payload["text"]}]
            }
        elif "parts" in payload:
            return {
                "messageId": str(int(time.time() * 1000)),
                "role": "user", 
                "parts": payload["parts"]
            }
        else:
            return {
                "messageId": str(int(time.time() * 1000)),
                "role": "user",
                "parts": [{"kind": "text", "text": str(payload)}]
            }

    def _extract_text_from_acp_response(self, response: Dict[str, Any]) -> str:
        """Extract text content from ACP response."""
        try:
            # ACP response format: {"events": [{"type": "agent_text_message", "data": "..."}]}
            events = response.get("events", [])
            for event in events:
                if event.get("type") == "agent_text_message":
                    return event.get("data", "")
            
            # Fallback: look for text in different response structures
            if "text" in response:
                return response["text"]
            elif "content" in response:
                return response["content"]
            elif "message" in response:
                return response["message"]
            
            return ""
        except Exception:
            return ""

    def _extract_user_input_from_context(self, context: Context) -> str:
        """Extract user input text from ACP context."""
        try:
            # This would use actual ACP SDK methods
            if hasattr(context, 'get_user_input'):
                return context.get_user_input() or ""
            return ""
        except Exception:
            return ""

    async def _run_acp_server(self, server: Server, host: str, port: int) -> None:
        """Run ACP server using uvicorn."""
        try:
            config = uvicorn.Config(
                server.create_app(),
                host=host,
                port=port,
                log_level="warning"
            )
            server_instance = uvicorn.Server(config)
            await server_instance.serve()
        except Exception as e:
            print(f"[ACPCommBackend] Server error on {host}:{port}: {e}")
