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

# ACP SDK imports (mandatory)
from acp_sdk.server import Server, Context, RunYield, RunYieldResume
from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart
import uvicorn


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
        self._local_agents = {}  # For locally spawned agents
        self._agent_names = {}  # agent_id -> registered server agent name

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
        message = self._to_acp_message(payload)

        # Use real ACP client for HTTP endpoints only
        agent_name = self._agent_names.get(dst_id, dst_id)
        client = Client(base_url=endpoint)
        run = client.run_sync(
            agent=agent_name,
            input=[message]
        )
        # Extract output messages if available
        text_content = ""
        raw_output = getattr(run, "output", None)
        if raw_output:
            for msg in raw_output:
                for part in getattr(msg, "parts", []) or []:
                    content = getattr(part, "content", None)
                    if isinstance(content, str):
                        text_content += content
        return {"raw": raw_output, "text": text_content}

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
        
        base_url = f"http://{host}:{port}"
        
        # Create ACP server with executor
        server = Server()
        
        @server.agent()
        async def privacy_agent(input: list[Message], context: Context):
            """ACP agent handler for privacy testing."""
            try:
                async for out in executor.execute(input, context):
                    # Directly yield acp_sdk Message
                    if isinstance(out, Message):
                        yield out
                        continue
                    # Map common simple outputs to Message
                    if isinstance(out, dict) and out.get("type") == "agent_text_message":
                        yield Message(parts=[MessagePart(content=out.get("data", ""), content_type="text/plain")])
                    elif isinstance(out, str):
                        yield Message(parts=[MessagePart(content=out, content_type="text/plain")])
                    else:
                        yield Message(parts=[MessagePart(content=str(out), content_type="text/plain")])
            except Exception as e:
                print(f"[ACPCommBackend] Agent execution error: {e}")
                yield Message(parts=[MessagePart(content="Internal server error", content_type="text/plain")])
        
        # Remember agent name used for client invocation
        self._agent_names[agent_id] = "privacy_agent"

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
    
    def _to_acp_message(self, payload: Dict[str, Any]):
        """Convert standard payload to ACP Message object."""
    # Create real ACP Message object
        if isinstance(payload, dict) and "text" in payload:
            return Message(parts=[MessagePart(content=payload["text"], content_type="text/plain")])
        elif "parts" in payload:
            # Convert parts to MessagePart objects
            message_parts = []
            for part in payload["parts"]:
                if isinstance(part, dict):
                    if "content" in part:
                        message_parts.append(MessagePart(content=part["content"], content_type=part.get("content_type", "text/plain")))
                    elif "text" in part:  # backward compat
                        message_parts.append(MessagePart(content=part["text"], content_type="text/plain"))
                elif hasattr(part, 'content'):
                    message_parts.append(part)
            return Message(parts=message_parts)
        else:
            return Message(parts=[MessagePart(content=str(payload), content_type="text/plain")])

    # Removed HTTP simulation helpers; ACP requires real SDK

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
