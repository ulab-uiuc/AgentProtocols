# -*- coding: utf-8 -*-
"""
A2A Comm Backend (HTTP + native event format)
Location: agent_network/script/streaming_queue/protocol_backend/a2a/comm.py

Notes:
- Directly uses the A2A Server's /message endpoint and event stream structure
- Integrates a lightweight host that can start a FastAPI + Uvicorn service in-process to host any AgentExecutor
- No longer uses a custom adapter
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

# A2A executor interface (required dependency)
from a2a.server.agent_execution import AgentExecutor  # type: ignore

# ------------------- Comm Base import -------------------
import sys
from pathlib import Path

# Add streaming_queue to path for imports
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parent.parent.parent  # Go up from a2a -> protocol_backend -> streaming_queue
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))

# Add comm path to avoid naming conflict with current file (same as ACP/ANP)
comm_path = streaming_queue_path / "comm"
if str(comm_path) not in sys.path:
    sys.path.insert(0, str(comm_path))

try:
    from base import BaseCommBackend  # type: ignore
except ImportError as e:
    raise ImportError(f"Cannot import BaseCommBackend from base: {e}")


# ==========================
# Embedded Host (FastAPI/Uvicorn)
# ==========================

# To minimize dependencies/fast startup, delay importing FastAPI and Uvicorn until they are needed.
# This way the Host functionality does not force heavy dependencies when not used.

# Remove simplified class, use the real A2A components

@dataclass
class A2AAgentHandle:
    agent_id: str
    host: str
    port: int
    base_url: str
    _server: Any | None
    _task: asyncio.Task | None

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass


async def _start_a2a_host(agent_id: str, host: str, port: int, executor: AgentExecutor) -> A2AAgentHandle:
    """
    Start a minimal A2A Host (/message + /health) using Starlette pure JSON,
    without relying on Pydantic/BaseModel.
    """
    # Delayed import (only depend on these libraries when host is needed)
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.requests import Request
    from starlette.routing import Route
    import uvicorn

    async def message_endpoint(request: Request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        # Compatible with A2A input structure: {"id": "...", "params": {"message": {...}}}
        msg = (payload.get("params") or {}).get("message") or {}
        parts = msg.get("parts") or []
        text = ""
        if parts and isinstance(parts, list) and isinstance(parts[0], dict):
            # Support {"kind":"text","text":"..."} / {"kind":"text","data":"..."}
            text = parts[0].get("text") or parts[0].get("data") or ""

        # Import the real A2A components
        from a2a.server.agent_execution import RequestContext
        from a2a.server.events import EventQueue
        from a2a.types import Message, MessageSendParams, Role, TextPart
        
        # Create the real A2A Message object
        if text:
            # Manually create a user message
            message = Message(
                role=Role.user,
                parts=[TextPart(text=text)],
                messageId=str(time.time_ns())
            )
        else:
            # Try to construct Message from original payload
            msg_data = payload.get("params", {}).get("message", {})
            if msg_data:
                # Construct Message object from dict
                message = Message.model_validate(msg_data)
            else:
                # Default status message
                message = Message(
                    role=Role.user,
                    parts=[TextPart(text="status")],
                    messageId=str(time.time_ns())
                )
        
        # Create MessageSendParams and RequestContext
        params = MessageSendParams(message=message)
        ctx = RequestContext(params)
        eq = EventQueue()
        
        await executor.execute(context=ctx, event_queue=eq)

        # Retrieve all events from the A2A EventQueue
        serializable_events = []
        try:
            while not eq.queue.empty():
                event = await eq.dequeue_event()
                if event:
                    if hasattr(event, 'model_dump'):
                        # Pydantic v2 style - use mode='json' to ensure enums etc. are serialized correctly
                        serializable_events.append(event.model_dump(mode='json'))
                    elif hasattr(event, 'dict'):
                        # Pydantic v1 style
                        serializable_events.append(event.dict())
                    else:
                        # If already a dict, use it directly
                        serializable_events.append(event)
        except:
            # If queue is empty or an exception occurs, fall back to empty list
            pass

        return JSONResponse({"events": serializable_events})

    async def health_endpoint(_request: Request):
        return JSONResponse({"ok": True, "agent_id": agent_id})

    routes = [
        Route("/message", message_endpoint, methods=["POST"]),
        Route("/health", health_endpoint, methods=["GET"]),
    ]
    app = Starlette(routes=routes)

    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    async def _serve():
        await server.serve()

    task = asyncio.create_task(_serve())
    await asyncio.sleep(0.3)  # Wait for the port to come up

    return A2AAgentHandle(
        agent_id=agent_id,
        host=host,
        port=port,
        base_url=f"http://{host}:{port}",
        _server=server,
        _task=task,
    )


# ==========================
# Comm Backend (including host management)
# ==========================

class A2ACommBackend(BaseCommBackend):
    """
    - Maintains agent_id -> base_url mapping
    - Responsible for /message calls (send/receive)
    - Can spawn a local A2A host in-process (spawn_local_agent)
    """

    def __init__(self, httpx_client: httpx.AsyncClient | None = None, request_timeout: float = 60.0):
        self._client = httpx_client or httpx.AsyncClient(timeout=request_timeout)
        self._own_client = httpx_client is None
        self._addr: Dict[str, str] = {}        # agent_id -> base_url
        self._hosts: Dict[str, A2AAgentHandle] = {}  # If started in-process, keep handle for shutdown

    # ---------- endpoint registration ----------
    async def register_endpoint(self, agent_id: str, address: str) -> None:
        self._addr[agent_id] = address.rstrip("/")

    # ---------- local Host management ----------
    async def spawn_local_agent(self, agent_id: str, host: str, port: int, executor: AgentExecutor) -> A2AAgentHandle:
        """
        Start a local A2A Host and automatically register its endpoint.
        """
        if agent_id in self._hosts:
            raise RuntimeError(f"[A2ACommBackend] local agent already exists: {agent_id}")
        handle = await _start_a2a_host(agent_id, host, port, executor)
        self._hosts[agent_id] = handle
        await self.register_endpoint(agent_id, handle.base_url)
        return handle

    async def stop_local_agent(self, agent_id: str) -> None:
        h = self._hosts.pop(agent_id, None)
        if h:
            await h.stop()
        self._addr.pop(agent_id, None)

    # ---------- send message ----------
    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """
        Send a message to the dst's A2A /message endpoint.
        payload supports:
          1) {"text":"..."} or {"parts":[{"kind":"text","text":"..."}]}
          2) Full A2A message structure (including role/parts) which will be forwarded as params.message
        """
        base = self._addr.get(dst_id)
        if not base:
            raise RuntimeError(f"[A2ACommBackend] unknown dst_id={dst_id}. Did you register_endpoint()?")

        msg = self._to_a2a_message(payload)
        req = {
            "id": str(time.time_ns()),
            "params": {
                "message": msg
            }
        }
        resp = await self._client.post(f"{base}/message", json=req)
        resp.raise_for_status()
        data = resp.json()
        return {
            "raw": data,
            "text": self._extract_text_from_events(data)
        }

    # ---------- health check ----------
    async def health_check(self, agent_id: str) -> bool:
        base = self._addr.get(agent_id)
        if not base:
            return False
        # Try /health first
        try:
            r = await self._client.get(f"{base}/health")
            if r.status_code == 200:
                return True
        except Exception:
            pass
        # Fallback: send a status message to /message
        try:
            req = {
                "id": str(time.time_ns()),
                "params": {
                    "message": {
                        "role": "user",
                        "messageId": str(time.time_ns()),
                        "parts": [{"kind": "text", "text": "status"}],
                    }
                },
            }
            r = await self._client.post(f"{base}/message", json=req)
            return r.status_code == 200
        except Exception:
            return False

    # ---------- close ----------
    async def close(self) -> None:
        # Shutdown any hosts started in this process
        for aid in list(self._hosts.keys()):
            try:
                await self.stop_local_agent(aid)
            except Exception:
                pass
        if self._own_client:
            await self._client.aclose()

    # -------------------- helpers --------------------
    def _to_a2a_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Already an A2A message structure
        if all(k in payload for k in ("role", "parts")):
            return payload

        if "parts" in payload and isinstance(payload["parts"], list):
            parts = payload["parts"]
        else:
            text = payload.get("text") or payload.get("content") or ""
            parts = [{"kind": "text", "text": text}]

        return {
            "role": "user",
            "parts": parts,
            "messageId": str(time.time_ns()),
        }

    def _extract_text_from_events(self, data: Dict[str, Any]) -> str:
        events = data.get("events") or []
        for ev in events:
            # Align with a2a.utils.new_agent_text_message
            if ev.get("type") == "agent_text_message":
                return ev.get("data") or ev.get("text") or ""
            # Some implementations use {"kind":"message","parts":[{"type":"text","text":"..."}]}
            if ev.get("kind") == "message":
                parts = ev.get("parts") or []
                if parts and isinstance(parts[0], dict):
                    t = parts[0].get("text")
                    if t:
                        return t
        return ""


# ==========================
# A2A Network (concrete)
# ==========================
# NOTE: Using lazy import pattern to avoid circular dependency with core.network_base
# This is necessary because network_base.py imports comm.base, and if this file
# imports NetworkBase at module level, it creates a circular import chain.

def get_a2a_network_class():
    """
    Lazy constructor for A2ANetwork class to avoid circular imports.
    Only imports NetworkBase when the class is actually needed.
    """
    from typing import Optional
    try:
        from core.network_base import NetworkBase  # type: ignore
    except ImportError as e:
        raise ImportError(f"Cannot import NetworkBase from core.network_base: {e}")
    
    class A2ANetwork(NetworkBase):
        """
        Concrete A2A implementation of NetworkBase:
          - injects A2ACommBackend for communication capability
          - additionally provides spawn_local_agent() convenience to start an in-process HTTP service for an executor and auto-register it
        """
        def __init__(self, httpx_client: Optional[httpx.AsyncClient] = None, request_timeout: float = 60.0):
            backend = A2ACommBackend(httpx_client=httpx_client, request_timeout=request_timeout)
            super().__init__(comm_backend=backend)

        async def spawn_local_agent(self, agent_id: str, host: str, port: int, executor: AgentExecutor) -> A2AAgentHandle:
            """
            Convenience: start a FastAPI+Uvicorn host in-process to host the executor,
            and automatically register it with the current network.
            """
            # self._comm is of type A2ACommBackend
            handle = await self._comm.spawn_local_agent(agent_id, host, port, executor)  # type: ignore[attr-defined]
            await self.register_agent(agent_id, handle.base_url)
            return handle
    
    return A2ANetwork

# Make A2ANetwork available via module attribute access (lazy loading)
_A2ANetwork_class = None

def __getattr__(name):
    """Module-level __getattr__ for lazy imports."""
    global _A2ANetwork_class
    if name == "A2ANetwork":
        if _A2ANetwork_class is None:
            _A2ANetwork_class = get_a2a_network_class()
        return _A2ANetwork_class
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
