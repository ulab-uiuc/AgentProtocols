# -*- coding: utf-8 -*-
"""
ACP Communication Backend using ACP SDK 1.0.3.

This implementation uses the official ACP SDK which provides full Agent Communication Protocol support.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, List
import logging
import uuid

# Add streaming_queue to path for imports
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parent.parent.parent
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))

# Add comm path
streaming_queue_path = Path(__file__).resolve().parent.parent.parent
comm_path = streaming_queue_path / "comm"
if str(comm_path) not in sys.path:
    sys.path.insert(0, str(comm_path))
from base import BaseCommBackend

# Import ACP SDK components
try:
    import acp_sdk
    from acp_sdk import Session, Run, RunCreateRequest, Message, RunMode
except ImportError as e:
    raise ImportError(
        f"ACP SDK is required but not available: {e}. "
        "Please install with: pip install acp-sdk"
    )


class ACPAgentHandle:
    """Handle for an ACP agent using ACP SDK."""
    
    def __init__(self, agent_id: str, session: Session, executor: Any):
        self.agent_id = agent_id
        self.session = session
        self.executor = executor
        self.base_url = f"acp://{agent_id}"
        self._runs: Dict[str, Run] = {}
    
    async def create_run(self, request: RunCreateRequest) -> str:
        """Create a new run in this agent's session and return run_id."""
        run_id = str(uuid.uuid4())
        run = Run(
            id=run_id,
            session_id=self.session.id,
            agent_name=self.agent_id,
            status="in-progress",
            mode=request.mode or RunMode.ASYNC
        )
        self._runs[run_id] = run
        return run_id
    
    async def send_message(self, run_id: str, content: str) -> Message:
        """Send a message to a run."""
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")
        
        # Create message and process with executor
        message_id = str(uuid.uuid4())
        message = Message(
            id=message_id,
            run_id=run_id,
            parts=[{"type": "text", "text": content}]
        )
        
        # Process message with executor
        if hasattr(self.executor, 'process_message'):
            response = await self.executor.process_message(message, run_id)
            return response
        elif hasattr(self.executor, 'execute'):
            response = await self.executor.execute(content)
            response_id = str(uuid.uuid4())
            return Message(
                id=response_id,
                run_id=run_id,
                parts=[{"type": "text", "text": str(response)}]
            )
        
        return message
    
    async def stop(self):
        """Stop the ACP agent."""
        for run in self._runs.values():
            if hasattr(run, 'status') and run.status == "in-progress":
                run.status = "cancelled"


class ACPCommBackend(BaseCommBackend):
    """ACP communication backend using ACP SDK 1.0.3."""
    
    def __init__(self, **kwargs):
        self._agents: Dict[str, ACPAgentHandle] = {}
        self._sessions: Dict[str, Session] = {}
        self._endpoints: Dict[str, str] = {}
    
    async def register_endpoint(self, agent_id: str, address: str) -> None:
        """Register an agent endpoint."""
        self._endpoints[agent_id] = address
    
    async def connect(self, src_id: str, dst_id: str) -> None:
        """Connect two agents."""
        # In ACP, connections are managed through sessions and runs
        pass
    
    async def send_message(self, src_id: str, dst_id: str, message: str) -> Optional[Dict[str, Any]]:
        """Send a message from one agent to another and return serialized response."""
        if dst_id not in self._agents:
            return None
        
        dst_handle = self._agents[dst_id]
        
        # Create a run if needed
        active_runs = [r for r in dst_handle._runs.values() if r.status == "in-progress"]
        if not active_runs:
            run_request = RunCreateRequest(
                agent_name=dst_id,
                input=[{"type": "text", "parts": [{"type": "text", "text": message}]}],
                mode=RunMode.ASYNC
            )
            run_id = await dst_handle.create_run(run_request)
            run = dst_handle._runs[run_id]
        else:
            run = active_runs[0]
        
        # Send message
        try:
            # Get run_id from run or use the created one
            current_run_id = run_id if 'run_id' in locals() else next(
                (rid for rid, r in dst_handle._runs.items() if r == run),
                None
            )
            response_msg = await dst_handle.send_message(current_run_id, message)
            return self._serialize_message(response_msg)
        except Exception as e:
            logging.error(f"Failed to send message to {dst_id}: {e}")
            return {
                "error": str(e),
                "dst_id": dst_id,
                "message": message
            }
    
    async def spawn_local_agent(self, agent_id: str, host: str, port: int, executor: Any) -> ACPAgentHandle:
        """Spawn a local ACP agent."""
        # Create a session for this agent with proper UUID
        session_id = str(uuid.uuid4())
        session = Session(
            id=session_id,
            agent_id=agent_id
        )
        self._sessions[agent_id] = session
        
        # Create agent handle
        handle = ACPAgentHandle(agent_id, session, executor)
        self._agents[agent_id] = handle
        
        return handle
    
    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message between agents with timing instrumentation."""
        message_text = self._extract_message_text(payload)
        request_start = time.time()
        raw_response = await self.send_message(src_id, dst_id, message_text)
        request_end = time.time()

        request_timing = {
            "request_start": request_start,
            "request_end": request_end,
            "total_request_time": request_end - request_start,
            "rate_limited": False
        }

        if not raw_response:
            return {
                "raw": raw_response,
                "text": None,
                "error": "No response",
                "timing": request_timing
            }
        
        answer_text = raw_response.get("text") if isinstance(raw_response, dict) else None
        llm_timing = None
        if isinstance(raw_response, dict):
            llm_timing = raw_response.get("llm_timing")
        
        return {
            "raw": raw_response,
            "text": answer_text,
            "llm_timing": llm_timing,
            "timing": request_timing
        }
    
    async def health_check(self, agent_id: str) -> bool:
        """Check if an agent is healthy (required by BaseCommBackend)."""
        return agent_id in self._agents and self._agents[agent_id] is not None
    
    async def close(self) -> None:
        """Close all agent connections."""
        for handle in self._agents.values():
            await handle.stop()
        self._agents.clear()
        self._sessions.clear()

    def _serialize_message(self, response_msg: Any) -> Dict[str, Any]:
        """Convert ACP SDK Message to serializable dict with parsed timing metadata."""
        if response_msg is None:
            return {}
        parts: List[Dict[str, Any]] = []
        first_text: Optional[str] = None
        for part in getattr(response_msg, 'parts', []) or []:
            part_dict = {
                "type": getattr(part, 'type', None),
                "text": getattr(part, 'text', None)
            }
            parts.append(part_dict)
            if first_text is None and isinstance(part_dict.get("text"), str):
                first_text = part_dict["text"]
        payload = {
            "id": getattr(response_msg, 'id', None),
            "parts": parts,
            "text": first_text
        }

        # Attempt to parse JSON payload for structured data (answer + llm timing)
        if first_text and isinstance(first_text, str):
            try:
                decoded = json.loads(first_text)
                if isinstance(decoded, dict) and "answer" in decoded:
                    payload["text"] = decoded.get("answer")
                    payload["llm_timing"] = decoded.get("llm_timing")
                    payload["payload"] = decoded
            except json.JSONDecodeError:
                pass
        return payload

    def _extract_message_text(self, payload: Any) -> str:
        """Normalize coordinator payloads into plain text messages."""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            for key in ("text", "message", "content"):
                if key in payload and isinstance(payload[key], str):
                    return payload[key]
            if "params" in payload:
                params = payload.get("params", {})
                if isinstance(params, dict):
                    msg = params.get("message")
                    if isinstance(msg, dict):
                        parts = msg.get("parts") or []
                        if parts and isinstance(parts[0], dict):
                            txt = parts[0].get("text") or parts[0].get("data")
                            if isinstance(txt, str):
                                return txt
            # As a fallback, serialize to json for traceability
            try:
                return json.dumps(payload)
            except TypeError:
                pass
        return str(payload)