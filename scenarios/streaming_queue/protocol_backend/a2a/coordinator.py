# -*- coding: utf-8 -*-
"""
A2A Coordinator:
- Inherits QACoordinatorBase
- Implements send_to_worker() using the native A2A SDK
- Also provides a QACoordinator A2A Executor to expose the coordinator as an A2A Agent
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, Any, Optional, List

import sys
from pathlib import Path

# Add streaming_queue to path for imports
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parent.parent.parent  # Go up from a2a -> protocol_backend -> streaming_queue
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))

from core.qa_coordinator_base import QACoordinatorBase

# --- Native A2A SDK (required) ---
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message


# ============ A2A Coordinator ============

class A2AQACoordinator(QACoordinatorBase):
    """
    Use the A2A protocol to communicate with workers.
    It expects self.agent_network to expose a routing method:
        await agent_network.route_message(src_id, dst_id, payload) -> Dict
    where payload is a standard A2A message (see implementation).
    """

    async def send_to_worker(self, worker_id: str, question: str) -> Dict[str, Any]:
        if not self.agent_network:
            raise RuntimeError("No A2A network/router set. Call set_network() first.")

        # A2A user message (minimal text)
        payload = {
            "messageId": str(int(time.time() * 1000)),
            "role": "user",
            "parts": [
                {"type": "text", "text": question}
            ],
        }

        # Send via A2A router (interface compatible with original code)
        response = await self.agent_network.route_message(self.coordinator_id, worker_id, payload)

        # Parse A2A event stream and extract the text answer
        answer: Optional[str] = None
        try:
            # Process nested response structure, check for "raw" key
            if isinstance(response, dict):
                actual_response = response.get("raw", response)
                events = actual_response.get("events") if isinstance(actual_response, dict) else None
                
                if events:
                    for ev in events:
                        if ev.get("kind") == "message" and "parts" in ev:
                            parts = ev.get("parts") or []
                            if parts and isinstance(parts[0], dict) and "text" in parts[0]:
                                answer = parts[0]["text"]
                                break
        except Exception:
            answer = None

        return {
            "answer": answer or "No answer received",
            "raw": response,
        }


# ============ A2A Executor (expose Coordinator as an A2A Agent) ============

class QACoordinatorExecutor(AgentExecutor):
    """
    Native A2A Executor for interactive control of the coordinator.
    Supports commands:
      - "dispatch" / "start_dispatch"
      - "status"
    """

    def __init__(self, config: Optional[dict] = None, output=None, coordinator: Optional[A2AQACoordinator] = None):
        self.config = config or {}
        self.output = output
        self.coordinator = coordinator or A2AQACoordinator(self.config, self.output)

    # ---- A2A entrypoint ----
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:  # type: ignore[override]
        try:
            raw = context.get_user_input() if hasattr(context, "get_user_input") else "status"
            # Support both structured A2A input and plain string input
            cmd = self._extract_command(raw)

            if cmd in ("dispatch", "start_dispatch"):
                result = await self.coordinator.dispatch_round()
            elif cmd == "status":
                result = await self.coordinator.get_status()
            else:
                result = f"Unknown command: {cmd}. Available commands: dispatch, status"

            # A2A event - use the real A2A SDK (synchronous call)
            event_queue.enqueue_event(new_agent_text_message(result))

        except Exception as e:
            msg = f"QA Coordinator execution failed: {e}"
            event_queue.enqueue_event(new_agent_text_message(msg))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:  # type: ignore[override]
        event_queue.enqueue_event(new_agent_text_message("QA Coordinator operations cancelled."))

    # ---- helpers ----
    def _extract_command(self, user_input: Any) -> str:
        """Extract command text from the A2A context"""
        if isinstance(user_input, dict):
            # Common A2A pattern: {"parts":[{"type":"text","text":"dispatch"}], ...}
            if "parts" in user_input:
                for part in user_input["parts"]:
                    if part.get("type") == "text" or part.get("kind") == "text":
                        return (part.get("text") or "status").strip().lower()
            if "text" in user_input:
                return str(user_input["text"]).strip().lower()
            return "status"
        if not user_input:
            return "status"
        return str(user_input).strip().lower()
