# -*- coding: utf-8 -*-
"""
A2A Worker (protocol-specific)

- Inherits QAWorkerBase and wraps it using the native A2A SDK into an executable Agent
- Responsible only for the "communication adapter layer": extract user input text -> call base class answer() -> return via A2A events
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import sys
from pathlib import Path

# Add streaming_queue to path for imports
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parent.parent.parent  # Go up from a2a -> protocol_backend -> streaming_queue
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))

# Add core path
streaming_queue_path = Path(__file__).resolve().parent.parent.parent
core_path = streaming_queue_path / "core"
if str(core_path) not in sys.path:
    sys.path.insert(0, str(core_path))
from qa_worker_base import QAWorkerBase

# A2A SDK (required dependency)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message


class A2AQAWorker(QAWorkerBase):
    """No additional logic is needed at the moment; kept as an extension point."""
    pass


class QAAgentExecutor(AgentExecutor):
    """
    A2A native Executor wrapper:
      - Extract text from RequestContext (supports multiple A2A shapes)
      - Call A2AQAWorker.answer()
      - Return text messages via EventQueue
    """

    def __init__(self, config: Optional[dict] = None, output=None):
        self.worker = A2AQAWorker(config=config, output=output)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:  # type: ignore[override]
        try:
            user_input = context.get_user_input() if hasattr(context, "get_user_input") else None
            question = self._extract_text(user_input) or "What is artificial intelligence?"

            result_dict = await self.worker.answer(question)
            
            # Extract answer and timing info
            if isinstance(result_dict, dict):
                answer_text = result_dict.get("answer", str(result_dict))
                llm_timing = result_dict.get("llm_timing")
            else:
                answer_text = str(result_dict)
                llm_timing = None

            event_payload = self._build_event_payload(answer_text, llm_timing)
            await event_queue.enqueue_event(event_payload)
        except Exception as e:
            event_queue.enqueue_event(new_agent_text_message(f"[QAAgentExecutor] failed: {e}"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:  # type: ignore[override]
        event_queue.enqueue_event(new_agent_text_message("cancel not supported"))

    # ------------- helpers -------------
    def _extract_text(self, payload: Any) -> str:
        """
        Accept A2A shapes:
          1) str
          2) {"text": "..."} or {"content":"..."}
          3) {"parts":[{"type":"text","text":"..."}]}
          4) {"parts":[{"type":"json","text":{...}}]}  # serialize to string if dict/list
        """
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload

        if isinstance(payload, dict):
            # Prefer parts
            parts = payload.get("parts") or []
            for p in parts:
                if not isinstance(p, dict):
                    continue
                if "text" in p:
                    t = p.get("text")
                    # text could be str or dict/list if type=="json"
                    if isinstance(t, (dict, list)):
                        try:
                            import json
                            return json.dumps(t, ensure_ascii=False)
                        except Exception:
                            return str(t)
                    if isinstance(t, str):
                        return t
            # Fallbacks
            if isinstance(payload.get("text"), str):
                return payload["text"]
            if isinstance(payload.get("content"), str):
                return payload["content"]

        try:
            return str(payload)
        except Exception:
            return ""

    def _build_event_payload(self, answer_text: str, llm_timing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a serializable event carrying timing metadata."""
        try:
            event = new_agent_text_message(answer_text)
            if hasattr(event, "model_dump"):
                event_dict = event.model_dump(mode="json")
            elif hasattr(event, "dict"):
                event_dict = event.dict()
            else:
                event_dict = {
                    "type": getattr(event, "type", "agent_text_message"),
                    "data": getattr(event, "data", answer_text)
                }
        except Exception:
            event_dict = {
                "type": "agent_text_message",
                "data": answer_text
            }

        if llm_timing:
            event_dict["llm_timing"] = llm_timing
        return event_dict
