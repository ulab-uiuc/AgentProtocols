# -*- coding: utf-8 -*-
"""
ACP Coordinator implementation using ACP SDK 1.0.3.

This implementation uses the official ACP SDK which provides full Agent Communication Protocol support.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

# Add streaming_queue to path for imports
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parent.parent.parent
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))

from core.qa_coordinator_base import QACoordinatorBase

# Import ACP SDK components
try:
    from acp_sdk import Message
except ImportError as e:
    raise ImportError(
        f"ACP SDK is required but not available: {e}. "
        "Please install with: pip install acp-sdk"
    )


class ACPCoordinatorExecutor:
    """ACP Coordinator Executor using ACP SDK patterns."""
    
    def __init__(self, config: Dict[str, Any], output=None):
        self.config = config
        self.output = output
        self._coordinator = None
        
        # Import coordinator after path setup
        from core.qa_coordinator_base import QACoordinatorBase
        
        class ACPCoordinator(QACoordinatorBase):
            def __init__(self, config, output):
                super().__init__(config, output)
                self._backend = None
            
            def set_backend(self, backend):
                """Set the communication backend."""
                self._backend = backend
            
            async def send_to_worker(self, worker_id: str, question: str) -> Dict[str, Any]:
                """Send question to worker using ACP messaging with timing metadata."""
                if not self.agent_network and not self._backend:
                    return {"answer": None, "raw": {"error": "No communication channel available"}}

                payload = {"text": question}

                try:
                    if self.agent_network:
                        try:
                            response = await self.agent_network.route_message(self.coordinator_id, worker_id, payload)
                        except Exception as route_error:
                            if self._backend:
                                response = await self._backend.send("Coordinator-1", worker_id, payload)  # type: ignore[arg-type]
                            else:
                                raise route_error
                    else:
                        response = await self._backend.send("Coordinator-1", worker_id, payload)  # type: ignore[arg-type]
                except Exception as e:
                    return {
                        "answer": None,
                        "raw": {"error": str(e)},
                        "timing": {"request_timing": {}, "llm_timing": {}, "adapter_time": None}
                    }

                request_timing = {}
                llm_timing = None
                answer = None
                raw_payload: Dict[str, Any] = {}
                raw_response = response if isinstance(response, dict) else {"response": response}

                if isinstance(response, dict):
                    request_timing = response.get("timing", {}) or {}
                    llm_timing = response.get("llm_timing")
                    answer = response.get("text")
                    raw_payload = response.get("raw", response)
                    if not llm_timing:
                        llm_timing = self._extract_llm_timing(raw_payload)
                else:
                    answer = str(response)
                    raw_payload = {"response": response}
                    raw_response = raw_payload

                adapter_time = None
                if request_timing and llm_timing:
                    adapter_time = (
                        request_timing.get("total_request_time", 0.0)
                        - llm_timing.get("llm_execution_time", 0.0)
                    )

                return {
                    "answer": answer,
                    "raw": raw_response,
                    "timing": {
                        "request_timing": request_timing,
                        "llm_timing": llm_timing,
                        "adapter_time": adapter_time
                    }
                }

            def _extract_llm_timing(self, raw_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                if not isinstance(raw_payload, dict):
                    return None
                if raw_payload.get("llm_timing"):
                    return raw_payload.get("llm_timing")
                parts = raw_payload.get("parts") or []
                for part in parts:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_val = part.get("text")
                        if isinstance(text_val, str):
                            try:
                                decoded = json.loads(text_val)
                                if isinstance(decoded, dict) and decoded.get("llm_timing"):
                                    return decoded.get("llm_timing")
                            except json.JSONDecodeError:
                                continue
                return None
        
        self._coordinator = ACPCoordinator(config, output)
    
    def set_backend(self, backend):
        """Set the communication backend for the coordinator."""
        if hasattr(self._coordinator, 'set_backend'):
            self._coordinator.set_backend(backend)
    
    async def process_message(self, message: Message, run_id: str) -> Message:
        """Process an incoming message using ACP patterns."""
        try:
            # Extract text content
            text_content = ""
            if hasattr(message, 'parts') and message.parts:
                for part in message.parts:
                    if hasattr(part, 'type') and part.type == "text":
                        text_content += getattr(part, 'text', "")
            
            # Process with coordinator based on command
            if text_content.strip().lower() == "dispatch":
                result = await self._coordinator.dispatch_round()
            elif text_content.strip().lower() == "status":
                result = await self._coordinator.get_status()
            else:
                result = f"Unknown command: {text_content}"
            
            # Return response message
            response_id = str(uuid.uuid4())
            return Message(
                id=response_id,
                parts=[{"type": "text", "text": str(result)}]
            )
        except Exception as e:
            error_id = str(uuid.uuid4())
            return Message(
                id=error_id,
                parts=[{"type": "text", "text": f"Error: {str(e)}"}]
            )
    
    @property
    def coordinator(self):
        """Access to the underlying coordinator."""
        return self._coordinator