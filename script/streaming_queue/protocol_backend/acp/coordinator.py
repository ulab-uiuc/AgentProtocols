# -*- coding: utf-8 -*-
"""
ACP Coordinator implementation using ACP SDK 1.0.3.

This implementation uses the official ACP SDK which provides full Agent Communication Protocol support.
"""

from __future__ import annotations

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
    ACP_AVAILABLE = True
except ImportError:
    ACP_AVAILABLE = False


class ACPCoordinatorExecutor:
    """ACP Coordinator Executor using ACP SDK patterns."""
    
    def __init__(self, config: Dict[str, Any], output=None):
        if not ACP_AVAILABLE:
            raise RuntimeError("ACP SDK is not available. Please install acp-sdk.")
        
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
                """Send question to worker using ACP messaging."""
                print(f"[DEBUG] send_to_worker called: {worker_id} <- {question[:50]}...")
                if not self._backend:
                    return {"answer": None, "raw": {"error": "No backend available"}}
                
                try:
                    response = await self._backend.send_message("Coordinator-1", worker_id, question)
                    print(f"[DEBUG] Worker {worker_id} response: {response}")
                    return {
                        "answer": response,
                        "raw": {"response": response, "status": "success"}
                    }
                except Exception as e:
                    print(f"[DEBUG] Worker {worker_id} error: {e}")
                    return {
                        "answer": None,
                        "raw": {"error": str(e), "status": "failed"}
                    }
        
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