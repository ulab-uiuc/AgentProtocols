# -*- coding: utf-8 -*-
"""
ACP Worker implementation using ACP SDK 1.0.3.

This implementation uses the official ACP SDK which provides full Agent Communication Protocol support.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

# Add streaming_queue to path for imports
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parent.parent.parent  # Go up to streaming_queue
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))

# Also add the current directory for relative imports
if str(current_file.parent) not in sys.path:
    sys.path.insert(0, str(current_file.parent))

# Add core directory specifically
core_path = streaming_queue_path / "core"
if str(core_path) not in sys.path:
    sys.path.insert(0, str(core_path))

from qa_worker_base import QAWorkerBase

# Import ACP SDK components
try:
    from acp_sdk.models import Message, MessagePart
except ImportError as e:
    raise ImportError(
        f"ACP SDK is required but not available: {e}. "
        "Please install with: pip install acp-sdk"
    )


class ACPWorkerExecutor:
    """ACP Worker Executor using ACP SDK patterns."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._worker = None
        
        # Import worker after path setup - use already imported QAWorkerBase
        # QAWorkerBase is already imported at module level
        
        class ACPWorker(QAWorkerBase):
            def __init__(self, config):
                super().__init__(config)
        
        self._worker = ACPWorker(config)
    
    async def process_message(self, message: Message, run_id: str) -> Message:
        """Process an incoming message using ACP patterns."""
        try:
            # Extract text content
            text_content = ""
            if hasattr(message, 'parts') and message.parts:
                for part in message.parts:
                    if hasattr(part, 'type') and part.type == "text":
                        text_content += getattr(part, 'text', "")
            
            # Process with worker using LLM
            if hasattr(self._worker, 'answer'):
                result = await self._worker.answer(text_content)
            else:
                result = f"Error: Worker has no answer method"
            
            # Return response message with strongly typed MessagePart
            response_id = str(uuid.uuid4())
            return Message(
                id=response_id,
                parts=[MessagePart(type="text", text=str(result))]
            )
        except Exception as e:
            error_id = str(uuid.uuid4())
            return Message(
                id=error_id,
                parts=[MessagePart(type="text", text=f"Error: {str(e)}")]
            )
    
    @property
    def worker(self):
        """Access to the underlying worker."""
        return self._worker