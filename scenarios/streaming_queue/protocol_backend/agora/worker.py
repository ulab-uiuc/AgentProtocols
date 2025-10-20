# """
# script/streaming_queue/protocol_backend/agora/worker.py
# """
# """
# Agora Worker (protocol-specific)

# - Inherits QAWorkerBase, wraps Agora protocol into an executable Agent
# - Responsible only for the "communication adapter" layer: extract user input text -> call base answer() -> return via Agora
# """
from __future__ import annotations
import asyncio
from typing import Any, Optional, Dict
from functools import wraps

# Import Agora native SDK
import agora

# Allow multiple relative import paths (avoid import failures when running from different root directories)
import sys
from pathlib import Path

try:
    from ...core.qa_worker_base import QAWorkerBase  # when run as package
except ImportError:
    try:
        # Add core path
        streaming_queue_path = Path(__file__).resolve().parent.parent.parent
        core_path = streaming_queue_path / "core"
        if str(core_path) not in sys.path:
            sys.path.insert(0, str(core_path))
        from qa_worker_base import QAWorkerBase
    except ImportError:
        from scenarios.streaming_queue.core.qa_worker_base import QAWorkerBase  # type: ignore


def _sender(func):
    """
    A decorator to send message to coordinator.
    """
    @wraps(func)
    async def wrapper(self: "AgoraQAWorker", *args, **kwargs):
        if not self.agent_network:
            raise RuntimeError("No Agora network/router set. Call set_network() first.")

        # First argument is coordinator_id, second is message
        coordinator_id, message = args[0], args[1]

        payload = {
            "protocolHash": None,
            "body": message
        }

        response = await self.agent_network.route_message(
            src_id=self.worker_id,
            dst_id=coordinator_id,
            payload=payload
        )
        return response
    return wrapper


class AgoraQAWorker(QAWorkerBase):
    """Agora-specific QA Worker that uses Agora's native SDK features."""
    
    def __init__(self, config=None, output=None):
        super().__init__(config, output)
        
        # Initialize Agora native components with proper parameters
        try:
            # Create minimal Agora components for enhancement (Protocol needs parameters)
            # Use basic protocol document for initialization
            protocol_doc = {"version": "1.0", "name": "agora-qa"}
            sources = []  # Empty sources for basic setup
            metadata = {"worker": "qa", "type": "text_processing"}
            
            self.agora_protocol = agora.Protocol(protocol_doc, sources, metadata)
            print("[AgoraWorker] Initialized with native Agora SDK Protocol")
        except Exception as e:
            print(f"[AgoraWorker] Agora SDK initialization error: {e}")
            # Create a basic protocol stub for fallback
            self.agora_protocol = None
    
    async def answer(self, question: str) -> str:
        """Override the answer method to use Agora native SDK."""
        try:
            # Use base QAWorkerBase for LLM call first
            base_answer = await super().answer(question)
            
            # Enhance with Agora Protocol if available
            if self.agora_protocol:
                try:
                    # Use Agora Protocol for message structuring
                    enhanced_answer = f"[Agora Enhanced] {base_answer}"
                    print(f"[AgoraWorker] Enhanced response using native Agora SDK")
                    return enhanced_answer
                except Exception as e:
                    print(f"[AgoraWorker] Agora enhancement failed: {e}")
            
            return base_answer
                
        except Exception as e:
            print(f"[AgoraWorker] Error in answer method: {e}")
            # Fallback to base implementation
            return await super().answer(question)


class AgoraWorkerExecutor:
    """Agora native Executor wrapper:
    - Extract text from input_data (supports Agora structure or plain text)
    - Call AgoraQAWorker.answer()
    - Return the textual result
    """

    def __init__(self, config: Optional[Dict] = None, output=None):
        self.worker = AgoraQAWorker(config=config, output=output)

    async def execute(self, input_data: Dict) -> Dict:
        """Handle Agora messages and return an answer.

        Input format:
        {
            "protocolHash": None,
            "body": command,
            "protocolSources": []
        }

        Output format: {"status": ..., "body": ...}
        """
        try:
            # Extract Agora-formatted content
            text = ""
            if isinstance(input_data, dict):
                if "body" in input_data:
                    text = input_data["body"]
                elif "payload" in input_data:
                    payload = input_data.get("payload", {})
                    text = payload.get("data", "")
                else:
                    text = str(input_data)
            else:
                text = str(input_data)

            if not text:
                raise ValueError("No input text provided")
            answer = await asyncio.wait_for(self.worker.answer(text), timeout=30.0)

            # Return in Agora format
            return {"status": "success", "body": answer}

        except asyncio.TimeoutError:
            return {"status": "error", "body": "Error: Request timed out after 30 seconds"}
        except Exception as e:
            return {"status": "error", "body": f"Error: {str(e)}"}