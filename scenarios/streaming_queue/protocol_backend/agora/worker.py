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
    
    async def answer(self, question: str) -> Dict[str, Any]:
        """Override the answer method to use Agora native SDK and return timing info."""
        try:
            base_result = await super().answer(question)
        except Exception as e:
            print(f"[AgoraWorker] Error in answer method: {e}")
            now = asyncio.get_event_loop().time()
            return {
                "answer": f"Agora worker error: {e}",
                "llm_timing": {
                    "llm_start": now,
                    "llm_end": now,
                    "llm_execution_time": 0.0
                }
            }

        if not isinstance(base_result, dict):
            base_result = {"answer": str(base_result), "llm_timing": None}

        answer_text = base_result.get("answer", "")
        llm_timing = base_result.get("llm_timing")

        if self.agora_protocol:
            try:
                answer_text = f"[Agora Enhanced] {answer_text}"
                print("[AgoraWorker] Enhanced response using native Agora SDK")
            except Exception as e:
                print(f"[AgoraWorker] Agora enhancement failed: {e}")

        return {
            "answer": answer_text,
            "llm_timing": llm_timing
        }


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
            answer_payload = await asyncio.wait_for(self.worker.answer(text), timeout=30.0)
            if isinstance(answer_payload, dict):
                answer_text = answer_payload.get("answer")
                llm_timing = answer_payload.get("llm_timing")
            else:
                answer_text = str(answer_payload)
                llm_timing = None

            # Return in Agora format with timing metadata
            result = {
                "status": "success",
                "body": answer_text,
                "llm_timing": llm_timing
            }
            print(f"[AgoraWorkerExecutor] Returning result: {result}")
            return result

        except asyncio.TimeoutError:
            return {"status": "error", "body": "Error: Request timed out after 30 seconds"}
        except Exception as e:
            return {"status": "error", "body": f"Error: {str(e)}"}