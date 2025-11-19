"""
Agora Coordinator:
- Inherits QACoordinatorBase
- Implements send_to_worker() using Agora native SDK
- Uses real Agora SDK features: Sender, Receiver, Protocol
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
# Add streaming_queue to path for imports
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parent.parent.parent
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))
from core.qa_coordinator_base import QACoordinatorBase

# Import Agora native SDK components
import agora

class AgoraQACoordinator(QACoordinatorBase):
    """Communicate with workers using the Agora protocol.

    Assumes self.agent_network exposes a channel routing method::
        await agent_network.route_message(src_id, dst_id, payload) -> Dict

    Where payload follows Agora's standard message structure (see implementation).
    """

    async def send_to_worker(self, worker_id: str, question: str) -> Dict[str, Any]:
        if not self.agent_network:
            raise RuntimeError("No Agora network/router set. Call set_network() first.")

        # Agora user message (simple text)
        payload = {
            "protocolHash": None,
            "body": question
        }

        # Route the message via agent_network, using only the official SDK
        response = await self.agent_network.route_message(
            src_id=self.coordinator_id,
            dst_id=worker_id,
            payload=payload
        )
        
        print(f"[AgoraCoordinator] Received response for {worker_id}: {response}")

        answer = None
        request_timing = {}
        llm_timing = None
        raw_payload: Dict[str, Any] = {}
        raw_response = response if isinstance(response, dict) else {"response": response}

        if isinstance(response, dict):
            raw_payload = response.get("raw", response)
            request_timing = response.get("timing", {}) or {}
            llm_timing = response.get("llm_timing")
            try:
                if raw_payload and isinstance(raw_payload, dict) and "body" in raw_payload:
                    answer = raw_payload.get("body")
            except Exception:
                answer = None
            if not answer:
                answer = response.get("text")
            if not llm_timing:
                llm_timing = self._extract_llm_timing(raw_payload)
            if response and answer:
                response["text"] = answer
        else:
            raw_payload = {"response": response}
            answer = str(response)

        adapter_time = None
        if request_timing and llm_timing:
            adapter_time = (
                request_timing.get("total_request_time", 0.0)
                - llm_timing.get("llm_execution_time", 0.0)
            )
        
        final_result = {
            "answer": answer,
            "raw": raw_response,
            "timing": {
                "request_timing": request_timing,
                "llm_timing": llm_timing,
                "adapter_time": adapter_time
            }
        }
        print(f"[AgoraCoordinator] Final result for {worker_id}: answer={answer[:50] if answer else None}, timing present={bool(llm_timing)}")

        return final_result

    def _extract_llm_timing(self, raw_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(raw_payload, dict):
            return None
        if raw_payload.get("llm_timing"):
            return raw_payload.get("llm_timing")
        inner = raw_payload.get("raw")
        if isinstance(inner, dict) and inner.get("llm_timing"):
            return inner.get("llm_timing")
        return None
    
class AgoraCoordinatorExecutor:
    """Agora native Executor used for conversational coordinator control.

    Supported commands:
      - "dispatch" / "start_dispatch"
      - "status"
      - "setup_network Worker-1,Worker-2,Worker-3,Worker-4"
    """

    def __init__(self, config: Dict | None = None, output=None):
        self.coordinator = AgoraQACoordinator(config, output)

    async def execute(self, input_data: Dict) -> Dict:
        """
        Handle Agora messages and execute coordinator commands
        """
        # Extract command text from input
        user_text = ""
        if isinstance(input_data, dict):
            if "content" in input_data:
                # Handle content array format
                parts = input_data["content"]
                if isinstance(parts, list):
                    user_text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
                else:
                    user_text = str(parts)
            elif "body" in input_data:
                user_text = input_data["body"]
            else:
                user_text = "status"
        else:
            user_text = str(input_data)

        cmd = user_text.strip().lower()
        original_text = user_text.strip()  # Preserve original case for worker IDs
        


        try:
            if "dispatch" in cmd or "start dispatch" in cmd:
                # Check if already dispatching to avoid duplicate processing
                if hasattr(self.coordinator, '_dispatching') and self.coordinator._dispatching:
                    result = "Dispatch already in progress"
                else:
                    self.coordinator._dispatching = True
                    try:
                        result = await self.coordinator.dispatch_round()
                    finally:
                        self.coordinator._dispatching = False
            elif cmd.startswith("setup_network"):
                try:
                    worker_list = original_text.split(" ", 1)[1] if " " in original_text else ""
                    worker_ids = [w.strip() for w in worker_list.split(",") if w.strip()]

                    # Setup coordinator network access
                    self.coordinator.worker_ids = worker_ids
                    self.coordinator.coordinator_id = "Coordinator-1"

                    result = f"Agora network setup complete with {len(worker_ids)} workers: {worker_ids}"
                except Exception as e:
                    result = f"Agora network setup failed: {e}"
            else:
                result = await self.coordinator.get_status()

            return {
                "status": "success",
                "body": result
            }

        except Exception as e:
            return {
                "status": "error",
                "body": f"Agora Coordinator error: {str(e)}"
            }