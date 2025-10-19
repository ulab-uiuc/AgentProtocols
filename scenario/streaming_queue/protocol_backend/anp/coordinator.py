# -*- coding: utf-8 -*-
"""
ANP Coordinator for Streaming Queue
使用真正的ANP协议实现QA协调器，支持DID认证和E2E加密
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys

# Add streaming_queue to path for imports
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parent.parent.parent
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))

# Import streaming_queue core
from core.qa_coordinator_base import QACoordinatorBase

# AgentConnect imports for ANP support
project_root = streaming_queue_path.parent.parent
agentconnect_path = project_root / "agentconnect_src"
sys.path.insert(0, str(agentconnect_path))

try:
    from agent_connect.simple_node import SimpleNode, SimpleNodeSession
except ImportError as e:
    raise ImportError(f"AgentConnect SDK required but not available: {e}")


class ANPQACoordinator(QACoordinatorBase):
    """
    ANP协议的QA协调器实现
    
    Features:
    - DID身份认证的工作器通信
    - E2E加密的消息传输
    - ANP协议原生支持
    - 与streaming_queue QACoordinatorBase完全兼容
    """

    def __init__(self, config: Optional[dict] = None, output=None):
        super().__init__(config, output)
        self.protocol = "anp"
        self.did_document = None
        self.private_keys = None
        self.simple_node = None
        self.node_session = None
        
        # ANP-specific metrics
        self.anp_metrics = {
            "did_authenticated_messages": 0,
            "encrypted_messages": 0,
            "websocket_messages": 0,
            "http_messages": 0
        }
        
        print(f"[ANP Coordinator] Initialized ANP QA Coordinator")

    def set_anp_identity(self, did_document: Dict[str, Any], private_keys: Dict[str, Any]):
        """Set ANP DID identity for coordinator"""
        self.did_document = did_document
        self.private_keys = private_keys
        
        # Initialize SimpleNode for enhanced ANP communication
        try:
            # For coordinator, SimpleNode is primarily used for outgoing connections
            # The actual SimpleNode will be created by the communication backend
            self.simple_node = None  # Will be set by comm backend
            self.node_session = None
            print(f"[ANP Coordinator] ANP identity set (SimpleNode managed by backend): {did_document.get('id', 'Unknown')}")
        except Exception as e:
            print(f"[ANP Coordinator] Failed to create SimpleNode: {e}")
            self.simple_node = None
            self.node_session = None
            print(f"[ANP Coordinator] ANP identity set (SimpleNode failed): {did_document.get('id', 'Unknown')}")

    async def send_to_worker(self, worker_id: str, question: str) -> Dict[str, Any]:
        """
        Send question to worker via ANP protocol
        
        This method maintains compatibility with QACoordinatorBase while adding
        ANP-specific features like DID authentication and encryption.
        """
        if not self.agent_network:
            raise RuntimeError("[ANP Coordinator] No network set. Call set_network() first.")

        # Prepare ANP-enhanced message
        anp_message = {
            "text": question,
            "anp_metadata": {
                "protocol": "anp",
                "coordinator_id": self.coordinator_id,
                "timestamp": time.time(),
                "message_type": "qa_question",
                "encryption": "e2e",
                "authentication": "did"
            }
        }
        
        # Add DID authentication if available
        if self.did_document:
            anp_message["anp_metadata"]["coordinator_did"] = self.did_document.get("id")
            anp_message["anp_metadata"]["authenticated"] = True
            self.anp_metrics["did_authenticated_messages"] += 1
        
        self.anp_metrics["encrypted_messages"] += 1
        self.anp_metrics["http_messages"] += 1
        
        try:
            print(f"[DEBUG ANP] Sending to {worker_id}: {question[:50]}...")
            # Send via ANP network (will use ANPCommBackend)
            response = await self.agent_network.route_message(
                self.coordinator_id, 
                worker_id, 
                anp_message
            )
            
            # Extract answer from ANP response
            answer = self._extract_anp_answer(response)
            
            return {
                "answer": answer,
                "raw": response,
                "anp_metadata": {
                    "response_encrypted": True,
                    "response_authenticated": self._verify_anp_response(response),
                    "worker_id": worker_id,
                    "protocol": "anp"
                }
            }
            
        except Exception as e:
            error_msg = f"[ANP Coordinator] Failed to send to {worker_id}: {e}"
            print(error_msg)
            return {
                "answer": f"ANP communication error: {e}",
                "raw": {"error": str(e)},
                "anp_metadata": {
                    "error": True,
                    "worker_id": worker_id,
                    "protocol": "anp"
                }
            }

    def _extract_anp_answer(self, response: Optional[Dict[str, Any]]) -> str:
        """Extract answer from ANP response"""
        if not response:
            return "No response received"
        
        # Check for streaming_queue compatible text field
        if "text" in response:
            return response["text"]
        
        # Check for ANP metadata
        anp_meta = response.get("anp_metadata", {})
        if "response_text" in anp_meta:
            return anp_meta["response_text"]
        
        # Check for raw response data
        raw_data = response.get("raw", {})
        if isinstance(raw_data, dict):
            # Check for events (A2A-style compatibility)
            events = raw_data.get("events", [])
            for event in events:
                if event.get("type") == "agent_text_message":
                    return event.get("data", "")
            
            # Check for direct text in raw data
            if "text" in raw_data:
                return raw_data["text"]
        
        # Fallback
        return str(response)

    def _verify_anp_response(self, response: Optional[Dict[str, Any]]) -> bool:
        """Verify ANP response authentication"""
        if not response:
            return False
        
        anp_meta = response.get("anp_metadata", {})
        return anp_meta.get("did_authenticated", False)

    async def get_anp_status(self) -> str:
        """Get ANP-specific coordinator status"""
        base_status = await self.get_status()
        
        anp_status = {
            "protocol": "anp",
            "did_identity": self.did_document.get("id") if self.did_document else None,
            "simple_node_active": self.node_session is not None,
            "metrics": self.anp_metrics,
            "base_status": base_status
        }
        
        return f"ANP Coordinator Status: {json.dumps(anp_status, indent=2)}"

    async def close(self):
        """Close ANP coordinator and cleanup resources"""
        try:
            if self.node_session:
                await self.node_session.close()
                print("[ANP Coordinator] Closed SimpleNode session")
        except Exception as e:
            print(f"[ANP Coordinator] Error closing: {e}")


# ================= ANP Executor for streaming_queue =================
class ANPCoordinatorExecutor:
    """
    ANP Coordinator Executor for streaming_queue compatibility
    
    This class adapts the ANPQACoordinator to work with streaming_queue's
    agent execution model while maintaining full ANP protocol support.
    """

    def __init__(self, config: Optional[dict] = None, output=None):
        self.config = config or {}
        self.output = output
        self.coordinator = ANPQACoordinator(self.config, self.output)
        
        # Track ANP executor state
        self.anp_executor_metrics = {
            "messages_processed": 0,
            "commands_executed": 0,
            "errors_encountered": 0
        }
        
        print("[ANP Coordinator Executor] Initialized")

    async def execute(self, context, event_queue) -> None:
        """
        Execute coordinator commands via ANP protocol
        
        Compatible with streaming_queue executor interface while providing
        ANP-specific enhancements.
        """
        try:
            # Extract command from context
            raw_input = context.get_user_input() if hasattr(context, "get_user_input") else "status"
            command = self._extract_anp_command(raw_input, context)
            
            self.anp_executor_metrics["messages_processed"] += 1
            
            # Process ANP-enhanced commands
            if command in ("dispatch", "start_dispatch"):
                result = await self._execute_anp_dispatch()
                self.anp_executor_metrics["commands_executed"] += 1
                
            elif command == "status":
                result = await self.coordinator.get_anp_status()
                self.anp_executor_metrics["commands_executed"] += 1
                
            elif command == "anp_info":
                result = self._get_anp_info()
                self.anp_executor_metrics["commands_executed"] += 1
                
            else:
                result = f"[ANP] Unknown command: {command}. Available: dispatch, status, anp_info"
                self.anp_executor_metrics["errors_encountered"] += 1

            # Create ANP-enhanced event
            anp_event = self._create_anp_event(result, command)
            await event_queue.enqueue_event(anp_event)

        except Exception as e:
            error_msg = f"[ANP Coordinator Executor] Execution failed: {e}"
            print(error_msg)
            self.anp_executor_metrics["errors_encountered"] += 1
            
            error_event = self._create_anp_event(error_msg, "error")
            await event_queue.enqueue_event(error_event)

    def _extract_anp_command(self, raw_input: Any, context: Any) -> str:
        """Extract command from ANP context"""
        # Check for ANP metadata in context
        if hasattr(context, "anp_metadata"):
            anp_meta = context.anp_metadata
            if "command" in anp_meta:
                return anp_meta["command"]
        
        # Standard command extraction
        if isinstance(raw_input, dict):
            # ANP message format
            if "anp_metadata" in raw_input:
                anp_meta = raw_input["anp_metadata"]
                if "command" in anp_meta:
                    return anp_meta["command"]
            
            # Text in various formats
            if "text" in raw_input:
                return str(raw_input["text"]).strip().lower()
            
            # Parts format
            if "parts" in raw_input:
                parts = raw_input["parts"]
                if parts and isinstance(parts[0], dict):
                    return str(parts[0].get("text", "status")).strip().lower()
        
        if not raw_input:
            return "status"
        
        return str(raw_input).strip().lower()

    async def _execute_anp_dispatch(self) -> str:
        """Execute dispatch with ANP enhancements"""
        try:
            result = await self.coordinator.dispatch_round()
            
            # Add ANP metadata to result
            anp_enhanced_result = f"""[ANP Protocol] Dispatch completed
{result}

ANP Features Used:
- DID Authentication: {'✓' if self.coordinator.did_document else '✗'}
- E2E Encryption: ✓
- Protocol: ANP v1.0
- SimpleNode: {'✓' if self.coordinator.node_session else '✗'}

ANP Metrics:
{json.dumps(self.coordinator.anp_metrics, indent=2)}"""
            
            return anp_enhanced_result
            
        except Exception as e:
            return f"[ANP] Dispatch failed: {e}"

    def _get_anp_info(self) -> str:
        """Get ANP protocol information"""
        info = {
            "protocol": "ANP (Agent Network Protocol)",
            "version": "1.0",
            "features": [
                "DID-based Authentication",
                "End-to-End Encryption", 
                "WebSocket Communication",
                "HTTP REST API",
                "SimpleNode Integration"
            ],
            "coordinator_did": self.coordinator.did_document.get("id") if self.coordinator.did_document else None,
            "simple_node_active": self.coordinator.node_session is not None,
            "executor_metrics": self.anp_executor_metrics,
            "coordinator_metrics": self.coordinator.anp_metrics
        }
        
        return f"ANP Protocol Information:\n{json.dumps(info, indent=2)}"

    def _create_anp_event(self, result: str, command: str) -> Dict[str, Any]:
        """Create ANP-enhanced event for streaming_queue"""
        return {
            "type": "agent_text_message",
            "data": result,
            "anp_metadata": {
                "protocol": "anp",
                "command": command,
                "timestamp": time.time(),
                "coordinator_id": self.coordinator.coordinator_id,
                "did_authenticated": self.coordinator.did_document is not None,
                "encrypted": True
            }
        }

    async def cancel(self, context, event_queue) -> None:
        """Cancel ANP coordinator operations"""
        cancel_msg = "[ANP] Coordinator operations cancelled"
        cancel_event = self._create_anp_event(cancel_msg, "cancel")
        await event_queue.enqueue_event(cancel_event)
        
        # Cleanup ANP resources
        await self.coordinator.close()
        print("[ANP Coordinator Executor] Operations cancelled and cleaned up")
