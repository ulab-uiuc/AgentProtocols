# -*- coding: utf-8 -*-
"""
A2A Communication Backend for Privacy Protection Testing
Implements A2A protocol communication for privacy-aware agent interactions.
"""

from __future__ import annotations
import asyncio
import json
import uuid
import httpx
from typing import Any, Dict, Optional
from dataclasses import dataclass

# Import base communication interface
try:
    from ...comm.base import BaseCommBackend
except ImportError:
    from comm.base import BaseCommBackend

# A2A SDK imports - REQUIRED for A2A protocol operation
try:
    from a2a.client import A2AClient
    from a2a.utils import new_agent_text_message
    from a2a.types import SendMessageRequest
    print("[A2A] Successfully imported A2A SDK for privacy testing")
except ImportError as e:
    raise ImportError(
        f"A2A SDK is required but not available: {e}. "
        "Please install with: pip install a2a-sdk"
    )


def create_safe_privacy_message(message_type: str, payload: dict, target: str = None) -> dict:
    """Create a safe A2A message for privacy testing that is guaranteed to be JSON serializable"""
    safe_payload = {}
    if payload:
        # Ensure all payload values are JSON serializable
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                safe_payload[str(key)] = value
            elif isinstance(value, (list, tuple)):
                safe_payload[str(key)] = [str(item) if not isinstance(item, (int, float, bool)) else item for item in value]
            else:
                safe_payload[str(key)] = str(value)
    
    return {
        "type": str(message_type),
        "payload": safe_payload,
        "target": str(target) if target else None,
        "message_id": str(uuid.uuid4()),
        "protocol": "a2a_privacy"  # 标识隐私测试协议
    }


@dataclass
class A2APrivacyAgentHandle:
    """Handle for A2A privacy testing agent with connection info"""
    agent_id: str
    agent_type: str  # "receptionist", "doctor", "analyzer"
    endpoint_url: str
    is_connected: bool = False
    client: Optional[A2AClient] = None


class A2ACommBackend(BaseCommBackend):
    """A2A protocol communication backend for privacy protection testing"""

    def __init__(self, router_url: str = "http://localhost:8080", **kwargs):
        self.router_url = router_url
        self._endpoints: Dict[str, str] = {}
        self._agent_handles: Dict[str, A2APrivacyAgentHandle] = {}
        self._clients: Dict[str, A2AClient] = {}
        self._connected = False
        self._message_queue = asyncio.Queue()

    async def register_endpoint(self, agent_id: str, address: str) -> None:
        """Register privacy testing agent endpoint"""
        self._endpoints[agent_id] = address
        
        # Parse agent type from agent_id for privacy testing
        agent_type = "unknown"
        if "Receptionist" in agent_id:
            agent_type = "receptionist"
        elif "Doctor" in agent_id:
            agent_type = "doctor"
        elif "Analyzer" in agent_id:
            agent_type = "analyzer"
        
        # Create agent handle
        handle = A2APrivacyAgentHandle(
            agent_id=agent_id,
            agent_type=agent_type,
            endpoint_url=address
        )
        self._agent_handles[agent_id] = handle
        
        # Initialize A2A client
        try:
            httpx_client = httpx.AsyncClient(timeout=30.0)
            a2a_client = A2AClient(httpx_client, url=address)
            self._clients[agent_id] = a2a_client
            handle.client = a2a_client
            handle.is_connected = True
            print(f"[A2ACommBackend] Registered privacy agent {agent_id} ({agent_type}) @ {address}")
        except Exception as e:
            print(f"[A2ACommBackend] Failed to create A2A client for {agent_id}: {e}")

    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """Send privacy testing message via A2A protocol"""
        dst_endpoint = self._endpoints.get(dst_id)
        if not dst_endpoint:
            raise RuntimeError(f"Unknown destination agent: {dst_id}")

        try:
            # Create privacy-specific A2A message
            privacy_message = self._to_privacy_message(src_id, dst_id, payload)

            if dst_id not in self._clients:
                raise RuntimeError(f"No A2A client available for {dst_id}")

            # Send via A2A SDK
            message_text = json.dumps(privacy_message)
            a2a_message = new_agent_text_message(message_text)

            # Create SendMessageRequest with proper routing
            request = SendMessageRequest(
                id=str(uuid.uuid4()),
                jsonrpc="2.0",
                method="message/send",
                params={
                    "message": a2a_message,
                    "receiver_id": dst_id,
                    "sender_id": src_id,
                    "privacy_context": True  # 标识隐私测试上下文
                }
            )

            client = self._clients[dst_id]
            response = await client.send_message(request)

            # Extract response content for privacy analysis
            response_content = self._extract_privacy_response(response)

            print(f"[A2ACommBackend] Privacy message sent: {src_id} -> {dst_id}")
            return {
                "raw": response,
                "text": response_content,
                "privacy_safe": True
            }
                
        except Exception as e:
            print(f"[A2ACommBackend] Privacy message send failed {src_id} -> {dst_id}: {e}")
            return {
                "raw": None,
                "text": f"Privacy communication error: {e}",
                "privacy_safe": False
            }

    async def health_check(self, agent_id: str) -> bool:
        """Health check for A2A privacy testing agent"""
        if agent_id not in self._endpoints:
            return False
        
        endpoint = self._endpoints[agent_id]
        
        # For a2a:// mock endpoints (simulation mode)
        if endpoint.startswith("a2a://"):
            return True
        
        # Use A2A client for health check
        if agent_id in self._clients:
            try:
                client = self._clients[agent_id]
                # Simple ping via A2A
                ping_message = create_safe_privacy_message("HEALTH_CHECK", {"ping": "privacy_test"})
                request = SendMessageRequest(
                    id=str(uuid.uuid4()),
                    jsonrpc="2.0",
                    method="message/send",
                    params={
                        "message": new_agent_text_message(json.dumps(ping_message)),
                        "receiver_id": agent_id
                    }
                )
                response = await asyncio.wait_for(client.send_message(request), timeout=5.0)
                return response is not None
            except Exception as e:
                print(f"[A2ACommBackend] Health check failed for {agent_id}: {e}")
                return False

        # Fallback HTTP health check
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{endpoint}/health", timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close A2A privacy testing resources"""
        try:
            # Close all A2A clients
            for agent_id, client in self._clients.items():
                if hasattr(client, 'close'):
                    await client.close()
                print(f"[A2ACommBackend] Closed A2A client for {agent_id}")
            
            # Clear handles and endpoints
            self._clients.clear()
            self._agent_handles.clear()
            self._endpoints.clear()
            self._connected = False
            
            print("[A2ACommBackend] Privacy testing A2A backend closed")
        except Exception as e:
            print(f"[A2ACommBackend] Error during close: {e}")

    # Privacy-specific helper methods
    def _to_privacy_message(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convert payload to privacy-aware A2A message format"""
        # Extract privacy context from agent types
        src_handle = self._agent_handles.get(src_id)
        dst_handle = self._agent_handles.get(dst_id)
        
        privacy_context = {
            "src_agent_type": src_handle.agent_type if src_handle else "unknown",
            "dst_agent_type": dst_handle.agent_type if dst_handle else "unknown",
            "interaction_type": self._determine_interaction_type(src_handle, dst_handle),
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Determine message type based on payload
        if "text" in payload:
            message_type = "PRIVACY_QUERY" if "question" in payload.get("text", "").lower() else "PRIVACY_RESPONSE"
        else:
            message_type = "PRIVACY_MESSAGE"
        
        return create_safe_privacy_message(
            message_type=message_type,
            payload={
                **payload,
                "privacy_context": privacy_context,
                "src_id": src_id,
                "dst_id": dst_id
            },
            target=dst_id
        )

    def _determine_interaction_type(self, src_handle: Optional[A2APrivacyAgentHandle], 
                                   dst_handle: Optional[A2APrivacyAgentHandle]) -> str:
        """Determine the type of privacy interaction"""
        if not src_handle or not dst_handle:
            return "unknown"
        
        interaction_map = {
            ("receptionist", "doctor"): "privacy_consultation",
            ("doctor", "receptionist"): "medical_inquiry", 
            ("patient", "receptionist"): "initial_contact",
            ("analyzer", "receptionist"): "privacy_audit",
            ("analyzer", "doctor"): "privacy_audit"
        }
        
        return interaction_map.get((src_handle.agent_type, dst_handle.agent_type), "general_interaction")

    def _extract_privacy_response(self, response: Any) -> str:
        """Extract text content from A2A response for privacy analysis"""
        if not response:
            return ""
        
        # Handle different A2A response formats
        if isinstance(response, dict):
            # Direct JSON response
            if "content" in response:
                return str(response["content"])
            elif "text" in response:
                return str(response["text"])
            elif "result" in response:
                result = response["result"]
                if isinstance(result, dict) and "content" in result:
                    return str(result["content"])
                return str(result)
        elif hasattr(response, 'content'):
            return str(response.content)
        elif hasattr(response, 'text'):
            return str(response.text)
        
        return str(response)

    async def get_privacy_statistics(self) -> Dict[str, Any]:
        """Get privacy testing statistics from A2A backend"""
        return {
            "total_agents": len(self._agent_handles),
            "connected_agents": sum(1 for h in self._agent_handles.values() if h.is_connected),
            "agent_types": {agent_type: sum(1 for h in self._agent_handles.values() if h.agent_type == agent_type)
                           for agent_type in ["receptionist", "doctor", "analyzer"]},
            "protocol": "a2a"
        }
