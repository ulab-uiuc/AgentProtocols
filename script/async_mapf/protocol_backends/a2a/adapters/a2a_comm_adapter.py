import asyncio
import json
import uuid
from dataclasses import asdict
from typing import Any, Dict

# A2A SDK client imports - REQUIRED, no fallback allowed
try:
    from a2a.client import A2AClient
    from a2a.utils import new_agent_text_message
    from a2a.types import SendMessageRequest
    import httpx
    print("[A2A] Successfully imported A2A SDK")
except ImportError as e:
    print(f"[FATAL ERROR] A2A SDK is required but not available: {e}")
    print("Please install A2A SDK or ensure it's in the Python path")
    raise ImportError("A2A SDK is mandatory for this system") from e

# Assuming these core files are correctly located relative to the execution root
from script.async_mapf.core.comm import AbstractCommAdapter
from script.async_mapf.core.types import MoveCmd, MoveFeedback

def create_safe_a2a_message(message_type: str, payload: dict, target: str = None) -> dict:
    """Create a safe A2A message that is guaranteed to be JSON serializable"""
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
        "message_id": str(uuid.uuid4())
    }

class A2ACommAdapter(AbstractCommAdapter):
    """Real A2A adapter using A2A SDK Client"""

    def __init__(self, self_id: str, router_url: str):
        self.self_id = self_id
        self.router_url = router_url
        self._rx_q = asyncio.Queue()
        self._client = None
        self._httpx_client = None
        self._connected = False
        
        # Validate router URL or agent URL
        if not router_url:
            raise ValueError(f"router_url is required for A2A communication")
        
        # Remove any path from router_url - A2AClient expects base URL only
        if "/api/" in router_url:
            router_url = router_url.split("/api/")[0]
        
        # Initialize httpx client and A2A Client with base URL only
        self._httpx_client = httpx.AsyncClient()
        self._client = A2AClient(self._httpx_client, url=router_url)
        print(f"[{self_id}] A2ACommAdapter initialized with A2A Client for: {router_url} (will use /api/jsonrpc)")

    async def connect(self):
        """Initialize A2A connection (A2AClient is HTTP-based, no persistent connection needed)"""
        if self._client and not self._connected:
            try:
                # A2AClient is ready to use immediately, no explicit connect needed
                self._connected = True
                print(f"[{self.self_id}] A2A Client ready for HTTP communication to: {self.router_url}")
            except Exception as e:
                print(f"[{self.self_id}] Failed to initialize A2A Client: {e}")
                raise ConnectionError(f"Cannot initialize A2A Client for {self.router_url}") from e
    
    async def disconnect(self):
        """Disconnect from A2A system"""
        # Close httpx client
        if self._httpx_client:
            await self._httpx_client.aclose()
            
        self._connected = False
        print(f"[{self.self_id}] Disconnected from A2A system")

    # -------- AbstractCommAdapter methods --------
    async def send(self, obj: Any) -> None:
        """Send message via A2A system"""
        if not self._connected:
            raise ConnectionError(f"A2A Client not connected. Call connect() first.")
        
        try:
            # Serialize the object to A2A message format
            serialized_data = self._serialize(obj)
            
            # Create A2A message
            message_text = json.dumps(serialized_data)
            a2a_message = new_agent_text_message(message_text)
            
            # Create proper SendMessageRequest (JSON-RPC format) with receiver routing
            target_id = serialized_data.get('target', 'network')
            request = SendMessageRequest(
                id=str(uuid.uuid4()),
                jsonrpc="2.0",
                method="message/send",
                params={
                    "message": a2a_message,
                    "receiver_id": target_id  # Add receiver_id for Router routing
                }
            )
            
            # Send the message using A2A Client
            response = await self._client.send_message(request)
            print(f"[{self.self_id}] Sent A2A message: {serialized_data['type']} to {serialized_data.get('target', 'network')}")
                
        except Exception as e:
            print(f"[{self.self_id}] Error sending A2A message: {e}")
            raise

    async def recv(self) -> Any:
        return await self._rx_q.get()

    def recv_nowait(self) -> Any:
        return self._rx_q.get_nowait()

    # -------- internal helpers --------
    def handle_incoming_message(self, message_data: str):
        """Handle incoming A2A message (called by HTTP request handler)"""
        try:
            # Deserialize and put in rx queue
            deserialized = self._deserialize(message_data)
            if deserialized:
                # Use asyncio.create_task to put message in queue without blocking
                asyncio.create_task(self._rx_q.put(deserialized))
                print(f"[{self.self_id}] Received A2A message: {type(deserialized).__name__}")
        except Exception as e:
            print(f"[{self.self_id}] Error handling incoming message: {e}")

    def _serialize(self, obj: Any) -> Dict[str, Any]:
        """Serializes framework objects into A2A messages."""
        if isinstance(obj, MoveCmd):
            return create_safe_a2a_message("MOVE_CMD", {
                "agent_id": obj.agent_id,
                "new_pos": list(obj.new_pos)
            }, target="network")
        elif isinstance(obj, MoveFeedback):
            return create_safe_a2a_message("MOVE_FB", {
                "agent_id": obj.agent_id,
                "success": obj.success,
                "actual_pos": list(obj.actual_pos),
                "reason": obj.reason or ""
            })
        elif isinstance(obj, dict) and obj.get("type") == "CONTROL":
            # Add sender info for better logging
            obj_with_sender = obj.copy()
            obj_with_sender.setdefault("sender", self.self_id)
            return create_safe_a2a_message("CONTROL", obj_with_sender)
        elif isinstance(obj, dict) and obj.get("type") == "MOVE_REQUEST":
            # Handle concurrent move requests from agents
            return create_safe_a2a_message("MOVE_REQUEST", obj.get("payload", {}), target=obj.get("receiver_id", "network"))
        elif isinstance(obj, dict) and obj.get("type") == "MOVE_RESPONSE":
            # Handle concurrent move responses from NetworkBase
            return create_safe_a2a_message("MOVE_RESPONSE", obj.get("payload", {}), target=obj.get("receiver_id", "agent"))
        elif isinstance(obj, dict) and obj.get("type") == "CHAT":
            # Handle new unified CHAT format
            payload = obj.get("payload", {})
            target = str(payload.get("dst", "unknown"))
            return create_safe_a2a_message("CHAT", payload, target=target)
        elif isinstance(obj, dict) and "dst" in obj:
            # Handle legacy direct msg format (for backward compatibility)
            return create_safe_a2a_message("CHAT", obj, target=str(obj["dst"]))
        else:
            # Fallback for unknown types
            return create_safe_a2a_message("UNKNOWN", {"content": str(obj)})

    def _deserialize(self, msg) -> Any:
        """Deserializes A2A messages back into framework objects."""
        
        # 🔧 FIX: Check meta field first (where CONTROL messages are sent)
        if hasattr(msg, 'meta') and msg.meta:
            # If meta contains a dict with type, use that directly
            if isinstance(msg.meta, dict) and msg.meta.get("type"):
                print(f"[{self.self_id}] 🎯 Found message in meta field: {msg.meta}")
                msg = msg.meta  # Use meta content directly
            
        # Handle A2A text message format
        elif hasattr(msg, 'content') or hasattr(msg, 'text'):
            # Extract text content from A2A message
            text_content = getattr(msg, 'content', None) or getattr(msg, 'text', None)
            if text_content:
                try:
                    # Parse JSON from text content
                    parsed_msg = json.loads(text_content)
                    if isinstance(parsed_msg, dict):
                        msg = parsed_msg
                    else:
                        return text_content  # Return as plain text if not JSON
                except json.JSONDecodeError:
                    return text_content  # Return as plain text if JSON parsing fails
        
        # Handle dict format (either parsed from JSON or direct dict)
        if not isinstance(msg, dict):
            return str(msg)  # Convert to string if not dict
            
        t, body = msg.get("type"), msg.get("payload", {})
        if t == "MOVE_CMD":
            return MoveCmd(
                agent_id=body.get("agent_id"),
                new_pos=tuple(body.get("new_pos", [0, 0]))
            )
        elif t == "MOVE_FB":
            return MoveFeedback(
                agent_id=body.get("agent_id"),
                success=body.get("success", False),
                actual_pos=tuple(body.get("actual_pos", [0, 0])),
                reason=body.get("reason", "")
            )
        elif t == "CONTROL":
            # Return the full message dict for CONTROL (MAPFAgentExecutor expects meta with type field)
            return msg  # Return the complete control message with type field
        elif t == "MOVE_REQUEST":
            # Return the full message dict for MOVE_REQUEST (NetworkBase expects full structure)
            return msg  # Return the complete move request message
        elif t == "MOVE_RESPONSE":
            # Return the full message dict for MOVE_RESPONSE (Agent expects full structure)
            return msg  # Return the complete move response message
        elif t == "CHAT":
            return body
        
        print(f"[{self.self_id}] Received unhandled message type: {t}")
        return msg  # Return the original message if unhandled

    async def close(self):
        """Gracefully close the adapter."""
        await self.disconnect()
        print(f"[{self.self_id}] A2ACommAdapter closed")