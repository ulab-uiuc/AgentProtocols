"""
SimpleJSONAdapter - A minimal JSON protocol adapter implementation
"""

import json
import time
from typing import Any, Dict, Optional, AsyncIterator
from uuid import uuid4

import httpx
from .base_adapter import BaseProtocolAdapter


class SimpleJSONAdapter(BaseProtocolAdapter):
    """
    A simple JSON protocol adapter.
    
    Uses the simplest JSON message format, designed for testing and prototyping.
    No complex validation—focuses on the core functionality of message passing.
    """

    @property
    def protocol_name(self) -> str:
        return "simple_json"

    def __init__(
        self, 
        httpx_client: httpx.AsyncClient, 
        base_url: str,
        auth_headers: Optional[Dict[str, str]] = None,
        agent_id: str = "unknown"
    ):
        """
        Initialize the simple JSON adapter.
        
        Parameters
        ----------
        httpx_client : httpx.AsyncClient
            HTTP client
        base_url : str
            Base URL of the target agent
        auth_headers : Optional[Dict[str, str]]
            Authentication headers (optional)
        agent_id : str
            ID of the current agent
        """
        super().__init__(base_url=base_url, auth_headers=auth_headers or {})
        self.httpx_client = httpx_client
        self.base_url = base_url.rstrip('/')
        self.auth_headers = auth_headers or {}
        self.agent_id = agent_id

    async def initialize(self) -> None:
        """
        Initialize the adapter. No special steps are required for the simple JSON protocol.
        """
        # No complex initialization needed for the simple JSON protocol
        pass

    async def send_message(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        """
        Send a simple JSON message.
        
        Parameters
        ----------
        dst_id : str
            Target agent ID
        payload : Dict[str, Any]
            Message payload
        
        Returns
        -------
        Any
            Response data
        """
        # Build the simple JSON message format
        message_id = str(uuid4())
        simple_message = {
            "id": message_id,
            "from": self.agent_id,
            "to": dst_id,
            "timestamp": time.time(),
            "payload": payload
        }
        
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(self.auth_headers)
            
            response = await self.httpx_client.post(
                f"{self.base_url}/message",
                content=json.dumps(simple_message, separators=(',', ':')),
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.TimeoutException as e:
            raise TimeoutError(f"SimpleJSON message timeout to {dst_id} (msg_id: {message_id})") from e
        except httpx.HTTPStatusError as e:
            raise ConnectionError(f"SimpleJSON HTTP error {e.response.status_code}: {e.response.text} (msg_id: {message_id})") from e
        except Exception as e:
            raise RuntimeError(f"Failed to send SimpleJSON message to {dst_id}: {e}") from e

    async def send_message_streaming(self, dst_id: str, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Send a streaming JSON message (simplified version).
        """
        # For the simple protocol, streaming returns a single response
        result = await self.send_message(dst_id, payload)
        yield result

    async def receive_message(self) -> Dict[str, Any]:
        """Receive message (not applicable to client adapters)."""
        raise NotImplementedError("Client adapters do not receive messages directly")

    def convert_to_native(self, ute_message) -> Dict[str, Any]:
        """Convert UTE message to simple JSON format."""
        return {
            "id": ute_message.id,
            "from": ute_message.src,
            "to": ute_message.dst,
            "timestamp": ute_message.timestamp,
            "content": ute_message.content,
            "context": ute_message.context,
            "metadata": ute_message.metadata
        }

    def convert_from_native(self, native_message: Dict[str, Any]):
        """Convert a simple JSON message to UTE format."""
        from ..core.unified_message import UTE
        
        return UTE(
            id=native_message.get("id", str(uuid4())),
            src=native_message.get("from", "unknown"),
            dst=native_message.get("to", "unknown"),
            timestamp=native_message.get("timestamp", time.time()),
            content=native_message.get("content", {}),
            context=native_message.get("context", {}),
            metadata=native_message.get("metadata", {})
        )

    def get_agent_card(self) -> Dict[str, Any]:
        """Get agent capabilities and metadata."""
        return {
            "agent_id": self.agent_id,
            "protocol": "simple_json",
            "base_url": self.base_url,
            "capabilities": ["message_sending", "json_processing"],
            "supported_message_types": ["json"],
            "version": "1.0.0"
        }

    async def health_check(self) -> bool:
        """Check whether the adapter is healthy and reachable."""
        try:
            # Try sending a health check request to the target URL
            response = await self.httpx_client.get(
                f"{self.base_url}/health",
                headers=self.auth_headers,
                timeout=5.0
            )
            return response.status_code == 200
        except Exception:
            return False