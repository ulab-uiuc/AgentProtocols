"""
A2AAdapter - A2A协议适配器实现
"""

import json
from typing import Any, Dict, Optional, AsyncIterator
from uuid import uuid4

import httpx
from .base_adapter import BaseProtocolAdapter


class A2AAdapter(BaseProtocolAdapter):
    """
    Adapter for A2A (Agent-to-Agent) protocol.
    
    Translates protocol-agnostic message structures to A2A standard format
    and interacts with remote A2A Agents via HTTP (supports both one-shot and streaming).
    
    Instance granularity: One outbound edge = One Adapter instance.
    Different dst_id / base_url / protocol / Auth do not share instances.
    """

    @property
    def protocol_name(self) -> str:
        return "a2a"

    def __init__(
        self, 
        httpx_client: httpx.AsyncClient, 
        base_url: str,
        auth_headers: Optional[Dict[str, str]] = None,
        agent_card_path: str = "/.well-known/agent.json"
    ):
        """
        Initialize A2A adapter.
        
        Parameters
        ----------
        httpx_client : httpx.AsyncClient
            Shared HTTP client for connection pooling
        base_url : str
            Base URL of the A2A agent endpoint
        auth_headers : Optional[Dict[str, str]]
            Authentication headers (e.g., {"Authorization": "Bearer <token>"})
        agent_card_path : str
            Path to agent card endpoint (default: /.well-known/agent.json)
        """
        super().__init__(base_url=base_url, auth_headers=auth_headers or {})
        self.httpx_client = httpx_client
        self.base_url = base_url.rstrip('/')
        self.auth_headers = auth_headers or {}
        self.agent_card_path = agent_card_path
        self.agent_card: Dict[str, Any] = {}
        self._inbox_not_available = False  # Cache for inbox 404 status

    async def initialize(self) -> None:
        """
        Initialize by fetching the agent card from /.well-known/agent.json.
        
        If the server declares supportsAuthenticatedExtendedCard, 
        auth_headers can be injected during construction to pull extended card.
        
        Raises
        ------
        ConnectionError
            If agent card retrieval fails
        """
        try:
            headers = {}
            headers.update(self.auth_headers)
            
            response = await self.httpx_client.get(
                f"{self.base_url}{self.agent_card_path}",
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            self.agent_card = response.json()
        except httpx.TimeoutException as e:
            raise ConnectionError(f"Timeout fetching agent card from {self.base_url}: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ConnectionError(f"HTTP {e.response.status_code} fetching agent card: {e.response.text}") from e
        except Exception as e:
            raise ConnectionError(f"Failed to initialize A2A adapter: {e}") from e

    async def send_message(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        """
        Send message via A2A protocol using official A2A format.
        
        Parameters
        ----------
        dst_id : str
            Destination agent ID (used for routing context)
        payload : Dict[str, Any]
            Message payload containing message data
        
        Returns
        -------
        Any
            Complete JSON response from the agent
            
        Raises
        ------
        TimeoutError
            If request times out (default 30s)
        ConnectionError
            For HTTP 4xx/5xx errors
        RuntimeError
            For other send failures
        """
        # Construct A2A official message format
        # Note: Upper layer can customize message ID for retry idempotency
        request_id = str(uuid4())
        request_data = {
            "id": request_id,
            "params": {
                "message": payload.get("message", payload),  # Support both wrapped and direct message
                "context": payload.get("context", {}),
                "routing": {
                    "destination": dst_id,
                    "source": payload.get("source", "unknown")
                }
            }
        }
        
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(self.auth_headers)
            
            # Use compact JSON to reduce payload size (~7% reduction)
            response = await self.httpx_client.post(
                f"{self.base_url}/message",
                content=json.dumps(request_data, separators=(',', ':')),
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as e:
            raise TimeoutError(f"A2A message timeout to {dst_id} (req_id: {request_id})") from e
        except httpx.HTTPStatusError as e:
            raise ConnectionError(f"A2A HTTP error {e.response.status_code}: {e.response.text} (req_id: {request_id})") from e
        except Exception as e:
            raise RuntimeError(f"A2A send failed: {e} (req_id: {request_id})") from e

    async def send_message_streaming(self, dst_id: str, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Send message via A2A protocol and return streaming response.
        
        Parameters
        ----------
        dst_id : str
            Destination agent ID (used for routing context)
        payload : Dict[str, Any]
            Message payload containing message data
        
        Yields
        ------
        Dict[str, Any]
            Streaming events parsed line by line
            
        Raises
        ------
        TimeoutError
            If request times out
        ConnectionError
            For HTTP 4xx/5xx errors  
        RuntimeError
            For streaming interruption or other failures
        """
        # Construct A2A official message format (same as send_message)
        request_id = str(uuid4())
        request_data = {
            "id": request_id,
            "params": {
                "message": payload.get("message", payload),
                "context": payload.get("context", {}),
                "routing": {
                    "destination": dst_id,
                    "source": payload.get("source", "unknown")
                }
            }
        }
        
        try:
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
            headers.update(self.auth_headers)
            
            async with self.httpx_client.stream(
                "POST",
                f"{self.base_url}/message",
                content=json.dumps(request_data, separators=(',', ':')),
                headers=headers,
                timeout=httpx.Timeout(30.0, read=None)  # No read timeout for streaming
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():  # Skip empty lines
                        try:
                            # Handle SSE format: "data: {...}" prefix
                            clean_line = line.lstrip("data:").strip()
                            if clean_line:
                                event_data = json.loads(clean_line)
                                yield event_data
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue
                            
        except httpx.TimeoutException as e:
            raise TimeoutError(f"A2A streaming timeout to {dst_id} (req_id: {request_id})") from e
        except httpx.HTTPStatusError as e:
            raise ConnectionError(f"A2A streaming HTTP error {e.response.status_code}: {e.response.text} (req_id: {request_id})") from e
        except Exception as e:
            raise RuntimeError(f"A2A streaming failed: {e} (req_id: {request_id})") from e

    async def receive_message(self) -> Dict[str, Any]:
        """Receive messages (not applicable for client adapters)."""
        # This is a client adapter, so it doesn't receive in a server capacity.
        # However, for full UTE compatibility, we can simulate a decode of an empty message.
        raw_message = {"messages": []} 
        ute = DECODE_TABLE[self.protocol_name](raw_message)
        return {"messages": [ute]}

    def get_agent_card(self) -> Dict[str, Any]:
        """Return the cached agent card."""
        return self.agent_card.copy()

    async def health_check(self) -> bool:
        """
        Check if the A2A agent is responsive.
        
        Used by AgentNetwork.health_check() for periodic probing.
        
        Returns
        -------
        bool
            True if agent is responsive, False otherwise
        """
        try:
            headers = {}
            headers.update(self.auth_headers)
            
            response = await self.httpx_client.get(
                f"{self.base_url}/health",
                headers=headers,
                timeout=5.0
            )
            return response.status_code == 200
        except Exception:
            return False

    async def cleanup(self) -> None:
        """
        Clean up adapter resources.
        
        Note: Does not close httpx_client since it's shared across adapters.
        The caller (AgentNetwork or upper layer) is responsible for 
        client lifecycle management.
        """
        # Clear cached data
        self.agent_card.clear()
        self._inbox_not_available = False
        # Note: httpx_client lifecycle managed by upper layer

    def get_capabilities(self) -> Dict[str, Any]:
        """Extract capabilities from agent card."""
        return self.agent_card.get("capabilities", {})

    def get_endpoint_info(self) -> Dict[str, Any]:
        """Get endpoint information."""
        return {
            "base_url": self.base_url,
            "protocol": "A2A",
            "health_endpoint": f"{self.base_url}/health",
            "message_endpoint": f"{self.base_url}/message",
            "inbox_endpoint": f"{self.base_url}/inbox",
            "agent_card_endpoint": f"{self.base_url}{self.agent_card_path}",
            "supports_streaming": True,
            "supports_auth": bool(self.auth_headers)
        }

    def supports_authenticated_extended_card(self) -> bool:
        """Check if agent supports authenticated extended card."""
        return self.agent_card.get("supportsAuthenticatedExtendedCard", False)

    def get_protocol_version(self) -> str:
        """Get A2A protocol version from agent card."""
        version = self.agent_card.get("protocolVersion", "unknown")
        if version == "unknown":
            print(f"Warning: Unknown protocol version for {self.base_url}, check agent card")
        return version

    def __repr__(self) -> str:
        """Debug representation of A2AAdapter."""
        return (
            f"A2AAdapter(base_url='{self.base_url}', "
            f"auth={'enabled' if self.auth_headers else 'disabled'}, "
            f"card_loaded={bool(self.agent_card)})"
        ) 