"""
ACP (Agent Communication Protocol) client adapter implementation.
"""

import json
import time
import uuid
from typing import Any, Dict, Optional, AsyncIterator
import httpx

from .base_adapter import BaseProtocolAdapter
#except ImportError:
#    from base_adapter import BaseProtocolAdapter


class ACPAdapter(BaseProtocolAdapter):
    """
    Client adapter for the Agent Communication Protocol (ACP).
    """

    @property
    def protocol_name(self) -> str:
        return "acp"

    def __init__(
        self,
        httpx_client: httpx.AsyncClient,
        base_url: str,
        agent_id: str,
        auth_headers: Optional[Dict[str, str]] = None,
        agent_card_path: str = "/.well-known/agent.json",
        capabilities_path: str = "/acp/capabilities",
        message_path: str = "/acp/message",
        status_path: str = "/acp/status"
    ):
        """
        Initialize ACP adapter.

        Parameters
        ----------
        httpx_client : httpx.AsyncClient
            Shared HTTP client for connection pooling
        base_url : str
            Base URL of the ACP agent endpoint
        agent_id : str
            ID of this agent (sender)
        auth_headers : Optional[Dict[str, str]]
            Authentication headers (e.g., {"Authorization": "Bearer <token>"})
        agent_card_path : str
            Path to agent card endpoint
        capabilities_path : str
            Path to capabilities endpoint
        message_path : str
            Path to message endpoint
        status_path : str
            Path to status endpoint
        """
        super().__init__(
            base_url=base_url,
            agent_id=agent_id,
            auth_headers=auth_headers or {}
        )
        self.httpx_client = httpx_client
        self.base_url = base_url.rstrip('/')
        self.agent_id = agent_id
        self.auth_headers = auth_headers or {}

        # Endpoint paths
        self.agent_card_path = agent_card_path
        self.capabilities_path = capabilities_path
        self.message_path = message_path
        self.status_path = status_path

        # Cache
        self.agent_card: Dict[str, Any] = {}
        self.capabilities: Dict[str, Any] = {}
        self.remote_agent_id: Optional[str] = None

    async def initialize(self) -> None:
        """
        Initialize by fetching the agent card and capabilities.

        Raises
        ------
        ConnectionError
            If agent card or capabilities retrieval fails
        """
        try:
            # Fetch agent card
            await self._fetch_agent_card()

            # Fetch capabilities if available
            await self._fetch_capabilities()

            # Extract remote agent ID from card
            self.remote_agent_id = self.agent_card.get("agent_id", "unknown")

        except Exception as e:
            raise ConnectionError(f"Failed to initialize ACP adapter: {e}") from e

    async def _fetch_agent_card(self) -> None:
        """Fetch agent card from remote agent."""
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

    async def _fetch_capabilities(self) -> None:
        """Fetch capabilities from remote agent."""
        try:
            headers = {}
            headers.update(self.auth_headers)

            response = await self.httpx_client.get(
                f"{self.base_url}{self.capabilities_path}",
                headers=headers,
                timeout=30.0
            )

            if response.status_code == 200:
                self.capabilities = response.json()
            # If capabilities endpoint doesn't exist, it's okay

        except httpx.HTTPStatusError:
            # Capabilities endpoint might not exist, that's fine
            pass
        except Exception:
            # Other errors with capabilities are non-fatal
            pass

    async def send_message(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        """
        Send message via ACP protocol.

        Parameters
        ----------
        dst_id : str
            Destination agent ID (should match remote agent)
        payload : Dict[str, Any]
            Message payload containing message data

        Returns
        -------
        Any
            Complete JSON response from the agent

        Raises
        ------
        ConnectionError
            If message sending fails
        ValueError
            If payload format is invalid
        """
        try:
            # Generate message ID
            message_id = str(uuid.uuid4())

            # Determine message type from payload
            message_type = payload.get("message_type", "request")
            # Valid message types for ACP
            valid_types = ["request", "response", "notification", "heartbeat"]
            if message_type not in valid_types:
                message_type = "request"  # Default fallback

            # Construct ACP message
            acp_message = {
                "id": message_id,
                "type": message_type,
                "sender": self.agent_id,
                "receiver": dst_id,
                "payload": payload,
                "timestamp": time.time()
            }

            # Add optional fields
            if "correlation_id" in payload:
                acp_message["correlation_id"] = payload["correlation_id"]
            if "session_id" in payload:
                acp_message["session_id"] = payload["session_id"]
            if "metadata" in payload:
                acp_message["metadata"] = payload["metadata"]

            # Prepare headers
            headers = {
                "Content-Type": "application/json"
            }
            headers.update(self.auth_headers)

            # Check if streaming is requested
            if payload.get("stream", False):
                headers["Accept"] = "text/event-stream"
                return await self._send_streaming_message(acp_message, headers)
            else:
                return await self._send_json_message(acp_message, headers)

        except Exception as e:
            raise ConnectionError(f"Failed to send ACP message: {e}") from e

    async def _send_json_message(self, acp_message: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Send message and return JSON response."""
        try:
            response = await self.httpx_client.post(
                f"{self.base_url}{self.message_path}",
                json=acp_message,
                headers=headers,
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException as e:
            raise ConnectionError(f"Timeout sending message to {self.base_url}: {e}") from e
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.json().get("error", e.response.text)
            except:
                error_detail = e.response.text
            raise ConnectionError(f"HTTP {e.response.status_code} sending message: {error_detail}") from e

    async def _send_streaming_message(self, acp_message: Dict[str, Any], headers: Dict[str, str]) -> AsyncIterator[Dict[str, Any]]:
        """Send message and return streaming response."""
        try:
            async with self.httpx_client.stream(
                "POST",
                f"{self.base_url}{self.message_path}",
                json=acp_message,
                headers=headers,
                timeout=60.0
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    line = line.strip()
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])  # Remove "data: " prefix
                            yield data
                        except json.JSONDecodeError:
                            continue  # Skip malformed JSON

        except httpx.TimeoutException as e:
            raise ConnectionError(f"Timeout streaming from {self.base_url}: {e}") from e
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.json().get("error", e.response.text)
            except:
                error_detail = e.response.text
            raise ConnectionError(f"HTTP {e.response.status_code} streaming: {error_detail}") from e

    async def receive_message(self) -> Dict[str, Any]:
        """Receive messages from the ACP inbox."""
        try:
            response = await self.httpx_client.get(f"{self.base_url}/inbox")
            response.raise_for_status()
            raw_messages = response.json().get("messages", [])
            
            # Decode each raw message into a UTE
            utes = [DECODE_TABLE[self.protocol_name](msg) for msg in raw_messages]
            
            return {"messages": utes}
        except Exception:
            return {"messages": []}

    def get_agent_card(self) -> Dict[str, Any]:
        """
        Get cached agent capabilities and metadata.

        Returns
        -------
        Dict[str, Any]
            Agent card containing capabilities and metadata
        """
        return self.agent_card.copy()

    async def health_check(self) -> bool:
        """
        Check if the remote ACP agent is healthy and reachable.

        Returns
        -------
        bool
            True if agent is healthy, False otherwise
        """
        try:
            headers = {}
            headers.update(self.auth_headers)

            # Try health endpoint first
            response = await self.httpx_client.get(
                f"{self.base_url}/health",
                headers=headers,
                timeout=10.0
            )

            if response.status_code == 200:
                return True

            # Fallback to status endpoint
            response = await self.httpx_client.get(
                f"{self.base_url}{self.status_path}",
                headers=headers,
                timeout=10.0
            )

            if response.status_code == 200:
                status_data = response.json()
                return status_data.get("status", "").lower() == "healthy"

            return False

        except Exception:
            return False

    async def get_status(self) -> Dict[str, Any]:
        """
        Get detailed status from remote ACP agent.

        Returns
        -------
        Dict[str, Any]
            Status information from remote agent
        """
        try:
            headers = {}
            headers.update(self.auth_headers)

            response = await self.httpx_client.get(
                f"{self.base_url}{self.status_path}",
                headers=headers,
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            return {"error": str(e), "status": "unknown"}

    async def send_heartbeat(self) -> Dict[str, Any]:
        """
        Send a heartbeat message to the remote agent.

        Returns
        -------
        Dict[str, Any]
            Heartbeat response from remote agent
        """
        payload = {
            "message_type": "heartbeat",
            "timestamp": time.time(),
            "sender_id": self.agent_id
        }

        return await self.send_message(self.remote_agent_id or "unknown", payload)

    async def send_notification(self, notification_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a notification message to the remote agent.

        Parameters
        ----------
        notification_data : Dict[str, Any]
            Notification payload

        Returns
        -------
        Dict[str, Any]
            Response from remote agent
        """
        payload = {
            "message_type": "notification",
            "notification_data": notification_data,
            "timestamp": time.time()
        }

        return await self.send_message(self.remote_agent_id or "unknown", payload)

    async def cleanup(self) -> None:
        """Clean up resources."""
        # No specific cleanup needed for ACP adapter
        # HTTP client cleanup is handled by the caller
        pass

    def __repr__(self) -> str:
        return f"ACPAdapter(base_url={self.base_url}, agent_id={self.agent_id})"
