# -*- coding: utf-8 -*-
"""
Privacy Testing Communication Backend Base
Abstract communication layer for privacy protection testing.
"""

from __future__ import annotations
import abc
from typing import Any, Dict, Optional


class BaseCommBackend(abc.ABC):
    """
    Abstract communication backend for privacy testing:
      - register_endpoint: Register an agent's service address
      - connect: (Optional) Establish connection/warm-up
      - send: Send message from src to dst, return protocol-native response
      - health_check: Health check probe
      - close: Close resources
    """

    @abc.abstractmethod
    async def register_endpoint(self, agent_id: str, address: str) -> None:
        """Register agent endpoint address."""
        ...

    async def connect(self, src_id: str, dst_id: str) -> None:
        """Some protocols need explicit connection; default no-op."""
        return None

    @abc.abstractmethod
    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """
        Send message from src_id to dst_id.
        
        Args:
            src_id: Source agent ID
            dst_id: Destination agent ID  
            payload: Message payload with standardized format:
                     {"text": "..."} or {"parts": [...]}
        
        Returns:
            Standardized response: {"raw": protocol_response, "text": "extracted_text"}
        """
        ...

    @abc.abstractmethod
    async def health_check(self, agent_id: str) -> bool:
        """Check if agent is healthy and reachable."""
        ...

    async def close(self) -> None:
        """Close underlying resources (HTTP clients, sockets, etc.). Default no-op."""
        return None

    # Optional: For protocols that support local hosting
    async def spawn_local_agent(self, agent_id: str, host: str, port: int, executor: Any) -> Any:
        """
        Optional: Spawn a local agent service for protocols that support it.
        Returns handle/address that can be used for registration.
        """
        raise NotImplementedError("This protocol does not support local agent hosting")

