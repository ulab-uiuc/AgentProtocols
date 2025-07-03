"""
BaseProtocolAdapter - 协议适配器基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseProtocolAdapter(ABC):
    """Base class for all protocol adapters."""

    def __init__(self, **kwargs):
        """Initialize the adapter with protocol-specific parameters."""
        self.config = kwargs

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the adapter and establish connections."""
        pass

    @abstractmethod
    async def send_message(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        """Send a message to destination agent."""
        pass

    @abstractmethod
    async def receive_message(self) -> Dict[str, Any]:
        """Receive a message (if applicable for the protocol)."""
        pass

    @abstractmethod
    def get_agent_card(self) -> Dict[str, Any]:
        """Get agent capabilities and metadata."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the adapter is healthy and connected."""
        pass

    async def cleanup(self) -> None:
        """Clean up resources (default implementation)."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})" 