"""
BaseProtocolAdapter - 协议适配器基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

# ★ UTE Imports Added - use absolute imports to avoid relative import issues
try:
    from ..core.unified_message import UTE
    from ..core.protocol_converter import ENCODE_TABLE, DECODE_TABLE
except ImportError:
    # Fallback for standalone execution
    from src.core.unified_message import UTE
    from src.core.protocol_converter import ENCODE_TABLE, DECODE_TABLE


class BaseProtocolAdapter(ABC):
    """Base class for all protocol adapters."""

    def __init__(self, **kwargs):
        """Initialize the adapter with protocol-specific parameters."""
        self.config = kwargs

    @property
    @abstractmethod
    def protocol_name(self) -> str:
        """Return lower-case protocol key used in converter tables."""
        raise NotImplementedError

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