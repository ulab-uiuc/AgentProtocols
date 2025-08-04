"""Abstract base class for protocol adapters."""
import abc
from typing import Any, Dict


class ProtocolAdapter(abc.ABC):
    """Abstract interface for encoding and decoding network packets."""

    @abc.abstractmethod
    def encode(self, packet: Dict[str, Any]) -> bytes:
        """Encode a dictionary packet to bytes."""
        ...

    @abc.abstractmethod
    def decode(self, blob: bytes) -> Dict[str, Any]:
        """Decode bytes to a dictionary packet."""
        ...

    @abc.abstractmethod
    def header_size(self, packet: Dict[str, Any]) -> int:
        """Calculate protocol header size for metrics."""
        ...
