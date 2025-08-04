"""JSON protocol adapter implementation."""
import json
from typing import Any, Dict

from .base_adapter import ProtocolAdapter


class JsonAdapter(ProtocolAdapter):
    """Simple JSON protocol adapter for human-readable communication."""

    def encode(self, packet: Dict[str, Any]) -> bytes:
        """Encode packet to JSON bytes."""
        return json.dumps(packet, ensure_ascii=False).encode('utf-8')

    def decode(self, blob: bytes) -> Dict[str, Any]:
        """Decode JSON bytes to packet."""
        return json.loads(blob.decode('utf-8'))

    def header_size(self, packet: Dict[str, Any]) -> int:
        """JSON has no separate header, so overhead is zero."""
        return 0
