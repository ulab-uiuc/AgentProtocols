"""Agent Protocol adapter implementation."""
import json
from typing import Any, Dict

from .base_adapter import ProtocolAdapter


class AgentProtocolAdapter(ProtocolAdapter):
    """Agent Protocol adapter for standardized agent communication."""

    def encode(self, packet: Dict[str, Any]) -> bytes:
        """Encode packet using Agent Protocol format."""
        # Convert to Agent Protocol format
        ap_packet = {
            "type": packet.get("type", "message"),
            "data": packet,
            "metadata": {
                "agent_id": packet.get("agent_id"),
                "timestamp": packet.get("timestamp"),
                "token_used": packet.get("token_used", 0)
            }
        }
        return json.dumps(ap_packet, ensure_ascii=False).encode('utf-8')

    def decode(self, blob: bytes) -> Dict[str, Any]:
        """Decode Agent Protocol format to packet."""
        ap_packet = json.loads(blob.decode('utf-8'))
        
        # Extract original packet data
        packet = ap_packet.get("data", {})
        
        # Merge metadata back into packet for compatibility
        metadata = ap_packet.get("metadata", {})
        packet.update({
            "agent_id": metadata.get("agent_id"),
            "timestamp": metadata.get("timestamp"),
            "token_used": metadata.get("token_used", 0)
        })
        
        return packet

    def header_size(self, packet: Dict[str, Any]) -> int:
        """Calculate Agent Protocol header overhead."""
        # Estimate metadata overhead
        metadata_size = len(json.dumps({
            "type": "message",
            "metadata": {
                "agent_id": packet.get("agent_id"),
                "timestamp": packet.get("timestamp"),
                "token_used": packet.get("token_used", 0)
            }
        }))
        return metadata_size
