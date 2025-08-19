"""
SimpleJSON protocol adapter for fail-storm recovery scenarios.

Implements communication using simple JSON messages over local queues,
compatible with the existing simple_json protocol implementation.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, Tuple

from core.comm import AbstractCommAdapter


class SimpleJSONAdapter(AbstractCommAdapter):
    """
    SimpleJSON protocol adapter for local queue-based communication.
    
    Compatible with the existing simple_json protocol implementation
    used in the fail-storm recovery scenarios.
    """
    
    def __init__(self, agent_id: str, agent_card: Dict[str, Any] = None):
        """
        Initialize SimpleJSON adapter.
        
        Parameters
        ----------
        agent_id : str
            Unique identifier for this agent
        agent_card : Dict[str, Any], optional
            Agent card information for protocol compatibility
        """
        self.agent_id = agent_id
        self.agent_card = agent_card or {
            "protocol": "simple_json",
            "agent_id": agent_id,
            "capabilities": []
        }
        
        # Communication queues
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._outbox: asyncio.Queue = asyncio.Queue()
        self._connected = False
        
        # Registered agents for local communication
        self._registered_agents: Dict[str, asyncio.Queue] = {}
        
        # Message tracking
        self._message_id_counter = 0
        
    async def connect(self) -> None:
        """Initialize SimpleJSON connection."""
        self._connected = True
        print(f"[SimpleJSONAdapter] {self.agent_id} connected")
    
    async def disconnect(self) -> None:
        """Close SimpleJSON connection."""
        self._connected = False
        print(f"[SimpleJSONAdapter] {self.agent_id} disconnected")
    
    async def send(self, target: str, message: Dict[str, Any]) -> None:
        """Send message to specific target using SimpleJSON format."""
        if not self._connected:
            raise ConnectionError("Adapter not connected")
        
        if target not in self._registered_agents:
            print(f"[SimpleJSONAdapter] Target {target} not found")
            return
        
        # Create SimpleJSON message format
        simple_json_message = self._create_simple_json_message(message, target)
        
        # Send to target's inbox
        await self._registered_agents[target].put(simple_json_message)
        
        print(f"[SimpleJSONAdapter] {self.agent_id} -> {target}: {message.get('type', 'unknown')}")
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all registered agents using SimpleJSON format."""
        if not self._connected:
            raise ConnectionError("Adapter not connected")
        
        # Create SimpleJSON message format
        simple_json_message = self._create_simple_json_message(message, broadcast=True)
        
        # Send to all registered agents except self
        for agent_id, queue in self._registered_agents.items():
            if agent_id != self.agent_id:
                await queue.put(simple_json_message)
        
        print(f"[SimpleJSONAdapter] {self.agent_id} broadcast: {message.get('type', 'unknown')}")
    
    async def recv(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Receive message from SimpleJSON protocol (blocking)."""
        if not self._connected:
            return None
        
        try:
            simple_json_message = await self._inbox.get()
            return self._parse_simple_json_message(simple_json_message)
        except asyncio.CancelledError:
            return None
    
    def recv_nowait(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Receive message from SimpleJSON protocol (non-blocking)."""
        if not self._connected:
            return None
        
        try:
            simple_json_message = self._inbox.get_nowait()
            return self._parse_simple_json_message(simple_json_message)
        except asyncio.QueueEmpty:
            return None
    
    async def send_heartbeat(self, agent_id: str) -> None:
        """Send heartbeat message using SimpleJSON format."""
        heartbeat_message = {
            "type": "heartbeat",
            "agent_id": agent_id,
            "timestamp": time.time(),
            "topology_version": len(self._registered_agents)
        }
        await self.broadcast(heartbeat_message)
    
    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        return self._connected
    
    def register_agent(self, agent_id: str, queue: asyncio.Queue) -> None:
        """Register another agent for local communication."""
        self._registered_agents[agent_id] = queue
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        self._registered_agents.pop(agent_id, None)
    
    def get_inbox(self) -> asyncio.Queue:
        """Get the inbox queue for this adapter."""
        return self._inbox
    
    def _create_simple_json_message(self, message: Dict[str, Any], target: str = None, broadcast: bool = False) -> Dict[str, Any]:
        """Create SimpleJSON message format."""
        self._message_id_counter += 1
        
        # Base message structure
        simple_json_message = {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "user" if message.get("type") == "heartbeat" else "assistant",
                "parts": [
                    {
                        "type": "text",
                        "text": json.dumps(message)
                    }
                ]
            },
            "context": {
                "type": message.get("type", "unknown"),
                "sender": self.agent_id,
                "timestamp": time.time(),
                "message_id": self._message_id_counter
            },
            "source": self.agent_id
        }
        
        # Add target information
        if target:
            simple_json_message["context"]["target"] = target
        elif broadcast:
            simple_json_message["context"]["broadcast"] = True
        
        # Add agent card information
        simple_json_message["context"]["agent_card"] = self.agent_card
        
        return simple_json_message
    
    def _parse_simple_json_message(self, simple_json_message: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Parse SimpleJSON message format."""
        try:
            # Extract sender
            sender = simple_json_message.get("source", "unknown")
            
            # Extract context
            context = simple_json_message.get("context", {})
            
            # Extract message content
            parts = simple_json_message.get("message", {}).get("parts", [])
            if parts and parts[0].get("type") == "text":
                try:
                    # Try to parse JSON from text
                    message_content = json.loads(parts[0]["text"])
                except (json.JSONDecodeError, TypeError):
                    # Fallback to raw text
                    message_content = {"text": parts[0]["text"]}
            else:
                message_content = {}
            
            # Merge context information
            message_content.update({
                "sender": sender,
                "timestamp": context.get("timestamp"),
                "message_id": context.get("message_id")
            })
            
            return sender, message_content
            
        except Exception as e:
            print(f"[SimpleJSONAdapter] Error parsing message: {e}")
            return "unknown", {"error": "parse_error", "original": simple_json_message} 