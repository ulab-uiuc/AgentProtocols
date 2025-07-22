# script/async_mapf/protocol_backends/a2a/network.py
from __future__ import annotations
import asyncio
import json
from typing import Dict, Any, List, Tuple
from ...core.network_base import BaseNet
from ...core.world import GridWorld


class A2ANet(BaseNet):
    """
    A2A protocol implementation of BaseNet.
    
    Implements network coordination methods using A2A SDK while inheriting
    all coordination and conflict resolution logic from BaseNet.
    """
    
    def __init__(self, world: GridWorld, client, tick_ms: int = 10, **kwargs):
        """
        Initialize A2A network coordinator.
        
        Args:
            world: Shared world reference
            client: A2A client instance
            tick_ms: Time step duration in milliseconds
            **kwargs: Additional configuration
        """
        super().__init__(world, tick_ms)
        self.client = client
        self.client_config = kwargs
        
        # A2A specific setup
        self._recv_queue: "asyncio.Queue[Tuple[int, Dict[str, Any]]]" = asyncio.Queue()
        self._setup_message_handler()
        
        # Network settings
        self.controller_channel = kwargs.get("controller_channel", "mapf-controller")
        self.broadcast_channel = kwargs.get("broadcast_channel", "mapf-broadcast")
        
    def _setup_message_handler(self) -> None:
        """Setup A2A message handler for coordinator messages."""
        def message_handler(raw_message: str) -> None:
            try:
                message = json.loads(raw_message)
                sender_id = message.get("sender_id", -1)
                self._recv_queue.put_nowait((sender_id, message))
            except (json.JSONDecodeError, Exception) as e:
                print(f"A2A Network: Failed to parse message: {e}")
        
        # Register message handler for controller channel
        self.client.on_message(self.controller_channel, message_handler)
    
    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """
        Deliver message to specific agent via A2A protocol.
        
        Args:
            dst: Destination agent ID
            msg: Message payload
        """
        try:
            # Add network metadata
            message = {
                **msg,
                "sender_id": -1,  # Network coordinator
                "recipient_id": dst,
                "timestamp": self.world.timestamp
            }
            
            # Send to agent's channel
            channel = f"mapf-agent-{dst}"
            serialized = json.dumps(message)
            
            await self.client.send(channel, serialized)
            
        except Exception as e:
            print(f"A2A Network: Failed to deliver message to agent {dst}: {e}")
    
    async def poll(self) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Poll for incoming messages from agents.
        
        Returns:
            List of (sender_id, message) tuples
        """
        messages = []
        
        try:
            # Collect all available messages (non-blocking)
            while not self._recv_queue.empty():
                sender_id, message = self._recv_queue.get_nowait()
                messages.append((sender_id, message))
                
        except Exception as e:
            print(f"A2A Network: Error polling messages: {e}")
        
        return messages
    
    async def broadcast(self, msg: Dict[str, Any]) -> None:
        """
        Broadcast message to all active agents.
        
        Args:
            msg: Message to broadcast
        """
        try:
            # Add broadcast metadata
            message = {
                **msg,
                "sender_id": -1,  # Network coordinator
                "broadcast": True,
                "timestamp": self.world.timestamp
            }
            
            serialized = json.dumps(message)
            
            # Send to each active agent
            for agent_id in self.active_agents:
                channel = f"mapf-agent-{agent_id}"
                await self.client.send(channel, serialized)
                
        except Exception as e:
            print(f"A2A Network: Failed to broadcast message: {e}")
    
    async def connect(self) -> bool:
        """
        Establish A2A network connection.
        
        Returns:
            True if connection successful
        """
        try:
            await self.client.connect()
            print("A2A Network: Connected successfully")
            return True
        except Exception as e:
            print(f"A2A Network: Connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from A2A network."""
        try:
            await self.client.disconnect()
            print("A2A Network: Disconnected")
        except Exception as e:
            print(f"A2A Network: Disconnect error: {e}")
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network status and statistics."""
        return {
            "protocol": "A2A",
            "connected": self.client.is_connected() if hasattr(self.client, 'is_connected') else True,
            "controller_channel": self.controller_channel,
            "broadcast_channel": self.broadcast_channel,
            "active_agents": len(self.active_agents),
            "message_queue_size": self._recv_queue.qsize(),
            "tick": self.current_tick
        } 