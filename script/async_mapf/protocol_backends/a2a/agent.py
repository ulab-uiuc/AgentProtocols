# script/async_mapf/protocol_backends/a2a/agent.py
from __future__ import annotations
import asyncio
import json
from typing import Dict, Any, Optional, Tuple
from ...core.agent_base import BaseRobot
from ...core.world import GridWorld


class A2ARobot(BaseRobot):
    """
    A2A protocol implementation of BaseRobot.
    
    Implements communication methods using A2A SDK while inheriting
    all pathfinding and coordination algorithms from BaseRobot.
    """
    
    def __init__(self, aid: int, world: GridWorld, goal: Tuple[int, int], 
                 client, **kwargs):
        """
        Initialize A2A robot.
        
        Args:
            aid: Agent ID
            world: Shared world reference
            goal: Target position
            client: A2A client instance
            **kwargs: Additional configuration
        """
        super().__init__(aid, world, goal)
        self.client = client
        self.client_config = kwargs
        
        # A2A specific setup
        self._inbox: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        self._setup_message_handler()
        
        # Connection settings
        self.channel_prefix = kwargs.get("channel_prefix", "mapf-agent")
        self.timeout_ms = kwargs.get("timeout_ms", 1000)
        
    def _setup_message_handler(self) -> None:
        """Setup A2A message handler for incoming messages."""
        channel = f"{self.channel_prefix}-{self.aid}"
        
        def message_handler(raw_message: str) -> None:
            try:
                message = json.loads(raw_message)
                self._inbox.put_nowait(message)
            except (json.JSONDecodeError, Exception) as e:
                # Log error but don't crash
                print(f"A2A Agent {self.aid}: Failed to parse message: {e}")
        
        # Register message handler with A2A client
        self.client.on_message(channel, message_handler)
    
    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """
        Send message to another agent via A2A protocol.
        
        Args:
            dst: Destination agent ID
            payload: Message content
        """
        try:
            # Add metadata
            message = {
                **payload,
                "sender_id": self.aid,
                "recipient_id": dst,
                "timestamp": self.world.timestamp
            }
            
            # Serialize and send
            channel = f"{self.channel_prefix}-{dst}"
            serialized = json.dumps(message)
            
            await self.client.send(channel, serialized)
            
        except Exception as e:
            print(f"A2A Agent {self.aid}: Failed to send message to {dst}: {e}")
    
    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Receive message from A2A protocol with timeout.
        
        Args:
            timeout: Maximum wait time in seconds (0 = non-blocking)
            
        Returns:
            Received message or None if timeout
        """
        try:
            if timeout == 0.0:
                # Non-blocking receive
                if self._inbox.empty():
                    return None
                return self._inbox.get_nowait()
            else:
                # Blocking receive with timeout
                return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
                
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            print(f"A2A Agent {self.aid}: Failed to receive message: {e}")
            return None
    
    async def connect(self) -> bool:
        """
        Establish A2A connection.
        
        Returns:
            True if connection successful
        """
        try:
            await self.client.connect()
            print(f"A2A Agent {self.aid}: Connected successfully")
            return True
        except Exception as e:
            print(f"A2A Agent {self.aid}: Connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from A2A network."""
        try:
            await self.client.disconnect()
            print(f"A2A Agent {self.aid}: Disconnected")
        except Exception as e:
            print(f"A2A Agent {self.aid}: Disconnect error: {e}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status and statistics."""
        return {
            "agent_id": self.aid,
            "protocol": "A2A",
            "connected": self.client.is_connected() if hasattr(self.client, 'is_connected') else True,
            "inbox_size": self._inbox.qsize(),
            "channel": f"{self.channel_prefix}-{self.aid}"
        } 