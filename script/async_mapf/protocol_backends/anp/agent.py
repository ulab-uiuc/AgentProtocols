# script/async_mapf/protocol_backends/anp/agent.py
from __future__ import annotations
import asyncio
import json
from typing import Dict, Any, Optional, Tuple
from ...core.agent_base import BaseRobot
from ...core.world import GridWorld


class ANPRobot(BaseRobot):
    """
    ANP protocol implementation of BaseRobot.
    
    Implements communication methods using ANP SDK while inheriting
    all pathfinding and coordination algorithms from BaseRobot.
    """
    
    def __init__(self, aid: int, world: GridWorld, goal: Tuple[int, int], 
                 anp_client, **kwargs):
        """
        Initialize ANP robot.
        
        Args:
            aid: Agent ID
            world: Shared world reference
            goal: Target position
            anp_client: ANP client instance
            **kwargs: Additional configuration
        """
        super().__init__(aid, world, goal)
        self.anp_client = anp_client
        self.client_config = kwargs
        
        # ANP specific setup
        self._message_inbox: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        self._setup_anp_handler()
        
        # ANP connection settings
        self.did_document = kwargs.get("did_document")
        self.private_key = kwargs.get("private_key")
        self.node_url = kwargs.get("node_url", "ws://localhost:8080")
        
    def _setup_anp_handler(self) -> None:
        """Setup ANP message handler for incoming messages."""
        async def anp_message_handler(message_data: Dict[str, Any]) -> None:
            try:
                # ANP messages come pre-parsed
                await self._message_inbox.put(message_data)
            except Exception as e:
                print(f"ANP Agent {self.aid}: Failed to handle message: {e}")
        
        # Register handler with ANP client
        self.anp_client.set_message_handler(anp_message_handler)
    
    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """
        Send message to another agent via ANP protocol.
        
        Args:
            dst: Destination agent ID
            payload: Message content
        """
        try:
            # Get destination agent's DID
            dst_did = self._get_agent_did(dst)
            if not dst_did:
                print(f"ANP Agent {self.aid}: No DID found for agent {dst}")
                return
            
            # Prepare ANP message
            anp_message = {
                "type": "mapf_message",
                "from_agent": self.aid,
                "to_agent": dst,
                "payload": payload,
                "timestamp": self.world.timestamp
            }
            
            # Send through ANP client
            await self.anp_client.send_message(dst_did, anp_message)
            
        except Exception as e:
            print(f"ANP Agent {self.aid}: Failed to send message to {dst}: {e}")
    
    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Receive message from ANP protocol with timeout.
        
        Args:
            timeout: Maximum wait time in seconds (0 = non-blocking)
            
        Returns:
            Received message or None if timeout
        """
        try:
            if timeout == 0.0:
                # Non-blocking receive
                if self._message_inbox.empty():
                    return None
                message = self._message_inbox.get_nowait()
            else:
                # Blocking receive with timeout
                message = await asyncio.wait_for(self._message_inbox.get(), timeout=timeout)
            
            # Extract payload from ANP message
            if isinstance(message, dict) and "payload" in message:
                return message["payload"]
            else:
                return message
                
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            print(f"ANP Agent {self.aid}: Failed to receive message: {e}")
            return None
    
    def _get_agent_did(self, agent_id: int) -> Optional[str]:
        """
        Get DID for another agent.
        
        In a real implementation, this would query a registry or use
        a predetermined mapping of agent IDs to DIDs.
        """
        # Placeholder implementation
        return f"did:example:agent-{agent_id}"
    
    async def connect(self) -> bool:
        """
        Establish ANP connection.
        
        Returns:
            True if connection successful
        """
        try:
            if self.did_document and self.private_key:
                await self.anp_client.connect(
                    node_url=self.node_url,
                    did_document=self.did_document,
                    private_key=self.private_key
                )
            else:
                await self.anp_client.connect(node_url=self.node_url)
            
            print(f"ANP Agent {self.aid}: Connected successfully")
            return True
        except Exception as e:
            print(f"ANP Agent {self.aid}: Connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from ANP network."""
        try:
            await self.anp_client.disconnect()
            print(f"ANP Agent {self.aid}: Disconnected")
        except Exception as e:
            print(f"ANP Agent {self.aid}: Disconnect error: {e}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status and statistics."""
        return {
            "agent_id": self.aid,
            "protocol": "ANP",
            "connected": self.anp_client.is_connected() if hasattr(self.anp_client, 'is_connected') else True,
            "inbox_size": self._message_inbox.qsize(),
            "node_url": self.node_url,
            "has_did": bool(self.did_document)
        } 