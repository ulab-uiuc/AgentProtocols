# script/async_mapf/protocol_backends/anp/network.py
from __future__ import annotations
import asyncio
import json
from typing import Dict, Any, List, Tuple
from ...core.network_base import BaseNet
from ...core.world import GridWorld


class ANPNet(BaseNet):
    """
    ANP protocol implementation of BaseNet.
    
    Implements network coordination methods using ANP SDK while inheriting
    all coordination and conflict resolution logic from BaseNet.
    """
    
    def __init__(self, world: GridWorld, anp_node, tick_ms: int = 10, **kwargs):
        """
        Initialize ANP network coordinator.
        
        Args:
            world: Shared world reference
            anp_node: ANP node instance
            tick_ms: Time step duration in milliseconds
            **kwargs: Additional configuration
        """
        super().__init__(world, tick_ms)
        self.anp_node = anp_node
        self.node_config = kwargs
        
        # ANP specific setup
        self._message_buffer: "asyncio.Queue[Tuple[int, Dict[str, Any]]]" = asyncio.Queue()
        self._setup_anp_handler()
        
        # Network settings
        self.node_did = kwargs.get("node_did")
        self.node_private_key = kwargs.get("node_private_key")
        self.agent_registry: Dict[int, str] = {}  # agent_id -> DID mapping
        
    def _setup_anp_handler(self) -> None:
        """Setup ANP message handler for coordinator messages."""
        async def coordinator_message_handler(message_data: Dict[str, Any]) -> None:
            try:
                # Extract sender information from ANP message
                sender_agent_id = message_data.get("from_agent", -1)
                await self._message_buffer.put((sender_agent_id, message_data))
            except Exception as e:
                print(f"ANP Network: Failed to handle message: {e}")
        
        # Register coordinator message handler
        self.anp_node.set_coordinator_handler(coordinator_message_handler)
    
    def register_agent_did(self, agent_id: int, did: str) -> None:
        """Register an agent's DID for message routing."""
        self.agent_registry[agent_id] = did
        print(f"ANP Network: Registered agent {agent_id} with DID {did}")
    
    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """
        Deliver message to specific agent via ANP protocol.
        
        Args:
            dst: Destination agent ID
            msg: Message payload
        """
        try:
            # Get agent's DID
            dst_did = self.agent_registry.get(dst)
            if not dst_did:
                print(f"ANP Network: No DID registered for agent {dst}")
                return
            
            # Prepare ANP coordinator message
            anp_message = {
                "type": "coordinator_message",
                "from_coordinator": True,
                "to_agent": dst,
                "payload": msg,
                "timestamp": self.world.timestamp
            }
            
            # Send through ANP node
            await self.anp_node.send_to_agent(dst_did, anp_message)
            
        except Exception as e:
            print(f"ANP Network: Failed to deliver message to agent {dst}: {e}")
    
    async def poll(self) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Poll for incoming messages from agents.
        
        Returns:
            List of (sender_id, message) tuples
        """
        messages = []
        
        try:
            # Collect all available messages (non-blocking)
            while not self._message_buffer.empty():
                sender_id, message = self._message_buffer.get_nowait()
                
                # Extract actual payload if it's wrapped
                if isinstance(message, dict) and "payload" in message:
                    payload = message["payload"]
                else:
                    payload = message
                
                messages.append((sender_id, payload))
                
        except Exception as e:
            print(f"ANP Network: Error polling messages: {e}")
        
        return messages
    
    async def broadcast_to_agents(self, msg: Dict[str, Any]) -> None:
        """
        Broadcast message to all registered agents.
        
        Args:
            msg: Message to broadcast
        """
        try:
            # Prepare broadcast message
            anp_message = {
                "type": "coordinator_broadcast",
                "from_coordinator": True,
                "payload": msg,
                "timestamp": self.world.timestamp
            }
            
            # Send to each registered agent
            for agent_id, agent_did in self.agent_registry.items():
                if agent_id in self.active_agents:
                    await self.anp_node.send_to_agent(agent_did, anp_message)
                    
        except Exception as e:
            print(f"ANP Network: Failed to broadcast message: {e}")
    
    async def connect(self) -> bool:
        """
        Establish ANP network connection.
        
        Returns:
            True if connection successful
        """
        try:
            if self.node_did and self.node_private_key:
                await self.anp_node.start_coordinator(
                    did_document=self.node_did,
                    private_key=self.node_private_key
                )
            else:
                await self.anp_node.start_coordinator()
            
            print("ANP Network: Coordinator started successfully")
            return True
        except Exception as e:
            print(f"ANP Network: Failed to start coordinator: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect ANP network coordinator."""
        try:
            await self.anp_node.stop_coordinator()
            print("ANP Network: Coordinator stopped")
        except Exception as e:
            print(f"ANP Network: Error stopping coordinator: {e}")
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network status and statistics."""
        return {
            "protocol": "ANP",
            "connected": self.anp_node.is_running() if hasattr(self.anp_node, 'is_running') else True,
            "registered_agents": len(self.agent_registry),
            "active_agents": len(self.active_agents),
            "message_queue_size": self._message_buffer.qsize(),
            "has_coordinator_did": bool(self.node_did),
            "tick": self.current_tick
        } 