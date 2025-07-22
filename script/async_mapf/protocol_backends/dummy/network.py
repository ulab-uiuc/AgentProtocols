# script/async_mapf/protocol_backends/dummy/network.py
from __future__ import annotations
import asyncio
from typing import Dict, Any, List, Tuple
from ...core.network_base import BaseNet
from ...core.world import GridWorld


class DummyNet(BaseNet):
    """
    Dummy protocol implementation of BaseNet.
    
    Uses local message queues for testing without external dependencies.
    Inherits all coordination and conflict resolution logic from BaseNet.
    """
    
    # Global coordinator mailbox
    _coordinator_mailbox: "asyncio.Queue[Tuple[int, Dict[str, Any]]]" = None
    _next_coordinator_message_id = 0
    
    def __init__(self, world: GridWorld, tick_ms: int = 10, **kwargs):
        """
        Initialize Dummy network coordinator.
        
        Args:
            world: Shared world reference
            tick_ms: Time step duration in milliseconds
            **kwargs: Additional configuration
        """
        super().__init__(world, tick_ms)
        
        # Initialize global coordinator mailbox
        if DummyNet._coordinator_mailbox is None:
            DummyNet._coordinator_mailbox = asyncio.Queue()
        
        self._mailbox = DummyNet._coordinator_mailbox
        
        # Dummy-specific settings
        self.delivery_delay = kwargs.get("delivery_delay", 0.001)
        self.broadcast_delay = kwargs.get("broadcast_delay", 0.002)
        self.message_loss_rate = kwargs.get("message_loss_rate", 0.0)
        
        # Statistics
        self.delivered_messages = 0
        self.broadcast_messages = 0
        self.dropped_messages = 0
        
        # Agent mailbox references (for direct delivery)
        from .agent import DummyRobot
        self._agent_mailboxes = DummyRobot._global_mailboxes
    
    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """
        Deliver message to specific agent via dummy protocol.
        
        Args:
            dst: Destination agent ID
            msg: Message payload
        """
        try:
            # Simulate message loss
            import random
            if random.random() < self.message_loss_rate:
                print(f"Dummy Network: Message to agent {dst} lost (simulated)")
                self.dropped_messages += 1
                return
            
            # Prepare coordinator message
            DummyNet._next_coordinator_message_id += 1
            message = {
                **msg,
                "_meta": {
                    "sender_id": -1,  # Network coordinator
                    "recipient_id": dst,
                    "message_id": DummyNet._next_coordinator_message_id,
                    "timestamp": self.world.timestamp,
                    "protocol": "dummy",
                    "from_coordinator": True
                }
            }
            
            # Create target mailbox if it doesn't exist
            if dst not in self._agent_mailboxes:
                self._agent_mailboxes[dst] = asyncio.Queue()
            
            # Simulate delivery delay
            if self.delivery_delay > 0:
                await asyncio.sleep(self.delivery_delay)
            
            # Deliver message directly to agent mailbox
            await self._agent_mailboxes[dst].put(message)
            self.delivered_messages += 1
            
        except Exception as e:
            print(f"Dummy Network: Failed to deliver message to agent {dst}: {e}")
            self.dropped_messages += 1
    
    async def poll(self) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Poll for incoming messages from agents.
        
        Returns:
            List of (sender_id, message) tuples
        """
        messages = []
        
        try:
            # Collect all available messages (non-blocking)
            while not self._mailbox.empty():
                sender_id, message = self._mailbox.get_nowait()
                
                # Remove metadata before processing
                if isinstance(message, dict) and "_meta" in message:
                    clean_message = {k: v for k, v in message.items() if k != "_meta"}
                else:
                    clean_message = message
                
                messages.append((sender_id, clean_message))
                
        except Exception as e:
            print(f"Dummy Network: Error polling messages: {e}")
        
        return messages
    
    async def broadcast_to_agents(self, msg: Dict[str, Any]) -> None:
        """
        Broadcast message to all active agents.
        
        Args:
            msg: Message to broadcast
        """
        try:
            # Simulate broadcast delay
            if self.broadcast_delay > 0:
                await asyncio.sleep(self.broadcast_delay)
            
            # Send to each active agent
            broadcast_tasks = []
            for agent_id in self.active_agents:
                task = self.deliver(agent_id, {
                    **msg,
                    "_broadcast": True
                })
                broadcast_tasks.append(task)
            
            # Execute all broadcasts concurrently
            if broadcast_tasks:
                await asyncio.gather(*broadcast_tasks, return_exceptions=True)
                self.broadcast_messages += 1
                
        except Exception as e:
            print(f"Dummy Network: Failed to broadcast message: {e}")
    
    def send_to_coordinator(self, sender_id: int, msg: Dict[str, Any]) -> None:
        """
        Send message from agent to coordinator (synchronous).
        
        This is called by dummy agents to send messages to the coordinator.
        """
        try:
            # Simulate message loss
            import random
            if random.random() < self.message_loss_rate:
                return
            
            # Put message in coordinator mailbox
            self._mailbox.put_nowait((sender_id, msg))
            
        except Exception as e:
            print(f"Dummy Network: Failed to receive message from agent {sender_id}: {e}")
    
    def get_coordinator_mailbox_size(self) -> int:
        """Get current number of pending coordinator messages."""
        return self._mailbox.qsize()
    
    def clear_coordinator_mailbox(self) -> int:
        """Clear all pending coordinator messages."""
        count = 0
        while not self._mailbox.empty():
            try:
                self._mailbox.get_nowait()
                count += 1
            except:
                break
        return count
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network status and statistics."""
        total_agent_messages = sum(q.qsize() for q in self._agent_mailboxes.values())
        
        return {
            "protocol": "dummy",
            "connected": True,  # Dummy is always "connected"
            "coordinator_mailbox_size": self.get_coordinator_mailbox_size(),
            "total_agent_mailboxes": len(self._agent_mailboxes),
            "total_queued_agent_messages": total_agent_messages,
            "delivered_messages": self.delivered_messages,
            "broadcast_messages": self.broadcast_messages,
            "dropped_messages": self.dropped_messages,
            "active_agents": len(self.active_agents),
            "tick": self.current_tick,
            "delivery_delay": self.delivery_delay,
            "message_loss_rate": self.message_loss_rate
        }
    
    @classmethod
    def reset_global_state(cls) -> None:
        """Reset global coordinator state (useful for testing)."""
        if cls._coordinator_mailbox:
            while not cls._coordinator_mailbox.empty():
                try:
                    cls._coordinator_mailbox.get_nowait()
                except:
                    break
        cls._next_coordinator_message_id = 0
    
    @classmethod
    def get_global_coordinator_stats(cls) -> Dict[str, Any]:
        """Get global coordinator statistics."""
        mailbox_size = cls._coordinator_mailbox.qsize() if cls._coordinator_mailbox else 0
        return {
            "coordinator_mailbox_size": mailbox_size,
            "next_message_id": cls._next_coordinator_message_id
        } 