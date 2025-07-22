# script/async_mapf/protocol_backends/dummy/agent.py
from __future__ import annotations
import asyncio
from typing import Dict, Any, Optional, Tuple
from ...core.agent_base import BaseRobot
from ...core.world import GridWorld


class DummyRobot(BaseRobot):
    """
    Dummy protocol implementation of BaseRobot.
    
    Uses local message queues for testing without external dependencies.
    Inherits all pathfinding and coordination algorithms from BaseRobot.
    """
    
    # Class-level message routing system
    _global_mailboxes: Dict[int, "asyncio.Queue[Dict[str, Any]]"] = {}
    _next_message_id = 0
    
    def __init__(self, aid: int, world: GridWorld, goal: Tuple[int, int], **kwargs):
        """
        Initialize Dummy robot.
        
        Args:
            aid: Agent ID
            world: Shared world reference
            goal: Target position
            **kwargs: Additional configuration (ignored for dummy)
        """
        super().__init__(aid, world, goal)
        
        # Create mailbox for this agent
        if aid not in self._global_mailboxes:
            self._global_mailboxes[aid] = asyncio.Queue()
        
        self._mailbox = self._global_mailboxes[aid]
        
        # Dummy-specific settings
        self.delivery_delay = kwargs.get("delivery_delay", 0.001)  # Simulate network delay
        self.message_loss_rate = kwargs.get("message_loss_rate", 0.0)  # 0-1, 0=no loss
        self.max_message_size = kwargs.get("max_message_size", 1024 * 1024)  # 1MB
        
        # Statistics
        self.sent_messages = 0
        self.received_messages = 0
        self.dropped_messages = 0
    
    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """
        Send message to another agent via dummy protocol.
        
        Args:
            dst: Destination agent ID
            payload: Message content
        """
        try:
            # Simulate message size check
            import json
            message_size = len(json.dumps(payload).encode('utf-8'))
            if message_size > self.max_message_size:
                print(f"Dummy Agent {self.aid}: Message too large ({message_size} bytes)")
                self.dropped_messages += 1
                return
            
            # Simulate message loss
            import random
            if random.random() < self.message_loss_rate:
                print(f"Dummy Agent {self.aid}: Message to {dst} lost (simulated)")
                self.dropped_messages += 1
                return
            
            # Prepare message with metadata
            self._next_message_id += 1
            message = {
                **payload,
                "_meta": {
                    "sender_id": self.aid,
                    "recipient_id": dst,
                    "message_id": self._next_message_id,
                    "timestamp": self.world.timestamp,
                    "protocol": "dummy"
                }
            }
            
            # Create target mailbox if it doesn't exist
            if dst not in self._global_mailboxes:
                self._global_mailboxes[dst] = asyncio.Queue()
            
            # Simulate network delay
            if self.delivery_delay > 0:
                await asyncio.sleep(self.delivery_delay)
            
            # Deliver message
            await self._global_mailboxes[dst].put(message)
            self.sent_messages += 1
            
        except Exception as e:
            print(f"Dummy Agent {self.aid}: Failed to send message to {dst}: {e}")
            self.dropped_messages += 1
    
    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Receive message from dummy protocol with timeout.
        
        Args:
            timeout: Maximum wait time in seconds (0 = non-blocking)
            
        Returns:
            Received message or None if timeout
        """
        try:
            if timeout == 0.0:
                # Non-blocking receive
                if self._mailbox.empty():
                    return None
                message = self._mailbox.get_nowait()
            else:
                # Blocking receive with timeout
                message = await asyncio.wait_for(self._mailbox.get(), timeout=timeout)
            
            self.received_messages += 1
            
            # Remove metadata before returning
            if isinstance(message, dict) and "_meta" in message:
                clean_message = {k: v for k, v in message.items() if k != "_meta"}
                return clean_message
            else:
                return message
                
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            print(f"Dummy Agent {self.aid}: Failed to receive message: {e}")
            return None
    
    def get_mailbox_size(self) -> int:
        """Get current number of pending messages."""
        return self._mailbox.qsize()
    
    def clear_mailbox(self) -> int:
        """Clear all pending messages and return count of cleared messages."""
        count = 0
        while not self._mailbox.empty():
            try:
                self._mailbox.get_nowait()
                count += 1
            except:
                break
        return count
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status and statistics."""
        return {
            "agent_id": self.aid,
            "protocol": "dummy",
            "connected": True,  # Dummy is always "connected"
            "mailbox_size": self.get_mailbox_size(),
            "sent_messages": self.sent_messages,
            "received_messages": self.received_messages,
            "dropped_messages": self.dropped_messages,
            "delivery_delay": self.delivery_delay,
            "message_loss_rate": self.message_loss_rate
        }
    
    @classmethod
    def reset_global_state(cls) -> None:
        """Reset global message routing state (useful for testing)."""
        cls._global_mailboxes.clear()
        cls._next_message_id = 0
    
    @classmethod
    def get_global_stats(cls) -> Dict[str, Any]:
        """Get global statistics for all dummy agents."""
        total_queued = sum(q.qsize() for q in cls._global_mailboxes.values())
        return {
            "total_agents": len(cls._global_mailboxes),
            "total_queued_messages": total_queued,
            "next_message_id": cls._next_message_id
        } 