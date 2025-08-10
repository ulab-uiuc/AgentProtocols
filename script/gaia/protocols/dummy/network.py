"""
Dummy Network implementation for GAIA framework.
Implements broadcast and polling mechanisms for dummy protocol.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Tuple, Optional
import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.network import MeshNetwork
from protocols.dummy.agent import DummyAgent


class DummyNetwork(MeshNetwork):
    """
    Dummy Network implementation with broadcast and enhanced polling capabilities.
    """
    
    def __init__(self):
        super().__init__()
        self._broadcast_queue = asyncio.Queue()  # 专门的广播队列
    
    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """
        Deliver message to specific agent using dummy protocol.
        
        Args:
            dst: Destination agent ID
            msg: Message payload
        """
        target_agent = self.get_agent_by_id(dst)
        if target_agent and isinstance(target_agent, DummyAgent):
            try:
                # Add network metadata
                enhanced_msg = {
                    **msg,
                    "_network_meta": {
                        "delivered_by": "dummy_network",
                        "target_agent": dst,
                        "delivery_timestamp": time.time(),
                        "message_id": f"net_{int(time.time() * 1000000)}"
                    }
                }
                
                # Record message in message pool (inherited from base class)
                await self._record_input_message(dst, enhanced_msg)
                
                # Directly put message into agent's mailbox
                await target_agent._client._mailbox.put(enhanced_msg)
                
                # Update metrics
                self.pkt_cnt += 1
                msg_size = len(json.dumps(msg).encode('utf-8'))
                self.bytes_tx += msg_size
                
                print(f"📤 DummyNetwork delivered message to agent {dst}: {msg.get('type', 'unknown')}")
                
            except Exception as e:
                print(f"❌ DummyNetwork failed to deliver message to agent {dst}: {e}")
        else:
            print(f"❌ Agent {dst} not found or not a DummyAgent")
    
    async def broadcast(self, msg: Dict[str, Any], exclude_sender: Optional[int] = None) -> None:
        """
        Broadcast message to all agents in the network.
        
        Args:
            msg: Message payload to broadcast
            exclude_sender: Optional sender ID to exclude from broadcast
        """
        print(f"📡 Broadcasting message to all agents: {msg.get('type', 'unknown')}")
        
        # Add broadcast metadata
        broadcast_msg = {
            **msg,
            "_broadcast_meta": {
                "is_broadcast": True,
                "sender_id": exclude_sender,
                "broadcast_timestamp": time.time(),
                "broadcast_id": f"bc_{int(time.time() * 1000000)}"
            }
        }
        
        # Send to all agents except the sender
        delivered_count = 0
        for agent in self.agents:
            if exclude_sender is None or agent.id != exclude_sender:
                try:
                    await self.deliver(agent.id, broadcast_msg)
                    delivered_count += 1
                except Exception as e:
                    print(f"❌ Failed to broadcast to agent {agent.id}: {e}")
        
        print(f"📡 Broadcast completed to {delivered_count} agents")
    
    async def poll(self) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Enhanced polling that collects all available messages from all agents.
        
        Returns:
            List of (sender_id, message) tuples
        """
        all_messages = []
        
        # Poll each agent multiple times to get all available messages
        for agent in self.agents:
            agent_messages = []
            
            # Keep polling until no more messages
            while True:
                try:
                    msg = await agent.recv_msg(timeout=0.0)  # Non-blocking
                    if msg:
                        # Update metrics
                        self.pkt_cnt += 1
                        msg_size = len(json.dumps(msg).encode('utf-8'))
                        self.bytes_rx += msg_size
                        
                        agent_messages.append((agent.id, msg))
                    else:
                        break  # No more messages
                except Exception as e:
                    break  # Stop on error
            
            all_messages.extend(agent_messages)
            
            if agent_messages:
                print(f"📥 Polled {len(agent_messages)} messages from agent {agent.id}")
        
        return all_messages
