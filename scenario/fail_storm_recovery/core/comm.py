"""
Communication abstraction layer for fail-storm recovery scenarios.

Provides protocol-agnostic adapter interface that can be extended
for different communication protocols (SimpleJSON, A2A, ANP, etc.) without
requiring changes to the core fail-storm logic.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Tuple


class AbstractCommAdapter(ABC):
    """Protocol-agnostic adapter for agent communication in fail-storm scenarios."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection to the communication system."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the communication system."""
        pass
    
    @abstractmethod
    async def send(self, target: str, message: Dict[str, Any]) -> None:
        """Send message to specific target."""
        pass
    
    @abstractmethod
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all connected agents."""
        pass
    
    @abstractmethod
    async def recv(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Receive message from communication system (blocking)."""
        pass
    
    @abstractmethod
    def recv_nowait(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Receive message from communication system (non-blocking)."""
        pass
    
    @abstractmethod
    async def send_heartbeat(self, agent_id: str) -> None:
        """Send heartbeat message for fault detection."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        pass


class LocalQueueAdapter(AbstractCommAdapter):
    """
    Local queue-based adapter for single-process communication.
    
    Uses asyncio.Queue to pass messages within the same Python process.
    Suitable for testing and single-node scenarios.
    """
    
    def __init__(self, agent_id: str):
        """Initialize with agent ID and internal queues."""
        self.agent_id = agent_id
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._outbox: asyncio.Queue = asyncio.Queue()
        self._broadcast_queue: asyncio.Queue = asyncio.Queue()
        self._connected = False
        self._registered_agents: Dict[str, asyncio.Queue] = {}
        
    async def connect(self) -> None:
        """Initialize local connection."""
        self._connected = True
        print(f"[LocalQueueAdapter] {self.agent_id} connected")
    
    async def disconnect(self) -> None:
        """Close local connection."""
        self._connected = False
        print(f"[LocalQueueAdapter] {self.agent_id} disconnected")
    
    async def send(self, target: str, message: Dict[str, Any]) -> None:
        """Send message to specific target via local queue."""
        if not self._connected:
            raise ConnectionError("Adapter not connected")
        
        if target in self._registered_agents:
            # Add metadata to message
            message_with_metadata = {
                **message,
                "sender": self.agent_id,
                "target": target,
                "timestamp": asyncio.get_event_loop().time()
            }
            await self._registered_agents[target].put(message_with_metadata)
        else:
            print(f"[LocalQueueAdapter] Target {target} not found")
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all registered agents."""
        if not self._connected:
            raise ConnectionError("Adapter not connected")
        
        # Add metadata to message
        message_with_metadata = {
            **message,
            "sender": self.agent_id,
            "broadcast": True,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Send to all registered agents except self
        for agent_id, queue in self._registered_agents.items():
            if agent_id != self.agent_id:
                await queue.put(message_with_metadata)
    
    async def recv(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Receive message from local queue (blocking)."""
        if not self._connected:
            return None
        
        try:
            message = await self._inbox.get()
            sender = message.get("sender", "unknown")
            return sender, message
        except asyncio.CancelledError:
            return None
    
    def recv_nowait(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Receive message from local queue (non-blocking)."""
        if not self._connected:
            return None
        
        try:
            message = self._inbox.get_nowait()
            sender = message.get("sender", "unknown")
            return sender, message
        except asyncio.QueueEmpty:
            return None
    
    async def send_heartbeat(self, agent_id: str) -> None:
        """Send heartbeat message."""
        heartbeat_msg = {
            "type": "heartbeat",
            "agent_id": agent_id,
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.broadcast(heartbeat_msg)
    
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