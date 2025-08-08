"""
Communication abstraction layer for agent-network interaction.

Provides protocol-agnostic adapter interface that can be extended
for different communication protocols (A2A, ANP, etc.) without
requiring changes to NetworkBase.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class AbstractCommAdapter(ABC):
    """Protocol-agnostic adapter between Agent and Network."""
    
    @abstractmethod
    async def send(self, obj: Any) -> None:
        """Send object to the communication endpoint."""
        pass
    
    @abstractmethod
    async def recv(self) -> Any:
        """Receive object from the communication endpoint."""
        pass

    @abstractmethod
    def recv_nowait(self) -> Any:
        """Non-blocking receive from communication endpoint."""
        pass


class LocalQueueAdapter(AbstractCommAdapter):
    """
    Local queue-based adapter for single-process communication.
    
    Uses asyncio.Queue to pass messages within the same Python process.
    Suitable for testing and single-node scenarios.
    """
    
    def __init__(self):
        """Initialize with an async queue."""
        self._queue: asyncio.Queue = asyncio.Queue()
    
    async def send(self, obj: Any) -> None:
        """Put object into the queue."""
        await self._queue.put(obj)
    
    async def recv(self) -> Any:
        """Get object from the queue (blocks until available)."""
        return await self._queue.get()
    
    def recv_nowait(self) -> Any:
        """Get object from the queue without blocking."""
        return self._queue.get_nowait()
    
    def qsize(self) -> int:
        """Get current queue size (for debugging)."""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty() 