# script/streaming_queue/protocol_backend/anp/__init__.py
"""
ANP (Agent Network Protocol) Backend Implementation

This module provides ANP protocol support for the streaming_queue framework.

Components:
- ANPCommBackend: Communication backend implementing BaseCommBackend
- ANPCoordinatorExecutor: Coordinator executor for ANP protocol
- ANPWorkerExecutor: Worker executor for ANP protocol

ANP Protocol Features:
- Simple JSON message format
- Built-in load balancing and fault recovery
- Lightweight network overhead
- Synchronous and asynchronous communication support
"""

from .anp_comm import ANPCommBackend
from .coordinator import ANPCoordinatorExecutor, ANPQACoordinator
from .worker import ANPWorkerExecutor, ANPQAWorker

__all__ = [
    "ANPCommBackend",
    "ANPCoordinatorExecutor",
    "ANPQACoordinator", 
    "ANPWorkerExecutor",
    "ANPQAWorker"
]
