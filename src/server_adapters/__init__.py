"""
Server adapters for different protocols.

This package provides pluggable server adapters that allow BaseAgent to support
multiple communication protocols (A2A, IoA, ACP, etc.) without hardcoding
protocol-specific logic.

Available adapters:
- BaseServerAdapter: Abstract base class for all adapters
- A2AServerAdapter: Agent-to-Agent protocol adapter
- DummyServerAdapter: Testing adapter
"""

from .base_adapter import BaseServerAdapter
from .a2a_adapter import A2AServerAdapter, A2AStarletteApplication
from .dummy_adapter import DummyServerAdapter

__all__ = [
    "BaseServerAdapter",
    "A2AServerAdapter", 
    "A2AStarletteApplication",
    "DummyServerAdapter",
] 