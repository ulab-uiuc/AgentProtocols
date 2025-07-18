"""
Server adapters for different protocols.

This package provides pluggable server adapters that allow BaseAgent to support
multiple communication protocols (A2A, IoA, ACP, etc.) without hardcoding
protocol-specific logic.

Available adapters:
- BaseServerAdapter: Abstract base class for all adapters
- A2AServerAdapter: Agent-to-Agent protocol adapter
- ACPServerAdapter: Agent Communication Protocol adapter
- DummyServerAdapter: Testing adapter
"""

from .base_adapter import BaseServerAdapter
from .a2a_adapter import A2AServerAdapter, A2AStarletteApplication
from .acp_adapter import ACPServerAdapter

__all__ = [
    "BaseServerAdapter",
    "A2AServerAdapter",
    "ACPServerAdapter",
    "A2AStarletteApplication",
    "DummyServerAdapter",
]