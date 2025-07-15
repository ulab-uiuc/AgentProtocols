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

# Try to import adapters, but handle missing dependencies gracefully
__all__ = ["BaseServerAdapter"]

try:
    from .a2a_adapter import A2AServerAdapter, A2AStarletteApplication
    __all__.extend(["A2AServerAdapter", "A2AStarletteApplication"])
except ImportError:
    pass

try:
    from .acp_adapter import ACPServerAdapter, ACPStarletteApplication
    __all__.extend(["ACPServerAdapter", "ACPStarletteApplication"])
except ImportError:
    pass

try:
    from .dummy_adapter import DummyServerAdapter
    __all__.append("DummyServerAdapter")
except ImportError:
    pass