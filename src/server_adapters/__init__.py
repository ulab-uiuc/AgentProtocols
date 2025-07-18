"""
Server adapters for different protocols.

This package provides pluggable server adapters that allow BaseAgent to support
multiple communication protocols (A2A, IoA, ACP, ANP, etc.) without hardcoding
protocol-specific logic.

Available adapters:
- BaseServerAdapter: Abstract base class for all adapters
- A2AServerAdapter: Agent-to-Agent protocol adapter
- AgentProtocolServerAdapter: Agent Protocol v1 adapter
- ANPServerAdapter: Agent Network Protocol (ANP) adapter
- DummyServerAdapter: Testing adapter
"""

from .base_adapter import BaseServerAdapter
from .a2a_adapter import A2AServerAdapter, A2AStarletteApplication
from .agent_protocol_adapter import AgentProtocolServerAdapter, AgentProtocolStarletteApplication
from .dummy_adapter import DummyServerAdapter

# Import ANP server adapter with graceful fallback
try:
    from .anp_adapter import ANPServerAdapter, ANPExecutorWrapper, ANPSimpleNodeWrapper
    ANP_AVAILABLE = True
except ImportError:
    # AgentConnect not available, create placeholders
    ANPServerAdapter = None
    ANPExecutorWrapper = None
    ANPSimpleNodeWrapper = None
    ANP_AVAILABLE = False

if ANP_AVAILABLE:
    __all__ = [
        "BaseServerAdapter",
        "A2AServerAdapter", 
        "A2AStarletteApplication",
        "AgentProtocolServerAdapter",
        "AgentProtocolStarletteApplication",
        "ANPServerAdapter",
        "ANPExecutorWrapper",
        "ANPSimpleNodeWrapper",
        "DummyServerAdapter",
    ]
else:
    __all__ = [
        "BaseServerAdapter",
        "A2AServerAdapter", 
        "A2AStarletteApplication",
        "AgentProtocolServerAdapter",
        "AgentProtocolStarletteApplication", 
        "DummyServerAdapter",
    ]