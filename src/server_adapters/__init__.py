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

# Import ACP server adapter
from .acp_adapter import ACPServerAdapter

# Import ANP server adapter directly
from .anp_adapter import ANPServerAdapter, ANPExecutorWrapper, ANPSimpleNodeWrapper
ANP_AVAILABLE = True

# Import Simple JSON server adapter
from .simple_json_adapter import SimpleJSONServerAdapter

# Always export all symbols, even if some are None
__all__ = [
    "BaseServerAdapter",
    "A2AServerAdapter",
    "ACPServerAdapter",
    "A2AStarletteApplication",
    "AgentProtocolServerAdapter",
    "AgentProtocolStarletteApplication",
    "ANPServerAdapter",
    "ANPExecutorWrapper", 
    "ANPSimpleNodeWrapper",
    "DummyServerAdapter",
    "SimpleJSONServerAdapter",
    "ANP_AVAILABLE"
]
