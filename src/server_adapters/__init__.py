"""
Server adapters for different protocols.

This package provides pluggable server adapters that allow BaseAgent to support
multiple communication protocols (A2A, IoA, ACP, etc.) without hardcoding
protocol-specific logic.

Available adapters:
- BaseServerAdapter: Abstract base class for all adapters
- A2AServerAdapter: Agent-to-Agent protocol adapter
- AgentProtocolServerAdapter: Agent Protocol v1 adapter
- DummyServerAdapter: Testing adapter
"""

from .base_adapter import BaseServerAdapter
from .a2a_adapter import A2AServerAdapter, A2AStarletteApplication
from .agent_protocol_adapter import AgentProtocolServerAdapter, AgentProtocolStarletteApplication
from .dummy_adapter import DummyServerAdapter

__all__ = [
    "BaseServerAdapter",
    "A2AServerAdapter", 
    "A2AStarletteApplication",
    "AgentProtocolServerAdapter",
    "AgentProtocolStarletteApplication", 
    "DummyServerAdapter",
]