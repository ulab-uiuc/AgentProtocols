"""
协议适配器模块 - 统一不同Agent通信协议的接口
"""

from .base_adapter import BaseProtocolAdapter
from .a2a_adapter import A2AAdapter
from .agent_protocol_adapter import AgentProtocolAdapter

# Import ANP adapter with graceful fallback
try:
    from .anp_adapter import ANPAdapter, ANPMessageBuilder
    ANP_AVAILABLE = True
except ImportError:
    # AgentConnect not available, create placeholder
    ANPAdapter = None
    ANPMessageBuilder = None
    ANP_AVAILABLE = False

if ANP_AVAILABLE:
    __all__ = ["BaseProtocolAdapter", "A2AAdapter", "AgentProtocolAdapter", "ANPAdapter", "ANPMessageBuilder"]
else:
    __all__ = ["BaseProtocolAdapter", "A2AAdapter", "AgentProtocolAdapter"]