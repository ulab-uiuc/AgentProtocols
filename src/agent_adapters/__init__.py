"""
协议适配器模块 - 统一不同Agent通信协议的接口
Direct imports - fail fast if dependencies are missing
"""

from .base_adapter import BaseProtocolAdapter
from .a2a_adapter import A2AAdapter
from .agent_protocol_adapter import AgentProtocolAdapter
from .acp_adapter import ACPAdapter
from .agora_adapter import AgoraClientAdapter
from .anp_adapter import ANPAdapter, ANPMessageBuilder

__all__ = [
    "BaseProtocolAdapter",
    "A2AAdapter",
    "ACPAdapter",
    "AgentProtocolAdapter",
    "AgoraClientAdapter",
    "ANPAdapter",
    "ANPMessageBuilder"
]
