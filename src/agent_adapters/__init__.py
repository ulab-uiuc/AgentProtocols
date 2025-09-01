"""
协议适配器模块 - 统一不同Agent通信协议的接口
"""

from .base_adapter import BaseProtocolAdapter
from .a2a_adapter import A2AAdapter
from .agora_adapter import AgoraClientAdapter
from .agent_protocol_adapter import AgentProtocolAdapter
from .acp_adapter import ACPAdapter # Assuming this is a client adapter

# __all__ = ["BaseProtocolAdapter", "A2AAdapter"]

# __all__ = [
#     "BaseProtocolAdapter",
#     "A2AAdapter",
#     "AgoraClientAdapter",
#     "AgoraServerAdapter",
#     "AgoraServerWrapper"
# ]

"""
协议适配器模块 - 统一不同Agent通信协议的接口
"""

from .base_adapter import BaseProtocolAdapter
from .a2a_adapter import A2AAdapter

# Optional agora adapter import
try:
    from .agora_adapter import AgoraClientAdapter, AgoraServerAdapter, AgoraServerWrapper  # 添加
    AGORA_AVAILABLE = True
except ImportError:
    # Agora adapter not available, create placeholder classes
    class AgoraClientAdapter:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Agora adapter not available - missing 'agora' module")

    class AgoraServerAdapter:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Agora adapter not available - missing 'agora' module")

    class AgoraServerWrapper:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Agora adapter not available - missing 'agora' module")

    AGORA_AVAILABLE = False

__all__ = ["BaseProtocolAdapter", "A2AAdapter", "ACPAdapter" , "AgentProtocolAdapter" , "AgoraClientAdapter", "AgoraServerAdapter", "AgoraServerWrapper", "ANPAdapter", "ANPMessageBuilder"]
