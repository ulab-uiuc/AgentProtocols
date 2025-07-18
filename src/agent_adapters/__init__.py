"""
协议适配器模块 - 统一不同Agent通信协议的接口
"""

# from .base_adapter import BaseProtocolAdapter
# from .a2a_adapter import A2AAdapter

# __all__ = ["BaseProtocolAdapter", "A2AAdapter"] 


"""
协议适配器模块 - 统一不同Agent通信协议的接口
"""

from .base_adapter import BaseProtocolAdapter
from .a2a_adapter import A2AAdapter
from .agora_adapter import AgoraClientAdapter, AgoraServerAdapter, AgoraServerWrapper  # 添加

__all__ = [
    "BaseProtocolAdapter",
    "A2AAdapter",
    "AgoraClientAdapter",
    "AgoraServerAdapter",
    "AgoraServerWrapper"
]
