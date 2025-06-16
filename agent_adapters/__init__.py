"""
协议适配器模块 - 统一不同Agent通信协议的接口
"""

from .base_adapter import BaseProtocolAdapter
from .a2a_adapter import A2AAdapter

__all__ = ["BaseProtocolAdapter", "A2AAdapter"] 