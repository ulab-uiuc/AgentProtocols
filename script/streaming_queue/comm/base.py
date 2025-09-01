# -*- coding: utf-8 -*-
"""
Comm Backend Base
抽象出网络通信层，AgentNetwork 不再直接依赖具体协议/HTTP。
"""

from __future__ import annotations
import abc
from typing import Any, Dict, Optional


class BaseCommBackend(abc.ABC):
    """
    抽象通信后端：
      - register_endpoint: 注册一个 agent 的服务地址（或者 inproc 句柄）
      - connect: （可选）建立连接/预热
      - send: 从 src 向 dst 发送一条消息，返回协议原生响应
      - health_check: 探活
      - close: 关闭资源
    """

    @abc.abstractmethod
    async def register_endpoint(self, agent_id: str, address: str) -> None:
        ...

    async def connect(self, src_id: str, dst_id: str) -> None:
        """某些协议需要显式建连；默认 no-op。"""
        return None

    @abc.abstractmethod
    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        ...

    @abc.abstractmethod
    async def health_check(self, agent_id: str) -> bool:
        ...

    async def close(self) -> None:
        """关闭底层资源（HTTP 客户端、socket 等）。默认 no-op。"""
        return None
