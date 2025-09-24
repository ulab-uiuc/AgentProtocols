# -*- coding: utf-8 -*-
"""
统一协议后端接口与注册表

要求：
- 使用原生协议，实现真实网络调用；不得提供mock/fallback/简化实现
- 仅抽象最小共性：发送（数据面）
- 通过注册表为协调器提供按协议名的发送能力
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional


class BaseProtocolBackend(abc.ABC):
    """协议后端最小接口。

    约束：
    - 所有实现必须基于原生协议的真实端点进行调用
    - 不允许降级为mock或简单实现
    """

    @abc.abstractmethod
    async def send(self, endpoint: str, payload: Dict[str, Any], correlation_id: Optional[str] = None, probe_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """向协议后端发送一条业务消息并返回对端响应（若有）。

        参数：
        - endpoint: 目标服务基础地址，如 http://127.0.0.1:9002
        - payload: 上层业务原始载荷（包含 sender_id/receiver_id/text/body/content 等）
        - correlation_id: 消息关联ID，用于追踪和指标统计
        - probe_config: 可选探针配置，用于S2保密性测试（TLS降级、重放攻击、明文嗅探等）

        返回：
        - 标准化的响应字典 {"status": "success|error", "data": ..., "probe_results": {...}}
        """
        raise NotImplementedError

    # 以下为控制/生命周期接口（Runner 使用）。实现应基于原生SDK。
    async def spawn(self, role: str, port: int, **kwargs: Any) -> Dict[str, Any]:
        """启动本地协议服务（可选实现）。

        要求：
        - 使用该协议官方/原生服务端实现（如 ReceiverServer、acp-sdk server、A2A server、ANP SimpleNode shim）
        - 以子进程/线程方式启动
        
        返回格式：
        {"status": "success|error", "data": {"pid": int, "port": int}, "error": "..."}
        """
        raise NotImplementedError

    async def register(self, agent_id: str, endpoint: str, conversation_id: str, role: str, **kwargs: Any) -> Dict[str, Any]:
        """向 RG 注册（调用各协议 registration_adapter 或原生证明端点）。
        
        返回格式：
        {"status": "success|error", "data": {"agent_id": str, "verification_method": str, "verification_latency_ms": int}, "error": "..."}
        """
        raise NotImplementedError

    async def health(self, endpoint: str) -> Dict[str, Any]:
        """健康检查（调用 /health 或协议自带健康接口）。
        
        返回格式：
        {"status": "success|error", "data": {"healthy": bool, "response_time_ms": int, "details": {}}, "error": "..."}
        """
        raise NotImplementedError


class _BackendRegistry:
    def __init__(self) -> None:
        self._name_to_backend: Dict[str, BaseProtocolBackend] = {}

    def register(self, name: str, backend: BaseProtocolBackend) -> None:
        key = (name or '').strip().lower()
        if not key:
            raise ValueError("protocol name is empty")
        if not isinstance(backend, BaseProtocolBackend):
            raise TypeError("backend must be BaseProtocolBackend")
        self._name_to_backend[key] = backend

    def get(self, name: str) -> Optional[BaseProtocolBackend]:
        if not name:
            return None
        return self._name_to_backend.get(name.strip().lower())


_REGISTRY = _BackendRegistry()


def get_registry() -> _BackendRegistry:
    return _REGISTRY


def register_backend(name: str, backend: BaseProtocolBackend) -> None:
    _REGISTRY.register(name, backend)


