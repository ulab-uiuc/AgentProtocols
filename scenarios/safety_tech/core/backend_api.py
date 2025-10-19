# -*- coding: utf-8 -*-
"""
Unified Backend API used by runners to interact with protocol backends.
- spawn_backend(protocol, role, port, **kwargs)
- register_backend(protocol, agent_id, endpoint, conversation_id, role, **kwargs)
- health_backend(protocol, endpoint)
 - send_backend(protocol, endpoint, payload)

Backends are original/native implementations registered via protocol_backends.common.interfaces.register_backend.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _get_backend(protocol: str):
    from scenarios.safety_tech.protocol_backends.common.interfaces import get_registry
    
    # 确保所有协议后端都被导入，触发它们的注册代码
    # 每个协议单独 try-catch，避免一个失败影响其他协议
    protocols_to_import = [
        ('anp', 'scenario.safety_tech.protocol_backends.anp'),
        ('acp', 'scenario.safety_tech.protocol_backends.acp'),
        ('a2a', 'scenario.safety_tech.protocol_backends.a2a'),
        ('agora', 'scenario.safety_tech.protocol_backends.agora'),
    ]
    
    for proto_name, module_path in protocols_to_import:
        try:
            __import__(module_path)
        except ImportError as e:
            # 只在请求该协议时才报错，否则仅警告
            if protocol == proto_name:
                print(f"Error: Cannot load {proto_name} backend: {e}")
                raise RuntimeError(f"Protocol backend '{proto_name}' is not available. Missing dependency: {e}")
            else:
                # 其他协议导入失败只是警告，不影响当前协议
                pass
        except Exception as e:
            # 其他异常也类似处理
            if protocol == proto_name:
                print(f"Error: Cannot initialize {proto_name} backend: {e}")
                raise RuntimeError(f"Protocol backend '{proto_name}' failed to initialize: {e}")
    
    registry = get_registry()
    backend = registry.get(protocol)
    if backend is None:
        raise RuntimeError(f"No backend registered for protocol: {protocol}")
    return backend


async def spawn_backend(protocol: str, role: str, port: int, **kwargs: Any) -> Dict[str, Any]:
    backend = _get_backend(protocol)
    return await backend.spawn(role, port, **kwargs)


async def register_backend(protocol: str, agent_id: str, endpoint: str, conversation_id: str, role: str, **kwargs: Any) -> Dict[str, Any]:
    backend = _get_backend(protocol)
    return await backend.register(agent_id, endpoint, conversation_id, role, **kwargs)


async def health_backend(protocol: str, endpoint: str) -> Dict[str, Any]:
    backend = _get_backend(protocol)
    return await backend.health(endpoint)


async def send_backend(protocol: str, endpoint: str, payload: Dict[str, Any], correlation_id: Optional[str] = None, probe_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """向指定协议后端直接发送业务消息（跳过协调器）。

    用途：
    - 在Runner中进行协议直连发压/注入探针
    - 需要直接观察协议数据面行为而不经由路由器
    - S1负载测试：并发/RPS/背压点测试
    - S2保密性测试：TLS降级、重放攻击、明文嗅探探针
    
    参数：
    - correlation_id: 消息关联ID，用于追踪和指标统计
    - probe_config: 探针配置，如 {"tls_downgrade": True, "replay_nonce": "xxx", "plaintext_sniff": True}
    """
    backend = _get_backend(protocol)
    return await backend.send(endpoint, payload, correlation_id, probe_config)


