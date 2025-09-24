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
    from script.safety_tech.protocol_backends.common.interfaces import get_registry
    
    # 确保所有协议后端都被导入，触发它们的注册代码
    try:
        import script.safety_tech.protocol_backends.anp
        import script.safety_tech.protocol_backends.acp  
        import script.safety_tech.protocol_backends.a2a
        import script.safety_tech.protocol_backends.agora
    except ImportError as e:
        # 如果某个协议后端导入失败，记录但不阻断其他协议
        print(f"Warning: Failed to import protocol backend: {e}")
    
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


