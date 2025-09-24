# -*- coding: utf-8 -*-
"""
A2A Protocol Registration Adapter
使用原生 A2A 服务器接口进行注册配合（无 mock / 无 fallback）
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, Any, Optional

import httpx


class A2ARegistrationAdapter:
    """A2A 协议注册适配器（配合 RegistrationGateway）
    
    设计目标：
    - 使用原生 A2A 服务接口进行端点探测与证明构造（/health）
    - 不使用任何 mock、fallback 或虚拟模式
    - 与其他协议适配器提供一致的外部方法签名
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rg_endpoint = config.get('rg_endpoint', 'http://127.0.0.1:8001')

    async def register_agent(
        self,
        agent_id: str,
        endpoint: str,
        conversation_id: str,
        role: str = "doctor",
    ) -> Dict[str, Any]:
        """注册 A2A Agent 到 RG
        
        要求：endpoint 必须是原生 A2A 服务器根地址（可直接访问 /health）
        """
        # 探测 A2A 端点健康状态
        base = endpoint.rstrip("/")
        async with httpx.AsyncClient() as client:
            health_resp = await client.get(f"{base}/health", timeout=10.0)
            if health_resp.status_code != 200:
                raise RuntimeError(f"A2A /health probe failed: {health_resp.status_code}")
            
        # 构造 A2A 原生证明
        proof = {
            "timestamp": time.time(),
            "nonce": str(uuid.uuid4()),
            "a2a_token": f"token_{agent_id}_{int(time.time())}",
            "endpoint_health": True,
        }

        registration_request = {
            "protocol": "a2a",
            "agent_id": agent_id,
            "endpoint": endpoint,
            "conversation_id": conversation_id,
            "role": role,
            "protocolMeta": {
                "protocol_version": "1.0",
                "capabilities": ["message", "health"],
            },
            "proof": proof,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.rg_endpoint}/register",
                json=registration_request,
                timeout=30.0,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Registration failed: {resp.status_code} - {resp.text}")
            return resp.json()

    async def subscribe_observer(
        self,
        observer_id: str,
        conversation_id: str,
        endpoint: str = "",
    ) -> Dict[str, Any]:
        """订阅 Observer 到 RG（独立于 A2A 服务器）"""
        proof = {
            "timestamp": time.time(),
            "nonce": str(uuid.uuid4()),
            "observer_type": "passive_listener",
        }

        subscription_request = {
            "agent_id": observer_id,
            "conversation_id": conversation_id,
            "role": "observer",
            "endpoint": endpoint,
            "proof": proof,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.rg_endpoint}/subscribe",
                json=subscription_request,
                timeout=30.0,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Observer subscription failed: {resp.status_code} - {resp.text}")
            return resp.json()

    async def get_conversation_directory(self, conversation_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.rg_endpoint}/directory",
                params={"conversation_id": conversation_id},
                timeout=15.0,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Directory query failed: {resp.status_code} - {resp.text}")
            return resp.json()
