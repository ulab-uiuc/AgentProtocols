# -*- coding: utf-8 -*-
"""
ACP Protocol Registration Adapter
使用原生 ACP 服务器接口进行注册/订阅配合（无 mock / 无 fallback）
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, List

import httpx


class ACPRegistrationAdapter:
    """ACP 协议注册适配器（配合 RegistrationGateway）

    设计目标：
    - 使用原生 ACP 服务接口进行端点探测与证明构造（/agents, /ping）
    - 不使用任何 mock、fallback 或虚拟模式
    - 与 AgoraRegistrationAdapter 提供一致的外部方法签名
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
        acp_probe_endpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """注册 ACP Agent 到 RG

        要求：endpoint 必须是原生 ACP 服务器根地址（可直接访问 /agents 与 /ping）
        """
        # 探测 ACP 端点，获取可用的 agent 名称
        base = (acp_probe_endpoint or endpoint).rstrip("/")
        async with httpx.AsyncClient() as client:
            agents_resp = await client.get(f"{base}/agents", timeout=10.0)
            if agents_resp.status_code != 200:
                raise RuntimeError(f"ACP /agents probe failed: {agents_resp.status_code}")
            agents_payload = agents_resp.json()
            agent_names = self._extract_agent_names(agents_payload)
            if not agent_names:
                raise RuntimeError("ACP /agents returned empty list")

            ping_resp = await client.get(f"{base}/ping", timeout=10.0)
            if ping_resp.status_code != 200:
                raise RuntimeError(f"ACP /ping probe failed: {ping_resp.status_code}")

        # 选择一个 ACP agent 名称用于证明（优先同名，其次第一个）
        acp_agent_name = agent_id if agent_id in agent_names else agent_names[0]

        proof = {
            "timestamp": time.time(),
            "nonce": str(uuid.uuid4()),
            "acp_agent_name": acp_agent_name,
            "endpoint_ping": True,
            "acp_probe_endpoint": base,
        }

        registration_request = {
            "protocol": "acp",
            "agent_id": agent_id,
            "endpoint": endpoint,
            "conversation_id": conversation_id,
            "role": role,
            "protocolMeta": {
                "protocol_version": "1.0",
                "capabilities": ["runs", "agents", "stream"],
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
        """订阅 Observer 到 RG（独立于 ACP 服务器）"""
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

    def _extract_agent_names(self, agents_payload: Any) -> List[str]:
        """从 /agents Response中提取 agent 名称列表（兼容 acp_sdk 的模型结构）"""
        try:
            # acp_sdk /agents 返回形如 {"agents":[{"name": "...", ...}, ...]}
            agents = agents_payload.get("agents", []) if isinstance(agents_payload, dict) else []
            names = []
            for item in agents:
                name = item.get("name")
                if isinstance(name, str) and name:
                    names.append(name)
            return names
        except Exception:
            return []


