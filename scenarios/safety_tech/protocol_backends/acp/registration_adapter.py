# -*- coding: utf-8 -*-
"""
ACP Protocol Registration Adapter
Uses native ACP server interface for registration/subscription coordination (no mock / no fallback)
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, List

import httpx


class ACPRegistrationAdapter:
    """ACP protocol registration adapter (works with RegistrationGateway)

    Design goals:
    - Use native ACP service interface for endpoint probing and proof construction (/agents, /ping)
    - Do not use any mock, fallback or virtual mode
    - Provide consistent external method signature with AgoraRegistrationAdapter
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
        """Register ACP Agent to RG

        Requirements: endpoint must be native ACP server root address (can directly access /agents and /ping)
        """
        # Probe ACP endpoint to get available agent names
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

        # Select an ACP agent name for proof (prefer same name, otherwise first one)
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
        """Subscribe Observer to RG (independent of ACP server)"""
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
        """Extract agent name list from /agents response (compatible with acp_sdk model structure)"""
        try:
            # acp_sdk /agents returns format like {"agents":[{"name": "...", ...}, ...]}
            agents = agents_payload.get("agents", []) if isinstance(agents_payload, dict) else []
            names = []
            for item in agents:
                name = item.get("name")
                if isinstance(name, str) and name:
                    names.append(name)
            return names
        except Exception:
            return []


