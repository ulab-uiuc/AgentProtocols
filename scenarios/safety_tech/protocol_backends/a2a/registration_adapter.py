# -*- coding: utf-8 -*-
"""
A2A Protocol Registration Adapter
Uses native A2A server interface for registration coordination (no mock / no fallback)
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, Any, Optional

import httpx


class A2ARegistrationAdapter:
    """A2A protocol registration adapter (works with RegistrationGateway)
    
    Design goals:
    - Use native A2A service interface for endpoint probing and proof construction (/health)
    - Do not use any mock, fallback or virtual mode
    - Provide consistent external method signature with other protocol adapters
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
        """Register A2A Agent to RG
        
        Requirements: endpoint must be native A2A server root address (can directly access /health)
        """
        # Probe A2A endpoint health status
        base = endpoint.rstrip("/")
        async with httpx.AsyncClient() as client:
            health_resp = await client.get(f"{base}/health", timeout=10.0)
            if health_resp.status_code != 200:
                raise RuntimeError(f"A2A /health probe failed: {health_resp.status_code}")
            
        # Construct A2A native proof
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
        """Subscribe Observer to RG (independent of A2A server)"""
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
