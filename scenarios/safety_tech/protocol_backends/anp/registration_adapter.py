# -*- coding: utf-8 -*-
"""
ANP Protocol Registration Adapter
Use native ANP server endpoints for registration (no mocks, no fallbacks).
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, Any, Optional

import httpx


class ANPRegistrationAdapter:
    """ANP protocol registration adapter (works with RegistrationGateway)
    
    Design goals:
    - Use native ANP service endpoints for endpoint probing and proof construction (/health, /registration_proof)
    - Do not use any mocks, fallbacks, or simulation modes
    - Provide consistent external method signatures with other protocol adapters
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
        """Register an ANP agent to the RG.
        
        Requirements: endpoint must be the native ANP server base URL (must allow direct access to /health and /registration_proof).
        """
        # Probe ANP endpoint health
        base = endpoint.rstrip("/")
        async with httpx.AsyncClient() as client:
            health_resp = await client.get(f"{base}/health", timeout=10.0)
            if health_resp.status_code != 200:
                raise RuntimeError(f"ANP /health probe failed: {health_resp.status_code}")
            
            # Get native ANP DID proof
            proof_resp = await client.get(f"{base}/registration_proof", timeout=10.0)
            if proof_resp.status_code != 200:
                raise RuntimeError(f"ANP /registration_proof failed: {proof_resp.status_code}")
            
            native_proof = proof_resp.json()
            
        # Build a complete ANP registration proof
        proof = {
            "timestamp": time.time(),
            "nonce": str(uuid.uuid4()),
            "endpoint_health": True,
            **native_proof  # Include DID, signature, and other native fields
        }

        registration_request = {
            "protocol": "anp",
            "agent_id": agent_id,
            "endpoint": endpoint,
            "conversation_id": conversation_id,
            "role": role,
            "protocolMeta": {
                "protocol_version": "1.0",
                "capabilities": ["message", "health", "did"],
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
        """Subscribe an observer to the RG (independent of the ANP server)."""
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
