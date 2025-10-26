# -*- coding: utf-8 -*-
"""
Registration Gateway (RG) - Core registration service for multi-agent systems.

Unified agent registration and directory service with multi-protocol admission control and authentication.
Provides REST API endpoints for agent registration, subscription, and directory queries.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from pathlib import Path
import logging
import httpx

# Import ANP signature verification tools (must be available, otherwise raise error)
try:
    from agent_connect.utils.crypto_tool import (
        get_public_key_from_hex,
        verify_signature_for_json,
    )
except Exception as e:
    raise RuntimeError(f"Failed to import ANP crypto tools: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass 
class RegistrationRecord:
    """Registration record data structure"""
    agent_id: str
    protocol: str
    endpoint: str
    role: str  # doctor_a, doctor_b, observer
    protocol_meta: Dict[str, Any]
    proof: Dict[str, Any]
    conversation_id: str
    timestamp: float
    verified: bool = False
    session_token: Optional[str] = None


@dataclass
class ConversationSession:
    """Session information data structure"""
    conversation_id: str
    participants: List[RegistrationRecord]
    created_at: float
    last_activity: float
    status: str = "active"  # active, completed, expired


class RegistrationGateway:
    """Registration Gateway core class"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.app = FastAPI(title="Registration Gateway", version="1.0.0")
        self.setup_routes()
        
        # Storage structures
        self.registrations: Dict[str, RegistrationRecord] = {}  # agent_id -> record
        self.conversations: Dict[str, ConversationSession] = {}  # conversation_id -> session
        self.protocol_verifiers: Dict[str, callable] = {}
        self.used_nonces: Set[str] = set()
        # Validate mode: transparent | native_delegated | strict
        # Default uses native delegation (relies only on protocol-native capabilities, no RG fallback)
        self.verification_mode: str = str(self.config.get('verification_mode', 'native_delegated')).lower()
        # ANP optional DID document probe (baseline mode records only, strict mode enforces)
        self.anp_probe_did_doc: bool = bool(self.config.get('anp_probe_did_doc', False))
        # Replay/time window observation metrics (record only, no blocking)
        self.metrics = {
            'nonce_reuse_count': 0,
            'timestamp_expired_count': 0,
        }
        
        # Configuration parameters
        self.session_timeout = self.config.get('session_timeout', 3600)  # 1 hour
        self.max_observers_per_session = self.config.get('max_observers', 5)
        self.require_proof_for_observers = self.config.get('require_observer_proof', True)
        
        # Register protocol verifiers
        self._setup_protocol_verifiers()
        
        logger.info("Registration Gateway initialized")

    def _setup_protocol_verifiers(self):
        """Setup protocol verifiers"""
        self.protocol_verifiers = {
            'agora': self._verify_agora,
            'a2a': self._verify_a2a,
            'acp': self._verify_acp,
            'anp': self._verify_anp,
            'direct': self._verify_direct
        }
        

    def setup_routes(self):
        """Setup REST API routes"""
        
        @self.app.post("/register")
        async def register_agent(request: Dict[str, Any]):
            """Agent registration endpoint"""
            try:
                return await self._handle_register(request)
            except Exception as e:
                logger.error(f"Registration failed: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        # Observer subscription endpoint removed - new S2 design does not require Observer mechanism
        
        @self.app.get("/directory")
        async def get_directory(conversation_id: str):
            """Get session directory"""
            try:
                return await self._handle_directory(conversation_id)
            except Exception as e:
                logger.error(f"Directory query failed: {e}")
                raise HTTPException(status_code=404, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check"""
            return {"status": "healthy", "timestamp": time.time(), "verification_mode": self.verification_mode, "metrics": self.metrics}
        
        @self.app.post("/cleanup")
        async def cleanup_sessions(background_tasks: BackgroundTasks):
            """Clean up expired sessions"""
            background_tasks.add_task(self._cleanup_expired_sessions)
            return {"message": "Cleanup task scheduled"}

    async def _handle_register(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent registration request"""
        # Validate required fields
        required_fields = ['protocol', 'agent_id', 'endpoint', 'conversation_id']
        for field in required_fields:
            if field not in request:
                raise ValueError(f"Missing required field: {field}")
        
        protocol = request['protocol']
        agent_id = request['agent_id']
        endpoint = request['endpoint']
        conversation_id = request['conversation_id']
        role = request.get('role', 'doctor')  # Default role is doctor
        protocol_meta = request.get('protocolMeta', {})
        proof = request.get('proof', {})
        
        # Check protocol support
        if protocol not in self.protocol_verifiers:
            raise ValueError(f"Unsupported protocol: {protocol}")
        
        # Create registration record
        record = RegistrationRecord(
            agent_id=agent_id,
            protocol=protocol,
            endpoint=endpoint,
            role=role,
            protocol_meta=protocol_meta,
            proof=proof,
            conversation_id=conversation_id,
            timestamp=time.time()
        )
        
        # Protocol verification (record latency and attribution)
        _verify_start = time.time()
        try:
            verification_result = await self.protocol_verifiers[protocol](record)
        except Exception as e:
            _latency_ms = int((time.time() - _verify_start) * 1000)
            # Record attribution reason
            try:
                reason = str(e)
            except Exception:
                reason = "verification_exception"
            logger.error(f"Protocol verification error [{protocol}] blocked_by=protocol latency_ms={_latency_ms} reason={reason}")
            raise
        _latency_ms = int((time.time() - _verify_start) * 1000)
        record.verified = verification_result.get('verified', False)
        record.session_token = verification_result.get('session_token')
        
        if not record.verified:
            raise ValueError(f"Protocol verification failed: {verification_result.get('error', 'Unknown error')}")
        
        # Check session constraints
        await self._validate_session_constraints(record)
        
        # Store registration record
        self.registrations[agent_id] = record
        
        # Update session information
        await self._update_conversation_session(record)
        
        logger.info(f"Agent {agent_id} registered successfully for protocol {protocol}")
        
        return {
            "status": "registered",
            "agent_id": agent_id,
            "conversation_id": conversation_id,
            "session_token": record.session_token,
            "timestamp": record.timestamp,
            "verified": record.verified,
            "verification_method": verification_result.get('verification_method', 'unknown'),
            "verification_latency_ms": _latency_ms,
            "blocked_by": "none",
            "reason": None
        }

    async def _handle_subscribe(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle observer subscription request"""
        agent_id = request.get('agent_id')
        conversation_id = request.get('conversation_id')
        role = request.get('role', 'observer')
        proof = request.get('proof', {})
        
        if not agent_id or not conversation_id:
            raise ValueError("Missing agent_id or conversation_id")
        
        if role != 'observer':
            raise ValueError("Subscribe endpoint only supports observer role")
        
        # Check observer proof requirements
        if self.require_proof_for_observers and not proof:
            raise ValueError("Proof required for observer subscription")
        
        # Check if session exists
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        session = self.conversations[conversation_id]
        
        # Check observer count limit
        current_observers = len([p for p in session.participants if p.role == 'observer'])
        if current_observers >= self.max_observers_per_session:
            raise ValueError(f"Maximum observers ({self.max_observers_per_session}) reached for conversation")
        
        # Create observer registration record
        observer_record = RegistrationRecord(
            agent_id=agent_id,
            protocol='observer',  # Special protocol identifier
            endpoint=request.get('endpoint', ''),
            role='observer',
            protocol_meta={},
            proof=proof,
            conversation_id=conversation_id,
            timestamp=time.time(),
            verified=True  # Observers pass verification by default
        )
        
        # Store record
        self.registrations[agent_id] = observer_record
        session.participants.append(observer_record)
        session.last_activity = time.time()
        
        logger.info(f"Observer {agent_id} subscribed to conversation {conversation_id}")
        
        return {
            "status": "subscribed",
            "agent_id": agent_id,
            "conversation_id": conversation_id,
            "role": "observer",
            "timestamp": observer_record.timestamp
        }

    async def _handle_directory(self, conversation_id: str) -> Dict[str, Any]:
        """Handle directory query request"""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        session = self.conversations[conversation_id]
        
        # Build participant directory
        participants = []
        for record in session.participants:
            participant_info = {
                "agent_id": record.agent_id,
                "protocol": record.protocol,
                "endpoint": record.endpoint,
                "role": record.role,
                "verified": record.verified,
                "joined_at": record.timestamp
            }
            participants.append(participant_info)
        
        return {
            "conversation_id": conversation_id,
            "participants": participants,
            "total_participants": len(participants),
            "doctors": len([p for p in participants if p["role"] in ["doctor_a", "doctor_b", "doctor"]]),
            "observers": len([p for p in participants if p["role"] == "observer"]),
            "session_status": session.status,
            "created_at": session.created_at,
            "last_activity": session.last_activity
        }

    async def _validate_session_constraints(self, record: RegistrationRecord):
        """Validate session constraints"""
        conversation_id = record.conversation_id
        
        if conversation_id in self.conversations:
            session = self.conversations[conversation_id]
            
            # Check role conflicts
            for participant in session.participants:
                if participant.role == record.role and record.role in ["doctor_a", "doctor_b"]:
                    raise ValueError(f"Role {record.role} already taken in conversation {conversation_id}")

    async def _update_conversation_session(self, record: RegistrationRecord):
        """Update session information"""
        conversation_id = record.conversation_id
        
        if conversation_id not in self.conversations:
            # Create new session
            session = ConversationSession(
                conversation_id=conversation_id,
                participants=[record],
                created_at=time.time(),
                last_activity=time.time()
            )
            self.conversations[conversation_id] = session
        else:
            # Update existing session
            session = self.conversations[conversation_id]
            session.participants.append(record)
            session.last_activity = time.time()

    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for conversation_id, session in self.conversations.items():
            if current_time - session.last_activity > self.session_timeout:
                expired_sessions.append(conversation_id)
        
        for conversation_id in expired_sessions:
            del self.conversations[conversation_id]
            # Cleanup related registration records
            expired_agents = [aid for aid, record in self.registrations.items() 
                            if record.conversation_id == conversation_id]
            for agent_id in expired_agents:
                del self.registrations[agent_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


    # Protocol verifier implementations
    async def _verify_agora(self, record: RegistrationRecord) -> Dict[str, Any]:
        """Verify Agora protocol (baseline mode: only validate protocol_hash and protocol_sources; no RG-level anti-replay blocking)"""
        try:
            from agora.utils import download_and_verify_protocol
        except Exception as e:
            raise RuntimeError(f"Failed to import Agora native utils: {e}")

        proof = record.proof or {}

        # Protocol strict binding: only accept when record.protocol == 'agora'
        if record.protocol != 'agora':
            return {"verified": False, "error": "Protocol mismatch for Agora verification"}

        # Validate required fields (hash and sources only)
        required = ['protocol_hash', 'protocol_sources']
        for f in required:
            if f not in proof:
                return {"verified": False, "error": f"Missing required Agora proof field: {f}"}

        protocol_hash = proof.get('protocol_hash')
        sources = proof.get('protocol_sources') or []
        if not isinstance(sources, list) or len(sources) == 0:
            return {"verified": False, "error": "protocol_sources must be a non-empty list"}

        # Use Agora native tools to verify protocol hash and sources
        verified_source_found = False
        for src in sources:
            try:
                text = download_and_verify_protocol(protocol_hash, src)
                if text is not None:
                    verified_source_found = True
                    break
            except Exception:
                continue

        if not verified_source_found:
            return {"verified": False, "error": "Protocol hash verification failed for all sources"}

        # Optional: endpoint ownership proof (controlled by config, default off)
        require_endpoint_proof = bool(self.config.get('agora_require_endpoint_proof', False))
        if require_endpoint_proof:
            ownership = proof.get('endpoint_ownership_proof')
            if not ownership or not isinstance(ownership, str) or len(ownership) < 8:
                return {"verified": False, "error": "Invalid or missing endpoint ownership proof"}

        # Generate session token after passing verification
        session_token = f"agora_{record.agent_id}_{int(time.time())}"
        return {"verified": True, "session_token": session_token, "verification_method": "agora_protocol_hash"}

    async def _verify_a2a(self, record: RegistrationRecord) -> Dict[str, Any]:
        """Verify A2A protocol
        - Minimum requirements: a2a_token+timestamp+nonce (always enabled)
        - Optional: SDK native challenge-echo probe (a2a_enable_challenge, default False, to maintain fairness)
        """
        proof = record.proof or {}
        required = ['a2a_token', 'timestamp', 'nonce']
        for f in required:
            if f not in proof:
                return {"verified": False, "error": f"Missing required A2A proof field: {f}"}

        # Time window and nonce
        now = time.time()
        try:
            ts = float(proof.get('timestamp', 0))
        except Exception:
            return {"verified": False, "error": "Invalid A2A proof timestamp"}
        if abs(now - ts) > 300:
            # Record only
            self.metrics['timestamp_expired_count'] = self.metrics.get('timestamp_expired_count', 0) + 1
            return {"verified": False, "error": "A2A proof timestamp expired"}

        nonce = str(proof.get('nonce', ''))
        if not nonce or nonce in self.used_nonces:
            # Record replay suspicion
            self.metrics['nonce_reuse_count'] = self.metrics.get('nonce_reuse_count', 0) + 1
            return {"verified": False, "error": "Replay detected: nonce reused or missing"}
        self.used_nonces.add(nonce)

        # Optional: SDK native challenge-echo probe
        if bool(self.config.get('a2a_enable_challenge', False)):
            base = (record.endpoint or '').rstrip('/')
            if not base.startswith('http://') and not base.startswith('https://'):
                return {"verified": False, "error": "A2A endpoint must be http(s) URL for challenge"}
            try:
                async with httpx.AsyncClient() as client:
                    # Send standard A2A message body via /message, requiring server acknowledgment (uniformly wrapped by adapter)
                    payload = {
                        "params": {
                            "message": {
                                "parts": [{"type": "text", "text": f"challenge:{nonce}"}],
                                "messageId": f"chal_{int(time.time()*1000)}",
                                "role": "user"
                            }
                        }
                    }
                    r = await client.post(f"{base}/message", json=payload, timeout=5.0)
                    if r.status_code != 200:
                        return {"verified": False, "error": f"A2A challenge failed: status {r.status_code}"}
                    js = r.json() if r.content else {}
                    # Unified JSONResponse: contains events array; if it contains at least one event, it is considered acknowledgment success
                    if not isinstance(js, dict) or not js.get('events'):
                        return {"verified": False, "error": "A2A challenge no events returned"}
                    verification_method = "a2a_challenge_echo"
            except Exception as e:
                return {"verified": False, "error": f"A2A challenge error: {e}"}
        else:
            verification_method = "a2a_minimal_token"

        session_token = f"a2a_{record.agent_id}_{int(time.time())}"
        return {"verified": True, "session_token": session_token, "verification_method": verification_method}

    async def _verify_acp(self, record: RegistrationRecord) -> Dict[str, Any]:
        """Verify ACP protocol (native verification, no fallback)"""
        proof = record.proof or {}

        # 1) Protocol strict binding
        if record.protocol != 'acp':
            return {"verified": False, "error": "Protocol mismatch for ACP verification"}

        # 2) Required fields and time window/nonce anti-replay
        required = ['timestamp', 'nonce', 'acp_agent_name']
        for f in required:
            if f not in proof:
                return {"verified": False, "error": f"Missing required ACP proof field: {f}"}

        now = time.time()
        try:
            ts = float(proof.get('timestamp', 0))
        except Exception:
            return {"verified": False, "error": "Invalid ACP proof timestamp"}
        if abs(now - ts) > 300:
            return {"verified": False, "error": "ACP proof timestamp expired"}

        nonce = str(proof.get('nonce', ''))
        if not nonce or nonce in self.used_nonces:
            return {"verified": False, "error": "Replay detected: nonce reused or missing"}
        self.used_nonces.add(nonce)

        acp_agent_name = str(proof.get('acp_agent_name', '')).strip()
        if not acp_agent_name:
            return {"verified": False, "error": "acp_agent_name must be non-empty"}

        # 3) Native endpoint probe: /agents and /ping
        endpoint = record.endpoint or ''
        if not endpoint.startswith('http://') and not endpoint.startswith('https://'):
            return {"verified": False, "error": "ACP endpoint must be http(s) URL"}

        base = endpoint.rstrip('/')
        try:
            async with httpx.AsyncClient() as client:
                agents_resp = await client.get(f"{base}/agents", timeout=10.0)
                if agents_resp.status_code != 200:
                    return {"verified": False, "error": f"ACP /agents probe failed: {agents_resp.status_code}"}
                agents_payload = agents_resp.json() if agents_resp.content else {}
                agents = agents_payload.get('agents', []) if isinstance(agents_payload, dict) else []
                names = [a.get('name') for a in agents if isinstance(a, dict) and isinstance(a.get('name'), str)]
                if acp_agent_name not in names:
                    return {"verified": False, "error": f"ACP agent '{acp_agent_name}' not found in /agents"}
                # Enforce name binding: agent_id during registration must match acp_agent_name in /agents
                if record.agent_id != acp_agent_name:
                    return {"verified": False, "error": f"ACP agent name binding mismatch: agent_id='{record.agent_id}' != acp_agent_name='{acp_agent_name}'"}

                ping_resp = await client.get(f"{base}/ping", timeout=10.0)
                if ping_resp.status_code != 200:
                    return {"verified": False, "error": f"ACP /ping probe failed: {ping_resp.status_code}"}
        except Exception as e:
            return {"verified": False, "error": f"ACP endpoint probe error: {e}"}

        # Pass verification, issue session token
        session_token = f"acp_{record.agent_id}_{int(time.time())}"
        return {"verified": True, "session_token": session_token, "verification_method": "acp_native_endpoint_probe"}

    async def _verify_anp(self, record: RegistrationRecord) -> Dict[str, Any]:
        """Verify ANP protocol - use native DID signature verification, no RG fallback"""
        proof = record.proof or {}

        # Transparent mode: record only, do not pass (leave to upper layer decision), still return unverified here
        if self.verification_mode == 'transparent':
            raise ValueError("transparent_mode_no_verification")

        # Required fields
        required_fields = ['did_signature', 'did_public_key', 'timestamp', 'did']
        missing = [f for f in required_fields if f not in proof]
        if missing:
            raise ValueError(f"Missing ANP proof fields: {','.join(missing)}")

        # Time window check (record replay suspicion, but do not block as RG fallback in baseline mode)
        try:
            ts = float(proof.get('timestamp', 0))
        except Exception:
            raise ValueError("Invalid ANP proof timestamp")
        if abs(time.time() - ts) > 300:
            raise ValueError("ANP proof timestamp expired")

        did_value = str(proof.get('did'))
        pub_hex = str(proof.get('did_public_key'))
        signature = str(proof.get('did_signature'))
        try:
            pub_key = get_public_key_from_hex(pub_hex)
        except Exception as e:
            raise ValueError(f"Invalid public key: {e}")

        message = {"did": did_value, "timestamp": ts}
        try:
            ok = verify_signature_for_json(pub_key, message, signature)
        except Exception as e:
            raise ValueError(f"Signature verification error: {e}")

        if not ok:
            raise ValueError("ANP DID signature verification failed")

        # Optional DID document probe (strict mode enforces pass)
        did_doc_probe_ok = None
        if self.anp_probe_did_doc and record.endpoint:
            did_doc_probe_ok = False
            base = record.endpoint.rstrip('/')
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{base}/v1/did/{did_value}", timeout=5.0)
                    if resp.status_code == 200 and isinstance(resp.text, str) and did_value in resp.text:
                        did_doc_probe_ok = True
            except Exception:
                did_doc_probe_ok = False
            if self.verification_mode == 'strict' and not did_doc_probe_ok:
                raise ValueError("ANP DID document probe failed under strict mode")

        # Issue session token after passing
        session_token = f"anp_{record.agent_id}_{int(time.time())}"
        result = {"verified": True, "session_token": session_token, "verification_method": "anp_did_signature"}
        if did_doc_probe_ok is not None:
            result["verification_details"] = {"did_doc_probe_ok": did_doc_probe_ok}
        return result

    async def _verify_direct(self, record: RegistrationRecord) -> Dict[str, Any]:
        """Verify Direct protocol"""
        # Direct protocol has no verification (weakest)
        session_token = f"direct_{record.agent_id}_{int(time.time())}"
        return {"verified": True, "session_token": session_token, "verification_method": "direct_none"}

    def run(self, host: str = "127.0.0.1", port: int = 8000):
        """Run registration gateway service"""
        logger.info(f"Starting Registration Gateway on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="warning", access_log=False, lifespan="off", loop="asyncio", http="h11")


if __name__ == "__main__":
    # Example configuration
    config = {
        "session_timeout": 3600,
        "max_observers": 5,
        "require_observer_proof": True
    }
    
    rg = RegistrationGateway(config)
    rg.run(port=8001)  # Use port 8001 to avoid conflicts with other services
