# -*- coding: utf-8 -*-
"""
Agora Protocol Registration Adapter
Agora protocol registration adapter - handles Agora protocol-specific authentication and registration logic
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from typing import Dict, List, Any, Optional
import httpx
from pathlib import Path
from agora.utils import compute_hash, encode_as_data_uri

class AgoraRegistrationAdapter:
    """Agora protocol registration adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agora_config = config.get('agora', {})
        self.rg_endpoint = config.get('rg_endpoint', 'http://127.0.0.1:8001')
        
        # Agora protocol feature configuration
        self.toolformer_config = self.agora_config.get('toolformer', {})
        self.privacy_features = self.agora_config.get('privacy_features', {})
        self.agent_features = self.agora_config.get('agent_features', {})
        
        # Internal state
        self.session_token = None
        self.agent_id = None
        self.conversation_id = None
        
    async def register_agent(self, agent_id: str, endpoint: str, conversation_id: str, 
                           role: str = "doctor") -> Dict[str, Any]:
        """Register Agora protocol Agent"""
        self.agent_id = agent_id
        self.conversation_id = conversation_id
        
        # Generate Agora protocol proof
        proof = await self._generate_agora_proof(agent_id, role)
        
        # Generate protocol metadata
        protocol_meta = self._generate_protocol_meta()
        
        # Build registration request
        registration_request = {
            "protocol": "agora",
            "agent_id": agent_id,
            "endpoint": endpoint,
            "conversation_id": conversation_id,
            "role": role,
            "protocolMeta": protocol_meta,
            "proof": proof
        }
        
        # Send registration request to RG
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.rg_endpoint}/register",
                json=registration_request,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Registration failed: {response.status_code} - {response.text}")
            
            result = response.json()
            self.session_token = result.get('session_token')
            
            return result
    
    async def subscribe_observer(self, observer_id: str, conversation_id: str, 
                               endpoint: str = "") -> Dict[str, Any]:
        """Subscribe Observer role"""
        # Observer uses simplified proof
        proof = await self._generate_observer_proof(observer_id)
        
        subscription_request = {
            "agent_id": observer_id,
            "conversation_id": conversation_id,
            "role": "observer",
            "endpoint": endpoint,
            "proof": proof
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.rg_endpoint}/subscribe",
                json=subscription_request,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Observer subscription failed: {response.status_code} - {response.text}")
            
            return response.json()
    
    async def get_conversation_directory(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation directory"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.rg_endpoint}/directory",
                params={"conversation_id": conversation_id},
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Directory query failed: {response.status_code} - {response.text}")
            
            return response.json()
    
    async def _generate_agora_proof(self, agent_id: str, role: str) -> Dict[str, Any]:
        """Generate Agora protocol proof"""
        proof = {}
        
        # 1. Toolformer model signature
        if self.toolformer_config:
            toolformer_signature = await self._generate_toolformer_signature(agent_id, role)
            proof['toolformer_signature'] = toolformer_signature
        
        # 2. LangChain integration proof
        if self.agent_features.get('enable_langchain_integration', False):
            langchain_proof = await self._generate_langchain_proof(agent_id)
            proof['langchain_proof'] = langchain_proof
        
        # 3. Protocol hash verification (using Agora native hash + data URI as source)
        # Generate a minimal medical consultation protocol document, calculate hash and provide source
        protocol_document = """---\nname: medical-consultation\ndescription: Two-doctor medical consultation focusing on clinical analysis.\nmultiround: true\n---\n# Roles\n- doctor_a\n- doctor_b\n# Rules\n- Discuss clinical facts only.\n- No credentials/tokens/config sharing.\n"""
        protocol_hash_value = compute_hash(protocol_document)
        protocol_source = encode_as_data_uri(protocol_document)
        proof['protocol_hash'] = protocol_hash_value
        proof['protocol_sources'] = [protocol_source]
        
        # 4. Timestamp and nonce (anti-replay)
        proof['timestamp'] = time.time()
        proof['nonce'] = str(uuid.uuid4())
        
        # 5. Agent identity signature
        agent_signature = await self._generate_agent_signature(agent_id, role, proof['timestamp'])
        proof['agent_signature'] = agent_signature
        
        return proof
    
    async def _generate_observer_proof(self, observer_id: str) -> Dict[str, Any]:
        """Generate Observer proof (simplified version)"""
        proof = {
            'observer_id': observer_id,
            'timestamp': time.time(),
            'nonce': str(uuid.uuid4()),
            'observer_type': 'passive_listener'
        }
        
        # If configuration requires Observer proof, generate simple signature
        if self.privacy_features.get('enable_observer_authentication', False):
            observer_signature = await self._generate_observer_signature(observer_id, proof['timestamp'])
            proof['observer_signature'] = observer_signature
        
        return proof
    
    async def _generate_toolformer_signature(self, agent_id: str, role: str) -> str:
        """Generate Toolformer model signature"""
        model_name = self.toolformer_config.get('model', 'gpt-4o-mini')
        temperature = self.toolformer_config.get('temperature', 0.1)
        
        # Build signature data
        signature_data = {
            'agent_id': agent_id,
            'role': role,
            'model': model_name,
            'temperature': temperature,
            'timestamp': time.time()
        }
        
        # Generate signature hash
        signature_string = json.dumps(signature_data, sort_keys=True)
        signature_hash = hashlib.sha256(signature_string.encode()).hexdigest()
        
        # Simulate Toolformer signature format
        return f"toolformer_{model_name}_{signature_hash[:16]}"
    
    async def _generate_langchain_proof(self, agent_id: str) -> Dict[str, Any]:
        """Generate LangChain integration proof"""
        chain_id = f"agora_chain_{agent_id}_{int(time.time())}"
        
        langchain_proof = {
            'chain_id': chain_id,
            'chain_type': 'conversation_chain',
            'memory_type': 'buffer_window',
            'tools_enabled': self.privacy_features.get('enable_tool_filtering', False),
            'structured_responses': self.agent_features.get('enable_structured_responses', False)
        }
        
        # Generate chain proof hash
        proof_string = json.dumps(langchain_proof, sort_keys=True)
        proof_hash = hashlib.md5(proof_string.encode()).hexdigest()
        langchain_proof['proof_hash'] = proof_hash
        
        return langchain_proof
    
    async def _generate_protocol_hash(self, agent_id: str, role: str) -> str:
        """Generate protocol hash verification"""
        protocol_data = {
            'protocol': 'agora',
            'version': '1.0',
            'agent_id': agent_id,
            'role': role,
            'features': {
                'tool_filtering': self.privacy_features.get('enable_tool_filtering', False),
                'content_sanitization': self.privacy_features.get('enable_content_sanitization', False),
                'protocol_validation': self.privacy_features.get('enable_protocol_validation', False)
            }
        }
        
        protocol_string = json.dumps(protocol_data, sort_keys=True)
        return hashlib.sha256(protocol_string.encode()).hexdigest()
    
    async def _generate_agent_signature(self, agent_id: str, role: str, timestamp: float) -> str:
        """Generate Agent identity signature"""
        signature_data = f"{agent_id}:{role}:{timestamp}:agora"
        return hashlib.sha256(signature_data.encode()).hexdigest()[:32]
    
    async def _generate_observer_signature(self, observer_id: str, timestamp: float) -> str:
        """Generate Observer signature"""
        signature_data = f"{observer_id}:observer:{timestamp}:agora"
        return hashlib.sha256(signature_data.encode()).hexdigest()[:24]
    
    def _generate_protocol_meta(self) -> Dict[str, Any]:
        """Generate protocol metadata"""
        return {
            'protocol_version': '1.0',
            'supported_features': {
                'toolformer_integration': bool(self.toolformer_config),
                'langchain_integration': self.agent_features.get('enable_langchain_integration', False),
                'structured_responses': self.agent_features.get('enable_structured_responses', False),
                'privacy_features': {
                    'tool_filtering': self.privacy_features.get('enable_tool_filtering', False),
                    'content_sanitization': self.privacy_features.get('enable_content_sanitization', False),
                    'protocol_validation': self.privacy_features.get('enable_protocol_validation', False)
                }
            },
            'authentication_methods': ['toolformer_signature', 'langchain_proof', 'agent_signature'],
            'session_management': {
                'token_based': True,
                'timeout_supported': True,
                'renewal_supported': False
            }
        }
    
    # Attack simulation methods
    async def simulate_impersonation_attack(self, target_agent_id: str, conversation_id: str) -> Dict[str, Any]:
        """Simulate impersonation attack"""
        try:
            # Try to register with empty or forged proof
            fake_proof = {
                'toolformer_signature': 'fake_signature_123',
                'timestamp': time.time(),
                'nonce': str(uuid.uuid4())
            }
            
            registration_request = {
                "protocol": "agora",
                "agent_id": target_agent_id,
                "endpoint": "http://fake-endpoint:8000",
                "conversation_id": conversation_id,
                "role": "doctor_b",
                "protocolMeta": self._generate_protocol_meta(),
                "proof": fake_proof
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rg_endpoint}/register",
                    json=registration_request,
                    timeout=10.0
                )
                
                return {
                    "attack_type": "impersonation",
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "response": response.json() if response.status_code == 200 else response.text
                }
        except Exception as e:
            return {
                "attack_type": "impersonation",
                "success": False,
                "error": str(e)
            }
    
    async def simulate_credential_reuse_attack(self, stolen_token: str, conversation_id: str) -> Dict[str, Any]:
        """Simulate credential reuse attack"""
        try:
            # Try to register Observer using stolen token
            fake_proof = {
                'stolen_token': stolen_token,
                'timestamp': time.time(),
                'observer_type': 'malicious_listener'
            }
            
            subscription_request = {
                "agent_id": f"malicious_observer_{int(time.time())}",
                "conversation_id": conversation_id,
                "role": "observer",
                "endpoint": "http://malicious-endpoint:8000",
                "proof": fake_proof
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rg_endpoint}/subscribe",
                    json=subscription_request,
                    timeout=10.0
                )
                
                return {
                    "attack_type": "credential_reuse",
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "response": response.json() if response.status_code == 200 else response.text
                }
        except Exception as e:
            return {
                "attack_type": "credential_reuse",
                "success": False,
                "error": str(e)
            }
    
    async def simulate_replay_attack(self, old_registration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate replay attack"""
        try:
            # Replay old registration data
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rg_endpoint}/register",
                    json=old_registration_data,
                    timeout=10.0
                )
                
                return {
                    "attack_type": "replay",
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "response": response.json() if response.status_code == 200 else response.text
                }
        except Exception as e:
            return {
                "attack_type": "replay",
                "success": False,
                "error": str(e)
            }
    
    async def simulate_endpoint_substitution_attack(self, conversation_id: str) -> Dict[str, Any]:
        """Simulate endpoint substitution attack"""
        try:
            # Register malicious endpoint as Observer
            malicious_proof = await self._generate_observer_proof(f"malicious_observer_{int(time.time())}")
            
            registration_request = {
                "protocol": "agora",
                "agent_id": f"legitimate_observer_{int(time.time())}",
                "endpoint": "http://malicious-relay:8000",  # Malicious relay endpoint
                "conversation_id": conversation_id,
                "role": "observer",
                "protocolMeta": self._generate_protocol_meta(),
                "proof": malicious_proof
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rg_endpoint}/register",
                    json=registration_request,
                    timeout=10.0
                )
                
                return {
                    "attack_type": "endpoint_substitution",
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "response": response.json() if response.status_code == 200 else response.text
                }
        except Exception as e:
            return {
                "attack_type": "endpoint_substitution",
                "success": False,
                "error": str(e)
            }
    
    async def simulate_cross_protocol_attack(self, anp_did_proof: Dict[str, Any], conversation_id: str) -> Dict[str, Any]:
        """Simulate cross-protocol attack"""
        try:
            # Try to register Agora protocol using ANP's DID proof
            cross_protocol_request = {
                "protocol": "agora",
                "agent_id": f"cross_protocol_agent_{int(time.time())}",
                "endpoint": "http://cross-protocol-endpoint:8000",
                "conversation_id": conversation_id,
                "role": "doctor",
                "protocolMeta": self._generate_protocol_meta(),
                "proof": anp_did_proof  # Use ANP proof
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rg_endpoint}/register",
                    json=cross_protocol_request,
                    timeout=10.0
                )
                
                return {
                    "attack_type": "cross_protocol",
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "response": response.json() if response.status_code == 200 else response.text
                }
        except Exception as e:
            return {
                "attack_type": "cross_protocol",
                "success": False,
                "error": str(e)
            }
    
    async def simulate_observer_auto_admission_attack(self, conversation_id: str) -> Dict[str, Any]:
        """Simulate Observer auto-admission attack"""
        try:
            # Try to register Observer without proof
            no_proof_request = {
                "agent_id": f"unauthorized_observer_{int(time.time())}",
                "conversation_id": conversation_id,
                "role": "observer",
                "endpoint": "http://unauthorized-endpoint:8000",
                "proof": {}  # Empty proof
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rg_endpoint}/subscribe",
                    json=no_proof_request,
                    timeout=10.0
                )
                
                return {
                    "attack_type": "observer_auto_admission",
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "response": response.json() if response.status_code == 200 else response.text
                }
        except Exception as e:
            return {
                "attack_type": "observer_auto_admission",
                "success": False,
                "error": str(e)
            }
