# -*- coding: utf-8 -*-
"""
Agora Protocol Registration Adapter
Agora协议注册适配器 - 处理Agora协议特定的身份验证和注册逻辑
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
    """Agora协议注册适配器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agora_config = config.get('agora', {})
        self.rg_endpoint = config.get('rg_endpoint', 'http://127.0.0.1:8001')
        
        # Agora协议特性配置
        self.toolformer_config = self.agora_config.get('toolformer', {})
        self.privacy_features = self.agora_config.get('privacy_features', {})
        self.agent_features = self.agora_config.get('agent_features', {})
        
        # 内部状态
        self.session_token = None
        self.agent_id = None
        self.conversation_id = None
        
    async def register_agent(self, agent_id: str, endpoint: str, conversation_id: str, 
                           role: str = "doctor") -> Dict[str, Any]:
        """注册Agora协议Agent"""
        self.agent_id = agent_id
        self.conversation_id = conversation_id
        
        # 生成Agora协议证明
        proof = await self._generate_agora_proof(agent_id, role)
        
        # 生成协议元数据
        protocol_meta = self._generate_protocol_meta()
        
        # 构建注册请求
        registration_request = {
            "protocol": "agora",
            "agent_id": agent_id,
            "endpoint": endpoint,
            "conversation_id": conversation_id,
            "role": role,
            "protocolMeta": protocol_meta,
            "proof": proof
        }
        
        # Send注册请求到RG
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
        """订阅Observer角色"""
        # Observer使用简化的证明
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
        """Get会话目录"""
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
        """GenerateAgora协议证明"""
        proof = {}
        
        # 1. Toolformer模型签名
        if self.toolformer_config:
            toolformer_signature = await self._generate_toolformer_signature(agent_id, role)
            proof['toolformer_signature'] = toolformer_signature
        
        # 2. LangChain集成证明
        if self.agent_features.get('enable_langchain_integration', False):
            langchain_proof = await self._generate_langchain_proof(agent_id)
            proof['langchain_proof'] = langchain_proof
        
        # 3. 协议哈希验证（使用Agora原生hash + data URI 作为来源）
        # 生成一个最小的医疗会诊协议文档，计算hash并提供来源
        protocol_document = """---\nname: medical-consultation\ndescription: Two-doctor medical consultation focusing on clinical analysis.\nmultiround: true\n---\n# Roles\n- doctor_a\n- doctor_b\n# Rules\n- Discuss clinical facts only.\n- No credentials/tokens/config sharing.\n"""
        protocol_hash_value = compute_hash(protocol_document)
        protocol_source = encode_as_data_uri(protocol_document)
        proof['protocol_hash'] = protocol_hash_value
        proof['protocol_sources'] = [protocol_source]
        
        # 4. 时间戳和nonce（防重放）
        proof['timestamp'] = time.time()
        proof['nonce'] = str(uuid.uuid4())
        
        # 5. Agent身份签名
        agent_signature = await self._generate_agent_signature(agent_id, role, proof['timestamp'])
        proof['agent_signature'] = agent_signature
        
        return proof
    
    async def _generate_observer_proof(self, observer_id: str) -> Dict[str, Any]:
        """GenerateObserver证明（简化版）"""
        proof = {
            'observer_id': observer_id,
            'timestamp': time.time(),
            'nonce': str(uuid.uuid4()),
            'observer_type': 'passive_listener'
        }
        
        # If配置要求Observer证明，生成简单签名
        if self.privacy_features.get('enable_observer_authentication', False):
            observer_signature = await self._generate_observer_signature(observer_id, proof['timestamp'])
            proof['observer_signature'] = observer_signature
        
        return proof
    
    async def _generate_toolformer_signature(self, agent_id: str, role: str) -> str:
        """GenerateToolformer模型签名"""
        model_name = self.toolformer_config.get('model', 'gpt-4o-mini')
        temperature = self.toolformer_config.get('temperature', 0.1)
        
        # 构建签名数据
        signature_data = {
            'agent_id': agent_id,
            'role': role,
            'model': model_name,
            'temperature': temperature,
            'timestamp': time.time()
        }
        
        # 生成签名哈希
        signature_string = json.dumps(signature_data, sort_keys=True)
        signature_hash = hashlib.sha256(signature_string.encode()).hexdigest()
        
        # 模拟Toolformer签名格式
        return f"toolformer_{model_name}_{signature_hash[:16]}"
    
    async def _generate_langchain_proof(self, agent_id: str) -> Dict[str, Any]:
        """GenerateLangChain集成证明"""
        chain_id = f"agora_chain_{agent_id}_{int(time.time())}"
        
        langchain_proof = {
            'chain_id': chain_id,
            'chain_type': 'conversation_chain',
            'memory_type': 'buffer_window',
            'tools_enabled': self.privacy_features.get('enable_tool_filtering', False),
            'structured_responses': self.agent_features.get('enable_structured_responses', False)
        }
        
        # 生成链证明哈希
        proof_string = json.dumps(langchain_proof, sort_keys=True)
        proof_hash = hashlib.md5(proof_string.encode()).hexdigest()
        langchain_proof['proof_hash'] = proof_hash
        
        return langchain_proof
    
    async def _generate_protocol_hash(self, agent_id: str, role: str) -> str:
        """Generate协议哈希验证"""
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
        """GenerateAgent身份签名"""
        signature_data = f"{agent_id}:{role}:{timestamp}:agora"
        return hashlib.sha256(signature_data.encode()).hexdigest()[:32]
    
    async def _generate_observer_signature(self, observer_id: str, timestamp: float) -> str:
        """GenerateObserver签名"""
        signature_data = f"{observer_id}:observer:{timestamp}:agora"
        return hashlib.sha256(signature_data.encode()).hexdigest()[:24]
    
    def _generate_protocol_meta(self) -> Dict[str, Any]:
        """Generate协议元数据"""
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
    
    # 攻击模拟方法
    async def simulate_impersonation_attack(self, target_agent_id: str, conversation_id: str) -> Dict[str, Any]:
        """模拟冒名顶替攻击"""
        try:
            # Try使用空或伪造的证明注册
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
        """模拟凭证复用攻击"""
        try:
            # 使用窃取的令牌尝试注册Observer
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
        """模拟重放攻击"""
        try:
            # 重放旧的注册数据
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
        """模拟端点替换攻击"""
        try:
            # 注册恶意端点作为Observer
            malicious_proof = await self._generate_observer_proof(f"malicious_observer_{int(time.time())}")
            
            registration_request = {
                "protocol": "agora",
                "agent_id": f"legitimate_observer_{int(time.time())}",
                "endpoint": "http://malicious-relay:8000",  # 恶意中继端点
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
        """模拟跨协议攻击"""
        try:
            # 使用ANP的DID证明尝试注册Agora协议
            cross_protocol_request = {
                "protocol": "agora",
                "agent_id": f"cross_protocol_agent_{int(time.time())}",
                "endpoint": "http://cross-protocol-endpoint:8000",
                "conversation_id": conversation_id,
                "role": "doctor",
                "protocolMeta": self._generate_protocol_meta(),
                "proof": anp_did_proof  # 使用ANP证明
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
        """模拟Observer自动准入攻击"""
        try:
            # Try无证明注册Observer
            no_proof_request = {
                "agent_id": f"unauthorized_observer_{int(time.time())}",
                "conversation_id": conversation_id,
                "role": "observer",
                "endpoint": "http://unauthorized-endpoint:8000",
                "proof": {}  # 空证明
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
