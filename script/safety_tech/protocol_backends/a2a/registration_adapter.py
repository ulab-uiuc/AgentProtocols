# -*- coding: utf-8 -*-
"""
A2A Protocol Registration Adapter
A2A协议注册适配器 - 处理A2A协议特定的身份验证和注册逻辑，全方面使用A2A原生SDK
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

# A2A 原生 SDK 导入 (必需依赖)
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCapabilities, AgentCard, AgentSkill, AgentProvider,
    Message, MessagePart, TextPart, Role
)
from a2a.utils import new_agent_text_message, compute_hash
from a2a.client import Client as A2AClient


class A2ARegistrationAdapter:
    """A2A协议注册适配器 - 全方面使用A2A原生SDK"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.a2a_config = config.get('a2a', {})
        self.rg_endpoint = config.get('rg_endpoint', 'http://127.0.0.1:8001')
        
        # A2A协议特性配置
        self.agent_capabilities = self.a2a_config.get('capabilities', {})
        self.privacy_features = self.a2a_config.get('privacy_features', {})
        self.agent_features = self.a2a_config.get('agent_features', {})
        
        # A2A原生组件
        self.task_store = InMemoryTaskStore()
        self.request_handler = DefaultRequestHandler()
        self.a2a_client = None
        
        # 内部状态
        self.session_token = None
        self.agent_id = None
        self.conversation_id = None
        self.agent_card = None
        
    async def register_agent(self, agent_id: str, endpoint: str, conversation_id: str, 
                           role: str = "doctor") -> Dict[str, Any]:
        """注册A2A协议Agent"""
        self.agent_id = agent_id
        self.conversation_id = conversation_id
        
        # 创建A2A Agent Card
        self.agent_card = await self._create_agent_card(agent_id, role)
        
        # 生成A2A协议证明
        proof = await self._generate_a2a_proof(agent_id, role)
        
        # 生成协议元数据
        protocol_meta = self._generate_protocol_meta()
        
        # 构建注册请求
        registration_request = {
            "protocol": "a2a",
            "agent_id": agent_id,
            "endpoint": endpoint,
            "conversation_id": conversation_id,
            "role": role,
            "protocolMeta": protocol_meta,
            "proof": proof
        }
        
        # 发送注册请求到RG
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.rg_endpoint}/register",
                json=registration_request,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"A2A Registration failed: {response.status_code} - {response.text}")
            
            result = response.json()
            self.session_token = result.get('session_token')
            
            return result
    
    async def subscribe_observer(self, observer_id: str, conversation_id: str, 
                               endpoint: str = "") -> Dict[str, Any]:
        """订阅Observer角色"""
        # Observer使用简化的A2A证明
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
                raise Exception(f"A2A Observer subscription failed: {response.status_code} - {response.text}")
            
            return response.json()
    
    async def get_conversation_directory(self, conversation_id: str) -> Dict[str, Any]:
        """获取会话目录"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.rg_endpoint}/directory",
                params={"conversation_id": conversation_id},
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"A2A Directory query failed: {response.status_code} - {response.text}")
            
            return response.json()
    
    async def _create_agent_card(self, agent_id: str, role: str) -> AgentCard:
        """创建A2A Agent Card"""
        # 定义Agent技能
        skills = []
        if role == "doctor_a":
            skills = [
                AgentSkill(
                    name="medical_consultation",
                    description="Primary care medical consultation and diagnosis"
                ),
                AgentSkill(
                    name="patient_privacy_protection", 
                    description="Protect patient sensitive information during consultations"
                )
            ]
        elif role == "doctor_b":
            skills = [
                AgentSkill(
                    name="specialist_consultation",
                    description="Specialist medical consultation and expert opinion"
                ),
                AgentSkill(
                    name="clinical_analysis",
                    description="Advanced clinical case analysis and recommendations"
                )
            ]
        elif role == "observer":
            skills = [
                AgentSkill(
                    name="conversation_monitoring",
                    description="Monitor medical conversations for compliance"
                )
            ]
        
        # 定义Agent能力
        capabilities = AgentCapabilities(
            text_generation=True,
            structured_output=self.agent_features.get('enable_structured_responses', False),
            tool_use=self.agent_features.get('enable_tool_use', False),
            multimodal=False
        )
        
        # 创建Agent Card
        agent_card = AgentCard(
            agent_id=agent_id,
            name=f"A2A_{role.title()}_{agent_id}",
            description=f"A2A Protocol {role} agent for medical consultation privacy testing",
            provider=AgentProvider(
                name="safety_tech_framework",
                version="1.0.0"
            ),
            capabilities=capabilities,
            skills=skills
        )
        
        return agent_card
    
    async def _generate_a2a_proof(self, agent_id: str, role: str) -> Dict[str, Any]:
        """生成A2A协议证明 - 使用原生SDK组件"""
        proof = {}
        
        # 1. A2A Agent Card 哈希验证
        if self.agent_card:
            card_dict = {
                "agent_id": self.agent_card.agent_id,
                "name": self.agent_card.name,
                "description": self.agent_card.description,
                "capabilities": {
                    "text_generation": self.agent_card.capabilities.text_generation,
                    "structured_output": self.agent_card.capabilities.structured_output,
                    "tool_use": self.agent_card.capabilities.tool_use,
                    "multimodal": self.agent_card.capabilities.multimodal
                },
                "skills": [{"name": skill.name, "description": skill.description} 
                          for skill in self.agent_card.skills]
            }
            card_json = json.dumps(card_dict, sort_keys=True)
            card_hash = compute_hash(card_json)
            proof['agent_card_hash'] = card_hash
            proof['agent_card_data'] = card_dict
        
        # 2. A2A Task Store 状态证明
        task_store_state = {
            "store_type": "InMemoryTaskStore",
            "initialized": True,
            "agent_id": agent_id,
            "timestamp": time.time()
        }
        task_store_json = json.dumps(task_store_state, sort_keys=True)
        proof['task_store_hash'] = compute_hash(task_store_json)
        proof['task_store_state'] = task_store_state
        
        # 3. A2A Request Handler 签名
        handler_signature = await self._generate_handler_signature(agent_id, role)
        proof['request_handler_signature'] = handler_signature
        
        # 4. A2A Message 格式验证证明
        sample_message = Message(
            message_id=f"proof_{int(time.time())}",
            role=Role.USER,
            parts=[TextPart(text=f"A2A proof message for {agent_id}")]
        )
        message_dict = {
            "message_id": sample_message.message_id,
            "role": sample_message.role.value,
            "parts": [{"type": "text", "text": part.text} for part in sample_message.parts if hasattr(part, 'text')]
        }
        message_json = json.dumps(message_dict, sort_keys=True)
        proof['message_format_hash'] = compute_hash(message_json)
        proof['sample_message'] = message_dict
        
        # 5. 时间戳和nonce（防重放）
        proof['timestamp'] = time.time()
        proof['nonce'] = str(uuid.uuid4())
        
        # 6. A2A协议版本证明
        proof['a2a_protocol_version'] = "1.0"
        proof['sdk_components'] = {
            "apps": "A2AStarletteApplication",
            "request_handlers": "DefaultRequestHandler", 
            "tasks": "InMemoryTaskStore",
            "agent_execution": "AgentExecutor",
            "events": "EventQueue"
        }
        
        # 7. Agent身份签名
        agent_signature = await self._generate_agent_signature(agent_id, role, proof['timestamp'])
        proof['agent_signature'] = agent_signature
        
        return proof
    
    async def _generate_observer_proof(self, observer_id: str) -> Dict[str, Any]:
        """生成Observer证明（简化版A2A证明）"""
        proof = {
            'observer_id': observer_id,
            'timestamp': time.time(),
            'nonce': str(uuid.uuid4()),
            'observer_type': 'a2a_passive_listener',
            'a2a_protocol_version': '1.0'
        }
        
        # 如果配置要求Observer证明，生成A2A组件签名
        if self.privacy_features.get('enable_observer_authentication', False):
            observer_signature = await self._generate_observer_signature(observer_id, proof['timestamp'])
            proof['observer_signature'] = observer_signature
            
            # 简化的A2A组件证明
            proof['a2a_components'] = {
                "task_store": "InMemoryTaskStore",
                "client": "A2AClient"
            }
        
        return proof
    
    async def _generate_handler_signature(self, agent_id: str, role: str) -> str:
        """生成A2A Request Handler签名"""
        handler_data = {
            'agent_id': agent_id,
            'role': role,
            'handler_type': 'DefaultRequestHandler',
            'capabilities': {
                'async_execution': True,
                'event_streaming': True,
                'context_management': True
            },
            'timestamp': time.time()
        }
        
        # 生成签名哈希
        signature_string = json.dumps(handler_data, sort_keys=True)
        signature_hash = hashlib.sha256(signature_string.encode()).hexdigest()
        
        # A2A Handler签名格式
        return f"a2a_handler_{signature_hash[:16]}"
    
    async def _generate_agent_signature(self, agent_id: str, role: str, timestamp: float) -> str:
        """生成A2A Agent身份签名"""
        signature_data = f"{agent_id}:{role}:{timestamp}:a2a"
        return hashlib.sha256(signature_data.encode()).hexdigest()[:32]
    
    async def _generate_observer_signature(self, observer_id: str, timestamp: float) -> str:
        """生成A2A Observer签名"""
        signature_data = f"{observer_id}:observer:{timestamp}:a2a"
        return hashlib.sha256(signature_data.encode()).hexdigest()[:24]
    
    def _generate_protocol_meta(self) -> Dict[str, Any]:
        """生成A2A协议元数据"""
        return {
            'protocol_version': '1.0',
            'sdk_version': 'latest',
            'supported_features': {
                'agent_cards': True,
                'task_store': True,
                'request_handlers': True,
                'event_streaming': True,
                'message_routing': True,
                'privacy_features': {
                    'message_encryption': self.privacy_features.get('enable_message_encryption', False),
                    'identity_verification': self.privacy_features.get('enable_identity_verification', True),
                    'content_filtering': self.privacy_features.get('enable_content_filtering', False)
                }
            },
            'authentication_methods': ['agent_card_hash', 'task_store_hash', 'request_handler_signature', 'agent_signature'],
            'session_management': {
                'token_based': True,
                'timeout_supported': True,
                'renewal_supported': True
            },
            'a2a_native_components': {
                'apps': 'A2AStarletteApplication',
                'request_handlers': 'DefaultRequestHandler',
                'tasks': 'InMemoryTaskStore',
                'agent_execution': 'AgentExecutor',
                'events': 'EventQueue',
                'types': 'AgentCard, AgentCapabilities, Message',
                'utils': 'new_agent_text_message, compute_hash',
                'client': 'A2AClient'
            }
        }
    
    # ==================== 攻击模拟方法 ====================
    
    async def simulate_impersonation_attack(self, target_agent_id: str, conversation_id: str) -> Dict[str, Any]:
        """模拟冒名顶替攻击 - 使用伪造的A2A组件"""
        try:
            # 尝试使用伪造的A2A Agent Card
            fake_card = {
                "agent_id": target_agent_id,
                "name": f"Fake_A2A_{target_agent_id}",
                "description": "Fake A2A agent for impersonation attack",
                "capabilities": {"text_generation": True},
                "skills": [{"name": "fake_skill", "description": "Fake skill"}]
            }
            
            fake_proof = {
                'agent_card_hash': 'fake_hash_123',
                'agent_card_data': fake_card,
                'task_store_hash': 'fake_task_store_hash',
                'request_handler_signature': 'fake_handler_sig',
                'timestamp': time.time(),
                'nonce': str(uuid.uuid4()),
                'a2a_protocol_version': '1.0'
            }
            
            registration_request = {
                "protocol": "a2a",
                "agent_id": target_agent_id,
                "endpoint": "http://fake-a2a-endpoint:8000",
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
        """模拟凭证复用攻击 - 重用A2A会话令牌"""
        try:
            # 使用窃取的A2A令牌尝试注册Observer
            fake_proof = {
                'stolen_a2a_token': stolen_token,
                'timestamp': time.time(),
                'observer_type': 'malicious_a2a_listener',
                'a2a_protocol_version': '1.0',
                'fake_task_store': 'InMemoryTaskStore'
            }
            
            subscription_request = {
                "agent_id": f"malicious_a2a_observer_{int(time.time())}",
                "conversation_id": conversation_id,
                "role": "observer",
                "endpoint": "http://malicious-a2a-endpoint:8000",
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
        """模拟重放攻击 - 重放旧的A2A注册数据"""
        try:
            # 重放旧的A2A注册数据
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
        """模拟端点替换攻击 - 注册恶意A2A端点"""
        try:
            # 注册恶意A2A端点作为Observer
            malicious_proof = await self._generate_observer_proof(f"malicious_a2a_observer_{int(time.time())}")
            
            registration_request = {
                "protocol": "a2a",
                "agent_id": f"legitimate_a2a_observer_{int(time.time())}",
                "endpoint": "http://malicious-a2a-relay:8000",  # 恶意A2A中继端点
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
    
    async def simulate_cross_protocol_attack(self, agora_proof: Dict[str, Any], conversation_id: str) -> Dict[str, Any]:
        """模拟跨协议攻击 - 使用Agora证明尝试注册A2A协议"""
        try:
            # 使用Agora的证明尝试注册A2A协议
            cross_protocol_request = {
                "protocol": "a2a",
                "agent_id": f"cross_protocol_a2a_agent_{int(time.time())}",
                "endpoint": "http://cross-protocol-a2a-endpoint:8000",
                "conversation_id": conversation_id,
                "role": "doctor",
                "protocolMeta": self._generate_protocol_meta(),
                "proof": agora_proof  # 使用Agora证明
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
        """模拟Observer自动准入攻击 - 尝试无A2A证明注册Observer"""
        try:
            # 尝试无A2A证明注册Observer
            no_proof_request = {
                "agent_id": f"unauthorized_a2a_observer_{int(time.time())}",
                "conversation_id": conversation_id,
                "role": "observer",
                "endpoint": "http://unauthorized-a2a-endpoint:8000",
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
    
    # ==================== A2A原生SDK工具方法 ====================
    
    async def create_a2a_application(self, agent_id: str, executor: AgentExecutor) -> A2AStarletteApplication:
        """创建A2A Starlette应用"""
        app = A2AStarletteApplication(
            agent_id=agent_id,
            executor=executor,
            task_store=self.task_store,
            request_handler=self.request_handler
        )
        return app
    
    async def initialize_a2a_client(self, base_url: str) -> A2AClient:
        """初始化A2A客户端"""
        if not self.a2a_client:
            self.a2a_client = A2AClient(base_url=base_url)
        return self.a2a_client
    
    async def send_a2a_message(self, target_agent_id: str, message_text: str) -> Dict[str, Any]:
        """使用A2A客户端发送消息"""
        if not self.a2a_client:
            raise RuntimeError("A2A client not initialized. Call initialize_a2a_client() first.")
        
        message = Message(
            message_id=f"msg_{int(time.time())}",
            role=Role.USER,
            parts=[TextPart(text=message_text)]
        )
        
        try:
            response = await self.a2a_client.send_message(target_agent_id, message)
            return {"success": True, "response": response}
        except Exception as e:
            return {"success": False, "error": str(e)}

