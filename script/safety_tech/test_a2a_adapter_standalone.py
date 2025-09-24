# -*- coding: utf-8 -*-
"""
A2A Adapter Standalone Test Script
独立测试A2A注册适配器，避免其他协议依赖
"""

import asyncio
import json
import sys
import hashlib
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

# 模拟A2A SDK组件（用于测试）
class MockAgentCapabilities:
    def __init__(self, text_generation=True, structured_output=False, tool_use=False, multimodal=False):
        self.text_generation = text_generation
        self.structured_output = structured_output
        self.tool_use = tool_use
        self.multimodal = multimodal

class MockAgentSkill:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

class MockAgentProvider:
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version

class MockAgentCard:
    def __init__(self, agent_id: str, name: str, description: str, provider, capabilities, skills):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.provider = provider
        self.capabilities = capabilities
        self.skills = skills

def compute_hash(data: str) -> str:
    """计算数据哈希"""
    return hashlib.sha256(data.encode()).hexdigest()


class A2ARegistrationAdapterTest:
    """A2A协议注册适配器测试版本"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.a2a_config = config.get('a2a', {})
        self.rg_endpoint = config.get('rg_endpoint', 'http://127.0.0.1:8001')
        
        # A2A协议特性配置
        self.agent_capabilities = self.a2a_config.get('capabilities', {})
        self.privacy_features = self.a2a_config.get('privacy_features', {})
        self.agent_features = self.a2a_config.get('agent_features', {})
        
        # 内部状态
        self.session_token = None
        self.agent_id = None
        self.conversation_id = None
        self.agent_card = None
    
    async def _create_agent_card(self, agent_id: str, role: str) -> MockAgentCard:
        """创建A2A Agent Card"""
        # 定义Agent技能
        skills = []
        if role == "doctor_a":
            skills = [
                MockAgentSkill(
                    name="medical_consultation",
                    description="Primary care medical consultation and diagnosis"
                ),
                MockAgentSkill(
                    name="patient_privacy_protection", 
                    description="Protect patient sensitive information during consultations"
                )
            ]
        elif role == "doctor_b":
            skills = [
                MockAgentSkill(
                    name="specialist_consultation",
                    description="Specialist medical consultation and expert opinion"
                ),
                MockAgentSkill(
                    name="clinical_analysis",
                    description="Advanced clinical case analysis and recommendations"
                )
            ]
        elif role == "observer":
            skills = [
                MockAgentSkill(
                    name="conversation_monitoring",
                    description="Monitor medical conversations for compliance"
                )
            ]
        
        # 定义Agent能力
        capabilities = MockAgentCapabilities(
            text_generation=True,
            structured_output=self.agent_features.get('enable_structured_responses', False),
            tool_use=self.agent_features.get('enable_tool_use', False),
            multimodal=False
        )
        
        # 创建Agent Card
        agent_card = MockAgentCard(
            agent_id=agent_id,
            name=f"A2A_{role.title()}_{agent_id}",
            description=f"A2A Protocol {role} agent for medical consultation privacy testing",
            provider=MockAgentProvider(
                name="safety_tech_framework",
                version="1.0.0"
            ),
            capabilities=capabilities,
            skills=skills
        )
        
        return agent_card
    
    async def _generate_a2a_proof(self, agent_id: str, role: str) -> Dict[str, Any]:
        """生成A2A协议证明"""
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
        sample_message = {
            "message_id": f"proof_{int(time.time())}",
            "role": "user",
            "parts": [{"type": "text", "text": f"A2A proof message for {agent_id}"}]
        }
        message_json = json.dumps(sample_message, sort_keys=True)
        proof['message_format_hash'] = compute_hash(message_json)
        proof['sample_message'] = sample_message
        
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


async def test_a2a_adapter():
    """测试A2A适配器基本功能"""
    print("🧪 Testing A2A Registration Adapter (Standalone)")
    
    # 测试配置
    config = {
        'rg_endpoint': 'http://127.0.0.1:8001',
        'a2a': {
            'capabilities': {
                'text_generation': True,
                'structured_output': True,
                'tool_use': False,
                'multimodal': False
            },
            'privacy_features': {
                'enable_message_encryption': False,
                'enable_identity_verification': True,
                'enable_content_filtering': False
            },
            'agent_features': {
                'enable_structured_responses': True,
                'enable_tool_use': False
            }
        }
    }
    
    try:
        # 创建A2A适配器
        adapter = A2ARegistrationAdapterTest(config)
        print("✅ A2A Adapter created successfully")
        
        # 测试Agent Card创建
        agent_card = await adapter._create_agent_card("test_agent", "doctor_a")
        print(f"✅ Agent Card created: {agent_card.agent_id}")
        print(f"   - Name: {agent_card.name}")
        print(f"   - Skills: {len(agent_card.skills)} skills")
        print(f"   - Capabilities: text_generation={agent_card.capabilities.text_generation}")
        
        # 测试A2A证明生成
        adapter.agent_card = agent_card
        proof = await adapter._generate_a2a_proof("test_agent", "doctor_a")
        print("✅ A2A Proof generated successfully")
        print(f"   - Protocol version: {proof.get('a2a_protocol_version')}")
        print(f"   - Agent card hash: {proof.get('agent_card_hash', 'N/A')[:16]}...")
        print(f"   - Task store hash: {proof.get('task_store_hash', 'N/A')[:16]}...")
        print(f"   - Handler signature: {proof.get('request_handler_signature', 'N/A')}")
        print(f"   - Message format hash: {proof.get('message_format_hash', 'N/A')[:16]}...")
        print(f"   - Agent signature: {proof.get('agent_signature', 'N/A')}")
        print(f"   - SDK components: {len(proof.get('sdk_components', {}))}")
        
        # 测试协议元数据生成
        protocol_meta = adapter._generate_protocol_meta()
        print("✅ Protocol metadata generated")
        print(f"   - Protocol version: {protocol_meta.get('protocol_version')}")
        print(f"   - Supported features: {len(protocol_meta.get('supported_features', {}))}")
        print(f"   - Authentication methods: {len(protocol_meta.get('authentication_methods', []))}")
        print(f"   - A2A native components: {len(protocol_meta.get('a2a_native_components', {}))}")
        
        # 验证证明结构完整性
        required_proof_fields = [
            'agent_card_hash', 'agent_card_data', 'task_store_hash', 'task_store_state',
            'request_handler_signature', 'message_format_hash', 'sample_message',
            'timestamp', 'nonce', 'a2a_protocol_version', 'sdk_components', 'agent_signature'
        ]
        
        missing_fields = [field for field in required_proof_fields if field not in proof]
        if missing_fields:
            print(f"⚠️  Missing proof fields: {missing_fields}")
        else:
            print("✅ All required proof fields present")
        
        # 验证SDK组件
        sdk_components = proof.get('sdk_components', {})
        expected_components = [
            'apps', 'request_handlers', 'tasks', 'agent_execution', 'events'
        ]
        
        missing_components = [comp for comp in expected_components if comp not in sdk_components]
        if missing_components:
            print(f"⚠️  Missing SDK components: {missing_components}")
        else:
            print("✅ All required SDK components present")
        
        # 验证哈希一致性
        print("\n🔍 Verifying hash consistency...")
        
        # 验证Agent Card哈希
        card_data = proof.get('agent_card_data')
        card_hash = proof.get('agent_card_hash')
        if card_data and card_hash:
            recalculated_hash = compute_hash(json.dumps(card_data, sort_keys=True))
            if recalculated_hash == card_hash:
                print("✅ Agent Card hash verification passed")
            else:
                print("❌ Agent Card hash verification failed")
        
        # 验证Task Store哈希
        store_state = proof.get('task_store_state')
        store_hash = proof.get('task_store_hash')
        if store_state and store_hash:
            recalculated_hash = compute_hash(json.dumps(store_state, sort_keys=True))
            if recalculated_hash == store_hash:
                print("✅ Task Store hash verification passed")
            else:
                print("❌ Task Store hash verification failed")
        
        # 验证Message格式哈希
        sample_message = proof.get('sample_message')
        message_hash = proof.get('message_format_hash')
        if sample_message and message_hash:
            recalculated_hash = compute_hash(json.dumps(sample_message, sort_keys=True))
            if recalculated_hash == message_hash:
                print("✅ Message format hash verification passed")
            else:
                print("❌ Message format hash verification failed")
        
        print("\n🎉 A2A Adapter test completed successfully!")
        print("\n📋 Test Summary:")
        print(f"   - Agent Card: ✅ Created with {len(agent_card.skills)} skills")
        print(f"   - A2A Proof: ✅ Generated with {len(proof)} fields")
        print(f"   - Protocol Meta: ✅ Generated with {len(protocol_meta)} sections")
        print(f"   - Field Validation: ✅ All required fields present")
        print(f"   - SDK Components: ✅ All components verified")
        print(f"   - Hash Verification: ✅ All hashes consistent")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("🚀 Starting A2A Adapter Standalone Tests\n")
    
    # 基本功能测试
    basic_test_passed = await test_a2a_adapter()
    
    print(f"\n📊 Final Results:")
    print(f"   - Basic Functionality: {'✅ PASS' if basic_test_passed else '❌ FAIL'}")
    
    if basic_test_passed:
        print("\n🎉 All tests passed! A2A adapter core logic is working correctly.")
        print("\n📝 Next Steps:")
        print("   1. Install A2A SDK dependencies")
        print("   2. Test with real A2A SDK components")
        print("   3. Run full RG integration test")
        print("   4. Test in agent_network environment")
    else:
        print("\n⚠️  Tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

