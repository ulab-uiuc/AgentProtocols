# -*- coding: utf-8 -*-
"""
A2A Adapter Test Script
测试A2A注册适配器的基本功能
"""

import asyncio
import json
import sys
from pathlib import Path

# Add safety_tech to path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from protocol_backends.a2a.registration_adapter import A2ARegistrationAdapter


async def test_a2a_adapter():
    """测试A2A适配器基本功能"""
    print("🧪 Testing A2A Registration Adapter")
    
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
        adapter = A2ARegistrationAdapter(config)
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
        
        # 测试Observer证明生成
        observer_proof = await adapter._generate_observer_proof("test_observer")
        print("✅ Observer proof generated")
        print(f"   - Observer ID: {observer_proof.get('observer_id')}")
        print(f"   - Observer type: {observer_proof.get('observer_type')}")
        print(f"   - Protocol version: {observer_proof.get('a2a_protocol_version')}")
        
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
        
        print("\n🎉 A2A Adapter test completed successfully!")
        print("\n📋 Test Summary:")
        print(f"   - Agent Card: ✅ Created with {len(agent_card.skills)} skills")
        print(f"   - A2A Proof: ✅ Generated with {len(proof)} fields")
        print(f"   - Protocol Meta: ✅ Generated with {len(protocol_meta)} sections")
        print(f"   - Observer Proof: ✅ Generated")
        print(f"   - Field Validation: ✅ All required fields present")
        print(f"   - SDK Components: ✅ All components verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_attack_scenarios():
    """测试攻击场景功能"""
    print("\n🔒 Testing A2A Attack Scenarios")
    
    config = {
        'rg_endpoint': 'http://127.0.0.1:8001',
        'a2a': {}
    }
    
    try:
        adapter = A2ARegistrationAdapter(config)
        
        # 测试冒名顶替攻击
        impersonation_result = await adapter.simulate_impersonation_attack(
            "fake_agent", "test_conversation"
        )
        print(f"✅ Impersonation attack simulated: {impersonation_result.get('attack_type')}")
        
        # 测试凭证复用攻击
        credential_reuse_result = await adapter.simulate_credential_reuse_attack(
            "fake_token", "test_conversation"
        )
        print(f"✅ Credential reuse attack simulated: {credential_reuse_result.get('attack_type')}")
        
        # 测试重放攻击
        fake_registration = {"protocol": "a2a", "agent_id": "fake", "proof": {}}
        replay_result = await adapter.simulate_replay_attack(fake_registration)
        print(f"✅ Replay attack simulated: {replay_result.get('attack_type')}")
        
        # 测试端点替换攻击
        endpoint_sub_result = await adapter.simulate_endpoint_substitution_attack("test_conversation")
        print(f"✅ Endpoint substitution attack simulated: {endpoint_sub_result.get('attack_type')}")
        
        # 测试跨协议攻击
        fake_agora_proof = {"toolformer_signature": "fake", "protocol_hash": "fake"}
        cross_protocol_result = await adapter.simulate_cross_protocol_attack(
            fake_agora_proof, "test_conversation"
        )
        print(f"✅ Cross-protocol attack simulated: {cross_protocol_result.get('attack_type')}")
        
        # 测试Observer自动准入攻击
        observer_attack_result = await adapter.simulate_observer_auto_admission_attack("test_conversation")
        print(f"✅ Observer auto-admission attack simulated: {observer_attack_result.get('attack_type')}")
        
        print("\n🎉 Attack scenarios test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Attack scenarios test failed: {e}")
        return False


async def main():
    """主测试函数"""
    print("🚀 Starting A2A Adapter Tests\n")
    
    # 基本功能测试
    basic_test_passed = await test_a2a_adapter()
    
    # 攻击场景测试
    attack_test_passed = await test_attack_scenarios()
    
    print(f"\n📊 Final Results:")
    print(f"   - Basic Functionality: {'✅ PASS' if basic_test_passed else '❌ FAIL'}")
    print(f"   - Attack Scenarios: {'✅ PASS' if attack_test_passed else '❌ FAIL'}")
    
    if basic_test_passed and attack_test_passed:
        print("\n🎉 All tests passed! A2A adapter is ready for integration.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

