#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2保密性探针测试脚本
验证各协议client的探针功能是否正常工作
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
SAFETY_TECH = Path(__file__).parent
sys.path.insert(0, str(SAFETY_TECH.parent.parent))

from script.safety_tech.core.probe_config import (
    create_s2_tls_downgrade_config,
    create_s2_replay_attack_config,
    create_s2_mitm_config,
    create_s2_did_spoofing_config,
    create_comprehensive_probe_config
)
from script.safety_tech.core.backend_api import send_backend


async def test_protocol_probes(protocol: str, endpoint: str):
    """测试指定协议的探针功能"""
    print(f"\n🔍 === 测试 {protocol.upper()} 协议探针 ===")
    
    # 测试载荷
    test_payload = {
        'sender_id': f'{protocol.upper()}_Test_A',
        'receiver_id': f'{protocol.upper()}_Test_B', 
        'text': 'Test message with sensitive data: password=secret123, patient=John Doe',
        'message_id': 'test_msg_001'
    }
    correlation_id = 'test_corr_001'
    
    # 测试各种探针配置
    probe_configs = {
        'TLS降级': create_s2_tls_downgrade_config(),
        '重放攻击': create_s2_replay_attack_config('test_nonce'),
        'DID伪造': create_s2_did_spoofing_config(f'did:fake:{protocol}_test'),
        '综合测试': create_comprehensive_probe_config()
    }
    
    results = {}
    
    for probe_name, probe_config in probe_configs.items():
        print(f"\n  📡 测试 {probe_name} 探针...")
        try:
            result = await send_backend(
                protocol=protocol,
                endpoint=endpoint,
                payload=test_payload,
                correlation_id=correlation_id,
                probe_config=probe_config.to_dict()
            )
            
            probe_results = result.get('probe_results', {})
            if probe_results:
                print(f"    ✅ 探针结果: {probe_results}")
                results[probe_name] = probe_results
            else:
                print(f"    ❌ 无探针结果返回")
                results[probe_name] = None
                
        except Exception as e:
            print(f"    ❌ 探针测试失败: {e}")
            results[probe_name] = {'error': str(e)}
    
    return results


async def test_all_protocols():
    """测试所有协议的探针功能"""
    print("🚀 开始S2保密性探针测试")
    
    # 协议配置（使用虚拟端点进行测试）
    protocols = {
        'a2a': 'http://127.0.0.1:8001',
        'acp': 'http://127.0.0.1:8002', 
        'anp': 'http://127.0.0.1:8003',
        'agora': 'http://127.0.0.1:8004'
    }
    
    all_results = {}
    
    for protocol, endpoint in protocols.items():
        try:
            results = await test_protocol_probes(protocol, endpoint)
            all_results[protocol] = results
        except Exception as e:
            print(f"\n❌ {protocol.upper()} 协议测试失败: {e}")
            all_results[protocol] = {'error': str(e)}
    
    # 生成测试报告
    print(f"\n📊 === S2探针测试报告 ===")
    
    for protocol, results in all_results.items():
        print(f"\n🔸 {protocol.upper()} 协议:")
        if isinstance(results, dict) and 'error' in results:
            print(f"  ❌ 整体失败: {results['error']}")
            continue
            
        for probe_name, probe_result in results.items():
            if probe_result is None:
                print(f"  ❌ {probe_name}: 无结果")
            elif isinstance(probe_result, dict) and 'error' in probe_result:
                print(f"  ❌ {probe_name}: {probe_result['error']}")
            else:
                print(f"  ✅ {probe_name}: {probe_result}")
    
    # 统计成功率
    total_tests = sum(len(results) for results in all_results.values() if isinstance(results, dict) and 'error' not in results)
    successful_tests = 0
    
    for results in all_results.values():
        if isinstance(results, dict) and 'error' not in results:
            for probe_result in results.values():
                if probe_result is not None and not (isinstance(probe_result, dict) and 'error' in probe_result):
                    successful_tests += 1
    
    success_rate = successful_tests / total_tests if total_tests > 0 else 0
    print(f"\n📈 总体成功率: {successful_tests}/{total_tests} ({success_rate:.1%})")
    
    if success_rate >= 0.8:
        print("🎉 S2探针功能基本正常！")
    elif success_rate >= 0.5:
        print("⚠️ S2探针功能部分正常，需要进一步调试")
    else:
        print("🚨 S2探针功能存在重大问题，需要修复")
    
    return all_results


async def test_probe_config_schema():
    """测试探针配置Schema"""
    print("\n🧪 === 测试探针配置Schema ===")
    
    # 测试各种配置创建
    configs = [
        create_s2_tls_downgrade_config(),
        create_s2_replay_attack_config(),
        create_s2_mitm_config(),
        create_s2_did_spoofing_config(),
        create_comprehensive_probe_config()
    ]
    
    for i, config in enumerate(configs):
        print(f"  📋 配置 {i+1}: {config.to_dict()}")
    
    print("  ✅ 探针配置Schema测试完成")


if __name__ == "__main__":
    async def main():
        # 测试Schema
        await test_probe_config_schema()
        
        # 测试所有协议探针（注意：需要实际的服务端点才能完全测试）
        print("\n⚠️ 注意: 完整测试需要启动实际的协议服务端点")
        print("当前将测试探针配置和client逻辑，可能会因为端点不可达而部分失败")
        
        results = await test_all_protocols()
        
        print(f"\n🏁 S2探针测试完成")
        return results
    
    asyncio.run(main())
