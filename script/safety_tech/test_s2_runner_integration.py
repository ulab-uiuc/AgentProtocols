#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2探针Runner集成测试
验证S2探针在实际runner环境中的集成效果
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# 添加项目路径
SAFETY_TECH = Path(__file__).parent
sys.path.insert(0, str(SAFETY_TECH.parent.parent))

# 设置环境变量
os.environ['A2A_USE_DIRECT_SEND'] = 'true'
os.environ['A2A_ENABLE_S2_PROBES'] = 'true'
os.environ['A2A_S2_PROBE_TYPE'] = 'comprehensive'

from script.safety_tech.core.backend_api import send_backend
from script.safety_tech.core.probe_config import create_comprehensive_probe_config


async def test_runner_integration():
    """测试S2探针在runner环境中的集成"""
    print("🚀 开始S2探针Runner集成测试")
    print(f"📡 环境配置:")
    print(f"   A2A_USE_DIRECT_SEND: {os.environ.get('A2A_USE_DIRECT_SEND')}")
    print(f"   A2A_ENABLE_S2_PROBES: {os.environ.get('A2A_ENABLE_S2_PROBES')}")
    print(f"   A2A_S2_PROBE_TYPE: {os.environ.get('A2A_S2_PROBE_TYPE')}")
    
    # 模拟runner中的医疗案例数据
    medical_cases = [
        {
            'case_id': 'CASE_001',
            'patient_info': '患者: 张三, 年龄: 45岁, 主诉: 胸痛',
            'initial_question': '患者出现持续性胸痛，伴有呼吸困难，血压160/90，心电图显示ST段抬高，请协助诊断和制定治疗方案。'
        }
    ]
    
    # 测试配置
    test_configs = {
        'TLS降级探针': {'A2A_S2_PROBE_TYPE': 'tls_downgrade'},
        '重放攻击探针': {'A2A_S2_PROBE_TYPE': 'replay_attack'}, 
        'DID伪造探针': {'A2A_S2_PROBE_TYPE': 'did_spoofing'},
        '综合探针': {'A2A_S2_PROBE_TYPE': 'comprehensive'}
    }
    
    results = {}
    
    for test_name, config in test_configs.items():
        print(f"\n🔍 === 测试 {test_name} ===")
        
        # 更新环境变量
        for key, value in config.items():
            os.environ[key] = value
            
        # 模拟runner中的对话循环
        total_attempted = 0
        total_success = 0
        total_latencies = []
        probe_results_collection = []
        
        for case in medical_cases:
            print(f"\n📋 处理案例: {case['case_id']}")
            
            # 模拟5轮对话
            for r in range(5):
                total_attempted += 1
                text = f"[Round {r+1}] {case['initial_question'][:100]}..."
                
                try:
                    start_time = time.time()
                    _mid = f"msg_{int(time.time()*1000)}"
                    _cid = f"corr_{int(time.time()*1000)}_{r}"
                    
                    # 使用综合探针配置
                    probe_config = create_comprehensive_probe_config().to_dict()
                    
                    # 模拟数据面直连发送
                    payload = {
                        'sender_id': 'A2A_Doctor_A',
                        'receiver_id': 'A2A_Doctor_B', 
                        'text': text,
                        'message_id': _mid
                    }
                    
                    result = await send_backend(
                        'a2a', 
                        'http://127.0.0.1:8001',  # 虚拟端点
                        payload, 
                        _cid, 
                        probe_config=probe_config
                    )
                    
                    latency_ms = (time.time() - start_time) * 1000
                    total_latencies.append(latency_ms)
                    
                    # 检查探针结果
                    probe_results = result.get('probe_results', {})
                    if probe_results:
                        probe_results_collection.append(probe_results)
                        total_success += 1
                        print(f"   ✅ Round {r+1}: 成功 ({latency_ms:.1f}ms)")
                        print(f"      探针结果: {probe_results}")
                    else:
                        print(f"   ❌ Round {r+1}: 无探针结果")
                        
                except Exception as e:
                    print(f"   ❌ Round {r+1}: 错误 - {str(e)}")
        
        # 统计结果
        success_rate = total_success / total_attempted if total_attempted > 0 else 0
        avg_latency = sum(total_latencies) / len(total_latencies) if total_latencies else 0
        
        # 聚合探针结果（模拟runner中的统计逻辑）
        aggregated_probes = {}
        if probe_results_collection:
            aggregated_probes = {
                'total_probes': len(probe_results_collection),
                'tls_downgrade_attempts': len([p for p in probe_results_collection if p.get('tls_downgrade')]),
                'replay_attempts': len([p for p in probe_results_collection if p.get('replay_attack')]),
                'did_spoofing_attempts': len([p for p in probe_results_collection if p.get('did_spoofing')]),
                'plaintext_bytes_detected': sum(p.get('plaintext_detected', 0) for p in probe_results_collection),
                'sensitive_keywords_total': sum(len(p.get('sensitive_keywords_detected', [])) for p in probe_results_collection),
                'avg_sensitive_score': sum(p.get('sensitive_data_score', 0) for p in probe_results_collection) / len(probe_results_collection)
            }
        
        results[test_name] = {
            'success_rate': success_rate,
            'avg_latency_ms': avg_latency,
            'probe_results': aggregated_probes
        }
        
        print(f"\n📊 {test_name} 结果:")
        print(f"   成功率: {success_rate:.1%}")
        print(f"   平均延迟: {avg_latency:.1f}ms")
        print(f"   探针统计: {aggregated_probes}")
    
    # 生成综合报告
    print(f"\n📋 === S2探针Runner集成测试报告 ===")
    
    for test_name, result in results.items():
        print(f"\n🔸 {test_name}:")
        print(f"  成功率: {result['success_rate']:.1%}")
        print(f"  平均延迟: {result['avg_latency_ms']:.1f}ms")
        
        probe_stats = result['probe_results']
        if probe_stats:
            print(f"  探针统计:")
            print(f"    总探针次数: {probe_stats.get('total_probes', 0)}")
            print(f"    TLS降级尝试: {probe_stats.get('tls_downgrade_attempts', 0)}")
            print(f"    重放攻击尝试: {probe_stats.get('replay_attempts', 0)}")
            print(f"    DID伪造尝试: {probe_stats.get('did_spoofing_attempts', 0)}")
            print(f"    明文字节检测: {probe_stats.get('plaintext_bytes_detected', 0)}")
            print(f"    敏感关键字总数: {probe_stats.get('sensitive_keywords_total', 0)}")
            print(f"    平均敏感度评分: {probe_stats.get('avg_sensitive_score', 0):.1f}")
    
    # 评估集成效果
    total_success_rate = sum(r['success_rate'] for r in results.values()) / len(results)
    avg_latency_impact = sum(r['avg_latency_ms'] for r in results.values()) / len(results)
    
    print(f"\n🎯 === 集成效果评估 ===")
    print(f"总体成功率: {total_success_rate:.1%}")
    print(f"平均延迟影响: {avg_latency_impact:.1f}ms")
    
    if total_success_rate >= 0.9:
        print("🎉 S2探针Runner集成效果优秀！")
    elif total_success_rate >= 0.7:
        print("✅ S2探针Runner集成效果良好")
    else:
        print("⚠️ S2探针Runner集成需要优化")
        
    if avg_latency_impact < 200:
        print("⚡ 延迟影响在可接受范围内")
    else:
        print("🐌 延迟影响较大，需要优化")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_runner_integration())
