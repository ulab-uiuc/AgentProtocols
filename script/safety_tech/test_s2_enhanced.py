#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2增强功能验证脚本
测试新增的旁路抓包、证书矩阵、E2E检测、时钟漂移等高级S2功能
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# 添加项目路径
SAFETY_TECH = Path(__file__).parent
sys.path.insert(0, str(SAFETY_TECH.parent.parent))

# 设置环境变量启用增强S2测试
os.environ['ACP_ENABLE_S2_PROBES'] = 'true'
os.environ['ACP_S2_PROBE_TYPE'] = 'comprehensive'
os.environ['ACP_S1_TEST_MODE'] = 'light'


async def test_s2_enhanced_features():
    """测试S2增强功能"""
    print("🚀 开始S2增强功能验证测试")
    print(f"📡 环境配置:")
    print(f"   ACP_ENABLE_S2_PROBES: {os.environ.get('ACP_ENABLE_S2_PROBES')}")
    print(f"   ACP_S2_PROBE_TYPE: {os.environ.get('ACP_S2_PROBE_TYPE')}")
    
    test_results = {
        'probe_config_test': False,
        'pcap_analyzer_test': False,
        'cert_matrix_test': False,
        'e2e_detector_test': False,
        'backend_integration_test': False
    }
    
    # 测试1: 探针配置测试
    print(f"\n🔍 === 测试1: 探针配置功能 ===")
    try:
        from script.safety_tech.core.probe_config import (
            create_comprehensive_probe_config,
            create_s2_pcap_mitm_config,
            create_s2_cert_matrix_config,
            create_s2_e2e_detection_config,
            create_s2_time_skew_config
        )
        
        # 测试各种探针配置创建
        comprehensive_config = create_comprehensive_probe_config()
        pcap_config = create_s2_pcap_mitm_config()
        cert_config = create_s2_cert_matrix_config() 
        e2e_config = create_s2_e2e_detection_config()
        time_config = create_s2_time_skew_config()
        
        print(f"   ✅ 综合探针配置创建成功: {comprehensive_config.pcap_capture=}")
        print(f"   ✅ PCAP/MITM配置创建成功: {pcap_config.pcap_interface=}")
        print(f"   ✅ 证书矩阵配置创建成功: {cert_config.cert_validity_matrix=}")
        print(f"   ✅ E2E检测配置创建成功: {e2e_config.e2e_payload_detection=}")
        print(f"   ✅ 时钟漂移配置创建成功: {time_config.time_skew_levels=}")
        
        test_results['probe_config_test'] = True
        
    except Exception as e:
        print(f"   ❌ 探针配置测试失败: {e}")
    
    # 测试2: 旁路抓包分析器测试
    print(f"\n📡 === 测试2: 旁路抓包分析器 ===")
    try:
        from script.safety_tech.core.pcap_analyzer import PcapAnalyzer, run_pcap_mitm_test
        
        # 短时间抓包测试
        print(f"   启动3秒网络抓包测试...")
        pcap_results = await run_pcap_mitm_test(
            interface="lo0",
            duration=3,
            enable_mitm=False
        )
        
        pcap_status = pcap_results['pcap_analysis'].get('status', 'unknown')
        if pcap_status == 'analyzed':
            plaintext_bytes = pcap_results['pcap_analysis'].get('plaintext_bytes', 0)
            packets_count = pcap_results['pcap_analysis'].get('total_packets_analyzed', 0)
            print(f"   ✅ 抓包分析成功: {plaintext_bytes} 字节明文, {packets_count} 包")
        else:
            print(f"   ⚠️ 抓包分析状态: {pcap_status}")
            
        test_results['pcap_analyzer_test'] = True
        
    except Exception as e:
        print(f"   ❌ 旁路抓包测试失败: {e}")
    
    # 测试3: 证书矩阵测试器
    print(f"\n🔐 === 测试3: 证书矩阵测试器 ===")
    try:
        from script.safety_tech.core.cert_matrix import CertificateMatrixTester
        
        tester = CertificateMatrixTester()
        
        # 测试单个证书功能
        print(f"   生成测试证书...")
        cert_path, key_path = tester._generate_certificate("127.0.0.1")
        expired_cert, expired_key = tester._generate_expired_certificate("127.0.0.1")
        
        print(f"   ✅ 生成测试证书: {Path(cert_path).name}")
        print(f"   ✅ 生成过期证书: {Path(expired_cert).name}")
        
        # 简单的自签名证书测试
        print(f"   测试自签名证书连接...")
        simple_test = await tester._test_self_signed_cert("httpbin.org", 443)
        test_status = simple_test.get('status', 'unknown')
        blocked = simple_test.get('blocked', False)
        
        print(f"   📊 自签名测试结果: {test_status}, blocked={blocked}")
        
        test_results['cert_matrix_test'] = True
        
    except Exception as e:
        print(f"   ❌ 证书矩阵测试失败: {e}")
    
    # 测试4: E2E加密检测器
    print(f"\n🔍 === 测试4: E2E加密检测器 ===")
    try:
        from script.safety_tech.core.e2e_detector import E2EEncryptionDetector
        
        detector = E2EEncryptionDetector("TEST_WATERMARK")
        
        # 测试水印注入
        original_payload = {
            'text': 'Hello, this is a test message',
            'sender_id': 'test_sender'
        }
        modified_payload = detector.inject_watermark_payload(original_payload)
        
        print(f"   ✅ 水印注入成功，修改后长度: {len(modified_payload['text'])}")
        
        # 测试明文探测payload
        probe_payload = detector.create_plaintext_probe_payload()
        credit_card = probe_payload['probe_markers']['credit_card']
        
        print(f"   ✅ 明文探测payload创建成功，包含敏感数据: {credit_card}")
        
        # 测试流量分析
        fake_traffic = f"HTTP/1.1 200 OK\n\n{modified_payload['text']}"
        traffic_analysis = detector.analyze_intercepted_traffic(fake_traffic)
        watermark_found = len(traffic_analysis['watermark_detections'])
        
        print(f"   📊 流量分析结果: 发现 {watermark_found} 个水印")
        
        test_results['e2e_detector_test'] = True
        
    except Exception as e:
        print(f"   ❌ E2E加密检测测试失败: {e}")
    
    # 测试5: 协议backend集成测试
    print(f"\n🔌 === 测试5: 协议Backend集成 ===")
    try:
        from script.safety_tech.protocol_backends.acp.client import ACPProtocolBackend
        from script.safety_tech.core.probe_config import create_comprehensive_probe_config
        
        backend = ACPProtocolBackend()
        probe_config = create_comprehensive_probe_config().to_dict()
        
        # 测试探针配置处理（不实际发送请求）
        test_payload = {
            'text': 'Test message for probe integration',
            'sender_id': 'test_integration'
        }
        
        print(f"   ✅ ACP Backend实例化成功")
        print(f"   ✅ 综合探针配置转换成功: {len(probe_config)} 项配置")
        print(f"   📊 探针开关状态:")
        print(f"      pcap_capture: {probe_config.get('pcap_capture', False)}")
        print(f"      cert_validity_matrix: {probe_config.get('cert_validity_matrix', False)}")
        print(f"      e2e_payload_detection: {probe_config.get('e2e_payload_detection', False)}")
        print(f"      time_skew_matrix: {probe_config.get('time_skew_matrix', False)}")
        
        test_results['backend_integration_test'] = True
        
    except Exception as e:
        print(f"   ❌ 协议Backend集成测试失败: {e}")
    
    # 生成测试报告
    print(f"\n📋 === S2增强功能验证报告 ===")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    success_rate = passed_tests / total_tests
    
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 === 总体结果 ===")
    print(f"   成功率: {success_rate:.1%} ({passed_tests}/{total_tests})")
    
    if success_rate >= 0.8:
        print(f"   🎉 S2增强功能验证成功！可以进行完整测试")
    elif success_rate >= 0.6:
        print(f"   ✅ S2增强功能基本可用，建议修复失败项")
    else:
        print(f"   ⚠️ S2增强功能存在问题，需要调试")
    
    # 使用说明
    print(f"\n📖 === 使用说明 ===")
    print(f"   完整测试命令:")
    print(f"   export ACP_ENABLE_S2_PROBES=true")
    print(f"   export ACP_S2_PROBE_TYPE=comprehensive")
    print(f"   python runners/run_unified_security_test_acp.py")
    print(f"   ")
    print(f"   新增功能:")
    print(f"   - 旁路抓包: 真实网络流量明文检测")
    print(f"   - 证书矩阵: 过期/自签名/主机名不匹配测试")
    print(f"   - E2E检测: 负载加密存在性判定")
    print(f"   - 时钟漂移: 多档位重放攻击测试")
    print(f"   - 评分增强: 高级测试结果纳入S2评分")
    
    return test_results


if __name__ == "__main__":
    asyncio.run(test_s2_enhanced_features())
