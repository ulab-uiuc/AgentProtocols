#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S1新版业务连续性测试验证脚本
验证新的S1测试框架是否正常工作
"""

import asyncio
import time
import json
from pathlib import Path

# 添加路径以导入模块
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from script.safety_tech.core.s1_business_continuity import (
    S1BusinessContinuityTester, LoadMatrixConfig, NetworkDisturbanceConfig, 
    AttackNoiseConfig, LoadPattern, MessageType
)
from script.safety_tech.core.s1_config_factory import create_s1_tester


async def mock_send_function(payload):
    """模拟发送函数"""
    # 模拟网络延迟
    await asyncio.sleep(0.05 + (0.1 * (hash(str(payload)) % 10) / 10))
    
    # 模拟一些失败
    if hash(str(payload)) % 20 == 0:  # 5%失败率
        return {"status": "error", "error": "Simulated network error"}
    
    return {"status": "success", "response": {"message": "OK"}}


async def test_light_configuration():
    """测试轻量配置"""
    print("🧪 测试轻量配置...")
    
    # 创建快速版本的轻量配置
    from script.safety_tech.core.s1_config_factory import S1ConfigFactory
    
    config = S1ConfigFactory.create_light_test_config()
    # 大幅缩短测试时间
    config['load_config'].test_duration_seconds = 2  # 从30秒缩短到2秒
    config['load_config'].base_rps = 5  # 降低RPS
    # 禁用攻击噪声
    config['attack_config'].enable_all = False
    
    tester = S1ConfigFactory.create_tester_from_config('acp', config)
    
    # 模拟端口
    rg_port = 8080
    coord_port = 8081
    obs_port = 8082
    
    try:
        results = await tester.run_full_test_matrix(
            send_func=mock_send_function,
            sender_id='Test_Doctor_A',
            receiver_id='Test_Doctor_B',
            rg_port=rg_port,
            coord_port=coord_port,
            obs_port=obs_port
        )
        
        report = tester.generate_comprehensive_report()
        
        print(f"✅ 轻量测试完成")
        print(f"   组合数: {len(results)}")
        print(f"   总请求: {report['test_summary']['total_requests']}")
        print(f"   完成率: {report['test_summary']['overall_completion_rate']:.1%}")
        print(f"   平均延迟: {report['latency_analysis']['avg_ms']:.1f}ms")
        
        return report
        
    except Exception as e:
        print(f"❌ 轻量测试失败: {e}")
        return None


async def test_protocol_optimized_configuration():
    """测试协议优化配置"""
    print("\n🧪 测试协议优化配置...")
    
    # 创建一个快速版本的协议优化配置
    from script.safety_tech.core.s1_config_factory import S1ConfigFactory
    from script.safety_tech.core.s1_business_continuity import S1BusinessContinuityTester
    
    config = S1ConfigFactory.create_protocol_optimized_config('acp')
    # 大幅简化配置以加快验证
    config['load_config'].test_duration_seconds = 3  # 从60秒缩短到3秒
    config['load_config'].concurrent_levels = [4]  # 只测试一个并发级别
    config['load_config'].base_rps = 5  # 降低RPS
    # 禁用攻击噪声以避免复杂的异步任务
    config['attack_config'].enable_all = False
    
    tester = S1ConfigFactory.create_tester_from_config('acp', config)
    
    # 模拟端口
    rg_port = 8080
    coord_port = 8081
    obs_port = 8082
    
    try:
        results = await tester.run_full_test_matrix(
            send_func=mock_send_function,
            sender_id='Test_Doctor_A',
            receiver_id='Test_Doctor_B',
            rg_port=rg_port,
            coord_port=coord_port,
            obs_port=obs_port
        )
        
        report = tester.generate_comprehensive_report()
        
        print(f"✅ 协议优化测试完成")
        print(f"   组合数: {len(results)}")
        print(f"   总请求: {report['test_summary']['total_requests']}")
        print(f"   完成率: {report['test_summary']['overall_completion_rate']:.1%}")
        print(f"   平均延迟: {report['latency_analysis']['avg_ms']:.1f}ms")
        
        # 检查维度分析
        print("\n   维度分析:")
        for level, data in report['dimensional_analysis']['by_concurrent_level'].items():
            print(f"     并发{level}: 完成率{data['avg_completion_rate']:.1%}, "
                  f"延迟{data['avg_latency_ms']:.1f}ms")
        
        return report
        
    except Exception as e:
        print(f"❌ 协议优化测试失败: {e}")
        return None


async def test_network_disturbance():
    """测试网络扰动功能"""
    print("\n🧪 测试网络扰动功能...")
    
    # 创建专门的网络扰动配置
    load_config = LoadMatrixConfig(
        concurrent_levels=[4],
        rps_patterns=[LoadPattern.CONSTANT],
        message_types=[MessageType.SHORT],
        test_duration_seconds=10,
        base_rps=5
    )
    
    disturbance_config = NetworkDisturbanceConfig(
        jitter_ms_range=(50, 100),
        packet_loss_rate=0.1,  # 10%丢包
        reorder_probability=0.05,  # 5%乱序
        enable_connection_drops=False  # 简化测试
    )
    
    attack_config = AttackNoiseConfig(enable_all=False)  # 禁用攻击噪声
    
    tester = S1BusinessContinuityTester(
        protocol_name='test',
        load_config=load_config,
        disturbance_config=disturbance_config,
        attack_config=attack_config
    )
    
    try:
        # 启动网络扰动
        await tester.start_network_disturbance()
        
        # 测试扰动效果
        effects_count = 0
        for i in range(10):
            try:
                effects = await tester.apply_network_disturbance(delay_before_send=True)
                if effects:
                    effects_count += 1
                    print(f"   扰动效果 {i+1}: {effects}")
            except Exception as e:
                print(f"   扰动导致异常 {i+1}: {type(e).__name__}")
                effects_count += 1
        
        await tester.stop_network_disturbance()
        
        print(f"✅ 网络扰动测试完成，{effects_count}/10 次产生效果")
        return True
        
    except Exception as e:
        print(f"❌ 网络扰动测试失败: {e}")
        return False


def test_config_factory():
    """测试配置工厂"""
    print("\n🧪 测试配置工厂...")
    
    try:
        from script.safety_tech.core.s1_config_factory import S1ConfigFactory
        
        # 测试所有预定义配置
        configs = S1ConfigFactory.get_available_configs()
        print(f"   可用配置: {configs}")
        
        for config_name in configs:
            config = S1ConfigFactory.create_config_by_name(config_name)
            print(f"   ✅ {config_name}: {len(config['load_config'].concurrent_levels)} 并发级别")
        
        # 测试协议优化配置
        for protocol in ['acp', 'anp', 'a2a', 'agora']:
            config = S1ConfigFactory.create_protocol_optimized_config(protocol)
            print(f"   ✅ {protocol} 优化配置: RPS={config['load_config'].base_rps}")
        
        print("✅ 配置工厂测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 配置工厂测试失败: {e}")
        return False


async def test_correlation_tracking():
    """测试关联跟踪"""
    print("\n🧪 测试关联跟踪...")
    
    tester = create_s1_tester('test', 'light')
    
    try:
        # 创建跟踪器
        tracker1 = tester.create_correlation_tracker('sender1', 'receiver1', MessageType.SHORT)
        tracker2 = tester.create_correlation_tracker('sender2', 'receiver2', MessageType.LONG)
        
        print(f"   跟踪器1: {tracker1.correlation_id}")
        print(f"   跟踪器2: {tracker2.correlation_id}")
        
        # 检查活跃跟踪器
        print(f"   活跃跟踪器数量: {len(tester.active_trackers)}")
        
        # 模拟收到回执
        success = tester.check_response_received(
            tracker1.correlation_id, 
            f"{tracker1.receiver_id} response: received your message"
        )
        print(f"   回执检查结果: {success}")
        
        # 清理过期跟踪器
        expired = tester.cleanup_expired_trackers()
        print(f"   清理过期跟踪器: {expired} 个")
        
        print("✅ 关联跟踪测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 关联跟踪测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("🚀 开始S1新版业务连续性测试验证")
    print("="*60)
    
    # 设置总体超时
    import signal
    
    def timeout_handler(signum, frame):
        print("\n⏰ 测试超时，强制退出")
        raise TimeoutError("Test timeout")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60秒超时
    
    # 测试配置工厂
    config_factory_ok = test_config_factory()
    
    # 测试关联跟踪
    correlation_ok = await test_correlation_tracking()
    
    # 测试网络扰动
    disturbance_ok = await test_network_disturbance()
    
    # 测试轻量配置
    light_report = await test_light_configuration()
    
    # 测试协议优化配置
    optimized_report = await test_protocol_optimized_configuration()
    
    # 取消超时alarm
    signal.alarm(0)
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试总结:")
    print(f"   配置工厂: {'✅' if config_factory_ok else '❌'}")
    print(f"   关联跟踪: {'✅' if correlation_ok else '❌'}")
    print(f"   网络扰动: {'✅' if disturbance_ok else '❌'}")
    print(f"   轻量测试: {'✅' if light_report else '❌'}")
    print(f"   协议优化测试: {'✅' if optimized_report else '❌'}")
    
    all_passed = all([config_factory_ok, correlation_ok, disturbance_ok, 
                      light_report is not None, optimized_report is not None])
    
    if all_passed:
        print("\n🎉 所有测试通过！新版S1业务连续性测试框架可以使用。")
        
        # 保存测试报告
        if optimized_report:
            output_dir = Path(__file__).parent / "test_output"
            output_dir.mkdir(exist_ok=True)
            
            report_file = output_dir / f"s1_test_verification_{int(time.time())}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(optimized_report, f, indent=2, ensure_ascii=False)
            
            print(f"📄 详细测试报告已保存到: {report_file}")
    else:
        print("\n❌ 部分测试失败，需要检查和修复。")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
