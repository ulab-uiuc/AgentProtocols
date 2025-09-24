# -*- coding: utf-8 -*-
"""
统一后端API基础连通性测试

测试所有协议（ACP/ANP/A2A/Agora）的统一后端API：
- spawn_backend: 启动服务
- health_backend: 健康检查
- register_backend: 注册到RG
- 基础消息路由测试
"""

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import httpx

HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(SAFETY_TECH))

# 导入统一后端API
try:
    from core.backend_api import spawn_backend, register_backend, health_backend
    from core.rg_coordinator import RGCoordinator
except ImportError:
    from script.safety_tech.core.backend_api import spawn_backend, register_backend, health_backend
    from script.safety_tech.core.rg_coordinator import RGCoordinator

# 导入协议后端（触发注册）
import script.safety_tech.protocol_backends.acp
import script.safety_tech.protocol_backends.anp
import script.safety_tech.protocol_backends.a2a
import script.safety_tech.protocol_backends.agora


async def test_protocol_backend(protocol: str, base_port: int) -> Dict[str, Any]:
    """测试单个协议后端的完整流程 - 启动两个医生agent"""
    print(f"\n🧪 测试 {protocol.upper()} 协议后端...")
    
    results = {
        'protocol': protocol,
        'spawn_success': False,
        'health_success': False,
        'register_success': False,
        'doctor_a': {'spawn': False, 'health': False, 'register': False},
        'doctor_b': {'spawn': False, 'health': False, 'register': False},
        'errors': []
    }
    
    try:
        # 1. 启动两个医生服务
        print(f"   1️⃣ 启动 {protocol} 双医生服务...")
        await spawn_backend(protocol, 'doctor_a', base_port)
        await spawn_backend(protocol, 'doctor_b', base_port + 1)
        await asyncio.sleep(3.0)  # 等待服务启动
        results['spawn_success'] = True
        results['doctor_a']['spawn'] = True
        results['doctor_b']['spawn'] = True
        print(f"   ✅ {protocol} 双医生服务启动成功")
        
        # 2. 健康检查
        print(f"   2️⃣ 检查 {protocol} 服务健康状态...")
        health_a = await health_backend(protocol, f"http://127.0.0.1:{base_port}")
        health_b = await health_backend(protocol, f"http://127.0.0.1:{base_port + 1}")
        results['health_success'] = True
        results['doctor_a']['health'] = True
        results['doctor_b']['health'] = True
        results['health_data'] = {'doctor_a': health_a, 'doctor_b': health_b}
        print(f"   ✅ {protocol} 双医生健康检查成功")
        
        # 3. 注册到RG
        print(f"   3️⃣ 注册 {protocol} 双医生到RG...")
        register_a = await register_backend(
            protocol, 
            f'{protocol.upper()}_Doctor_A', 
            f"http://127.0.0.1:{base_port}",
            f'test_conv_{protocol}',  # 每个协议使用独立会话
            'doctor_a',
            rg_endpoint='http://127.0.0.1:8001'
        )
        register_b = await register_backend(
            protocol, 
            f'{protocol.upper()}_Doctor_B', 
            f"http://127.0.0.1:{base_port + 1}",
            f'test_conv_{protocol}',  # 每个协议使用独立会话
            'doctor_b',
            rg_endpoint='http://127.0.0.1:8001'
        )
        results['register_success'] = True
        results['doctor_a']['register'] = True
        results['doctor_b']['register'] = True
        results['register_data'] = {'doctor_a': register_a, 'doctor_b': register_b}
        print(f"   ✅ {protocol} 双医生注册成功")
        
    except Exception as e:
        error_msg = f"{protocol} 测试失败: {str(e)}"
        results['errors'].append(error_msg)
        print(f"   ❌ {error_msg}")
    
    return results


async def test_coordinator_routing() -> Dict[str, Any]:
    """测试Coordinator消息路由功能"""
    print(f"\n📡 测试Coordinator消息路由...")
    
    results = {
        'route_success': False,
        'history_success': False,
        'errors': []
    }
    
    try:
        # 发送测试消息（同协议内通信：ACP doctor_a -> ACP doctor_b）
        async with httpx.AsyncClient() as client:
            message_data = {
                'sender_id': 'ACP_Doctor_A',
                'receiver_id': 'ACP_Doctor_B',  # 同协议内通信
                'text': 'Test message from unified backend test',
                'message_id': f'test_msg_{int(time.time()*1000)}',
                'correlation_id': f'test_corr_{int(time.time()*1000)}'
            }
            
            # 路由消息
            route_resp = await client.post(
                "http://127.0.0.1:8888/route_message", 
                json=message_data, 
                timeout=10.0
            )
            
            if route_resp.status_code in (200, 202):
                results['route_success'] = True
                print(f"   ✅ 消息路由成功: {route_resp.status_code}")
            else:
                results['errors'].append(f"路由失败: {route_resp.status_code}")
            
            # 检查消息历史
            await asyncio.sleep(1.0)
            hist_resp = await client.get(
                "http://127.0.0.1:8888/message_history",
                params={'limit': 10},
                timeout=5.0
            )
            
            if hist_resp.status_code == 200:
                history = hist_resp.json()
                results['history_success'] = True
                results['message_count'] = len(history) if isinstance(history, list) else 0
                print(f"   ✅ 消息历史获取成功: {results['message_count']} 条消息")
            else:
                results['errors'].append(f"历史获取失败: {hist_resp.status_code}")
                
    except Exception as e:
        error_msg = f"Coordinator路由测试失败: {str(e)}"
        results['errors'].append(error_msg)
        print(f"   ❌ {error_msg}")
    
    return results


async def main():
    """主测试函数"""
    print("🚀 开始统一后端API基础连通性测试...")
    
    # 启动RG
    print("\n📋 启动Registration Gateway...")
    rg_proc = None
    try:
        import subprocess
        rg_proc = subprocess.Popen([
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, '{PROJECT_ROOT}'); "
            "from script.safety_tech.core.registration_gateway import RegistrationGateway; "
            f"RegistrationGateway({{'session_timeout':3600,'max_observers':5,'require_observer_proof':False}}).run(host='127.0.0.1', port=8001)"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 等待RG启动
        for i in range(15):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get("http://127.0.0.1:8001/health", timeout=2.0)
                    if resp.status_code == 200:
                        print("   ✅ RG启动成功")
                        break
            except Exception:
                await asyncio.sleep(1.0)
        else:
            raise Exception("RG启动超时")
    except Exception as e:
        print(f"   ❌ RG启动失败: {e}")
        return
    
    # 启动Coordinator
    print("\n🎛️ 启动Coordinator...")
    coordinator = None
    try:
        coordinator = RGCoordinator({
            'rg_endpoint': 'http://127.0.0.1:8001',
            'conversation_id': 'test_conv_acp',  # 默认监听ACP会话，稍后可以路由到其他会话
            'coordinator_port': 8888
        })
        await coordinator.start()
        
        # 验证Coordinator启动
        await asyncio.sleep(2.0)
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://127.0.0.1:8888/health", timeout=2.0)
            if resp.status_code == 200:
                print("   ✅ Coordinator启动成功")
            else:
                raise Exception(f"Coordinator健康检查失败: {resp.status_code}")
    except Exception as e:
        print(f"   ❌ Coordinator启动失败: {e}")
        return
    
    # 测试各协议后端 (每个协议需要2个端口：doctor_a, doctor_b)
    test_results = []
    protocols = [
        ('acp', 9001),    # 9001, 9002
        ('anp', 9101),    # 9101, 9102 
        ('a2a', 9201),    # 9201, 9202
        ('agora', 9301)   # 9301, 9302
    ]
    
    for protocol, port in protocols:
        try:
            result = await test_protocol_backend(protocol, port)
            test_results.append(result)
        except Exception as e:
            print(f"❌ {protocol} 测试异常: {e}")
            test_results.append({
                'protocol': protocol,
                'spawn_success': False,
                'health_success': False, 
                'register_success': False,
                'errors': [str(e)]
            })
    
    # 测试Coordinator路由
    routing_result = await test_coordinator_routing()
    
    # 生成测试报告
    print("\n" + "="*80)
    print("📊 统一后端API测试报告")
    print("="*80)
    
    total_protocols = len(protocols)
    successful_protocols = len([r for r in test_results if r['spawn_success'] and r['health_success'] and r['register_success']])
    
    print(f"📋 协议测试结果: {successful_protocols}/{total_protocols} 成功")
    for result in test_results:
        protocol = result['protocol'].upper()
        spawn_status = "✅" if result['spawn_success'] else "❌"
        health_status = "✅" if result['health_success'] else "❌" 
        register_status = "✅" if result['register_success'] else "❌"
        print(f"   {protocol}: Spawn{spawn_status} Health{health_status} Register{register_status}")
        
        # 显示双医生详细状态
        if 'doctor_a' in result and 'doctor_b' in result:
            da_status = "✅" if all(result['doctor_a'].values()) else "❌"
            db_status = "✅" if all(result['doctor_b'].values()) else "❌"
            print(f"      Doctor_A{da_status} Doctor_B{db_status}")
            
        if result['errors']:
            for error in result['errors']:
                print(f"      ⚠️ {error}")
    
    print(f"\n📡 Coordinator路由测试:")
    route_status = "✅" if routing_result['route_success'] else "❌"
    history_status = "✅" if routing_result['history_success'] else "❌"
    print(f"   路由{route_status} 历史{history_status}")
    if routing_result['errors']:
        for error in routing_result['errors']:
            print(f"      ⚠️ {error}")
    
    # 整体评估
    overall_success = (
        successful_protocols == total_protocols and 
        routing_result['route_success'] and 
        routing_result['history_success']
    )
    
    print(f"\n🎯 整体测试结果: {'✅ 通过' if overall_success else '❌ 失败'}")
    
    if overall_success:
        print("🎉 所有协议的统一后端API工作正常！")
        print("📝 解耦完成状态:")
        print("   ✅ 统一接口 (spawn/register/health/send)")
        print("   ✅ 协议后端分离")
        print("   ✅ Runner迁移完成")
        print("   ✅ Coordinator协议无关化")
        print("\n🔧 最小可用解耦已完成，可以进行进一步的测试场景开发")
    else:
        print("⚠️ 部分功能存在问题，需要进一步调试")
    
    print("="*80)
    
    # 保存详细报告
    report = {
        'timestamp': time.time(),
        'protocol_results': test_results,
        'routing_result': routing_result,
        'overall_success': overall_success,
        'successful_protocols': successful_protocols,
        'total_protocols': total_protocols
    }
    
    report_file = SAFETY_TECH / 'output' / f'unified_backend_test_{int(time.time())}.json'
    report_file.parent.mkdir(exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 详细报告保存至: {report_file}")
    
    # 清理
    try:
        if coordinator:
            await coordinator.stop()
        if rg_proc:
            rg_proc.send_signal(signal.SIGTERM)
            rg_proc.wait(timeout=5)
    except Exception as e:
        print(f"清理资源时出错: {e}")


if __name__ == '__main__':
    asyncio.run(main())
