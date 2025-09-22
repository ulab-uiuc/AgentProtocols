#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试协调器与RG的集成
"""

import asyncio
import subprocess
import sys
import os
import time
import httpx
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from script.safety_tech.core.backend_api import register_backend, spawn_backend, health_backend

async def wait_http_ok(url: str, timeout: float = 10.0):
    """等待HTTP端点可用"""
    start_time = time.time()
    last_err = None
    
    while time.time() - start_time < timeout:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=5.0)
                if response.status_code == 200:
                    return
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.5)
    
    raise RuntimeError(f"Timeout waiting {url}: {last_err}")

async def test_coordinator_rg_integration():
    """测试协调器与RG的集成"""
    
    # 端口配置
    rg_port = 8001
    coord_port = 8888
    a_port = 9002
    b_port = 9003
    conv_id = "test_conv_integration"
    
    procs = []
    
    try:
        print("🚀 启动RG...")
        # 启动RG
        rg_proc = subprocess.Popen([
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, '{PROJECT_ROOT}'); "
            "from script.safety_tech.core.registration_gateway import RegistrationGateway; "
            f"RegistrationGateway({{'session_timeout':3600,'max_observers':5,'require_observer_proof':True}}).run(host='127.0.0.1', port={rg_port})"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs.append(rg_proc)
        print(f"RG PID: {rg_proc.pid}")
        
        await wait_http_ok(f"http://127.0.0.1:{rg_port}/health")
        print("✅ RG启动成功")
        
        print("🚀 启动协调器...")
        # 启动协调器
        coord_code = f"""
import sys
sys.path.insert(0, '{PROJECT_ROOT}')
from script.safety_tech.core.rg_coordinator import RGCoordinator
import asyncio
import logging

# 启用调试日志
logging.basicConfig(level=logging.INFO)

async def run():
    coord = RGCoordinator({{
        'rg_endpoint': 'http://127.0.0.1:{rg_port}',
        'conversation_id': '{conv_id}',
        'coordinator_port': {coord_port}
    }})
    await coord.start()
    print(f"Coordinator started on port {coord_port}")
    
    # 等待一段时间让目录轮询工作
    await asyncio.sleep(5)
    
    # 打印参与者信息
    print(f"Participants: {{len(coord.participants)}}")
    for agent_id, participant in coord.participants.items():
        print(f"  - {{agent_id}}: {{participant.role}} @ {{participant.endpoint}}")
    
    # 保持运行
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("Coordinator shutting down")

if __name__ == "__main__":
    asyncio.run(run())
"""
        coord_proc = subprocess.Popen([
            sys.executable, "-c", coord_code
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs.append(coord_proc)
        print(f"协调器 PID: {coord_proc.pid}")
        
        await wait_http_ok(f"http://127.0.0.1:{coord_port}/health")
        print("✅ 协调器启动成功")
        
        print("🚀 启动Agent...")
        # 启动Agent
        await spawn_backend('acp', 'doctor_a', a_port, coord_endpoint=f"http://127.0.0.1:{coord_port}")
        await spawn_backend('acp', 'doctor_b', b_port, coord_endpoint=f"http://127.0.0.1:{coord_port}")
        await health_backend('acp', f"http://127.0.0.1:{a_port}")
        await health_backend('acp', f"http://127.0.0.1:{b_port}")
        print("✅ Agent启动成功")
        
        print("🔐 注册Agent...")
        # 注册Agent
        resp_a = await register_backend('acp', 'ACP_Doctor_A', f'http://127.0.0.1:{a_port}', conv_id, 'doctor_a', rg_endpoint=f'http://127.0.0.1:{rg_port}')
        resp_b = await register_backend('acp', 'ACP_Doctor_B', f'http://127.0.0.1:{b_port}', conv_id, 'doctor_b', rg_endpoint=f'http://127.0.0.1:{rg_port}')
        
        print(f"Doctor A注册: {resp_a.get('status', 'unknown')}")
        print(f"Doctor B注册: {resp_b.get('status', 'unknown')}")
        
        # 等待协调器轮询更新
        print("⏳ 等待协调器轮询更新...")
        await asyncio.sleep(8)
        
        print("🔍 检查RG目录...")
        # 检查RG目录
        async with httpx.AsyncClient() as client:
            directory_resp = await client.get(f"http://127.0.0.1:{rg_port}/directory", 
                                            params={"conversation_id": conv_id}, timeout=5.0)
            if directory_resp.status_code == 200:
                directory = directory_resp.json()
                print(f"RG目录: {directory['total_participants']} 个参与者")
                for p in directory['participants']:
                    print(f"  - {p['agent_id']}: {p['role']} @ {p['endpoint']}")
            else:
                print(f"RG目录查询失败: {directory_resp.status_code}")
        
        print("🔍 检查协调器状态...")
        # 检查协调器状态
        async with httpx.AsyncClient() as client:
            coord_health = await client.get(f"http://127.0.0.1:{coord_port}/health", timeout=5.0)
            print(f"协调器健康: {coord_health.status_code}")
            
        print("📨 测试消息路由...")
        # 测试消息路由
        test_payload = {
            "sender_id": "ACP_Doctor_A",
            "receiver_id": "ACP_Doctor_B", 
            "content": "Hello from integration test",
            "correlation_id": "test_corr_123"
        }
        
        async with httpx.AsyncClient() as client:
            route_resp = await client.post(f"http://127.0.0.1:{coord_port}/route_message", 
                                         json=test_payload, timeout=10.0)
            print(f"路由测试: {route_resp.status_code}")
            if route_resp.status_code == 200:
                print(f"路由响应: {route_resp.json()}")
                print("✅ 集成测试成功！")
            else:
                print(f"路由失败: {route_resp.text}")
                print("❌ 集成测试失败")
        
        print("🎉 测试完成，等待5秒后退出...")
        await asyncio.sleep(5)
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("🧹 清理进程...")
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

if __name__ == "__main__":
    asyncio.run(test_coordinator_rg_integration())
