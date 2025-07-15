#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多协议兼容性测试 - 测试 A2A 和 Agent Protocol 的集成
"""

import asyncio
import json
import logging
from typing import Any, Dict

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入框架组件
from src.base_agent import BaseAgent
from src.server_adapters import A2AServerAdapter, AgentProtocolServerAdapter

# 简单的测试执行器
class TestExecutor:
    """测试执行器，同时支持 A2A 和 Agent Protocol 接口"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
    
    # A2A SDK 原生接口
    async def execute(self, context, event_queue):
        """A2A SDK 原生执行接口"""
        message = getattr(context, 'message', 'Unknown message')
        logger.info(f"[A2A] {self.agent_name} 处理消息: {message}")
        
        # 发送响应事件
        try:
            from a2a.utils import new_agent_text_message
            response = new_agent_text_message(f"A2A 响应来自 {self.agent_name}: 已处理 '{message}'")
            await event_queue.put(response)
        except ImportError:
            # 如果没有 A2A SDK，创建简单响应
            response = {
                "type": "text",
                "content": f"A2A 响应来自 {self.agent_name}: 已处理 '{message}'"
            }
            await event_queue.put(response)
    
    # Agent Protocol 接口
    async def handle_task_creation(self, task):
        """处理 Agent Protocol 任务创建"""
        logger.info(f"[Agent Protocol] {self.agent_name} 创建任务: {task.task_id}")
        task.status = "ready"
    
    async def execute_step(self, step):
        """执行 Agent Protocol 步骤"""
        logger.info(f"[Agent Protocol] {self.agent_name} 执行步骤: {step.step_id}")
        
        result = {
            "output": f"Agent Protocol 响应来自 {self.agent_name}: 已处理 '{step.input}'",
            "status": "completed",
            "is_last": True,
            "artifacts": []
        }
        
        return result


async def test_a2a_to_a2a():
    """测试 A2A 智能体之间的通信"""
    print("\n🧪 测试 A2A 到 A2A 通信")
    print("=" * 50)
    
    try:
        # 创建两个 A2A 智能体
        agent1 = await BaseAgent.create_a2a(
            agent_id="a2a-agent-1",
            host="localhost",
            port=8001,
            executor=TestExecutor("A2A Agent 1")
        )
        
        agent2 = await BaseAgent.create_a2a(
            agent_id="a2a-agent-2", 
            host="localhost",
            port=8002,
            executor=TestExecutor("A2A Agent 2")
        )
        
        # 建立连接
        await agent1.add_connection(
            dst_id="a2a-agent-2",
            base_url="http://localhost:8002",
            protocol="a2a"
        )
        
        # 发送消息
        response = await agent1.send("a2a-agent-2", {
            "message": "Hello from A2A Agent 1!"
        })
        
        print(f"✅ A2A 到 A2A 通信成功")
        print(f"   响应: {response}")
        
        # 清理
        await agent1.stop()
        await agent2.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ A2A 到 A2A 通信失败: {e}")
        return False


async def test_agent_protocol_to_agent_protocol():
    """测试 Agent Protocol 智能体之间的通信"""
    print("\n🧪 测试 Agent Protocol 到 Agent Protocol 通信")
    print("=" * 50)
    
    try:
        # 创建两个 Agent Protocol 智能体
        agent1 = await BaseAgent.create_agent_protocol(
            agent_id="ap-agent-1",
            host="localhost", 
            port=8003,
            executor=TestExecutor("AP Agent 1")
        )
        
        agent2 = await BaseAgent.create_agent_protocol(
            agent_id="ap-agent-2",
            host="localhost",
            port=8004,
            executor=TestExecutor("AP Agent 2")
        )
        
        # 建立连接
        await agent1.add_connection(
            dst_id="ap-agent-2",
            base_url="http://localhost:8004",
            protocol="agent_protocol"
        )
        
        # 发送消息
        response = await agent1.send("ap-agent-2", {
            "message": "Hello from Agent Protocol Agent 1!"
        })
        
        print(f"✅ Agent Protocol 到 Agent Protocol 通信成功")
        print(f"   响应: {response}")
        
        # 清理
        await agent1.stop()
        await agent2.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ Agent Protocol 到 Agent Protocol 通信失败: {e}")
        return False


async def test_a2a_to_agent_protocol():
    """测试 A2A 到 Agent Protocol 跨协议通信"""
    print("\n🧪 测试 A2A 到 Agent Protocol 跨协议通信")
    print("=" * 50)
    
    try:
        # 创建 A2A 智能体
        a2a_agent = await BaseAgent.create_a2a(
            agent_id="a2a-client",
            host="localhost",
            port=8005,
            executor=TestExecutor("A2A Client")
        )
        
        # 创建 Agent Protocol 智能体
        ap_agent = await BaseAgent.create_agent_protocol(
            agent_id="ap-server",
            host="localhost",
            port=8006,
            executor=TestExecutor("AP Server")
        )
        
        # A2A 智能体连接到 Agent Protocol 智能体
        await a2a_agent.add_connection(
            dst_id="ap-server",
            base_url="http://localhost:8006",
            protocol="agent_protocol"  # 明确指定协议
        )
        
        # 发送消息
        response = await a2a_agent.send("ap-server", {
            "message": "Cross-protocol message from A2A to Agent Protocol!"
        })
        
        print(f"✅ A2A 到 Agent Protocol 跨协议通信成功")
        print(f"   响应: {response}")
        
        # 清理
        await a2a_agent.stop()
        await ap_agent.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ A2A 到 Agent Protocol 跨协议通信失败: {e}")
        return False


async def test_agent_protocol_to_a2a():
    """测试 Agent Protocol 到 A2A 跨协议通信"""
    print("\n🧪 测试 Agent Protocol 到 A2A 跨协议通信")
    print("=" * 50)
    
    try:
        # 创建 Agent Protocol 智能体
        ap_agent = await BaseAgent.create_agent_protocol(
            agent_id="ap-client",
            host="localhost",
            port=8007,
            executor=TestExecutor("AP Client")
        )
        
        # 创建 A2A 智能体
        a2a_agent = await BaseAgent.create_a2a(
            agent_id="a2a-server",
            host="localhost",
            port=8008,
            executor=TestExecutor("A2A Server")
        )
        
        # Agent Protocol 智能体连接到 A2A 智能体
        await ap_agent.add_connection(
            dst_id="a2a-server",
            base_url="http://localhost:8008",
            protocol="a2a"  # 明确指定协议
        )
        
        # 发送消息
        response = await ap_agent.send("a2a-server", {
            "message": "Cross-protocol message from Agent Protocol to A2A!"
        })
        
        print(f"✅ Agent Protocol 到 A2A 跨协议通信成功")
        print(f"   响应: {response}")
        
        # 清理
        await ap_agent.stop()
        await a2a_agent.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ Agent Protocol 到 A2A 跨协议通信失败: {e}")
        return False


async def test_auto_protocol_detection():
    """测试自动协议检测功能"""
    print("\n🧪 测试自动协议检测")
    print("=" * 50)
    
    try:
        # 创建不同协议的智能体
        a2a_agent = await BaseAgent.create_a2a(
            agent_id="a2a-auto",
            host="localhost",
            port=8009,
            executor=TestExecutor("A2A Auto")
        )
        
        ap_agent = await BaseAgent.create_agent_protocol(
            agent_id="ap-auto",
            host="localhost",
            port=8010,
            executor=TestExecutor("AP Auto")
        )
        
        # 创建客户端智能体
        client = await BaseAgent.create_a2a(
            agent_id="auto-client",
            host="localhost",
            port=8011,
            executor=TestExecutor("Auto Client")
        )
        
        # 自动检测 A2A 协议
        await client.add_connection(
            dst_id="a2a-auto",
            base_url="http://localhost:8009",
            protocol="auto"  # 自动检测
        )
        
        # 自动检测 Agent Protocol 协议
        await client.add_connection(
            dst_id="ap-auto",
            base_url="http://localhost:8010",
            protocol="auto"  # 自动检测
        )
        
        # 测试连接
        response1 = await client.send("a2a-auto", {"message": "Auto-detected A2A!"})
        response2 = await client.send("ap-auto", {"message": "Auto-detected Agent Protocol!"})
        
        print(f"✅ 自动协议检测成功")
        print(f"   A2A 响应: {response1}")
        print(f"   Agent Protocol 响应: {response2}")
        
        # 显示连接信息
        connections = client.get_connection_info()
        print("\n📊 连接信息:")
        for dst_id, info in connections.items():
            print(f"   {dst_id}: {info['protocol']} 协议")
        
        # 清理
        await client.stop()
        await a2a_agent.stop()
        await ap_agent.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ 自动协议检测失败: {e}")
        return False


async def main():
    """运行所有测试"""
    print("🚀 多协议兼容性测试")
    print("============================================================")
    
    tests = [
        ("A2A 到 A2A", test_a2a_to_a2a),
        ("Agent Protocol 到 Agent Protocol", test_agent_protocol_to_agent_protocol),
        ("A2A 到 Agent Protocol", test_a2a_to_agent_protocol),
        ("Agent Protocol 到 A2A", test_agent_protocol_to_a2a),
        ("自动协议检测", test_auto_protocol_detection),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
            
            # 短暂延迟以避免端口冲突
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"❌ 测试 '{test_name}' 出现异常: {e}")
            results.append((test_name, False))
    
    # 总结结果
    print("\n📋 测试结果总结")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {status} - {test_name}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！两种协议已成功兼容。")
    else:
        print("⚠️  部分测试失败，请检查错误信息。")


if __name__ == "__main__":
    asyncio.run(main())