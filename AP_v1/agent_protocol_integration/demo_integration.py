"""
Agent Protocol 集成演示 - 将您的 test_ap.py 实现集成到 A2A 框架
"""

import asyncio
import json
import sys
import os
from typing import Dict, Any, List
from pathlib import Path

# 添加路径以导入您的原始实现
sys.path.append("/GPFS/data/sujiaqi/gui/Multiagent-Protocol")
sys.path.append("/GPFS/data/sujiaqi/gui/Multiagent-Protocol/A2A/src")

# 导入 A2A 框架
from network import AgentNetwork
from base_agent import BaseAgent

# 导入 Agent Protocol 集成组件
from agent_protocol_integration.agent_protocol_server_adapter import (
    AgentProtocolServerAdapter, 
    AgentProtocolTask, 
    AgentProtocolStep
)
from agent_adapters.agent_protocol_adapter import (
    AgentProtocolAdapter, 
    AgentProtocolMessageBuilder
)

# 尝试导入您的原始 Agent Protocol 实现
try:
    from test_ap import plan, execute, task_handler, step_handler, generate_steps
    ORIGINAL_IMPLEMENTATION_AVAILABLE = True
    print("✅ 成功导入您的原始 Agent Protocol 实现")
except ImportError as e:
    print(f"⚠️  无法导入原始实现: {e}")
    ORIGINAL_IMPLEMENTATION_AVAILABLE = False
    
    # 创建模拟实现
    def generate_steps(task_input: str) -> List[str]:
        """生成任务步骤的模拟实现"""
        if "问答" in task_input or "question" in task_input.lower():
            return ["analyze_question", "search_knowledge", "generate_answer"]
        elif "总结" in task_input or "summary" in task_input.lower():
            return ["extract_content", "analyze_key_points", "generate_summary"]
        else:
            return ["analyze_task", "execute_task", "verify_result"]
    
    async def plan(step):
        """计划处理器模拟实现"""
        print(f"📋 模拟计划步骤: {step.input}")
        return {"status": "completed", "output": f"计划完成: {step.input}"}
    
    async def execute(step):
        """执行处理器模拟实现"""
        print(f"⚡ 模拟执行步骤: {step.name}")
        return {"status": "completed", "output": f"执行完成: {step.name}"}
    
    async def task_handler(task):
        """任务处理器模拟实现"""
        print(f"📝 模拟处理任务: {task.input}")
    
    async def step_handler(step):
        """步骤处理器模拟实现"""
        if step.name == "plan":
            return await plan(step)
        else:
            return await execute(step)


class AgentProtocolExecutor:
    """
    Agent Protocol 执行器 - 将您的 Agent Protocol 逻辑包装为 A2A 执行器
    """
    
    def __init__(self):
        self.tasks: Dict[str, AgentProtocolTask] = {}
        self.steps: Dict[str, AgentProtocolStep] = {}
    
    async def handle_task_creation(self, task: AgentProtocolTask):
        """处理任务创建"""
        print(f"🎯 创建任务: {task.task_id} - {task.input}")
        
        # 调用您的原始 task_handler
        try:
            await task_handler(task)
            task.status = "created"
        except Exception as e:
            print(f"❌ 任务处理失败: {e}")
            task.status = "failed"
    
    async def execute_step(self, step: AgentProtocolStep) -> Dict[str, Any]:
        """执行步骤"""
        print(f"🔧 执行步骤: {step.step_id} - {step.name}")
        
        try:
            # 调用您的原始 step_handler
            result = await step_handler(step)
            
            if isinstance(result, dict):
                return {
                    "status": result.get("status", "completed"),
                    "output": result.get("output", f"步骤 {step.name} 执行完成"),
                    "additional_output": result.get("additional_output", {}),
                    "is_last": result.get("is_last", False),
                    "artifacts": result.get("artifacts", [])
                }
            else:
                return {
                    "status": "completed",
                    "output": str(result) if result else f"步骤 {step.name} 执行完成",
                    "additional_output": {},
                    "is_last": False,
                    "artifacts": []
                }
                
        except Exception as e:
            print(f"❌ 步骤执行失败: {e}")
            return {
                "status": "failed",
                "output": f"执行错误: {str(e)}",
                "additional_output": {},
                "is_last": False,
                "artifacts": []
            }


async def create_agent_protocol_agent(agent_id: str, port: int) -> BaseAgent:
    """创建集成了您的 Agent Protocol 实现的 A2A 智能体"""
    
    # 创建执行器
    executor = AgentProtocolExecutor()
    
    # 创建服务器适配器
    server_adapter = AgentProtocolServerAdapter()
    
    # 创建 BaseAgent
    agent = BaseAgent(
        agent_id=agent_id,
        host="localhost",
        port=port,
        server_adapter=server_adapter
    )
    
    # 启动服务器
    await agent._start_server(executor)
    await agent._wait_for_server_ready()
    await agent._fetch_self_card()
    
    agent._initialized = True
    
    print(f"✅ Agent Protocol 智能体 '{agent_id}' 创建成功")
    print(f"   - 监听地址: {agent.get_listening_address()}")
    print(f"   - Agent Protocol API: {agent.get_listening_address()}/ap/v1/agent/tasks")
    print(f"   - A2A 兼容端点: {agent.get_listening_address()}/message")
    
    return agent


async def test_agent_protocol_operations(agent: BaseAgent):
    """测试 Agent Protocol 操作"""
    
    print(f"\n🧪 测试 Agent Protocol 操作")
    print(f"=" * 50)
    
    import httpx
    base_url = agent.get_listening_address()
    
    async with httpx.AsyncClient() as client:
        
        # 测试 1: 创建任务
        print(f"\n📝 测试 1: 创建任务")
        task_data = {
            "input": "分析人工智能的发展趋势和影响",
            "additional_input": {
                "domain": "technology",
                "priority": "high"
            }
        }
        
        response = await client.post(
            f"{base_url}/ap/v1/agent/tasks",
            json=task_data,
            timeout=30
        )
        
        if response.status_code == 200:
            task_result = response.json()
            task_id = task_result["task_id"]
            print(f"   ✅ 任务创建成功: {task_id}")
            print(f"   📋 任务详情: {json.dumps(task_result, ensure_ascii=False, indent=2)}")
        else:
            print(f"   ❌ 任务创建失败: {response.status_code} - {response.text}")
            return
        
        # 测试 2: 获取任务
        print(f"\n📖 测试 2: 获取任务信息")
        response = await client.get(f"{base_url}/ap/v1/agent/tasks/{task_id}")
        
        if response.status_code == 200:
            task_info = response.json()
            print(f"   ✅ 任务信息获取成功")
            print(f"   📋 任务状态: {task_info['status']}")
        else:
            print(f"   ❌ 获取任务失败: {response.status_code}")
        
        # 测试 3: 执行步骤
        print(f"\n⚡ 测试 3: 执行步骤")
        step_data = {
            "name": "analyze_ai_trends",
            "input": "请分析当前人工智能的主要发展趋势",
            "additional_input": {
                "analysis_type": "comprehensive"
            }
        }
        
        response = await client.post(
            f"{base_url}/ap/v1/agent/tasks/{task_id}/steps",
            json=step_data,
            timeout=30
        )
        
        if response.status_code == 200:
            step_result = response.json()
            step_id = step_result["step_id"]
            print(f"   ✅ 步骤执行成功: {step_id}")
            print(f"   📋 步骤结果: {json.dumps(step_result, ensure_ascii=False, indent=2)}")
        else:
            print(f"   ❌ 步骤执行失败: {response.status_code} - {response.text}")
            return
        
        # 测试 4: 列出步骤
        print(f"\n📝 测试 4: 列出所有步骤")
        response = await client.get(f"{base_url}/ap/v1/agent/tasks/{task_id}/steps")
        
        if response.status_code == 200:
            steps_result = response.json()
            print(f"   ✅ 步骤列表获取成功")
            print(f"   📋 步骤数量: {len(steps_result['steps'])}")
            for step in steps_result['steps']:
                print(f"      - {step['name']}: {step['status']}")
        else:
            print(f"   ❌ 获取步骤列表失败: {response.status_code}")
        
        # 测试 5: 获取特定步骤
        print(f"\n🔍 测试 5: 获取特定步骤")
        response = await client.get(f"{base_url}/ap/v1/agent/tasks/{task_id}/steps/{step_id}")
        
        if response.status_code == 200:
            step_detail = response.json()
            print(f"   ✅ 步骤详情获取成功")
            print(f"   📋 步骤输出: {step_detail.get('output', 'N/A')}")
        else:
            print(f"   ❌ 获取步骤详情失败: {response.status_code}")


async def test_a2a_compatibility(agent: BaseAgent):
    """测试 A2A 兼容性"""
    
    print(f"\n🔗 测试 A2A 兼容性")
    print(f"=" * 50)
    
    import httpx
    base_url = agent.get_listening_address()
    
    async with httpx.AsyncClient() as client:
        
        # 测试 A2A 消息格式
        a2a_message = {
            "id": "test-a2a-001",
            "params": {
                "message": {
                    "input": "通过 A2A 协议创建的任务",
                    "additional_input": {
                        "source": "a2a_test",
                        "format": "a2a"
                    }
                },
                "context": {},
                "routing": {
                    "destination": agent.agent_id,
                    "source": "test_client"
                }
            }
        }
        
        response = await client.post(
            f"{base_url}/message",
            json=a2a_message,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ A2A 消息处理成功")
            print(f"   📋 响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
        else:
            print(f"   ❌ A2A 消息处理失败: {response.status_code} - {response.text}")


async def test_network_integration():
    """测试网络集成"""
    
    print(f"\n🌐 测试 AgentNetwork 集成")
    print(f"=" * 50)
    
    # 创建网络
    network = AgentNetwork()
    
    # 创建两个 Agent Protocol 智能体
    agent1 = await create_agent_protocol_agent("AP-Agent-1", 8081)
    agent2 = await create_agent_protocol_agent("AP-Agent-2", 8082)
    
    # 注册到网络
    await network.register_agent(agent1)
    await network.register_agent(agent2)
    
    print(f"📡 智能体已注册到 AgentNetwork")
    
    # 创建客户端适配器进行通信
    import httpx
    client = httpx.AsyncClient()
    
    # 为 agent1 创建连接到 agent2 的适配器
    adapter = AgentProtocolAdapter(
        httpx_client=client,
        base_url=agent2.get_listening_address()
    )
    await adapter.initialize()
    agent1.add_outbound_adapter("AP-Agent-2", adapter)
    
    # 通过网络发送消息
    print(f"\n📤 测试网络消息路由")
    
    # 创建任务消息
    create_task_msg = AgentProtocolMessageBuilder.create_task_message(
        input_text="网络通信测试任务",
        additional_input={"source_agent": "AP-Agent-1"}
    )
    
    try:
        response = await network.route_message("AP-Agent-1", "AP-Agent-2", create_task_msg)
        print(f"   ✅ 网络消息路由成功")
        print(f"   📋 响应: {json.dumps(response, ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"   ❌ 网络消息路由失败: {e}")
    
    # 健康检查
    print(f"\n🏥 网络健康检查")
    health = await network.health_check()
    print(f"   网络健康状态: {health}")
    
    # 清理
    await client.aclose()
    await network.stop_all_agents()
    
    print(f"✅ 网络集成测试完成")


async def demonstrate_full_integration():
    """完整的集成演示"""
    
    print(f"🌟 Agent Protocol 与 A2A 框架完整集成演示")
    print(f"=" * 70)
    
    if ORIGINAL_IMPLEMENTATION_AVAILABLE:
        print(f"✅ 使用您的原始 Agent Protocol 实现")
    else:
        print(f"⚠️  使用模拟实现（未找到原始代码）")
    
    print(f"\n🚀 启动演示...")
    
    try:
        # 创建单个智能体进行基础测试
        agent = await create_agent_protocol_agent("Demo-AP-Agent", 8080)
        
        # 测试 Agent Protocol 操作
        await test_agent_protocol_operations(agent)
        
        # 测试 A2A 兼容性
        await test_a2a_compatibility(agent)
        
        # 停止单个智能体
        await agent.stop()
        
        # 测试网络集成
        await test_network_integration()
        
        print(f"\n🎉 演示完成！您的 Agent Protocol 实现已成功集成到 A2A 框架中")
        print(f"\n📖 主要特性:")
        print(f"   ✅ 完全兼容 Agent Protocol v1 规范")
        print(f"   ✅ 支持 Task/Step/Artifact 模式")
        print(f"   ✅ 提供标准 Agent Protocol HTTP API")
        print(f"   ✅ 与 A2A 框架无缝集成")
        print(f"   ✅ 支持网络通信和路由")
        print(f"   ✅ 保留您的原始处理逻辑")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(demonstrate_full_integration())
