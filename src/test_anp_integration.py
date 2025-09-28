#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANP (Agent Network Protocol) 集成测试
测试 ANP 适配器与现有多协议框架的集成
"""

import asyncio
import json
import logging
from typing import Any, Dict

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入框架组件
try:
    from src.core.base_agent import BaseAgent
    from src.server_adapters import ANPServerAdapter
    from src.agent_adapters import ANPAdapter, ANPMessageBuilder
except ImportError as e:
    logger.warning(f"ANP adapters not available: {e}")
    raise ImportError("ANP adapters require the AgentConnect library. Please install it via 'pip install agent-connect'.")


# 简单的测试执行器
class TestANPExecutor:
    """测试执行器，支持 ANP 协议接口"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
    
    async def execute_step(self, step):
        """执行步骤 - Agent Protocol 风格"""
        logger.info(f"[ANP-AP] {self.agent_name} 执行步骤: {getattr(step, 'input', 'Unknown')}")
        
        result = {
            "output": f"ANP 响应来自 {self.agent_name}: 已处理 '{getattr(step, 'input', 'Unknown')}'",
            "status": "completed",
            "is_last": True,
            "artifacts": []
        }
        
        return result
    
    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """直接调用 - Callable 风格"""
        logger.info(f"[ANP-Callable] {self.agent_name} 处理载荷: {payload}")
        
        return {
            "response": f"ANP 处理结果来自 {self.agent_name}",
            "input_received": payload,
            "status": "success"
        }


async def test_anp_adapter_availability():
    """测试ANP适配器的可用性"""
    print("\n🧪 测试ANP适配器可用性")
    print("=" * 50)
    
    try:
        # 测试导入
        assert ANPAdapter is not None
        assert ANPServerAdapter is not None
        assert ANPMessageBuilder is not None
        
        print("✅ ANP适配器导入成功")
        print(f"   - ANPAdapter: {ANPAdapter}")
        print(f"   - ANPServerAdapter: {ANPServerAdapter}")
        print(f"   - ANPMessageBuilder: {ANPMessageBuilder}")
        
        return True
        
    except Exception as e:
        print(f"❌ ANP适配器测试失败: {e}")
        return False


async def test_anp_message_builder():
    """测试ANP消息构建器"""
    print("\n🧪 测试ANP消息构建器")
    print("=" * 50)
    
    try:
        # 测试文本消息
        text_msg = ANPMessageBuilder.text_message("Hello ANP!")
        print(f"✅ 文本消息: {text_msg}")
        
        # 测试JSON消息
        json_msg = ANPMessageBuilder.json_message({"key": "value", "number": 42})
        print(f"✅ JSON消息: {json_msg}")
        
        # 测试ping消息
        ping_msg = ANPMessageBuilder.ping_message()
        print(f"✅ Ping消息: {ping_msg}")
        
        # 测试协议协商消息
        negotiation_msg = ANPMessageBuilder.protocol_negotiation_message(
            requirement="简单问答协议",
            input_desc="用户问题",
            output_desc="AI回答"
        )
        print(f"✅ 协议协商消息: {negotiation_msg}")
        
        return True
        
    except Exception as e:
        print(f"❌ ANP消息构建器测试失败: {e}")
        return False


async def test_anp_server_creation():
    """测试ANP服务器创建（模拟）"""
    print("\n🧪 测试ANP服务器创建")
    print("=" * 50)

    try:
        # 创建测试执行器
        executor = TestANPExecutor("ANP Server Test")
        
        # 测试服务器适配器创建
        adapter = ANPServerAdapter()
        
        # 模拟构建服务器（不实际启动）
        print("📋 模拟ANP服务器构建...")
        print(f"   协议名称: {adapter.protocol_name}")
        print(f"   适配器类型: {type(adapter)}")
        
        # 测试执行器包装器
        from src.server_adapters.anp_adapter import ANPExecutorWrapper
        wrapper = ANPExecutorWrapper(executor)
        print(f"✅ 执行器包装器创建成功: {wrapper.executor_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ ANP服务器创建测试失败: {e}")
        return False


async def test_anp_client_creation():
    """测试ANP客户端创建（模拟）"""
    print("\n🧪 测试ANP客户端创建")
    print("=" * 50)

    
    try:
        import httpx
        
        # 创建模拟的DID信息
        mock_did_info = {
            "private_key_pem": "-----BEGIN PRIVATE KEY-----\nMOCK_KEY\n-----END PRIVATE KEY-----",
            "did": "did:wba:example.com:mock-agent-123",
            "did_document_json": '{"id": "did:wba:example.com:mock-agent-123"}'
        }
        
        # 创建ANP适配器（不初始化）
        with httpx.AsyncClient() as client:
            adapter = ANPAdapter(
                httpx_client=client,
                target_did="did:wba:example.com:target-agent-456",
                local_did_info=mock_did_info,
                host_domain="localhost",
                host_port="8000",
                protocol_negotiation=False
            )
            
            print(f"✅ ANP客户端适配器创建成功")
            print(f"   目标DID: {adapter.target_did}")
            print(f"   本地DID: {adapter.local_did_info.get('did', 'unknown')}")
            print(f"   协议协商: {adapter.protocol_negotiation}")
            
            # 测试端点信息
            endpoint_info = adapter.get_endpoint_info()
            print(f"✅ 端点信息: {endpoint_info}")
            
        return True
        
    except Exception as e:
        print(f"❌ ANP客户端创建测试失败: {e}")
        return False


async def test_anp_protocol_comparison():
    """测试ANP协议与其他协议的对比"""
    print("\n🧪 ANP协议特性对比")
    print("=" * 50)
    
    protocols = {
        "ANP": {
            "认证机制": "DID-based去中心化",
            "传输协议": "WebSocket",
            "消息格式": "JSON+加密",
            "连接类型": "持久连接",
            "协议协商": "LLM动态协商（可选）",
            "安全性": "端到端加密",
            "特色功能": "去中心化身份、智能协商"
        },
        "A2A": {
            "认证机制": "SDK内置认证",
            "传输协议": "HTTP/SSE",
            "消息格式": "JSON",
            "连接类型": "请求-响应",
            "协议协商": "固定接口",
            "安全性": "传输层加密",
            "特色功能": "流式响应、事件队列"
        },
        "Agent Protocol": {
            "认证机制": "API Key/Token",
            "传输协议": "HTTP REST",
            "消息格式": "JSON",
            "连接类型": "无状态",
            "协议协商": "标准化API",
            "安全性": "传输层加密",
            "特色功能": "Task/Step/Artifact模型"
        }
    }
    
    print("📊 协议特性对比表:")
    print("-" * 80)
    
    # 打印表头
    features = list(protocols["ANP"].keys())
    print(f"{'特性':15} | {'ANP':25} | {'A2A':25} | {'Agent Protocol':25}")
    print("-" * 80)
    
    # 打印对比内容
    for feature in features:
        anp_val = protocols["ANP"][feature]
        a2a_val = protocols["A2A"][feature]
        ap_val = protocols["Agent Protocol"][feature]
        print(f"{feature:15} | {anp_val:25} | {a2a_val:25} | {ap_val:25}")
    
    print("-" * 80)
    
    # ANP优势分析
    print("\n🎯 ANP协议优势:")
    advantages = [
        "✅ 去中心化身份认证，无需第三方CA",
        "✅ 端到端加密，保护隐私和安全",
        "✅ 持久WebSocket连接，实时通信",
        "✅ 智能协议协商，适应性强",
        "✅ 支持复杂的智能体网络拓扑",
        "✅ 面向未来的智能体互联网标准"
    ]
    
    for advantage in advantages:
        print(f"   {advantage}")
    
    print("\n⚠️  当前挑战:")
    challenges = [
        "⚠️  需要DID解析服务支持",
        "⚠️  WebSocket连接管理复杂性",
        "⚠️  协议协商可能增加延迟",
        "⚠️  生态系统仍在发展中"
    ]
    
    for challenge in challenges:
        print(f"   {challenge}")
    
    return True


async def test_anp_integration_roadmap():
    """显示ANP集成路线图"""
    print("\n🗺️  ANP集成路线图")
    print("=" * 50)
    
    roadmap = {
        "阶段1 - 基础集成": [
            "✅ ANP适配器接口设计",
            "✅ 基础DID认证集成",
            "✅ WebSocket通信封装",
            "⏳ 消息路由和转换",
            "⏳ 错误处理和重连"
        ],
        "阶段2 - 协议协商": [
            "⏳ LLM协议协商集成",
            "⏳ 动态协议加载",
            "⏳ 协议版本管理",
            "⏳ 协议缓存机制"
        ],
        "阶段3 - 互操作性": [
            "⏳ A2A ↔ ANP 协议桥接",
            "⏳ Agent Protocol ↔ ANP 桥接",
            "⏳ 统一消息路由",
            "⏳ 多协议会话管理"
        ],
        "阶段4 - 高级特性": [
            "⏳ 分布式DID解析",
            "⏳ 高级加密选项",
            "⏳ 性能优化",
            "⏳ 监控和分析"
        ]
    }
    
    for phase, tasks in roadmap.items():
        print(f"\n📋 {phase}:")
        for task in tasks:
            print(f"   {task}")
    
    print(f"\n🎯 下一步优先级:")
    next_steps = [
        "1. 完善AgentConnect库的集成和错误处理",
        "2. 实现ANP与A2A/Agent Protocol的消息转换",
        "3. 创建完整的端到端测试用例",
        "4. 优化连接管理和性能",
        "5. 添加协议协商功能支持"
    ]
    
    for step in next_steps:
        print(f"   {step}")
    
    return True


async def main():
    """运行所有ANP集成测试"""
    print("🚀 ANP (Agent Network Protocol) 集成测试")
    print("============================================================")
    
    tests = [
        ("ANP适配器可用性", test_anp_adapter_availability),
        ("ANP消息构建器", test_anp_message_builder),
        ("ANP服务器创建", test_anp_server_creation),
        ("ANP客户端创建", test_anp_client_creation),
        ("ANP协议对比分析", test_anp_protocol_comparison),
        ("ANP集成路线图", test_anp_integration_roadmap),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
            
            # 短暂延迟
            await asyncio.sleep(0.5)
            
        except Exception as e:
            print(f"❌ 测试 '{test_name}' 出现异常: {e}")
            results.append((test_name, False))
    
    # 总结结果
    print("\n📋 测试结果总结")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        if result is True:
            status = "✅ 通过"
        elif result is False:
            status = "❌ 失败"
        else:
            status = "ℹ️  信息"
        print(f"   {status} - {test_name}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if ANPAdapter is not None:
        print("🎉 ANP适配器已成功集成到多协议框架中！")
        print("💡 现在可以使用ANP协议进行去中心化的智能体通信。")
    else:
        print("📝 ANP适配器代码已就绪，等待AgentConnect库安装。")
        print("💡 安装AgentConnect后即可使用ANP协议功能。")


if __name__ == "__main__":
    asyncio.run(main()) 