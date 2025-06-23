# AgentNetwork - 多智能体通信框架

## 概述

AgentNetwork是一个先进的多智能体通信框架，旨在使用各种协议促进分布式智能体之间的无缝交互。该框架为构建复杂的多智能体系统提供了灵活、可扩展的架构，支持不同的通信协议、网络拓扑和监控功能。

## 核心特性

- **协议无关**: 支持多种通信协议(A2A、IoA等)
- **灵活拓扑**: 星型、网状和自定义网络拓扑
- **实时监控**: 全面的指标和健康监控
- **容错机制**: 内置故障检测和恢复机制
- **可扩展架构**: 支持水平扩展的模块化设计
- **可插拔适配器**: 轻松集成新协议和服务器类型

## 核心组件

1. **AgentNetwork** (`network.py`) - 中央协调器和拓扑管理器
2. **BaseAgent** (`base_agent.py`) - 双重角色智能体实现(服务器+客户端)
3. **指标系统** (`metrics.py`) - 基于Prometheus的监控和可观测性
4. **智能体适配器** (`agent_adapters/`) - 协议特定的客户端适配器
5. **服务器适配器** (`server_adapters/`) - 协议特定的服务器实现

## 架构图

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   基础智能体 A   │    │   基础智能体 B   │    │   基础智能体 C   │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │A2A服务器  │  │    │  │A2A服务器  │  │    │  │A2A服务器  │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │A2A客户端  │  │    │  │A2A客户端  │  │    │  │A2A客户端  │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  智能体网络     │
                    │    协调器       │
                    │                 │
                    │ ┌─────────────┐ │
                    │ │ 拓扑        │ │
                    │ │ 管理器      │ │
                    │ └─────────────┘ │
                    │ ┌─────────────┐ │
                    │ │ 健康        │ │
                    │ │ 监控器      │ │
                    │ └─────────────┘ │
                    │ ┌─────────────┐ │
                    │ │ 指标        │ │
                    │ │ 收集器      │ │
                    │ └─────────────┘ │
                    └─────────────────┘
```

## 快速开始

```python
import asyncio
from agent_network import AgentNetwork, BaseAgent

async def main():
    # 创建网络
    network = AgentNetwork()
    
    # 创建智能体
    agent1 = await BaseAgent.create_a2a(
        agent_id="agent-1",
        host="localhost", 
        port=8001,
        executor=your_executor
    )
    
    agent2 = await BaseAgent.create_a2a(
        agent_id="agent-2",
        host="localhost",
        port=8002, 
        executor=your_executor
    )
    
    # 注册智能体
    await network.register_agent(agent1)
    await network.register_agent(agent2)
    
    # 设置拓扑
    network.setup_star_topology("agent-1")
    
    # 发送消息
    response = await network.route_message(
        "agent-1", "agent-2",
        {"message": "来自agent-1的问候"}
    )
    
    print(f"响应: {response}")

asyncio.run(main())
```

## 详细文档

- [网络架构](docs/network_zh_CN.md)
- [基础智能体实现](docs/base_agent_zh_CN.md)  
- [指标系统](docs/metrics_zh_CN.md)
- [智能体适配器](docs/agent_adapters_zh_CN.md)
- [服务器适配器](docs/server_adapters_zh_CN.md)

## 语言版本

- [English](README.md)
- [中文文档](README_zh_CN.md) (当前)

## 许可证

MIT License

## 贡献指南

请阅读[CONTRIBUTING.md](CONTRIBUTING.md)了解我们的行为准则和提交拉取请求的流程。 