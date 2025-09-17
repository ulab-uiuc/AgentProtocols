# GAIA Multi-Agent Framework

GAIA (General Agent Interaction Architecture) 是一个灵活的多智能体框架，支持多种通信协议和智能体互操作。

## 🌟 核心特性

- **🔌 多协议支持**: 支持多种通信协议（Dummy、Agent Protocol、AP、ACP等）
- **🤖 智能体抽象**: 统一的智能体接口，支持工具调用和内存管理
- **🌐 网络拓扑**: 灵活的网络拓扑和消息路由机制
- **📊 性能监控**: 实时性能指标和执行状态监控
- **🔄 工作流引擎**: 支持复杂的多智能体工作流编排
- **📝 记忆管理**: 智能体级别和网络级别的内存池管理

## 🏗️ 架构设计

```
gaia/
├── core/                    # 核心框架代码
│   ├── agent.py            # 智能体基类和抽象接口
│   ├── network.py          # 网络基类和消息传递
│   ├── schema.py           # 数据模型和类型定义
│   ├── llm.py             # LLM接口和管理
│   ├── planner.py         # 任务规划和执行
│   └── prompt.py          # 提示词管理
├── protocol_backends/       # 协议后端实现
│   ├── protocol_factory.py # 协议工厂和管理器
│   ├── dummy/             # 虚拟协议实现
│   ├── ap/               # Agent Protocol实现
│   └── [custom]/         # 自定义协议扩展
├── config/                 # 配置文件
│   ├── default.yaml       # 默认配置
│   └── [protocol].yaml   # 协议特定配置
├── tools/                 # 工具集合
├── workspaces/           # 工作空间和输出
└── docs/                 # 文档
```

### 核心组件

#### 1. 智能体架构 (Agent Architecture)

**MeshAgent** 是所有智能体的基类，提供：

- **通信接口**: `send_msg()` 和 `recv_msg()` 方法
- **工具调用**: 基于 ReAct 模式的工具执行能力
- **内存管理**: 消息历史和上下文管理
- **生命周期**: 启动、运行、停止的完整生命周期管理

```python
class MeshAgent(ToolCallAgent):
    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """发送消息到目标智能体"""
        
    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """接收消息，支持超时设置"""
        
    async def process_messages(self):
        """处理收到的消息并生成响应"""
        
    def get_memory_messages(self) -> List[Message]:
        """获取内存中的所有消息"""
```

#### 2. 网络架构 (Network Architecture)

**MeshNetwork** 是网络通信的基类，提供：

- **消息投递**: `deliver()` 方法用于点对点消息传递
- **网络管理**: 智能体注册、发现和生命周期管理
- **工作流执行**: 支持复杂的多智能体工作流编排

```python
class MeshNetwork(ABC):
    @abstractmethod
    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """投递消息到指定智能体"""
        
    async def execute_workflow(self, config: Dict[str, Any], initial_task: str) -> str:
        """执行多智能体工作流"""
        
    def register_agent(self, agent: MeshAgent) -> None:
        """注册智能体到网络中"""
```

#### 3. 协议系统 (Protocol System)

**协议工厂模式** 支持多种通信协议：

```python
class ProtocolFactory:
    def create_multi_agent_system(self, agents_config, task_id, protocol):
        """创建完整的多智能体系统"""
        
    def get_available_protocols(self) -> List[str]:
        """获取可用的协议列表"""
```

## 🚀 快速开始

### 1. 运行测试

```bash
# 测试默认协议 (dummy)
cd /root/Multiagent-Protocol/script/gaia/protocol_backends
python test_protocol.py

# 测试特定协议
python test_protocol.py --protocol dummy

# 列出所有可用协议
python test_protocol.py --list
```

### 2. 基本使用

```python
from protocol_backends.protocol_factory import protocol_factory

# 创建智能体配置
agents_config = [
    {'id': 0, 'name': 'Agent_0', 'tool': 'create_chat_completion', 'port': 9000},
    {'id': 1, 'name': 'Agent_1', 'tool': 'create_chat_completion', 'port': 9001},
]

# 创建多智能体系统
network, agents = protocol_factory.create_multi_agent_system(
    agents_config, 
    task_id="demo_001", 
    protocol="dummy"
)

# 启动网络
await network.start()

# 发送消息
await network.deliver(1, {
    "type": "task", 
    "content": "Process this data"
})

# 停止网络
await network.stop()
```

### 3. 工作流执行

```python
# 定义工作流配置
workflow_config = {
    "workflow": {
        "start_agent": 0,
        "execution_pattern": "sequential",
        "message_flow": [
            {"from": 0, "to": [1], "message_type": "task"},
            {"from": 1, "to": [2], "message_type": "result"},
            {"from": 2, "to": "final", "message_type": "final_result"}
        ]
    },
    "agents": agents_config
}

# 执行工作流
result = await network.execute_workflow(workflow_config, "Initial task")
```

## 📊 支持的协议

| 协议名称 | 描述 | 状态 | 使用场景 |
|---------|------|------|----------|
| dummy | 虚拟协议，用于测试 | ✅ 完成 | 开发测试 |
| ap | Agent Protocol HTTP API | 🚧 开发中 | Web服务 |
| acp | Agent Communication Protocol | 🚧 开发中 | 企业级应用 |
| a2a | Agent-to-Agent Direct | 🚧 开发中 | 高性能通信 |
...
## 🛠️ 开发自定义协议

查看 [协议开发指南](protocol_backends/README.md) 了解如何开发和集成自定义协议。

## 📈 性能监控

框架提供丰富的性能指标：

- **网络指标**: 包传输量、字节数、延迟等
- **智能体指标**: Token使用量、处理时间、成功率等
- **工作流指标**: 步骤完成率、总执行时间等

## 🔧 配置

参考 `config/default.yaml` 了解完整的配置选项，包括：

- 网络设置（端口范围、超时等）
- 智能体配置（Token限制、优先级等）
- LLM模型配置
- 性能和评估参数

## 📚 文档

- [智能体开发指南](docs/agent_development.md)
- [网络架构说明](docs/network_architecture.md)
- [协议开发指南](protocol_backends/README.md)
- [工作流配置](docs/workflow_configuration.md)

## 🤝 贡献

欢迎贡献代码！请参考 [贡献指南](CONTRIBUTING.md) 了解详细信息。

## 📄 许可证

本项目采用 MIT 许可证。详情请参考 [LICENSE](LICENSE) 文件。

---

**GAIA Framework** - 让多智能体协作更简单、更高效！ 🚀
