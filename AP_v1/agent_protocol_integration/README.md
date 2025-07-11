# Agent Protocol 适配器 - A2A框架集成

## 概述

本项目为A2A（Agent-to-Agent）多智能体通信框架开发了完整的Agent Protocol适配器，使您能够：

- 🔌 **无缝集成**: 将现有的Agent Protocol实现集成到A2A框架中
- 📋 **标准兼容**: 完全支持Agent Protocol v1规范 (Task/Step/Artifact模式)
- 🌐 **网络通信**: 获得A2A框架的分布式智能体通信能力
- 🔄 **双向兼容**: 同时支持Agent Protocol和A2A协议
- 🛠️ **易于使用**: 提供简单的API和辅助工具

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    A2A Framework                            │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   AgentNetwork  │◄──►│   BaseAgent     │                │
│  │                 │    │   (Enhanced)    │                │
│  └─────────────────┘    └─────────────────┘                │
│                                │                            │
│                                ▼                            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │         Agent Protocol Integration                      ││
│  │                                                         ││
│  │  ┌──────────────────┐    ┌──────────────────┐          ││
│  │  │Server Adapter    │    │Client Adapter    │          ││
│  │  │- AP v1 Endpoints │    │- HTTP Client     │          ││
│  │  │- Task Management │    │- Message Builder │          ││
│  │  │- Step Execution  │    │- Error Handling  │          ││
│  │  └──────────────────┘    └──────────────────┘          ││
│  └─────────────────────────────────────────────────────────┘│
│                                │                            │
│                                ▼                            │
│           您的 Agent Protocol 实现                          │
│        (plan, execute, task_handler, step_handler)         │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. Agent Protocol Client Adapter (`agent_protocol_adapter.py`)

**功能**: 作为客户端连接到其他Agent Protocol智能体

**主要特性**:
- ✅ 支持所有Agent Protocol v1端点
- ✅ 完整的Task/Step/Artifact操作
- ✅ 错误处理和超时管理
- ✅ 消息构建辅助工具

**使用示例**:
```python
import httpx
from agent_adapters.agent_protocol_adapter import AgentProtocolAdapter, AgentProtocolMessageBuilder

# 创建客户端适配器
client = httpx.AsyncClient()
adapter = AgentProtocolAdapter(
    httpx_client=client,
    base_url="http://target-agent:8080"
)
await adapter.initialize()

# 创建任务
task_msg = AgentProtocolMessageBuilder.create_task_message(
    input_text="分析数据趋势",
    additional_input={"category": "analysis"}
)
response = await adapter.send_message("target-agent", task_msg)
```

### 2. Agent Protocol Server Adapter (`agent_protocol_server_adapter.py`)

**功能**: 将A2A智能体转换为Agent Protocol兼容的服务器

**支持的端点**:
- `POST /ap/v1/agent/tasks` - 创建任务
- `GET /ap/v1/agent/tasks/{task_id}` - 获取任务
- `POST /ap/v1/agent/tasks/{task_id}/steps` - 执行步骤
- `GET /ap/v1/agent/tasks/{task_id}/steps` - 列出步骤
- `GET /ap/v1/agent/tasks/{task_id}/steps/{step_id}` - 获取步骤
- `GET /ap/v1/agent/tasks/{task_id}/artifacts` - 列出工件
- `GET /ap/v1/agent/tasks/{task_id}/artifacts/{artifact_id}` - 下载工件

**使用示例**:
```python
from agent_protocol_integration.agent_protocol_server_adapter import AgentProtocolServerAdapter
from base_agent import BaseAgent

# 创建执行器
class MyExecutor:
    async def handle_task_creation(self, task):
        # 您的任务处理逻辑
        pass
    
    async def execute_step(self, step):
        # 您的步骤执行逻辑
        return {"status": "completed", "output": "步骤完成"}

# 创建智能体
executor = MyExecutor()
server_adapter = AgentProtocolServerAdapter()

agent = BaseAgent(
    agent_id="my-ap-agent",
    host="localhost",
    port=8080,
    server_adapter=server_adapter
)

await agent._start_server(executor)
```

## 快速开始

### 1. 安装依赖

```bash
pip install httpx starlette uvicorn
```

### 2. 运行基础测试

```bash
cd /GPFS/data/sujiaqi/gui/Multiagent-Protocol/A2A/src/agent_protocol_integration
python test_integration.py --test simple
```

### 3. 运行完整演示

```bash
python test_integration.py --test full
```

### 4. 集成您的实现

```python
# 导入您的原始实现
from test_ap import plan, execute, task_handler, step_handler

# 创建执行器包装器
class YourAgentProtocolExecutor:
    async def handle_task_creation(self, task):
        await task_handler(task)
    
    async def execute_step(self, step):
        result = await step_handler(step)
        return {
            "status": "completed",
            "output": str(result),
            "is_last": False
        }

# 创建智能体
executor = YourAgentProtocolExecutor()
agent = await create_agent_protocol_agent("your-agent", 8080, executor)
```

## API 参考

### Client Adapter API

#### AgentProtocolAdapter

```python
class AgentProtocolAdapter(BaseProtocolAdapter):
    async def send_message(dst_id: str, payload: Dict[str, Any]) -> Any
    async def health_check() -> bool
    def get_agent_card() -> Dict[str, Any]
    def get_endpoint_info() -> Dict[str, Any]
```

#### 消息类型

| 类型 | 描述 | 必需参数 |
|------|------|----------|
| `create_task` | 创建新任务 | `input` |
| `execute_step` | 执行步骤 | `task_id` |
| `get_task` | 获取任务信息 | `task_id` |
| `get_steps` | 获取步骤列表 | `task_id` |
| `get_step` | 获取特定步骤 | `task_id`, `step_id` |
| `get_artifacts` | 获取工件列表 | `task_id` |
| `get_artifact` | 下载工件 | `task_id`, `artifact_id` |

#### AgentProtocolMessageBuilder

```python
# 创建任务消息
task_msg = AgentProtocolMessageBuilder.create_task_message(
    input_text="任务描述",
    additional_input={"key": "value"}
)

# 执行步骤消息
step_msg = AgentProtocolMessageBuilder.execute_step_message(
    task_id="task-uuid",
    input_text="步骤输入"
)

# 获取任务消息
get_msg = AgentProtocolMessageBuilder.get_task_message("task-uuid")
```

### Server Adapter API

#### AgentProtocolServerAdapter

```python
class AgentProtocolServerAdapter(BaseServerAdapter):
    def build_app(agent_card: Dict[str, Any], executor: Any) -> Starlette
    def get_default_agent_card(agent_id: str, host: str, port: int) -> Dict[str, Any]
```

#### 执行器接口

```python
class YourExecutor:
    async def handle_task_creation(self, task: AgentProtocolTask):
        """处理任务创建"""
        pass
    
    async def execute_step(self, step: AgentProtocolStep) -> Dict[str, Any]:
        """执行步骤，返回结果字典"""
        return {
            "status": "completed",  # "created", "completed", "failed"
            "output": "步骤输出",
            "additional_output": {},
            "is_last": False,
            "artifacts": []
        }
```

## 网络集成示例

### 创建Agent Protocol智能体网络

```python
import asyncio
from network import AgentNetwork
from agent_protocol_integration.demo_integration import create_agent_protocol_agent

async def create_ap_network():
    # 创建网络
    network = AgentNetwork()
    
    # 创建Agent Protocol智能体
    agent1 = await create_agent_protocol_agent("AP-Agent-1", 8081)
    agent2 = await create_agent_protocol_agent("AP-Agent-2", 8082)
    
    # 注册到网络
    await network.register_agent(agent1)
    await network.register_agent(agent2)
    
    # 建立连接
    await network.connect_agents("AP-Agent-1", "AP-Agent-2")
    
    # 发送任务
    task_msg = {
        "type": "create_task",
        "input": "网络协作任务",
        "additional_input": {"source": "AP-Agent-1"}
    }
    
    response = await network.route_message("AP-Agent-1", "AP-Agent-2", task_msg)
    print(f"任务响应: {response}")
    
    return network

# 运行网络
network = await create_ap_network()
```

### 混合协议通信

```python
# Agent Protocol智能体
ap_agent = await create_agent_protocol_agent("AP-Agent", 8081)

# A2A智能体
a2a_agent = await BaseAgent.create_a2a("A2A-Agent", port=8082, executor=a2a_executor)

# 注册到同一网络
await network.register_agent(ap_agent)
await network.register_agent(a2a_agent)

# 跨协议通信
await network.connect_agents("AP-Agent", "A2A-Agent")

# Agent Protocol消息到A2A智能体
ap_message = {"type": "create_task", "input": "来自AP的任务"}
response1 = await network.route_message("AP-Agent", "A2A-Agent", ap_message)

# A2A消息到Agent Protocol智能体
a2a_message = {"message": "来自A2A的消息"}
response2 = await network.route_message("A2A-Agent", "AP-Agent", a2a_message)
```

## 测试和验证

### 运行测试套件

```bash
# 运行所有测试
python test_integration.py --test all

# 单独测试组件
python test_integration.py --test simple    # 基础功能测试
python test_integration.py --test client    # 客户端适配器测试
python test_integration.py --test full      # 完整集成演示
```

### 手动测试

```bash
# 启动Agent Protocol智能体
python demo_integration.py

# 在另一个终端测试API
curl -X POST http://localhost:8080/ap/v1/agent/tasks \
  -H "Content-Type: application/json" \
  -d '{"input": "测试任务", "additional_input": {"priority": "high"}}'

# 获取任务信息
curl http://localhost:8080/ap/v1/agent/tasks/{task_id}

# 执行步骤
curl -X POST http://localhost:8080/ap/v1/agent/tasks/{task_id}/steps \
  -H "Content-Type: application/json" \
  -d '{"name": "test_step", "input": "步骤输入"}'
```

## 故障排除

### 常见问题

1. **导入错误**: 确保正确设置了Python路径
   ```python
   import sys
   sys.path.append("/path/to/A2A/src")
   ```

2. **端口冲突**: 为每个智能体分配不同端口
   ```python
   agent1 = await create_agent_protocol_agent("agent1", 8081)
   agent2 = await create_agent_protocol_agent("agent2", 8082)
   ```

3. **依赖缺失**: 安装必要依赖
   ```bash
   pip install httpx starlette uvicorn
   ```

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查智能体状态
health = await agent.health_check()
card = agent.get_card()
print(f"Agent health: {health}")
print(f"Agent card: {card}")

# 查看适配器信息
adapter_info = adapter.get_endpoint_info()
print(f"Adapter info: {adapter_info}")
```

## 扩展和自定义

### 添加自定义端点

```python
class CustomAgentProtocolStarletteApplication(AgentProtocolStarletteApplication):
    def build(self) -> Starlette:
        app = super().build()
        
        # 添加自定义路由
        app.routes.append(
            Route("/custom/endpoint", self.custom_handler, methods=["POST"])
        )
        return app
    
    async def custom_handler(self, request: Request) -> JSONResponse:
        # 自定义处理逻辑
        return JSONResponse({"message": "Custom endpoint"})
```

### 自定义执行器

```python
class AdvancedExecutor:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def handle_task_creation(self, task):
        # 使用LLM分析任务
        analysis = await self.llm.analyze(task.input)
        task.additional_input["llm_analysis"] = analysis
    
    async def execute_step(self, step):
        # 基于步骤类型选择不同处理逻辑
        if step.name == "analyze":
            return await self._analyze_step(step)
        elif step.name == "generate":
            return await self._generate_step(step)
        else:
            return await self._default_step(step)
```

## 贡献和支持

本Agent Protocol适配器是A2A框架的扩展，旨在提供完整的Agent Protocol v1支持。

### 功能特性

- ✅ 完整的Agent Protocol v1规范支持
- ✅ Task/Step/Artifact生命周期管理
- ✅ 标准HTTP API端点
- ✅ A2A框架无缝集成
- ✅ 错误处理和日志记录
- ✅ 认证和安全支持
- ✅ 网络通信和路由
- ✅ 混合协议支持
- ✅ 消息构建辅助工具
- ✅ 全面的测试覆盖

### 下一步开发

- 🔄 流式响应支持
- 📁 高级工件管理
- 🔐 增强认证机制
- 📊 性能监控集成
- 🌐 WebSocket支持

通过这个适配器，您的Agent Protocol实现可以充分利用A2A框架的强大网络通信能力，同时保持与Agent Protocol标准的完全兼容性。
