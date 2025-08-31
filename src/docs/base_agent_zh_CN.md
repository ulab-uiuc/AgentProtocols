# BaseAgent - 双重角色智能体实现

## 概述

`BaseAgent`类是一个复杂的双重角色智能体实现，既可以作为服务器（接收消息）也可以作为多客户端（使用各种协议向不同目标发送消息）。它为智能体通信提供统一接口，同时支持多个协议适配器和服务器实现。

## 关键特性

- **双重角色架构**: 既可作为HTTP服务器也可作为客户端
- **协议无关**: 通过适配器支持多种通信协议
- **A2A SDK集成**: 原生支持A2A（智能体到智能体）协议
- **并发操作**: 异步消息处理和处理
- **健康监控**: 内置健康检查端点
- **自动发现**: 智能体卡片发现和能力交换
- **资源管理**: 适当的生命周期管理和清理

## 类架构

```python
class BaseAgent:
    def __init__(self, agent_id: str, host: str = "0.0.0.0", port: Optional[int] = None, 
                 httpx_client: Optional[httpx.AsyncClient] = None, 
                 server_adapter: Optional[BaseServerAdapter] = None):
        self.agent_id = agent_id                          # 智能体ID
        self._host = host                                 # 服务器主机
        self._port = port or self._find_free_port()       # 服务器端口
        self._httpx_client = httpx_client or httpx.AsyncClient(timeout=30.0)  # HTTP客户端
        self._server_adapter = server_adapter or A2AServerAdapter()           # 服务器适配器
        
        # 多适配器支持: dst_id -> adapter
        self._outbound: Dict[str, BaseProtocolAdapter] = {}
        
        # 服务器组件
        self._server_task: Optional[asyncio.Task] = None
        self._server_instance: Optional[uvicorn.Server] = None
        self._self_agent_card: Optional[Dict[str, Any]] = None
```

## 工厂方法

### A2A智能体创建

```python
@classmethod
async def create_a2a(
    cls,
    agent_id: str,
    host: str = "0.0.0.0", 
    port: Optional[int] = None,
    executor: Optional[Any] = None,
    httpx_client: Optional[httpx.AsyncClient] = None,
    server_adapter: Optional[BaseServerAdapter] = None
) -> "BaseAgent":
    """创建具有A2A服务器功能的BaseAgent"""
```

**功能:**
- 验证执行器实现SDK原生接口
- 启动带有A2A端点的HTTP服务器
- 获取并缓存智能体卡片
- 返回完全初始化的智能体实例

### 遗留方法（已弃用）

```python
@classmethod
async def from_a2a(cls, agent_id: str, base_url: str, 
                   httpx_client: Optional[httpx.AsyncClient] = None) -> "BaseAgent":
    """已弃用: v0.x兼容性方法（仅客户端模式）"""
```

## 核心操作

### 消息发送

```python
async def send(self, dst_id: str, payload: Dict[str, Any]) -> Any:
    """使用适当的出站适配器向目标智能体发送消息"""
```

**流程:**
1. 验证智能体已初始化
2. 找到目标的适当出站适配器
3. 用源信息丰富载荷
4. 记录指标和时间
5. 委托给协议适配器
6. 处理错误并记录失败

### 健康监控

```python
async def health_check(self) -> bool:
    """检查智能体服务器是否健康和响应"""
```

**功能:**
- 服务器响应性检查
- HTTP端点验证
- 超时处理
- 状态码验证

### 智能体发现

```python
def get_card(self) -> Dict[str, Any]:
    """获取此智能体的卡片（来自本地服务器）"""
```

**智能体卡片内容:**
- 智能体ID和元数据
- 服务器地址和能力
- 协议支持信息
- 连接统计

## 服务器架构

### A2A服务器端点

- `/.well-known/agent.json` - 智能体卡片发现
- `/health` - 健康检查端点
- `/message` - 消息处理端点

### 执行器接口

```python
# 必需的执行器接口
async def execute(context: RequestContext, event_queue: EventQueue) -> None:
    """处理传入请求并将响应事件排队"""
```

**要求:**
- 必须实现异步execute方法
- 接受RequestContext和EventQueue参数
- 处理消息处理和响应生成
- 支持JSON和流式响应

## 连接管理

### 出站适配器

```python
def add_outbound_adapter(self, dst_id: str, adapter: BaseProtocolAdapter) -> None:
    """为连接到目标智能体添加出站适配器"""

def remove_outbound_adapter(self, dst_id: str) -> None:
    """移除出站适配器"""
    
def get_outbound_adapters(self) -> Dict[str, BaseProtocolAdapter]:
    """获取所有出站适配器（用于调试/监控）"""
```

### 协议支持

- **A2A (智能体到智能体)**: 主要协议支持
- **IoA (智能体互联网)**: 计划未来支持
- **自定义协议**: 通过适配器模式可扩展

## 生命周期管理

### 启动过程

1. **端口分配**: 如果未指定则查找可用端口
2. **服务器创建**: 使用执行器初始化服务器适配器
3. **服务器启动**: 在后台任务中启动uvicorn服务器
4. **健康等待**: 等待服务器准备就绪
5. **卡片获取**: 从服务器检索智能体卡片
6. **初始化**: 标记智能体为已初始化

### 关闭过程

```python
async def stop(self) -> None:
    """优雅地停止智能体服务器并清理所有资源"""
```

**步骤:**
1. 发出服务器关闭信号
2. 等待优雅关闭（带超时）
3. 如需要则强制取消
4. 清理所有适配器
5. 清除跟踪数据
6. 标记为未初始化

## 错误处理

### 连接错误
- 指数退避自动重试
- 失败端点的断路器模式
- 部分失败的优雅降级

### 服务器错误
- 健康检查失败检测
- 自动服务器重启功能
- 错误日志记录和指标

### 协议错误
- 协议特定的错误处理
- 向调用者传播错误
- 标准化错误格式

## 性能优化

### 连接池
- 适配器间共享httpx客户端
- 持久连接以提高效率
- 连接限制管理

### 并发处理
- 异步消息处理
- 非阻塞操作
- 并行适配器管理

### 资源管理
- 自动清理未使用的连接
- 内存高效的数据结构
- 垃圾回收集成

## 实现示例

```python
import asyncio
from agent_network import BaseAgent
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

class SimpleExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 获取用户输入
        user_input = context.get_user_input()
        
        # 处理消息
        response = f"回音: {user_input}"
        
        # 发送响应
        await event_queue.enqueue_event(new_agent_text_message(response))
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(new_agent_text_message("操作已取消"))

async def example():
    # 使用执行器创建智能体
    agent = await BaseAgent.create_a2a(
        agent_id="example-agent",
        host="localhost",
        port=8080,
        executor=SimpleExecutor()
    )
    
    # 智能体现在正在运行并准备接收消息
    print(f"智能体运行地址: {agent.get_listening_address()}")
    print(f"智能体卡片: {agent.get_card()}")
    
    # 保持运行
    try:
        await asyncio.sleep(3600)  # 运行1小时
    finally:
        await agent.stop()  # 清理关闭

asyncio.run(example())
```

## 最佳实践

1. **执行器设计**: 保持执行器轻量且专注
2. **错误处理**: 始终在执行器中实现适当的错误处理
3. **资源清理**: 使用try/finally块进行适当的清理
4. **健康监控**: 在生产环境中定期检查智能体健康
5. **协议选择**: 根据用例要求选择适当的协议 