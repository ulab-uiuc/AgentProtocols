# 新协议实现开发指南

本文档说明如何在 Agent Network 框架中实现新协议类型的兼容，包括所需的步骤、接口实现和代码示例。

## 概览

Agent Network 框架采用适配器模式来支持不同的协议类型。每个协议需要实现两个适配器：

1. **客户端适配器** (`agent_adapters`): 用于连接到其他代理
2. **服务器适配器** (`server_adapters`): 用于构建本地服务器

## 实现步骤

### 步骤1: 创建客户端适配器

在 `src/agent_adapters/` 目录下创建新的协议适配器文件，例如 `your_protocol_adapter.py`：

```python
from typing import Any, Dict, Optional
from .base_adapter import BaseProtocolAdapter

class YourProtocolAdapter(BaseProtocolAdapter):
    """Your protocol adapter implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize protocol-specific configurations
        self.endpoint_url = kwargs.get('endpoint_url')
        self.auth_config = kwargs.get('auth_config', {})
        # Add other protocol-specific parameters
    
    async def initialize(self) -> None:
        """Initialize the adapter and establish connections."""
        # Implement connection initialization
        # e.g., authenticate, fetch capabilities, etc.
        pass
    
    async def send_message(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        """Send a message to destination agent."""
        # Implement message sending logic
        # Transform payload to protocol-specific format
        # Send via protocol-specific transport
        # Return response
        pass
    
    async def receive_message(self) -> Dict[str, Any]:
        """Receive a message (if applicable for the protocol)."""
        # Implement message receiving logic
        # Return received messages in standard format
        pass
    
    def get_agent_card(self) -> Dict[str, Any]:
        """Get agent capabilities and metadata."""
        # Return agent card information
        pass
    
    async def health_check(self) -> bool:
        """Check if the adapter is healthy and connected."""
        # Implement health check logic
        pass
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        # Implement cleanup logic
        pass
```

### 步骤2: 创建服务器适配器

在 `src/server_adapters/` 目录下创建服务器适配器文件，例如 `your_protocol_adapter.py`：

```python
from typing import Any, Dict, Tuple
import uvicorn
from .base_adapter import BaseServerAdapter

class YourProtocolServerAdapter(BaseServerAdapter):
    """Server adapter for your protocol."""
    
    protocol_name = "YourProtocol"
    
    def build(
        self,
        host: str,
        port: int,
        agent_id: str,
        executor: Any,
        **kwargs
    ) -> Tuple[uvicorn.Server, Dict[str, Any]]:
        """
        Build and configure a protocol-specific server.
        
        Returns:
            Tuple of (uvicorn.Server instance, agent_card dict)
        """
        # Create agent card
        agent_card = {
            "name": f"Agent {agent_id}",
            "url": f"http://{host}:{port}/",
            "protocolVersion": "1.0.0",
            "protocol": self.protocol_name,
            "capabilities": {
                # Define protocol-specific capabilities
            }
        }
        
        # Create server application (FastAPI, Starlette, etc.)
        app = self._create_app(agent_card, executor)
        
        # Configure uvicorn server
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="error"
        )
        server = uvicorn.Server(config)
        
        return server, agent_card
    
    def _create_app(self, agent_card: Dict[str, Any], executor: Any):
        """Create the web application with protocol-specific endpoints."""
        # Implement your protocol-specific web application
        # Define routes and handlers
        # Integrate with the executor
        pass
```

### 步骤3: 更新模块导入

在 `src/agent_adapters/__init__.py` 中添加新适配器的导入：

```python
from .your_protocol_adapter import YourProtocolAdapter

__all__ = [
    "BaseProtocolAdapter",
    "A2AAdapter", 
    "YourProtocolAdapter",  # 添加新适配器
]
```

在 `src/server_adapters/__init__.py` 中添加服务器适配器的导入：

```python
from .your_protocol_adapter import YourProtocolServerAdapter

__all__ = [
    "BaseServerAdapter",
    "A2AServerAdapter",
    "YourProtocolServerAdapter",  # 添加新服务器适配器
]
```

### 步骤4: 在BaseAgent中添加工厂方法

在 `src/base_agent.py` 中添加新协议的工厂方法：

```python
@classmethod
async def create_your_protocol(
    cls,
    agent_id: str,
    host: str = "0.0.0.0",
    port: Optional[int] = None,
    executor: Optional[Any] = None,
    protocol_config: Optional[Dict[str, Any]] = None,
    httpx_client: Optional[httpx.AsyncClient] = None
) -> "BaseAgent":
    """
    Factory method: Create BaseAgent with YourProtocol server capability.
    """
    # Validate executor
    if executor is None:
        raise ValueError("executor parameter is required")
    
    # Create server adapter
    server_adapter = YourProtocolServerAdapter()
    
    # Create BaseAgent instance
    agent = cls(
        agent_id=agent_id,
        host=host,
        port=port,
        httpx_client=httpx_client,
        server_adapter=server_adapter
    )
    
    # Start server
    await agent._start_server(executor)
    await agent._fetch_self_card()
    
    agent._initialized = True
    return agent
```

### 步骤5: 创建连接适配器

在 `src/base_agent.py` 中的 `add_outbound_adapter` 方法中，可以添加协议检测逻辑：

```python
def add_outbound_adapter_for_protocol(
    self, 
    dst_id: str, 
    protocol_type: str,
    config: Dict[str, Any]
) -> None:
    """Add outbound adapter based on protocol type."""
    if protocol_type == "A2A":
        adapter = A2AAdapter(**config)
    elif protocol_type == "YourProtocol":
        adapter = YourProtocolAdapter(**config)
    else:
        raise ValueError(f"Unsupported protocol: {protocol_type}")
    
    self.add_outbound_adapter(dst_id, adapter)
```

## 协议接口规范

### 客户端适配器接口

每个客户端适配器必须实现 `BaseProtocolAdapter` 的所有抽象方法：

- `initialize()`: 初始化连接和认证
- `send_message(dst_id, payload)`: 发送消息
- `receive_message()`: 接收消息（可选，取决于协议）
- `get_agent_card()`: 获取代理能力卡片
- `health_check()`: 健康检查
- `cleanup()`: 资源清理

### 服务器适配器接口

每个服务器适配器必须实现 `BaseServerAdapter` 的抽象方法：

- `build(host, port, agent_id, executor, **kwargs)`: 构建服务器实例和代理卡片

## 消息格式标准化

### 输入消息格式

适配器接收的 `payload` 应该是标准化的格式：

```python
{
    "message": "actual message content",
    "context": {"key": "value"},
    "source": "source_agent_id",
    "routing": {
        "destination": "dst_agent_id",
        "protocol": "protocol_name"
    }
}
```

### 输出消息格式

适配器返回的响应应该包含：

```python
{
    "result": "response data",
    "status": "success|error",
    "metadata": {
        "protocol": "protocol_name",
        "timestamp": "ISO8601",
        "request_id": "unique_id"
    }
}
```

## 测试要求

为新协议创建测试文件：

1. 在 `tests/` 目录下创建 `test_your_protocol_adapter.py`
2. 测试客户端适配器的所有方法
3. 测试服务器适配器的构建和运行
4. 测试端到端的消息传递

## 示例: 简单HTTP协议实现

参考 `a2a_adapter.py` 的实现，这里提供一个简化的HTTP协议示例：

```python
# SimpleHTTPAdapter示例
class SimpleHTTPAdapter(BaseProtocolAdapter):
    def __init__(self, base_url: str, **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def send_message(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        response = await self.client.post(
            f"{self.base_url}/message",
            json=payload
        )
        return response.json()
```

## 注意事项

1. **线程安全**: 确保适配器在异步环境中是线程安全的
2. **错误处理**: 实现适当的错误处理和重试机制  
3. **性能优化**: 考虑连接池、批处理等性能优化
4. **安全性**: 实现认证、加密等安全措施
5. **可观测性**: 添加日志、指标等监控功能
6. **协议版本**: 支持协议版本检测和兼容性处理

## 集成到AgentNetwork

新协议实现完成后，可以通过以下方式集成到 `AgentNetwork` 中：

```python
from agent_network import AgentNetwork
from agent_adapters import YourProtocolAdapter

# 创建网络
network = AgentNetwork()

# 添加支持新协议的代理
agent = await BaseAgent.create_your_protocol(
    agent_id="agent1",
    executor=your_executor,
    protocol_config={"param": "value"}
)

network.add_agent(agent)

# 添加连接
await network.add_connection(
    "agent1", "agent2", 
    protocol="YourProtocol",
    config={"endpoint": "http://agent2:8080"}
)
```

## 总结

实现新协议支持的核心步骤：

1. 创建客户端适配器类继承 `BaseProtocolAdapter`
2. 创建服务器适配器类继承 `BaseServerAdapter` 
3. 更新模块导入
4. 在 `BaseAgent` 中添加工厂方法
5. 编写测试用例
6. 更新文档

遵循这些步骤，您就可以成功地为 Agent Network 框架添加新的协议支持。 