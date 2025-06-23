# 智能体适配器 - 协议特定的客户端适配器

## 概述

智能体适配器提供协议特定的客户端实现，使BaseAgent实例能够使用各种协议与不同类型的智能体通信。它们将不同通信协议的复杂性抽象到统一接口之后，允许无缝集成和协议切换。

## 架构

适配器系统遵循**适配器模式**，为不同协议实现提供一致的接口：

```
BaseProtocolAdapter (抽象接口)
    ├── A2AAdapter (智能体到智能体协议)
    ├── IoAAdapter (智能体互联网 - 未来)
    └── CustomAdapter (用户定义协议)
```

## 基础协议适配器

`BaseProtocolAdapter`类定义了所有协议适配器必须实现的通用接口：

```python
class BaseProtocolAdapter:
    """所有协议适配器的基类"""
    
    def __init__(self, base_url: str, auth_headers: Dict[str, str] = None):
        self.base_url = base_url
        self.auth_headers = auth_headers or {}
    
    async def initialize(self) -> None:
        """初始化适配器（获取能力、设置连接等）"""
        raise NotImplementedError
    
    async def send_message(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        """向目标智能体发送消息"""
        raise NotImplementedError
    
    async def receive_message(self) -> Dict[str, Any]:
        """接收传入消息（对于支持轮询的协议）"""
        raise NotImplementedError
    
    async def health_check(self) -> bool:
        """检查目标智能体是否健康"""
        raise NotImplementedError
    
    async def cleanup(self) -> None:
        """清理适配器资源"""
        raise NotImplementedError
```

## A2A适配器实现

`A2AAdapter`是智能体到智能体协议的主要实现：

### 核心特性

- **基于HTTP的通信**: 使用HTTP/HTTPS进行智能体通信
- **JSON消息格式**: 标准化JSON载荷结构
- **流式支持**: 一次性和流式消息传递
- **智能体发现**: 自动智能体卡片获取和能力发现
- **身份验证**: 支持自定义身份验证头
- **连接池**: 通过共享httpx客户端高效连接重用

### 类结构

```python
class A2AAdapter(BaseProtocolAdapter):
    def __init__(
        self, 
        httpx_client: httpx.AsyncClient, 
        base_url: str,
        auth_headers: Optional[Dict[str, str]] = None,
        agent_card_path: str = "/.well-known/agent.json"
    ):
        super().__init__(base_url=base_url, auth_headers=auth_headers or {})
        self.httpx_client = httpx_client
        self.base_url = base_url.rstrip('/')
        self.auth_headers = auth_headers or {}
        self.agent_card_path = agent_card_path
        self.agent_card: Dict[str, Any] = {}
        self._inbox_not_available = False
```

### 消息格式

A2A适配器使用官方A2A消息格式：

```python
# 出站消息结构
request_data = {
    "id": request_id,
    "params": {
        "message": payload.get("message", payload),
        "context": payload.get("context", {}),
        "routing": {
            "destination": dst_id,
            "source": payload.get("source", "unknown")
        }
    }
}
```

### 关键方法

#### 消息发送

```python
async def send_message(self, dst_id: str, payload: Dict[str, Any]) -> Any:
    """使用A2A协议和官方A2A格式发送消息"""
    
    # 构造A2A官方消息格式
    request_id = str(uuid4())
    request_data = {
        "id": request_id,
        "params": {
            "message": payload.get("message", payload),
            "context": payload.get("context", {}),
            "routing": {
                "destination": dst_id,
                "source": payload.get("source", "unknown")
            }
        }
    }
    
    try:
        headers = {"Content-Type": "application/json"}
        headers.update(self.auth_headers)
        
        response = await self.httpx_client.post(
            f"{self.base_url}/message",
            content=json.dumps(request_data, separators=(',', ':')),
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise RuntimeError(f"A2A发送失败: {e}") from e
```

#### 流式消息

```python
async def send_message_streaming(self, dst_id: str, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
    """通过A2A协议发送消息并返回流式响应"""
    
    async with self.httpx_client.stream(
        "POST",
        f"{self.base_url}/message",
        content=json.dumps(request_data, separators=(',', ':')),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        timeout=httpx.Timeout(30.0, read=None)
    ) as response:
        response.raise_for_status()
        
        async for line in response.aiter_lines():
            if line.strip():
                try:
                    clean_line = line.lstrip("data:").strip()
                    if clean_line:
                        event_data = json.loads(clean_line)
                        yield event_data
                except json.JSONDecodeError:
                    continue
```

#### 智能体发现

```python
async def initialize(self) -> None:
    """通过从/.well-known/agent.json获取智能体卡片来初始化"""
    
    try:
        headers = {}
        headers.update(self.auth_headers)
        
        response = await self.httpx_client.get(
            f"{self.base_url}{self.agent_card_path}",
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        self.agent_card = response.json()
    except Exception as e:
        raise ConnectionError(f"A2A适配器初始化失败: {e}") from e
```

#### 健康监控

```python
async def health_check(self) -> bool:
    """检查A2A智能体是否响应"""
    
    try:
        headers = {}
        headers.update(self.auth_headers)
        
        response = await self.httpx_client.get(
            f"{self.base_url}/health",
            headers=headers,
            timeout=5.0
        )
        return response.status_code == 200
    except Exception:
        return False
```

## 适配器生命周期

### 初始化过程

1. **适配器创建**: 使用目标智能体URL实例化适配器
2. **连接设置**: 配置HTTP客户端和身份验证
3. **智能体发现**: 获取智能体卡片和能力
4. **能力验证**: 验证协议兼容性
5. **就绪状态**: 适配器准备好进行消息交换

### 运行时操作

- **消息路由**: 基于目标智能体ID路由消息
- **错误处理**: 处理网络故障和超时
- **健康监控**: 定期检查目标智能体健康
- **指标收集**: 跟踪性能和错误指标

### 清理过程

- **连接终止**: 关闭活动连接
- **资源释放**: 释放分配的资源
- **缓存清理**: 清除缓存的智能体信息

## 与BaseAgent的集成

BaseAgent通过出站适配器注册表使用适配器：

```python
class BaseAgent:
    def __init__(self, ...):
        # 多适配器支持: dst_id -> adapter
        self._outbound: Dict[str, BaseProtocolAdapter] = {}
    
    def add_outbound_adapter(self, dst_id: str, adapter: BaseProtocolAdapter) -> None:
        """为连接到目标智能体添加出站适配器"""
        self._outbound[dst_id] = adapter
        self.outgoing_edges.add(dst_id)
    
    async def send(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        """使用适当的出站适配器向目标智能体发送消息"""
        
        if dst_id not in self._outbound:
            raise RuntimeError(f"未找到目标{dst_id}的出站适配器")
        
        adapter = self._outbound[dst_id]
        
        # 向载荷添加源信息
        enriched_payload = payload.copy()
        enriched_payload.setdefault("source", self.agent_id)
        
        # 委托给协议适配器
        response = await adapter.send_message(dst_id, enriched_payload)
        return response
```

## 错误处理

### 连接错误

```python
try:
    response = await adapter.send_message(dst_id, payload)
except httpx.TimeoutException as e:
    raise TimeoutError(f"A2A消息超时到{dst_id}") from e
except httpx.HTTPStatusError as e:
    raise ConnectionError(f"A2A HTTP错误 {e.response.status_code}") from e
except Exception as e:
    raise RuntimeError(f"A2A发送失败: {e}") from e
```

### 协议错误

- **无效消息格式**: A2A消息结构验证
- **身份验证失败**: 身份认证相关错误处理
- **协议不匹配**: 不兼容协议版本检测

## 性能优化

### 连接池

- **共享HTTP客户端**: 每个适配器实例一个httpx客户端
- **连接限制**: 可配置的连接池大小
- **保持活动**: 持久连接以获得更好性能

### 消息优化

- **JSON压缩**: 紧凑JSON序列化
- **批量操作**: 尽可能组合多个消息
- **异步操作**: 非阻塞消息发送

## 使用示例

### 创建和使用A2A适配器

```python
import asyncio
import httpx
from agent_network.agent_adapters import A2AAdapter

async def example():
    # 创建共享HTTP客户端
    client = httpx.AsyncClient(timeout=30.0)
    
    # 创建A2A适配器
    adapter = A2AAdapter(
        httpx_client=client,
        base_url="http://target-agent:8080",
        auth_headers={"Authorization": "Bearer token123"}
    )
    
    # 初始化适配器
    await adapter.initialize()
    
    # 发送消息
    response = await adapter.send_message(
        "target-agent-id",
        {"message": "来自源智能体的问候"}
    )
    
    print(f"响应: {response}")
    
    # 检查健康
    is_healthy = await adapter.health_check()
    print(f"智能体健康: {is_healthy}")
    
    # 清理
    await adapter.cleanup()
    await client.aclose()

asyncio.run(example())
```

### 自定义适配器实现

```python
from agent_network.agent_adapters import BaseProtocolAdapter

class CustomProtocolAdapter(BaseProtocolAdapter):
    """自定义协议适配器实现"""
    
    async def initialize(self) -> None:
        """初始化自定义协议连接"""
        # 自定义初始化逻辑
        pass
    
    async def send_message(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        """使用自定义协议发送消息"""
        # 自定义消息发送逻辑
        return {"status": "success", "data": "自定义响应"}
    
    async def receive_message(self) -> Dict[str, Any]:
        """使用自定义协议接收消息"""
        # 自定义消息接收逻辑
        return {"message": "自定义传入消息"}
    
    async def health_check(self) -> bool:
        """使用自定义协议检查智能体健康"""
        # 自定义健康检查逻辑
        return True
    
    async def cleanup(self) -> None:
        """清理自定义协议资源"""
        # 自定义清理逻辑
        pass
```

## 最佳实践

1. **连接重用**: 始终跨适配器重用HTTP客户端
2. **错误处理**: 实现带重试的全面错误处理
3. **超时配置**: 为不同操作类型设置适当的超时
4. **资源清理**: 关闭时始终清理适配器
5. **健康监控**: 定期检查适配器和目标健康状态 