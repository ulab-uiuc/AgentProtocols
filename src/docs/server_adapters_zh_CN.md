# 服务器适配器 - 协议特定的服务器实现

## 概述

服务器适配器提供协议特定的服务器实现，使BaseAgent实例能够使用各种协议接收和处理传入消息。它们处理服务器端通信、请求解析、执行器集成和不同协议标准的响应格式化。

## 架构

服务器适配器系统遵循可插拔架构模式：

```
BaseServerAdapter (抽象接口)
    ├── A2AServerAdapter (智能体到智能体协议服务器)
    ├── DummyServerAdapter (测试/开发服务器)
    └── CustomServerAdapter (用户定义协议)
```

## 基础服务器适配器

`BaseServerAdapter`类为所有服务器适配器实现定义接口：

```python
class BaseServerAdapter:
    """协议特定服务器适配器的基类"""
    
    protocol_name = "BaseProtocol"
    
    def build(
        self,
        host: str,
        port: int,
        agent_id: str,
        executor: Any,
        **kwargs
    ) -> Tuple[uvicorn.Server, Dict[str, Any]]:
        """
        构建和配置服务器实例
        
        返回:
            (uvicorn.Server, agent_card)的元组
        """
        raise NotImplementedError("子类必须实现build方法")
```

## A2A服务器适配器

`A2AServerAdapter`是智能体到智能体协议服务器的主要实现：

### 核心特性

- **A2A协议合规**: 完全符合A2A规范
- **SDK原生集成**: 与A2A SDK执行器接口直接集成
- **双响应模式**: 支持JSON和SSE流式响应
- **智能体卡片生成**: 自动智能体卡片创建和服务
- **健康监控**: 内置健康检查端点
- **请求验证**: 全面的请求验证和错误处理

### 类结构

```python
class A2AServerAdapter(BaseServerAdapter):
    """具有SDK原生接口的A2A（智能体到智能体）协议服务器适配器"""
    
    protocol_name = "A2A"
    
    def build(
        self,
        host: str,
        port: int,
        agent_id: str,
        executor: Any,
        **kwargs
    ) -> Tuple[uvicorn.Server, Dict[str, Any]]:
        """使用SDK原生执行器接口构建A2A服务器"""
```

### 智能体卡片生成

适配器自动生成A2A兼容的智能体卡片：

```python
agent_card = {
    "name": f"智能体 {agent_id}",
    "url": f"http://{host}:{port}/",
    "protocolVersion": "1.0.0",
    "skills": [
        {
            "id": "agent_execution",
            "name": "智能体执行",
            "description": "使用A2A SDK原生接口执行智能体任务"
        }
    ],
    "capabilities": {
        "streaming": True,
        "supportsAuthenticatedExtendedCard": False,
        "nativeSDK": True
    }
}
```

### 服务器端点

A2A服务器公开以下端点：

- **`/.well-known/agent.json`** - 智能体卡片发现端点
- **`/health`** - 健康检查端点
- **`/message`** - 消息处理端点

### 请求处理流程

1. **请求接收**: 在`/message`端点接收HTTP POST请求
2. **请求解析**: 从JSON正文提取A2A消息结构
3. **上下文创建**: 从请求数据创建SDK RequestContext
4. **执行器调用**: 调用executor.execute(context, event_queue)
5. **响应收集**: 从事件队列收集事件
6. **响应格式化**: 基于Accept头格式化响应
7. **响应传递**: 向客户端发送JSON或SSE响应

## A2A Starlette应用

核心服务器实现使用Starlette框架：

```python
class A2AStarletteApplication:
    """使用SDK原生执行器接口的A2A服务器实现"""
    
    def __init__(self, agent_card: Dict[str, Any], executor: Any):
        self.agent_card = agent_card
        self.executor = executor
        
    def build(self) -> Starlette:
        """构建Starlette应用"""
        routes = [
            Route("/.well-known/agent.json", self.get_agent_card, methods=["GET"]),
            Route("/health", self.health_check, methods=["GET"]),
            Route("/message", self.handle_message, methods=["POST"]),
        ]
        
        return Starlette(routes=routes)
```

### 消息处理

```python
async def handle_message(self, request: Request) -> JSONResponse | StreamingResponse:
    """使用SDK原生执行器接口处理传入消息"""
    
    try:
        # 解析请求正文
        body = await request.json()
        
        # 创建SDK RequestContext
        if 'params' in body and 'message' in body['params']:
            message_data = body['params']['message']
            message = Message(**message_data)
            params = MessageSendParams(message=message)
            ctx = RequestContext(params)
        else:
            # 简单消息的回退方案
            text = body.get('text', str(body))
            message = new_agent_text_message(text, role=Role.user)
            params = MessageSendParams(message=message)
            ctx = RequestContext(params)
        
        # 创建EventQueue
        queue = EventQueue()
        
        # 调用SDK原生执行器
        await self.executor.execute(ctx, queue)
        
        # 基于Accept头处理响应
        accept_header = request.headers.get("accept", "")
        if "text/event-stream" in accept_header:
            return StreamingResponse(
                self._sse_generator(queue),
                media_type="text/event-stream"
            )
        else:
            # 收集事件并返回JSON
            events = []
            try:
                while True:
                    event = await queue.dequeue_event(no_wait=True)
                    events.append(self._event_to_dict(event))
            except asyncio.QueueEmpty:
                pass
            
            return JSONResponse({"events": events})
            
    except Exception as e:
        return JSONResponse(
            {"error": f"消息处理失败: {e}"},
            status_code=500
        )
```

### SSE流式支持

```python
async def _sse_generator(self, queue: EventQueue):
    """从EventQueue生成SSE事件"""
    
    try:
        while True:
            try:
                event = await queue.dequeue_event(no_wait=True)
                event_data = self._event_to_dict(event)
                yield f"data: {json.dumps(event_data)}\n\n"
            except asyncio.QueueEmpty:
                try:
                    event = await queue.dequeue_event(no_wait=False)
                    event_data = self._event_to_dict(event)
                    yield f"data: {json.dumps(event_data)}\n\n"
                except asyncio.QueueEmpty:
                    break
    except Exception as e:
        error_data = {"error": str(e), "type": "stream_error"}
        yield f"data: {json.dumps(error_data)}\n\n"
```

## 虚拟服务器适配器

为了测试和开发目的，提供了虚拟服务器适配器：

```python
class DummyServerAdapter(BaseServerAdapter):
    """用于测试目的的虚拟服务器适配器"""
    
    protocol_name = "Dummy"
    
    def build(
        self,
        host: str,
        port: int,
        agent_id: str,
        executor: Any,
        **kwargs
    ) -> Tuple[uvicorn.Server, Dict[str, Any]]:
        """构建简单的虚拟服务器进行测试"""
        
        # 创建简单的虚拟应用
        app = self._create_dummy_app(agent_id, executor)
        
        # 生成虚拟智能体卡片
        agent_card = {
            "name": f"虚拟智能体 {agent_id}",
            "url": f"http://{host}:{port}/",
            "protocol": "dummy",
            "capabilities": ["echo", "ping"]
        }
        
        # 配置uvicorn服务器
        config = uvicorn.Config(app, host=host, port=port, log_level="error")
        server = uvicorn.Server(config)
        
        return server, agent_card
```

## 服务器生命周期管理

### 启动过程

1. **适配器选择**: 选择适当的服务器适配器
2. **执行器验证**: 验证执行器接口兼容性
3. **服务器构建**: 构建服务器实例和智能体卡片
4. **后台任务**: 在asyncio后台任务中启动服务器
5. **健康轮询**: 等待服务器就绪
6. **卡片获取**: 检索和缓存智能体卡片

### 运行时操作

- **请求处理**: 处理传入的协议请求
- **执行器集成**: 桥接请求到执行器接口
- **响应处理**: 格式化和传递响应
- **健康监控**: 响应健康检查请求
- **错误处理**: 处理和报告处理错误

### 关闭过程

- **优雅关闭**: 信号服务器停止接受请求
- **请求完成**: 等待活动请求完成
- **资源清理**: 清理服务器资源
- **任务取消**: 取消服务器后台任务

## 与BaseAgent的集成

服务器适配器无缝集成到BaseAgent中：

```python
class BaseAgent:
    async def _start_server(self, executor: Any) -> None:
        """使用可插拔适配器启动内部服务器"""
        
        # 使用服务器适配器构建服务器和智能体卡片
        self._server_instance, self._self_agent_card = self._server_adapter.build(
            host=self._host,
            port=self._port,
            agent_id=self.agent_id,
            executor=executor
        )
        
        # 在后台任务中启动服务器
        self._server_task = asyncio.create_task(self._server_instance.serve())
        
        # 等待服务器准备就绪
        await self._wait_for_server_ready()
```

## 错误处理

### 请求验证错误

```python
try:
    body = await request.json()
    # 验证A2A消息结构
    if not self._validate_a2a_message(body):
        return JSONResponse(
            {"error": "无效的A2A消息格式"},
            status_code=400
        )
except json.JSONDecodeError:
    return JSONResponse(
        {"error": "请求正文中的JSON无效"},
        status_code=400
    )
```

### 执行器错误

```python
try:
    await self.executor.execute(ctx, queue)
except Exception as e:
    logger.error(f"执行器错误: {e}")
    return JSONResponse(
        {"error": f"执行失败: {str(e)}"},
        status_code=500
    )
```

### 协议错误

- **不支持的方法**: 返回405 Method Not Allowed
- **缺失头部**: 返回400 Bad Request并详细说明
- **协议违规**: 返回400 Bad Request并详细说明违规

## 性能优化

### 请求处理

- **异步处理器**: 非阻塞请求处理
- **连接池**: 高效连接管理
- **响应流**: 内存高效的流式响应
- **请求批处理**: 适用时的批处理

### 资源管理

- **内存效率**: 每个请求的最小内存占用
- **CPU优化**: 高效的JSON解析和序列化
- **连接限制**: 可配置的连接限制

## 使用示例

### 创建自定义服务器适配器

```python
from agent_network.server_adapters import BaseServerAdapter
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

class CustomServerAdapter(BaseServerAdapter):
    """自定义协议服务器适配器"""
    
    protocol_name = "Custom"
    
    def build(self, host: str, port: int, agent_id: str, executor: Any, **kwargs):
        """构建自定义协议服务器"""
        
        # 创建自定义Starlette应用
        app = self._create_custom_app(agent_id, executor)
        
        # 生成自定义智能体卡片
        agent_card = {
            "name": f"自定义智能体 {agent_id}",
            "url": f"http://{host}:{port}/",
            "protocol": "custom-v1",
            "capabilities": ["custom_processing", "custom_streaming"]
        }
        
        # 配置服务器
        config = uvicorn.Config(app, host=host, port=port, log_level="error")
        server = uvicorn.Server(config)
        
        return server, agent_card
    
    def _create_custom_app(self, agent_id: str, executor: Any) -> Starlette:
        """创建自定义Starlette应用"""
        
        async def custom_handler(request):
            """处理自定义协议请求"""
            try:
                body = await request.json()
                # 自定义处理逻辑
                result = await self._process_custom_message(body, executor)
                return JSONResponse({"result": result})
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
        
        routes = [
            Route("/custom", custom_handler, methods=["POST"]),
            Route("/health", lambda r: JSONResponse({"status": "ok"}), methods=["GET"])
        ]
        
        return Starlette(routes=routes)
    
    async def _process_custom_message(self, message: dict, executor: Any):
        """使用自定义协议处理消息"""
        # 实现自定义消息处理
        return {"processed": True, "message": message}
```

### 使用自定义服务器适配器

```python
import asyncio
from agent_network import BaseAgent

async def example():
    # 使用自定义服务器适配器创建智能体
    custom_adapter = CustomServerAdapter()
    
    agent = await BaseAgent.create_a2a(
        agent_id="custom-agent",
        host="localhost",
        port=8080,
        executor=your_custom_executor,
        server_adapter=custom_adapter
    )
    
    print(f"自定义智能体运行地址: {agent.get_listening_address()}")
    
    # 智能体现在提供自定义协议服务
    await asyncio.sleep(3600)  # 保持运行
    
    await agent.stop()

asyncio.run(example())
```

## 配置示例

### A2A服务器配置

```python
# A2A服务器的Uvicorn配置
config = uvicorn.Config(
    app,
    host=host,
    port=port,
    log_level="error",  # 最小化日志记录
    access_log=False,   # 禁用访问日志
    loop="uvloop",      # 使用uvloop以获得更好的性能
    http="httptools"    # 使用httptools进行HTTP解析
)
```

### 服务器适配器注册表

```python
# 注册自定义服务器适配器
SERVER_ADAPTERS = {
    "a2a": A2AServerAdapter,
    "dummy": DummyServerAdapter,
    "custom": CustomServerAdapter
}

def get_server_adapter(protocol: str) -> BaseServerAdapter:
    """根据协议名称获取服务器适配器"""
    if protocol not in SERVER_ADAPTERS:
        raise ValueError(f"未知协议: {protocol}")
    return SERVER_ADAPTERS[protocol]()
```

## 最佳实践

1. **执行器验证**: 服务器启动前始终验证执行器接口
2. **错误处理**: 为所有端点实现全面的错误处理
3. **健康端点**: 始终提供健康检查端点用于监控
4. **资源限制**: 设置适当的连接和请求限制
5. **安全性**: 实现适当的身份验证和输入验证 