# 智能体开发指南

本指南将帮助您理解和开发 GAIA 框架中的智能体组件。

## 智能体架构概述

GAIA 框架采用分层的智能体架构，从基础到具体实现分为以下层次：

```
智能体继承层次:
ReActAgent (基础抽象类)
    ↓
ToolCallAgent (工具调用智能体) 
    ↓
MeshAgent (网络智能体抽象类)
    ↓
[Protocol]Agent (协议特定实现)
```

## 核心智能体类

### 1. ReActAgent - 基础抽象智能体

提供最基本的思考-行动循环：

```python
class ReActAgent(BaseModel, ABC):
    """基础 ReAct 智能体"""
    name: str
    description: Optional[str] = None
    state: AgentState = AgentState.IDLE
    max_steps: int = 10
    current_step: int = 0

    @abstractmethod
    async def think(self) -> bool:
        """处理当前状态并决定下一步行动"""

    @abstractmethod
    async def act(self) -> str:
        """执行决定的行动"""

    async def step(self) -> str:
        """执行单步：思考和行动"""
        should_act = await self.think()
        if not should_act:
            return "Thinking complete - no action needed"
        return await self.act()
```

### 2. ToolCallAgent - 工具调用智能体

在基础智能体之上添加了 LLM 和工具调用能力：

```python
class ToolCallAgent(ReActAgent):
    """具备工具调用能力的智能体"""
    
    # LLM 配置
    llm: Optional[LLM] = Field(default_factory=LLM)
    available_tools: ToolCollection = Field(default_factory=lambda: ToolRegistry().available_tools)
    
    # 消息处理
    messages: List[Message] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    
    async def think(self) -> bool:
        """使用 LLM 进行思考和工具选择"""
        # 调用 LLM API 获取响应
        response = await self.llm.ask_tool(
            messages=self.messages,
            tools=self.available_tools.to_params(),
            tool_choice=self.tool_choices
        )
        
        # 处理工具调用
        self.tool_calls = response.get("tool_calls", [])
        return bool(self.tool_calls)
    
    async def act(self) -> str:
        """执行工具调用"""
        results = []
        for command in self.tool_calls:
            result = await self.execute_tool(command)
            results.append(result)
        return "\n\n".join(results)
```

### 3. MeshAgent - 网络智能体

添加了网络通信和消息处理能力：

```python
class MeshAgent(ToolCallAgent, ABC):
    """网络通信智能体"""
    
    # 网络身份
    id: int = Field(description="唯一智能体标识符")
    port: int = Field(description="监听端口")
    
    # 通信能力
    memory: Memory = Field(default_factory=Memory, description="消息内存")
    result_callback: Optional[callable] = Field(default=None)
    
    # 抽象通信方法
    @abstractmethod
    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """发送消息到另一个智能体"""
    
    @abstractmethod
    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """接收消息，支持超时"""
    
    # 消息处理循环
    async def process_messages(self):
        """处理收到的消息并生成响应"""
        msg = await self.recv_msg(timeout=0.0)
        if not msg:
            return
        
        # 添加用户消息到内存
        user_msg = Message.user_message(msg.get("content", ""))
        self.messages.append(user_msg)
        self.memory.add_message(user_msg)
        
        # 使用 ToolCallAgent 的逻辑处理消息
        while self.current_step < self.max_steps and self.state != AgentState.FINISHED:
            await self.step()
            self.current_step += 1
        
        # 提取结果并存储到内存
        final_result = self._extract_final_result()
        assistant_msg = Message.assistant_message(content=final_result)
        self.memory.add_message(assistant_msg)
        
        # 通过回调返回结果（非阻塞）
        if self.result_callback:
            await self._notify_result(final_result)
```

## 内存管理系统

GAIA 框架提供两级内存管理：

### 1. 智能体级内存 (Agent Memory)

每个智能体维护自己的消息历史：

```python
class Memory(BaseModel):
    """智能体内存"""
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)
    
    def add_message(self, message: Message) -> None:
        """添加消息到内存"""
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_recent_messages(self, n: int) -> List[Message]:
        """获取最近n条消息"""
        return self.messages[-n:]
    
    def clear(self) -> None:
        """清空内存"""
        self.messages.clear()
```

### 2. 网络级内存池 (Network Memory Pool)

网络维护所有智能体的执行记录：

```python
class NetworkMemoryPool(BaseModel):
    """网络级内存池"""
    step_executions: Dict[int, StepExecution] = Field(default_factory=dict)
    
    def add_step_execution(self, step: int, agent_id: str, agent_name: str, 
                          task_id: str, user_message: str) -> str:
        """添加步骤执行记录"""
        
    def update_step_status(self, step: int, status: ExecutionStatus, 
                          messages: Optional[List[Message]] = None) -> bool:
        """更新步骤状态"""
        
    def get_step_chain_context(self, current_step: int) -> List[Dict[str, Any]]:
        """获取前序步骤的上下文"""
```

## 工具系统集成

### 工具注册和调用

智能体通过 `ToolRegistry` 获取可用工具：

```python
# 在智能体初始化时
self.tool_registry = ToolRegistry()
self.available_tools = self.tool_registry.available_tools

# 工具执行
async def execute_tool(self, command) -> str:
    """执行工具调用"""
    name = command.function.name
    args = json.loads(command.function.arguments)
    
    result = await self.available_tools.execute(name=name, tool_input=args)
    return f"Tool '{name}' result: {result}"
```

### 特殊工具处理

框架支持特殊工具的自定义处理：

```python
class MeshAgent:
    special_tool_names: list[str] = Field(default_factory=lambda: ["CreateChatCompletion"])
    
    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """处理特殊工具执行"""
        if self._is_special_tool(name):
            # 自定义处理逻辑
            if self._should_finish_execution(name=name, result=result):
                self.state = AgentState.FINISHED
```

## 生命周期管理

### 智能体启动

```python
async def start(self):
    """启动智能体"""
    self._log(f"Starting agent {self.name} (ID: {self.id})")
    self.running = True
    
    try:
        # 主执行循环
        while self.running:
            await self.process_messages()
            await self._monitor_agent_health()
            await asyncio.sleep(0.05)  # 防止忙等待
    finally:
        await self.stop()
```

### 智能体停止

```python
async def stop(self):
    """停止智能体"""
    self.running = False
    await self.cleanup()
    
async def cleanup(self):
    """清理资源"""
    for tool_name, tool_instance in self.available_tools.tool_map.items():
        if hasattr(tool_instance, "cleanup"):
            await tool_instance.cleanup()
```

## 协议特定实现

不同协议需要实现各自的通信逻辑：

### Dummy 协议示例

```python
class DummyAgent(MeshAgent):
    """Dummy 协议智能体实现"""
    
    def __init__(self, node_id: int, name: str, tool: str, port: int, 
                 config: Dict[str, Any], task_id: Optional[str] = None, 
                 router_url: str = "dummy://localhost:8000"):
        super().__init__(node_id, name, tool, port, config, task_id)
        self._client = DummyClient(router_url, str(node_id))
    
    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """通过 Dummy 协议发送消息"""
        if not self._connected:
            await self.connect()
        
        success = await self._client.send_message(str(dst), payload)
        if success:
            self._log(f"Successfully sent message to agent {dst}")
    
    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """通过 Dummy 协议接收消息"""
        if not self._connected:
            await self.connect()
        
        return await self._client.receive_message(timeout)
```

## 状态管理

### 智能体状态

```python
class AgentState(str, Enum):
    """智能体执行状态"""
    IDLE = "IDLE"           # 空闲状态
    RUNNING = "RUNNING"     # 运行状态  
    FINISHED = "FINISHED"   # 完成状态
    ERROR = "ERROR"         # 错误状态
```

### 执行状态

```python
class ExecutionStatus(str, Enum):
    """执行状态"""
    PENDING = "pending"         # 等待执行
    PROCESSING = "processing"   # 正在处理
    SUCCESS = "success"         # 执行成功
    ERROR = "error"            # 执行错误
    TIMEOUT = "timeout"        # 执行超时
```

## 最佳实践

### 1. 错误处理

```python
async def process_messages(self):
    try:
        msg = await self.recv_msg(timeout=0.0)
        if msg:
            # 处理消息
            result = await self.step()
    except Exception as e:
        self._log(f"Error processing message: {e}")
        error_msg = Message.assistant_message(f"Error: {e}")
        self.memory.add_message(error_msg)
```

### 2. 资源管理

```python
async def cleanup(self):
    """正确清理资源"""
    for tool_name, tool_instance in self.available_tools.tool_map.items():
        if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(tool_instance.cleanup):
            try:
                await tool_instance.cleanup()
            except Exception as e:
                self._log(f"Error cleaning up tool '{tool_name}': {e}")
```

### 3. 性能监控

```python
async def _monitor_agent_health(self):
    """监控智能体健康状态"""
    if self.token_used > self.max_tokens * 0.9:
        self._log(f"Warning: Token usage approaching limit ({self.token_used}/{self.max_tokens})")
```

### 4. 日志记录

```python
def _log(self, message: str):
    """带上下文的日志记录"""
    logger.info(f"[Agent-{self.id}:{self.name}] {message}")
```

## 开发新智能体类型

要开发新的智能体类型，请遵循以下步骤：

1. **继承 MeshAgent**: 从 `MeshAgent` 基类开始
2. **实现通信方法**: 必须实现 `send_msg()` 和 `recv_msg()`
3. **添加协议逻辑**: 根据您的协议实现特定的通信逻辑
4. **处理生命周期**: 正确实现 `connect()`, `disconnect()`, `start()`, `stop()`
5. **集成工具系统**: 利用现有的工具注册和执行机制
6. **测试验证**: 使用 `test_protocol.py` 验证实现

通过遵循这些原则，您可以创建功能强大且可靠的智能体实现。
