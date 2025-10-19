# 网络架构说明

本文档详细说明 GAIA 框架的网络架构设计，包括核心概念、通信机制和扩展方法。

## 网络架构概述

GAIA 框架采用 **网状网络 (Mesh Network)** 架构，支持多种通信协议和拓扑结构：

```
网络架构层次:
Protocol Factory (协议工厂)
    ↓
MeshNetwork (抽象网络基类)
    ↓
[Protocol]Network (协议特定实现)
    ↓
MeshAgent (网络智能体)
```

## 核心组件

### 1. MeshNetwork - 网络基类

`MeshNetwork` 是所有网络实现的抽象基类，定义了标准的网络接口：

```python
class MeshNetwork(ABC):
    """网状网络基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.agents: List[MeshAgent] = []
        self.config: Dict[str, Any] = config
        self.memory: Memory = Memory()
        self.network_memory: NetworkMemoryPool = NetworkMemoryPool()
        self.running = False
        
        # 网络指标
        self.bytes_tx = 0      # 发送字节数
        self.bytes_rx = 0      # 接收字节数
        self.pkt_cnt = 0       # 包计数
        self.start_ts = None   # 开始时间戳
        self.done_ts = None    # 结束时间戳

    # 抽象方法：必须由子类实现
    @abstractmethod
    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """投递消息到指定智能体"""
        pass
```

### 2. 消息传递机制

#### 点对点消息投递

```python
async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
    """
    投递消息到指定智能体
    
    Args:
        dst: 目标智能体 ID
        msg: 消息载荷
    """
    # 实现协议特定的消息投递逻辑
    target_agent = self.get_agent_by_id(dst)
    if target_agent:
        # 添加网络元数据
        enhanced_msg = {
            **msg,
            "_network_meta": {
                "delivered_by": "network",
                "target_agent": dst,
                "delivery_timestamp": time.time()
            }
        }
        
        # 协议特定的投递实现
        await self._protocol_specific_deliver(target_agent, enhanced_msg)
        
        # 更新网络指标
        self._update_metrics(msg)
```

#### 消息轮询机制

```python
async def poll(self) -> List[Tuple[int, Dict[str, Any]]]:
    """
    轮询所有智能体的消息
    
    Returns:
        List of (sender_id, message) tuples
    """
    all_messages = []
    
    for agent in self.agents:
        try:
            msg = await agent.recv_msg(timeout=0.0)  # 非阻塞
            if msg:
                all_messages.append((agent.id, msg))
                self._update_rx_metrics(msg)
        except Exception as e:
            # 处理接收错误
            pass
    
    return all_messages
```

### 3. 智能体管理

#### 智能体注册

```python
def register_agent(self, agent: MeshAgent) -> None:
    """注册新智能体到网络"""
    self.agents.append(agent)
    print(f"📝 Registered agent {agent.id} ({agent.name}) with tool {agent.tool_name}")

def unregister_agent(self, agent_id: int) -> None:
    """从网络中移除智能体"""
    self.agents = [agent for agent in self.agents if agent.id != agent_id]
    print(f"🗑️ Unregistered agent {agent_id}")

def get_agent_by_id(self, agent_id: int) -> Optional[MeshAgent]:
    """根据 ID 获取智能体"""
    for agent in self.agents:
        if agent.id == agent_id:
            return agent
    return None
```

#### 网络生命周期

```python
async def start(self):
    """启动网络和所有智能体"""
    print("🌐 Starting multi-agent network...")
    self.start_ts = time.time() * 1000
    
    # 并发启动所有智能体
    agent_tasks = []
    for agent in self.agents:
        task = asyncio.create_task(agent.start())
        agent_tasks.append(task)
    
    self._agent_tasks = agent_tasks
    self.running = True
    print("🚀 Network started successfully")

async def stop(self):
    """停止网络和所有智能体"""
    print("🛑 Stopping network...")
    self.running = False
    
    # 取消所有智能体任务
    if hasattr(self, '_agent_tasks'):
        for task in self._agent_tasks:
            if not task.done():
                task.cancel()
    
    # 并发停止所有智能体
    stop_tasks = []
    for agent in self.agents:
        task = asyncio.create_task(agent.stop())
        stop_tasks.append(task)
    
    if stop_tasks:
        await asyncio.gather(*stop_tasks, return_exceptions=True)
    
    print("✅ Network stopped")
```

## 工作流执行引擎

### 工作流配置

GAIA 支持声明式的工作流配置：

```yaml
workflow:
  start_agent: 0                    # 起始智能体
  execution_pattern: "sequential"   # 执行模式：sequential/parallel
  message_flow:                     # 消息流向定义
    - from: 0                       # 源智能体
      to: [1]                       # 目标智能体列表
      message_type: "task"          # 消息类型
    - from: 1
      to: [2]
      message_type: "result"
    - from: 2
      to: "final"                   # "final" 表示工作流结束
      message_type: "final_result"
```

### 工作流执行

```python
async def execute_workflow(self, config: Dict[str, Any], initial_task: str = None) -> str:
    """
    执行工作流
    
    Args:
        config: 工作流配置
        initial_task: 初始任务
        
    Returns:
        最终执行结果
    """
    workflow = config.get('workflow', {})
    message_flow = workflow.get('message_flow', [])
    
    print(f"🚀 Starting workflow execution")
    
    current_input = initial_task or "Begin task execution"
    workflow_results = {}
    final_result = None
    
    # 按照消息流执行步骤
    for step_idx, flow_step in enumerate(message_flow):
        from_agent_id = flow_step.get('from')
        to_agents = flow_step.get('to')
        message_type = flow_step.get('message_type', 'task_result')
        
        print(f"🔄 Processing step {step_idx + 1}: Agent {from_agent_id} -> {to_agents}")
        
        try:
            # 构建上下文消息
            context_message = self._build_workflow_context(current_input, step_idx)
            
            # 执行智能体步骤
            agent_result = await self._execute_agent_step(from_agent_id, context_message, step_idx)
            workflow_results[f'step_{step_idx}'] = agent_result
            
            # 检查是否为最终步骤
            if to_agents == 'final' or to_agents == ['final']:
                final_result = agent_result
                break
            
            current_input = agent_result
            
        except Exception as e:
            print(f"❌ Agent {from_agent_id} failed: {e}")
            current_input = f"Previous agent failed: {e}"
            workflow_results[f'step_{step_idx}'] = f"FAILED: {e}"
            continue
    
    await self._log_message_pool_to_workspace()
    return final_result or "No results generated"
```

### 智能体步骤执行

```python
async def _execute_agent_step(self, agent_id: int, context_message: str, step_idx: int) -> str:
    """
    执行单个智能体步骤
    
    Args:
        agent_id: 目标智能体 ID
        context_message: 上下文消息
        step_idx: 步骤索引
        
    Returns:
        智能体执行结果
    """
    agent = self.get_agent_by_id(agent_id)
    if not agent:
        raise AgentTaskFailed(f"Agent {agent_id} not found")
    
    print(f"🔄 Executing agent {agent_id} for step {step_idx + 1}")
    
    # 添加步骤执行记录
    self.network_memory.add_step_execution(
        step=step_idx, 
        agent_id=str(agent_id), 
        agent_name=agent.name,
        task_id=self.task_id,
        user_message=context_message
    )
    
    # 设置结果捕获
    result_container = {"result": None, "received": False}
    
    async def capture_result(message_data):
        result_container["result"] = message_data["assistant_response"]
        result_container["received"] = True
    
    try:
        # 更新步骤状态
        self.network_memory.update_step_status(step_idx, ExecutionStatus.PROCESSING)
        
        # 设置回调并发送消息
        agent.set_result_callback(capture_result)
        await self.deliver(agent_id, {
            "type": "task_execution",
            "sender_id": 0,
            "content": context_message,
            "step": step_idx,
            "timestamp": time.time()
        })
        
        # 等待结果
        timeout = 10.0
        elapsed = 0.0
        poll_interval = 0.1
        
        while not result_container["received"] and elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        
        agent.set_result_callback(None)
        
        if not result_container["received"]:
            raise AgentTaskTimeout(f"Agent {agent_id} timeout")
        
        result = result_container["result"]
        if not result or not result.strip():
            result = "No meaningful result generated"
            self.network_memory.update_step_status(step_idx, ExecutionStatus.ERROR)
        else:
            messages = agent.get_memory_messages()
            self.network_memory.update_step_status(step_idx, ExecutionStatus.SUCCESS, messages=messages)
        
        print(f"✅ Agent {agent_id} completed step {step_idx + 1}")
        return result
        
    except Exception as e:
        agent.set_result_callback(None)
        self.network_memory.update_step_status(step_idx, ExecutionStatus.ERROR, error_message=str(e))
        raise AgentTaskFailed(f"Agent {agent_id} execution failed: {e}")
```

## 内存管理系统

### 网络级内存池

```python
class NetworkMemoryPool(BaseModel):
    """网络级内存池，跟踪所有智能体执行"""
    
    step_executions: Dict[int, StepExecution] = Field(default_factory=dict)
    max_executions: int = Field(default=1000)
    
    def add_step_execution(self, step: int, agent_id: str, agent_name: str, 
                          task_id: str, user_message: str) -> str:
        """添加步骤执行记录"""
        step_execution = StepExecution(
            step=step,
            agent_id=agent_id,
            agent_name=agent_name
        )
        self.step_executions[step] = step_execution
        return f"{agent_id}_{task_id}_{step}"
    
    def get_step_chain_context(self, current_step: int) -> List[Dict[str, Any]]:
        """获取前序步骤的上下文"""
        context = []
        for step in range(current_step):
            if step in self.step_executions:
                step_exec = self.step_executions[step]
                if step_exec.is_completed() and step_exec.messages:
                    assistant_messages = [msg for msg in step_exec.messages if msg.role == "assistant"]
                    if assistant_messages:
                        context.append({
                            "step": step,
                            "agent_id": step_exec.agent_id,
                            "agent_name": step_exec.agent_name,
                            "result": assistant_messages[-1].content,
                            "status": step_exec.execution_status.value
                        })
        return context
```

### 上下文构建

```python
def _build_workflow_context(self, current_input: str, step_idx: int) -> str:
    """构建工作流上下文消息"""
    context_parts = [f"Step {step_idx + 1} - Task Input:"]
    context_parts.append(current_input)
    
    # 添加前序步骤的上下文
    if step_idx > 0:
        previous_context = self.network_memory.get_step_chain_context(step_idx)
        
        if previous_context:
            context_parts.append("\nPrevious Steps Context:")
            for ctx in previous_context:
                step_num = ctx["step"] + 1
                agent_name = ctx["agent_name"]
                result = ctx["result"]
                status = ctx["status"]
                
                context_parts.append(f"Step {step_num} ({agent_name} - {status}): {result[:200]}...")
    
    return "\n".join(context_parts)
```

## 性能指标和监控

### 网络指标

```python
class MeshNetwork:
    def _update_metrics(self, msg: Dict[str, Any]):
        """更新网络指标"""
        self.pkt_cnt += 1
        msg_size = len(json.dumps(msg).encode('utf-8'))
        self.bytes_tx += msg_size
    
    def _update_rx_metrics(self, msg: Dict[str, Any]):
        """更新接收指标"""
        msg_size = len(json.dumps(msg).encode('utf-8'))
        self.bytes_rx += msg_size
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """获取网络性能指标"""
        elapsed_ms = 0
        if self.done_ts and self.start_ts:
            elapsed_ms = self.done_ts - self.start_ts
        elif self.start_ts:
            elapsed_ms = time.time() * 1000 - self.start_ts
        
        return {
            "bytes_tx": self.bytes_tx,
            "bytes_rx": self.bytes_rx,
            "pkt_cnt": self.pkt_cnt,
            "elapsed_ms": elapsed_ms,
            "num_agents": len(self.agents)
        }
```

### 消息池日志

```python
async def _log_message_pool_to_workspace(self):
    """记录消息池到工作空间"""
    try:
        task_id = self.config.get("task_id", "unknown")
        logs_dir = Path(f"workspaces/{task_id}/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存网络内存和步骤执行记录
        network_log = {
            "network_memory": [msg.to_dict() for msg in self.memory.messages],
            "workflow_progress": self.network_memory.get_workflow_progress(),
            "step_executions": {},
            "timestamp": time.time()
        }
        
        # 保存步骤执行记录
        for step, step_exec in self.network_memory.step_executions.items():
            network_log["step_executions"][f"step_{step}"] = {
                "step": step_exec.step,
                "agent_id": step_exec.agent_id,
                "agent_name": step_exec.agent_name,
                "status": step_exec.execution_status.value,
                "duration": step_exec.duration(),
                "messages": [msg.to_dict() for msg in step_exec.messages],
                "error_message": step_exec.error_message
            }
        
        # 写入文件
        log_file = logs_dir / "network_execution_log.json"
        with open(log_file, "w", encoding='utf-8') as f:
            json.dump(network_log, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Network execution log saved to {log_file}")
        
    except Exception as e:
        print(f"❌ Failed to log message pool: {e}")
```

## 协议扩展指南

### 创建新协议

1. **继承 MeshNetwork**: 从 `MeshNetwork` 基类开始
2. **实现抽象方法**: 必须实现 `deliver()` 方法
3. **添加协议逻辑**: 实现协议特定的通信机制
4. **注册到工厂**: 在 `ProtocolFactory` 中注册新协议

### 示例协议实现

```python
class CustomProtocolNetwork(MeshNetwork):
    """自定义协议网络实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config=config)
        self.protocol_name = "custom"
        # 协议特定初始化
    
    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """协议特定的消息投递实现"""
        target_agent = self.get_agent_by_id(dst)
        if target_agent:
            # 添加协议元数据
            enhanced_msg = {
                **msg,
                "_protocol_meta": {
                    "protocol": self.protocol_name,
                    "delivery_timestamp": time.time()
                }
            }
            
            # 实现您的投递逻辑
            await self._custom_deliver_logic(target_agent, enhanced_msg)
            
            # 更新指标
            self._update_metrics(msg)
    
    async def _custom_deliver_logic(self, target_agent, enhanced_msg):
        """实现协议特定的投递逻辑"""
        # 根据您的协议实现具体的投递机制
        pass
```

## 错误处理和恢复

### 智能体故障处理

```python
class AgentTaskTimeout(Exception):
    """智能体任务超时异常"""
    pass

class AgentTaskFailed(Exception):
    """智能体任务失败异常"""
    pass

# 在工作流执行中处理异常
try:
    agent_result = await self._execute_agent_step(from_agent_id, context_message, step_idx)
except (AgentTaskTimeout, AgentTaskFailed) as e:
    print(f"❌ Agent {from_agent_id} failed: {e}")
    # 继续下一个智能体或采取恢复措施
    current_input = f"Previous agent failed: {e}"
    continue
```

### 网络恢复机制

```python
async def _monitor_network_health(self):
    """监控网络健康状态"""
    failed_agents = []
    
    for agent in self.agents:
        if not getattr(agent, 'running', False):
            failed_agents.append(agent.id)
    
    if failed_agents:
        print(f"⚠️ Detected failed agents: {failed_agents}")
        # 实现恢复逻辑
        await self._recover_failed_agents(failed_agents)

async def _recover_failed_agents(self, failed_agent_ids: List[int]):
    """恢复失败的智能体"""
    for agent_id in failed_agent_ids:
        agent = self.get_agent_by_id(agent_id)
        if agent:
            try:
                await agent.start()
                print(f"✅ Recovered agent {agent_id}")
            except Exception as e:
                print(f"❌ Failed to recover agent {agent_id}: {e}")
```

## 总结

GAIA 的网络架构提供了：

1. **统一接口**: 所有协议都基于相同的抽象接口
2. **灵活扩展**: 易于添加新的通信协议
3. **工作流支持**: 内置的工作流编排和执行引擎
4. **性能监控**: 完整的网络和智能体性能指标
5. **错误恢复**: 健壮的错误处理和恢复机制
6. **内存管理**: 多层级的内存和上下文管理

通过这种设计，GAIA 框架能够支持各种复杂的多智能体协作场景，同时保持良好的可维护性和扩展性。
