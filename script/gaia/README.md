# 协议调试指南 (Protocol Debugging Guide)

这个文档将指导您如何自定义协议并集成到 `test_protocol.py` 测试框架中。

## 目录

1. [协议架构概述](#协议架构概述)
2. [自定义协议开发](#自定义协议开发)
3. [集成到测试框架](#集成到测试框架)
4. [运行测试和预期结果](#运行测试和预期结果)
5. [故障排除](#故障排除)

## 协议架构概述

GAIA 框架使用分层协议架构，主要包含以下组件：

```
protocols/
├── protocol_factory.py     # 协议工厂，管理所有协议
├── dummy/                  # 示例协议实现
│   ├── network.py         # 网络层实现
│   ├── agent.py           # 代理实现
│   └── __init__.py
└── [your_protocol]/       # 您的自定义协议
    ├── network.py
    ├── agent.py
    └── __init__.py
```

### 核心接口

1. **MeshNetwork**: 基础网络类，提供消息传递和代理管理
2. **MeshAgent**: 基础代理类，定义代理行为接口
3. **ProtocolFactory**: 协议工厂，管理协议注册和创建

## 自定义协议开发

### 步骤 1: 创建协议目录结构

```bash
mkdir -p protocols/my_protocol
touch protocols/my_protocol/__init__.py
touch protocols/my_protocol/network.py
touch protocols/my_protocol/agent.py
```

### 步骤 2: 实现网络层 (`network.py`)

您的网络类必须继承 `MeshNetwork` 并实现以下关键方法：

```python
"""
My Protocol Network implementation.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Tuple, Optional
import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.network import MeshNetwork
from protocols.my_protocol.agent import MyProtocolAgent

def create_my_protocol_agent(agent_config: Dict[str, Any], task_id: str) -> MyProtocolAgent:
    """Create a MyProtocolAgent from configuration."""
    return MyProtocolAgent(
        node_id=agent_config['id'],
        name=agent_config['name'],
        tool=agent_config['tool'],
        port=agent_config['port'],
        config={
            'max_tokens': agent_config.get('max_tokens', 500),
            'specialization': agent_config.get('specialization', ''),
            'role': agent_config.get('role', 'agent'),
            'priority': agent_config.get('priority', 1),
            # 添加您的协议特定配置
            'custom_setting': agent_config.get('custom_setting', 'default_value')
        },
        task_id=task_id,
        router_url="my_protocol://localhost:8000"
    )

class MyProtocolNetwork(MeshNetwork):
    """
    自定义协议网络实现
    """
    
    def __init__(self):
        super().__init__()
        # 添加协议特定的初始化
        self.protocol_name = "my_protocol"
        self.protocol_version = "1.0"
    
    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """
        **必须实现**: 向指定代理投递消息
        
        Args:
            dst: 目标代理 ID
            msg: 消息载荷
        """
        target_agent = self.get_agent_by_id(dst)
        if target_agent and isinstance(target_agent, MyProtocolAgent):
            try:
                # 添加网络元数据
                enhanced_msg = {
                    **msg,
                    "_network_meta": {
                        "delivered_by": f"{self.protocol_name}_network",
                        "target_agent": dst,
                        "delivery_timestamp": time.time(),
                        "message_id": f"my_{int(time.time() * 1000000)}",
                        "protocol_version": self.protocol_version
                    }
                }
                
                # 记录消息到消息池（继承自基类）
                await self._record_input_message(dst, enhanced_msg)
                
                # 实现您的消息投递逻辑
                await self._custom_deliver_logic(target_agent, enhanced_msg)
                
                # 更新指标
                self.pkt_cnt += 1
                msg_size = len(json.dumps(msg).encode('utf-8'))
                self.bytes_tx += msg_size
                
                print(f"📤 {self.protocol_name.upper()} delivered message to agent {dst}: {msg.get('type', 'unknown')}")
                
            except Exception as e:
                print(f"❌ {self.protocol_name.upper()} failed to deliver message to agent {dst}: {e}")
        else:
            print(f"❌ Agent {dst} not found or not compatible with {self.protocol_name}")
    
    async def _custom_deliver_logic(self, target_agent, enhanced_msg):
        """实现您的自定义投递逻辑"""
        # 示例：这里我使用模拟协议，直接投递到代理的邮箱，大部分协议应该是直接与对应agent通信
        if hasattr(target_agent, '_client') and hasattr(target_agent._client, '_mailbox'):
            await target_agent._client._mailbox.put(enhanced_msg)
        else:
            # 或者使用代理的 send_msg 方法
            await target_agent.send_msg(target_agent.id, enhanced_msg)
    
    async def poll(self) -> List[Tuple[int, Dict[str, Any]]]:
        """
        **必须实现**: 轮询所有代理的消息
        
        Returns:
            List of (sender_id, message) tuples
        """
        all_messages = []
        
        for agent in self.agents:
            try:
                # 使用非阻塞接收
                msg = await agent.recv_msg(timeout=0.0)
                if msg:
                    # 更新指标
                    self.pkt_cnt += 1
                    msg_size = len(json.dumps(msg).encode('utf-8'))
                    self.bytes_rx += msg_size
                    
                    all_messages.append((agent.id, msg))
                    print(f"📥 {self.protocol_name.upper()} polled message from agent {agent.id}")
            except Exception as e:
                # 非阻塞操作，忽略超时错误
                pass
        
        return all_messages
    
    async def broadcast(self, msg: Dict[str, Any], exclude_sender: Optional[int] = None) -> None:
        """
        **可选实现**: 广播消息到所有代理
        """
        print(f"📡 {self.protocol_name.upper()} broadcasting message: {msg.get('type', 'unknown')}")
        
        broadcast_msg = {
            **msg,
            "_broadcast_meta": {
                "is_broadcast": True,
                "sender_id": exclude_sender,
                "broadcast_timestamp": time.time(),
                "protocol": self.protocol_name
            }
        }
        
        delivered_count = 0
        for agent in self.agents:
            if exclude_sender is None or agent.id != exclude_sender:
                try:
                    await self.deliver(agent.id, broadcast_msg)
                    delivered_count += 1
                except Exception as e:
                    print(f"❌ Failed to broadcast to agent {agent.id}: {e}")
        
        print(f"📡 {self.protocol_name.upper()} broadcast completed to {delivered_count} agents")
```

### 步骤 3: 实现代理层 (`agent.py`)

您的代理类必须继承 `MeshAgent` 并实现通信方法：

```python
"""
My Protocol Agent Implementation.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional
import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.agent_base import MeshAgent

class MyProtocolClient:
    """
    自定义协议客户端，或者使用SDK
    """
    
    def __init__(self, router_url: str, agent_id: str):
        self.router_url = router_url
        self.agent_id = agent_id
        self._connected = False
        
        # 协议特定配置
        self.protocol_settings = {
            'timeout': 30.0,
            'retry_count': 3,
            'compression': False
        }
        
        # 统计信息
        self.sent_messages = 0
        self.received_messages = 0
        
        # 创建消息队列（根据您的协议需求实现）
        self._mailbox = asyncio.Queue()
        
        print(f"[{agent_id}] MyProtocolClient initialized for: {router_url}")
    
    async def connect(self):
        """**必须实现**: 建立连接"""
        try:
            # 实现您的连接逻辑
            # 例如：连接到消息中间件、建立 WebSocket 连接等
            await self._establish_connection()
            self._connected = True
            print(f"[{self.agent_id}] MyProtocolClient connected")
        except Exception as e:
            print(f"[{self.agent_id}] Failed to connect: {e}")
            raise
    
    async def _establish_connection(self):
        """实现实际的连接逻辑"""
        # 示例：模拟连接延迟
        await asyncio.sleep(0.1)
        # 在这里实现您的实际连接逻辑
        pass
    
    async def disconnect(self):
        """**必须实现**: 断开连接"""
        self._connected = False
        print(f"[{self.agent_id}] MyProtocolClient disconnected")
    
    async def send_message(self, dst_id: str, message: Dict[str, Any]) -> bool:
        """**必须实现**: 发送消息"""
        try:
            # 实现您的消息发送逻辑
            enhanced_message = {
                **message,
                "_protocol_meta": {
                    "sender_id": self.agent_id,
                    "recipient_id": dst_id,
                    "timestamp": time.time(),
                    "protocol": "my_protocol"
                }
            }
            
            # 根据您的协议实现消息发送
            success = await self._do_send_message(dst_id, enhanced_message)
            
            if success:
                self.sent_messages += 1
                print(f"MyProtocolClient {self.agent_id}: Sent message to {dst_id}")
            
            return success
            
        except Exception as e:
            print(f"MyProtocolClient {self.agent_id}: Failed to send to {dst_id}: {e}")
            return False
    
    async def _do_send_message(self, dst_id: str, message: Dict[str, Any]) -> bool:
        """实现实际的消息发送逻辑"""
        # 示例：直接投递到目标队列
        # 在真实实现中，这里应该是通过网络发送
        await asyncio.sleep(0.05)  # 模拟网络延迟
        return True
    
    async def receive_message(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """**必须实现**: 接收消息"""
        try:
            if timeout == 0.0:
                if self._mailbox.empty():
                    return None
                message = self._mailbox.get_nowait()
            else:
                message = await asyncio.wait_for(self._mailbox.get(), timeout=timeout)
            
            self.received_messages += 1
            
            # 清理协议元数据
            if isinstance(message, dict) and "_protocol_meta" in message:
                clean_message = {k: v for k, v in message.items() if k != "_protocol_meta"}
                return clean_message
            
            return message
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            print(f"MyProtocolClient {self.agent_id}: Failed to receive: {e}")
            return None

class MyProtocolAgent(MeshAgent):
    """
    自定义协议代理实现
    """
    
    def __init__(self, node_id: int, name: str, tool: str, port: int, 
                 config: Dict[str, Any], task_id: Optional[str] = None, 
                 router_url: str = "my_protocol://localhost:8000"):
        """
        初始化自定义协议代理
        """
        super().__init__(node_id, name, tool, port, config, task_id)
        
        self.router_url = router_url
        self.agent_id = str(node_id)
        
        # 初始化协议客户端
        self._client = MyProtocolClient(router_url, self.agent_id)
        self._connected = False
        
        # 协议特定配置
        self.custom_setting = config.get("custom_setting", "default_value")
        
        self._log(f"MyProtocolAgent initialized with router: {router_url}")
    
    async def connect(self):
        """**必须实现**: 建立连接"""
        if not self._connected:
            try:
                await self._client.connect()
                self._connected = True
                self._log("MyProtocolAgent connected successfully")
            except Exception as e:
                self._log(f"Failed to connect MyProtocolAgent: {e}")
                raise
    
    async def disconnect(self):
        """**必须实现**: 断开连接"""
        if self._client:
            await self._client.disconnect()
        self._connected = False
        self._log("MyProtocolAgent disconnected")
    
    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """
        **必须实现**: 发送消息的具体实现
        """
        try:
            if not self._connected:
                await self.connect()
            
            dst_str = str(dst)
            success = await self._client.send_message(dst_str, payload)
            
            if success:
                self._log(f"Successfully sent message to agent {dst}: {payload.get('type', 'unknown')}")
            else:
                self._log(f"Failed to send message to agent {dst}")
                
        except Exception as e:
            self._log(f"Error sending message to agent {dst}: {e}")
    
    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        **必须实现**: 接收消息的具体实现
        """
        try:
            if not self._connected:
                await self.connect()
            
            message = await self._client.receive_message(timeout)
            
            if message:
                self._log(f"Received message: {message.get('type', 'unknown')}")
                return message
            
            return None
                
        except Exception as e:
            self._log(f"Error receiving message: {e}")
            return None
    
    def get_connection_status(self) -> Dict[str, Any]:
        """获取连接状态和统计信息"""
        return {
            "agent_id": self.id,
            "agent_name": self.name,
            "protocol": "my_protocol",
            "connected": self._connected,
            "router_url": self.router_url,
            "custom_setting": self.custom_setting,
            "statistics": {
                "sent_messages": self._client.sent_messages if self._client else 0,
                "received_messages": self._client.received_messages if self._client else 0,
            }
        }
```
**注意！以上两步均为protocol-specific的，只起到参考作用！！！**

### 步骤 4: 注册协议到工厂

编辑 `protocols/protocol_factory.py`，添加您的协议：

```python
# 在文件顶部添加导入
from protocols.my_protocol.network import MyProtocolNetwork, create_my_protocol_agent

class ProtocolFactory:
    def __init__(self):
        self.protocols = {
            'dummy': {
                'network_class': DummyNetwork,
                'create_agent_func': create_dummy_agent,
                'description': 'Dummy protocol for testing'
            },
            'my_protocol': {  # 添加您的协议
                'network_class': MyProtocolNetwork,
                'create_agent_func': create_my_protocol_agent,
                'description': 'My custom protocol implementation'
            }
        }
```

## 集成到测试框架

### 使用 test_protocol.py 测试

现在您可以使用 `test_protocol.py` 测试您的自定义协议：

```bash
# 测试您的自定义协议
cd /root/Multiagent-Protocol/script/gaia/scripts
python test_protocol.py --protocol my_protocol

# 列出所有可用协议
python test_protocol.py --list
```

### 测试脚本功能

`test_protocol.py` 会自动执行以下测试：

1. **协议工厂测试**: 验证协议是否正确注册
2. **代理创建测试**: 创建多个代理实例
3. **网络启动测试**: 启动网络和所有代理
4. **消息投递测试**: 测试点对点消息传递
5. **消息轮询测试**: 测试消息接收功能
6. **消息池测试**: 验证消息记录和统计
7. **网络指标测试**: 检查性能指标
8. **清理测试**: 正确停止网络和代理

## 运行测试和预期结果

### 成功运行的输出示例

```
🧪 Testing Network Deliver Functionality - Protocol: MY_PROTOCOL
--------------------------------------------------
✅ Created agent 0: TestAgent_0 (Protocol: my_protocol)
✅ Created agent 1: TestAgent_1 (Protocol: my_protocol)
✅ Created agent 2: TestAgent_2 (Protocol: my_protocol)

🌐 Starting network...
✅ Started agent 0 (TestAgent_0) on port 9000
✅ Started agent 1 (TestAgent_1) on port 9001  
✅ Started agent 2 (TestAgent_2) on port 9002
🚀 Network started successfully

📤 Testing message delivery...
📨 Sending message 1: Agent 0 -> Agent 1
   Message: {'type': 'greeting', 'content': 'Hello from test!'}
📤 MY_PROTOCOL delivered message to agent 1: greeting

📨 Sending message 2: Agent 1 -> Agent 2
   Message: {'type': 'task', 'content': 'Process this data', 'priority': 1}
📤 MY_PROTOCOL delivered message to agent 2: task

📨 Sending message 3: Agent 2 -> Agent 0
   Message: {'type': 'status', 'content': 'System ready'}
📤 MY_PROTOCOL delivered message to agent 0: status

📊 Checking message pool...
   Total input messages: 3
   Active agents: 3
   Conversation turns: 3

📥 Testing message polling...
📥 MY_PROTOCOL polled message from agent 1
📥 MY_PROTOCOL polled message from agent 2
📥 MY_PROTOCOL polled message from agent 0
   Polled 3 messages

✅ Message delivery test PASSED
   - Messages were successfully delivered
   - Message pool is recording correctly

📝 Testing message pool logging...
📝 Logging message pool to workspace...
📊 Message pool logged to:
  📄 Readable log: workspaces/test_protocol_001/message_pool_log_test_protocol_001.md
  📊 Raw data: workspaces/test_protocol_001/message_pool_data_test_protocol_001.json
✅ Message pool logging completed

📊 Network metrics:
   - Packets sent: 6
   - Bytes TX: 486
   - Bytes RX: 486

🛑 Stopping network...
✅ Stopped agent 0
✅ Stopped agent 1
✅ Stopped agent 2
✅ Network stopped
✅ Test completed

🎉 All tests completed!
```

### 错误情况和调试

#### 常见错误 1: 协议未注册
```
❌ Error: Protocol 'my_protocol' not available.
Available protocols: dummy
```
**解决方案**: 检查 `protocol_factory.py` 中是否正确注册了您的协议。

#### 常见错误 2: 导入错误
```
ModuleNotFoundError: No module named 'protocols.my_protocol.network'
```
**解决方案**: 
- 确保创建了 `__init__.py` 文件
- 检查文件路径和模块名称是否正确

#### 常见错误 3: 方法未实现
```
❌ Agent 1 does not support direct message delivery
```
**解决方案**: 确保您的代理类实现了所有必需的方法。

#### 常见错误 4: 连接失败
```
❌ Failed to connect MyProtocolAgent: Cannot connect to my_protocol://localhost:8000
```
**解决方案**: 检查您的连接逻辑，确保 `connect()` 方法正确实现。

### 性能指标说明

测试完成后，您会看到以下指标：

- **Packets sent**: 发送的数据包数量
- **Bytes TX**: 发送的字节数
- **Bytes RX**: 接收的字节数
- **Message pool**: 消息池统计信息
- **Agent metrics**: 每个代理的详细统计

## 必须实现的方法总结

### 网络层 (Network) 必须实现:

1. `async def deliver(self, dst: int, msg: Dict[str, Any]) -> None`
2. `async def poll(self) -> List[Tuple[int, Dict[str, Any]]]`

### 代理层 (Agent) 必须实现:

1. `async def connect(self) -> None`
2. `async def disconnect(self) -> None` 
3. `async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None`
4. `async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]`

### 工厂函数必须提供:

1. `create_[protocol_name]_agent(agent_config, task_id) -> YourAgent`

### 可选但推荐实现:

1. `async def broadcast(self, msg: Dict[str, Any], exclude_sender: Optional[int] = None) -> None`
2. `def get_connection_status(self) -> Dict[str, Any]`

## 故障排除

### 启用调试日志

在您的协议实现中添加详细日志：

```python
import logging

# 设置调试级别
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 在关键位置添加日志
logger.debug(f"Sending message: {message}")
logger.info(f"Agent {self.id} connected successfully")
logger.error(f"Failed to deliver message: {error}")
```

### 检查消息流

使用消息池功能检查消息流向：

```python
# 在测试脚本中添加
summary = network.get_message_pool_summary()
print(f"Message pool summary: {summary}")

# 获取特定代理的上下文
agent_context = network.get_agent_context(agent_id)
print(f"Agent {agent_id} context: {agent_context}")
```

### 验证协议兼容性

确保您的协议与基础框架兼容：

```python
# 检查代理是否正确继承
assert isinstance(your_agent, MeshAgent)

# 检查网络是否正确继承  
assert isinstance(your_network, MeshNetwork)

# 检查必需方法是否存在
assert hasattr(your_agent, 'send_msg')
assert hasattr(your_agent, 'recv_msg')
assert hasattr(your_network, 'deliver')
assert hasattr(your_network, 'poll')
```

## 高级功能

### 消息池集成

您的协议会自动获得消息池功能，包括：

- 消息历史记录
- 代理上下文跟踪
- 工作流程分析
- 性能指标收集

### 工作流程支持

您的协议将自动支持工作流程执行：

```python
# test_protocol.py 会自动测试工作流程功能
workflow_config = {
    'workflow': {
        'start_agent': 0,
        'execution_pattern': 'sequential',
        'message_flow': [
            {'from': 0, 'to': [1], 'message_type': 'task'},
            {'from': 1, 'to': [2], 'message_type': 'result'},
            {'from': 2, 'to': 'final', 'message_type': 'final_result'}
        ]
    }
}

result = await network.execute_workflow(workflow_config, "Initial task")
```

通过遵循本指南，您应该能够成功创建和测试自定义协议。如果遇到问题，请检查日志输出并确保所有必需的方法都已正确实现。
