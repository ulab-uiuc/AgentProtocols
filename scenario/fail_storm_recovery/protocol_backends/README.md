# Protocol Backends 扩展指南

## 概述

本指南详细说明如何为 Fail-Storm Recovery 场景添加新的通信协议支持。协议后端系统采用插件化架构，通过抽象基类和工厂模式实现协议的热插拔。

## 🏗️ 架构设计

### 核心组件

```
protocol_backends/
├── base_runner.py                 # 抽象基类，定义通用接口
├── __init__.py                   # 工厂注册和导入
├── anp/                         # ANP 协议实现示例
│   ├── __init__.py
│   ├── runner.py               # ANPRunner 具体实现
│   └── config.yaml            # 协议特定配置
└── simple_json/                # Simple JSON 协议实现示例
    ├── __init__.py
    ├── runner.py              # SimpleJsonRunner 具体实现
    └── config.yaml           # 协议特定配置
```

### 设计原则

1. **协议无关性**: 核心逻辑与具体协议解耦
2. **统一接口**: 所有协议实现相同的抽象接口
3. **配置分离**: 每个协议维护独立的配置文件
4. **可扩展性**: 新增协议无需修改现有代码

## 📋 添加新协议的步骤

### 第一步：创建协议目录

```bash
mkdir protocol_backends/your_protocol/
touch protocol_backends/your_protocol/__init__.py
touch protocol_backends/your_protocol/runner.py
touch protocol_backends/your_protocol/config.yaml
```

### 第二步：实现协议 Runner

创建 `protocol_backends/your_protocol/runner.py`：

```python
import asyncio
from typing import Dict, Any, Optional, Set
from pathlib import Path

from ..base_runner import FailStormRunnerBase

class YourProtocolRunner(FailStormRunnerBase):
    """Your Protocol 的 Fail-Storm 场景运行器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)
        
        # 协议特定的初始化
        self.config["scenario"]["protocol"] = "your_protocol"
        
        # 添加协议特定的属性
        self.your_protocol_specific_data = {}
    
    # === 必须实现的抽象方法 ===
    
    async def create_agent(self, agent_id: str, port: int, data_file: str) -> bool:
        """
        创建协议特定的 Agent
        
        Args:
            agent_id: Agent 唯一标识符
            port: 分配给 Agent 的端口
            data_file: Agent 的数据文件路径
            
        Returns:
            bool: 创建是否成功
        """
        try:
            # 示例：创建你的协议的 Agent
            # agent = YourProtocolAgent(agent_id, port, data_file)
            # await agent.start()
            
            # 记录到 agents 字典
            # self.agents[agent_id] = agent
            
            self.output.success(f"🚀 [YOUR_PROTOCOL] Created {agent_id} on port {port}")
            return True
            
        except Exception as e:
            self.output.error(f"❌ [YOUR_PROTOCOL] Failed to create {agent_id}: {e}")
            return False
    
    def get_protocol_info(self) -> Dict[str, Any]:
        """返回协议的显示信息"""
        return {
            "name": "Your Protocol",
            "description": "Your protocol description",
            "features": ["feature1", "feature2"],
            "agent_count": len(self.agents)
        }
    
    def get_reconnection_info(self) -> Dict[str, Any]:
        """返回重连过程的显示信息"""
        return {
            "reconnection_method": "Your reconnection method",
            "authentication": "Your auth method",
            "estimated_time": "~X seconds"
        }
    
    # === 可选的协议特定方法 ===
    
    async def _setup_mesh_topology(self) -> bool:
        """建立协议特定的网格拓扑"""
        try:
            # 实现你的协议的网格连接逻辑
            self.output.info("🔗 [YOUR_PROTOCOL] Setting up mesh topology...")
            
            # 示例：连接所有 agents
            for agent_id in self.agents:
                for other_id in self.agents:
                    if agent_id != other_id:
                        # await self.connect_agents(agent_id, other_id)
                        pass
            
            self.output.success("🔗 [YOUR_PROTOCOL] Mesh topology established")
            return True
            
        except Exception as e:
            self.output.error(f"❌ [YOUR_PROTOCOL] Mesh setup failed: {e}")
            return False
    
    async def _broadcast_document(self) -> bool:
        """协议特定的文档广播"""
        try:
            self.output.info("📡 [YOUR_PROTOCOL] Broadcasting document...")
            
            # 实现你的协议的文档广播逻辑
            # success_count = await self.mesh_network.broadcast_document(...)
            
            self.output.success("📡 [YOUR_PROTOCOL] Document broadcast completed")
            return True
            
        except Exception as e:
            self.output.error(f"❌ [YOUR_PROTOCOL] Document broadcast failed: {e}")
            return False
    
    async def _execute_normal_phase(self, duration: float) -> None:
        """执行正常阶段的 QA 任务"""
        try:
            self.output.info(f"🔍 [YOUR_PROTOCOL] Running QA for {duration}s...")
            
            # 启动所有 agents 的 QA 任务
            tasks = []
            for agent_id in self.agents:
                task = asyncio.create_task(self._run_qa_task_for_agent(agent_id, duration))
                tasks.append(task)
            
            # 等待所有任务完成
            await asyncio.gather(*tasks, return_exceptions=True)
            
            self.output.success(f"🔍 [YOUR_PROTOCOL] Normal phase completed in {duration:.2f}s")
            
        except Exception as e:
            self.output.error(f"❌ [YOUR_PROTOCOL] Normal phase failed: {e}")
    
    async def _inject_faults(self, kill_count: int) -> Set[str]:
        """注入故障，返回被杀死的 agent IDs"""
        import random
        
        # 随机选择要杀死的 agents
        available_agents = list(self.agents.keys())
        victims = random.sample(available_agents, min(kill_count, len(available_agents)))
        
        killed_agents = set()
        
        for agent_id in victims:
            try:
                self.output.warning(f"💥 [YOUR_PROTOCOL] Killing {agent_id}...")
                
                # 实现协议特定的清理逻辑
                # await self.cleanup_agent(agent_id)
                
                # 终止进程
                agent = self.agents[agent_id]
                # await agent.terminate()
                
                killed_agents.add(agent_id)
                
                self.output.warning(f"💥 [YOUR_PROTOCOL] Killed {agent_id}")
                
            except Exception as e:
                self.output.error(f"❌ [YOUR_PROTOCOL] Failed to kill {agent_id}: {e}")
        
        return killed_agents
    
    async def _monitor_recovery(self) -> None:
        """监控恢复过程"""
        recovery_duration = self.config.get("scenario", {}).get("recovery_duration", 60)
        start_time = time.time()
        
        try:
            # 启动幸存 agents 的恢复任务
            surviving_agents = [aid for aid in self.agents if aid not in self.killed_agents]
            
            # 监控重连过程
            while time.time() - start_time < recovery_duration:
                await asyncio.sleep(5)
                elapsed = time.time() - start_time
                remaining = recovery_duration - elapsed
                
                self.output.info(f"🔄 [YOUR_PROTOCOL] Recovery: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
                
                # 检查并重连被杀死的 agents
                await self._attempt_reconnections()
            
            self.output.success(f"🔄 [YOUR_PROTOCOL] Recovery completed")
            
        except Exception as e:
            self.output.error(f"❌ [YOUR_PROTOCOL] Recovery monitoring failed: {e}")
    
    async def _attempt_reconnections(self) -> None:
        """尝试重连被杀死的 agents"""
        for agent_id in list(self.killed_agents):
            try:
                # 实现协议特定的重连逻辑
                # success = await self.reconnect_agent(agent_id)
                # if success:
                #     self.killed_agents.remove(agent_id)
                #     self.output.success(f"✅ [YOUR_PROTOCOL] {agent_id} reconnected")
                pass
                
            except Exception as e:
                self.output.error(f"❌ [YOUR_PROTOCOL] Failed to reconnect {agent_id}: {e}")
    
    # === 辅助方法 ===
    
    async def _run_qa_task_for_agent(self, agent_id: str, duration: float) -> None:
        """为单个 agent 运行 QA 任务"""
        # 实现协议特定的 QA 任务逻辑
        pass
```

### 第三步：创建协议配置

创建 `protocol_backends/your_protocol/config.yaml`：

```yaml
scenario:
  protocol: "your_protocol"
  agent_count: 8
  runtime: 120.0
  fault_time: 60.0
  recovery_duration: 60
  fault_percentage: 0.25

llm:
  type: "nvidia"
  model: "nvdev/nvidia/llama-3.1-nemotron-70b-instruct"
  base_url: "https://integrate.api.nvidia.com/v1"
  nvidia_api_key: "${NVIDIA_API_KEY}"
  max_tokens: 1000
  temperature: 0.1

your_protocol:
  # 协议特定的配置参数
  connection_timeout: 30
  retry_attempts: 3
  custom_parameter: "value"

network:
  base_port: 9000
  heartbeat_interval: 5.0
  connection_timeout: 10.0

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 第四步：确保类命名规范

确保您的 Runner 类遵循命名规范：

```python
# 在 protocol_backends/your_protocol/runner.py 中

class YourProtocolRunner(FailStormRunnerBase):  # ✅ 正确的命名
    """Your Protocol 的 Fail-Storm 场景运行器"""
    # ... 实现代码 ...
```

### 第五步：注册协议

修改 `fail_storm_runner.py` 中的 `ProtocolRunnerFactory.RUNNERS` 字典，添加你的协议：

```python
class ProtocolRunnerFactory:
    RUNNERS = {
        "simple_json": SimpleJsonRunner,
        "anp": ANPRunner,
        "your_protocol": YourProtocolRunner,  # 添加这行
        # Add more protocols here as they are implemented:
        # "a2a": A2ARunner,
        # "acp": ACPRunner,
    }
```

同时在文件顶部添加导入：

```python
# Import protocol-specific runners
from protocol_backends.simple_json.runner import SimpleJsonRunner
from protocol_backends.anp.runner import ANPRunner
from protocol_backends.your_protocol.runner import YourProtocolRunner  # 添加这行
```

## 🧪 测试新协议

### 基本功能测试

```bash
# 测试协议创建
python fail_storm_runner.py --protocol your_protocol --agents 3 --runtime 30 --fault-time 15

# 测试参数帮助
python fail_storm_runner.py --help
```

### 验证检查清单

- [ ] 协议能够成功创建指定数量的 Agent
- [ ] Mesh 网络拓扑正确建立
- [ ] 文档广播功能正常
- [ ] QA 任务能够正常执行
- [ ] 故障注入机制工作正常
- [ ] 重连和恢复逻辑正确
- [ ] 指标收集和报告完整
- [ ] 配置文件格式正确
- [ ] 错误处理健壮

## 📊 关键接口说明

### 必须实现的抽象方法

| 方法 | 作用 | 返回值 |
|------|------|--------|
| `create_agent()` | 创建协议特定的 Agent | `bool` |
| `get_protocol_info()` | 返回协议显示信息 | `Dict[str, Any]` |
| `get_reconnection_info()` | 返回重连信息 | `Dict[str, Any]` |

### 可选重写的方法

| 方法 | 作用 | 默认行为 |
|------|------|----------|
| `_setup_mesh_topology()` | 建立网格拓扑 | 调用基类实现 |
| `_broadcast_document()` | 广播文档 | 调用基类实现 |
| `_execute_normal_phase()` | 执行正常阶段 | 通用 QA 任务 |
| `_inject_faults()` | 注入故障 | 随机杀死 Agent |
| `_monitor_recovery()` | 监控恢复 | 基本重连逻辑 |

## 🔧 调试和日志

### 日志输出规范

使用统一的日志格式：

```python
# 成功操作
self.output.success(f"✅ [YOUR_PROTOCOL] Operation successful")

# 信息提示
self.output.info(f"🔍 [YOUR_PROTOCOL] Processing...")

# 警告消息
self.output.warning(f"⚠️ [YOUR_PROTOCOL] Warning message")

# 错误消息
self.output.error(f"❌ [YOUR_PROTOCOL] Error occurred")

# 进度更新
self.output.progress(f"📊 [YOUR_PROTOCOL] Progress: {percent}%")
```

### 调试技巧

1. **使用较少的 Agent**: 开发时使用 3-4 个 Agent
2. **缩短运行时间**: 使用 30-60 秒进行快速测试
3. **启用详细日志**: 在配置中设置 `logging.level: DEBUG`
4. **逐步测试**: 先测试 Agent 创建，再测试网络，最后测试故障恢复

## 📚 参考实现

### Simple JSON 协议
- **位置**: `protocol_backends/simple_json/`
- **特点**: 简单的 HTTP JSON 通信
- **适用**: 快速原型和基础测试

### ANP 协议
- **位置**: `protocol_backends/anp/`
- **特点**: DID 认证、E2E 加密、混合通信
- **适用**: 高安全性要求的生产环境

## 🤝 贡献指南

1. **遵循命名规范**: 使用小写字母和下划线
2. **完善错误处理**: 确保所有异常都被正确处理
3. **编写单元测试**: 为关键功能编写测试
4. **更新文档**: 同步更新相关文档
5. **性能考虑**: 注意资源使用和清理

## ❓ 常见问题

### Q: 如何处理协议特定的依赖？
A: 将依赖放在协议目录下的 `requirements.txt` 中，并在 Runner 初始化时动态导入。

### Q: 如何自定义指标收集？
A: 重写 `_finalize_scenario()` 方法，添加协议特定的指标到结果字典中。

### Q: 如何处理不同的认证机制？
A: 在协议配置中定义认证参数，在 `create_agent()` 方法中实现特定的认证逻辑。

### Q: 如何优化大量 Agent 的性能？
A: 使用异步批处理、连接池和资源复用技术，避免创建过多的并发连接。