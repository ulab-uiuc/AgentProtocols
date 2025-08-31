# Async-MAPF Skeleton

一个协议无关的异步多智能体路径规划（MAPF）框架，支持多种通信后端的热插拔。

## 1 目录总览

```
script/async_mapf/
├── core/                     # ★ 核心算法，协议无关
│   ├── world.py              # GridWorld & collision rules
│   ├── agent_base.py         # BaseRobot (communication stubs)
│   ├── network_base.py       # BaseNet    (communication stubs)
│   └── utils.py              # helpers / shared data structs
├── protocol_backends/        # 🔌 协议实现（每种协议一个子目录）
│   ├── a2a/                  # A2A backend: agent.py + network.py
│   ├── anp/                  # ANP backend: ditto
│   └── dummy/                # In‑memory mock, used by CI / tests
├── runners/                  # 🚀 entry points
│   ├── local_runner.py       # single‑process demo
│   └── distributed_runner.py # placeholder for multi‑node
├── metrics/                  # 📊 runtime metrics
│   ├── recorder.py           # append‑only logger
│   ├── analyzer.py           # offline analysis helpers
│   └── dashboard.py          # (optional) live UI
├── config/                   # ⚙️ YAML configs (select protocol / params)
│   ├── dummy.yaml
│   ├── a2a.yaml
│   ├── anp.yaml
│   └── distributed.yaml
├── examples/                 # 📚 quick demos / notebooks
│   ├── single_node_demo.py
│   ├── protocol_comparison.py
│   └── README.md
└── docs/                     # 📖 Sphinx / MkDocs 源文件
    └── README.md
```

## 2 核心设计原则

### 算法‑协议解耦

`core/` 内文件自洽完成寻路、碰撞检测、时序裁决；仅在通信处保留 **4 个抽象方法**：

- `BaseRobot.send_msg()` / `BaseRobot.recv_msg()`
- `BaseNet.deliver()` / `BaseNet.poll()`

**协议实现者只需覆写这四个方法**即可插入任何通讯栈。

### 每协议一文件夹

在 `protocol_backends/<proto>/` 中提供 `agent.py` 和 `network.py`：

```python
class XxxRobot(BaseRobot):    # 实现两个 send/recv
    async def send_msg(self, dst, payload): ...
    async def recv_msg(self, timeout=0.0): ...

class XxxNet(BaseNet):        # 实现 deliver/poll
    async def deliver(self, dst, msg): ...
    async def poll(self): ...
```

其余算法逻辑保持继承，**不得修改**。

### 配置热插拔

YAML 里声明待加载类的**全限定路径**：

```yaml
agent_cls: "script.async_mapf.protocol_backends.a2a.agent:A2ARobot"
net_cls:   "script.async_mapf.protocol_backends.a2a.network:A2ANet"
```

Runner 使用 `importlib` 动态加载，**替换协议 = 换配置文件**。

### 统一日志 / .gitignore

- 所有运行日志写入 `script/async_mapf/logs/`，由 `metrics/recorder.py` 管理
- 根目录一份 `.gitignore` 即可；已为 logs/、虚拟环境、`__pycache__` 等通用条目留空位

## 3 各目录职责

| 目录 | 说明 | 二次开发关注点 |
|------|------|----------------|
| `core/` | 网格模型、基础寻路、调度循环 | 不可修改算法；如需新特性请先提 Issue |
| `protocol_backends/` | 协议具体实现 | 只写通信层；严禁复制/修改核心算法 |
| `runners/` | 工程入口；解析 YAML → 装配对象 | 如需 CLI 参数、新 Runner 可在此扩展 |
| `metrics/` | 指标收集、输出、可视化 | 记录格式遵循 recorder.py 中的 schema |
| `config/` | 场景 & 协议组合 | 新协议请附带一个最简 YAML |
| `examples/` | 教学或 benchmark 脚本 | 可放入 *.ipynb 或实验脚本 |

## 4 如何新增一个协议

### 创建目录与文件

```bash
mkdir script/async_mapf/protocol_backends/myproto
touch script/async_mapf/protocol_backends/myproto/{__init__,agent,network}.py
```

### 继承并实现通信方法

```python
# agent.py
from ...core.agent_base import BaseRobot

class MyProtoRobot(BaseRobot):
    async def send_msg(self, dst, payload): 
        # 实现发送逻辑
        pass
        
    async def recv_msg(self, timeout=0.0): 
        # 实现接收逻辑
        pass
```

```python
# network.py
from ...core.network_base import BaseNet

class MyProtoNet(BaseNet):
    async def deliver(self, dst, msg): 
        # 实现消息投递
        pass
        
    async def poll(self): 
        # 实现消息轮询
        pass
```

### 添加 YAML 配置

```yaml
# config/myproto.yaml
agent_cls: "script.async_mapf.protocol_backends.myproto.agent:MyProtoRobot"
net_cls:   "script.async_mapf.protocol_backends.myproto.network:MyProtoNet"

# 协议特定配置
myproto_config:
  server_endpoint: "tcp://localhost:5555"
  auth_token: "your-token"
```

### 运行

```bash
python script/async_mapf/runners/local_runner.py --config script/async_mapf/config/myproto.yaml
```

## 5 日志 & 指标

### 指标记录

使用 `metrics.recorder.log(metric_name, value, ts)` 统一落盘（CSV）：

```python
from script.async_mapf.metrics.recorder import MetricsRecorder

recorder = MetricsRecorder("output_dir")
recorder.start_recording()

# 记录自定义指标
recorder.record_metric("custom_metric", 42, agent_id=0)

# 记录网络指标
recorder.record_network_metrics(network.get_performance_metrics())

recorder.stop_recording()
```

### 指标分析

- `metrics.analyzer` 提供简单聚合
- `metrics.dashboard` 计划接入 rich 或 plotly 实时展示
- 默认目录：`script/async_mapf/logs/<date>/`

## 6 代码风格 & 约定

| 项目 | 约定 |
|------|------|
| **语言** | Python ≥ 3.11，全部打开 `from __future__ import annotations` |
| **类型** | 必写函数签名；使用 `mypy --strict` 通过 |
| **文档** | Google style docstring，英文 |
| **依赖** | 核心只依赖标准库；协议层如需外部包请在 `requirements-<proto>.txt` 单独列出 |
| **单测** | pytest; mock 协议请使用 `protocol_backends/dummy` |

### 代码示例

```python
from __future__ import annotations

import asyncio
from typing import Dict, Any, Optional

class ExampleRobot(BaseRobot):
    """Example robot implementation.
    
    This class demonstrates the proper way to implement
    a protocol backend for the MAPF framework.
    """
    
    def __init__(self, aid: int, world: GridWorld, goal: tuple[int, int]) -> None:
        """Initialize the robot.
        
        Args:
            aid: Agent ID
            world: Shared world reference
            goal: Target position (x, y)
        """
        super().__init__(aid, world, goal)
    
    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """Send message to another agent.
        
        Args:
            dst: Destination agent ID
            payload: Message content
        """
        # Implementation here
        pass
    
    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """Receive message with timeout.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            Received message or None if timeout
        """
        # Implementation here
        return None
```

## 7 快速开始

### 安装依赖

```bash
pip install asyncio pyyaml numpy matplotlib pandas
```

### 运行基本示例

```bash
cd script/async_mapf
python examples/single_node_demo.py basic
```

### 查看实时仪表板

```bash
python examples/single_node_demo.py dashboard
```

### 协议比较基准测试

```bash
python examples/protocol_comparison.py
```

## 8 API参考

### BaseRobot接口

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List

class BaseRobot(ABC):
    """Abstract base class for MAPF agents."""
    
    @abstractmethod
    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """Send message to another agent."""
        pass
    
    @abstractmethod
    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """Receive message with timeout."""
        pass
    
    # 算法方法（已实现，不可修改）
    def compute_path(self) -> List[Tuple[int, int]]:
        """Compute A* path from current position to goal."""
        
    async def move_next(self) -> bool:
        """Execute next move in planned path."""
        
    def is_at_goal(self) -> bool:
        """Check if agent has reached its goal."""
```

### BaseNet接口

```python
class BaseNet(ABC):
    """Abstract base class for network coordinators."""
    
    @abstractmethod
    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """Deliver message to specific agent."""
        pass
    
    @abstractmethod
    async def poll(self) -> List[Tuple[int, Dict[str, Any]]]:
        """Poll for incoming messages from agents."""
        pass
    
    # 协调方法（已实现，不可修改）
    def register_agent(self, agent_id: int, start_pos: Tuple[int, int], 
                      goal_pos: Tuple[int, int]) -> None:
        """Register a new agent in the system."""
        
    async def resolve_conflicts(self, conflicts: List[ConflictInfo]) -> None:
        """Resolve movement conflicts using configured strategy."""
```

## 9 支持的协议

### Dummy协议（测试用）
- **特点**：内存队列，无外部依赖
- **用途**：开发、测试、CI环境
- **配置**：`config/dummy.yaml`

### A2A协议（智能体间通信）
- **特点**：HTTP/WebSocket通信
- **用途**：中等规模部署
- **配置**：`config/a2a.yaml`

### ANP协议（安全网络协议）
- **特点**：DID身份验证，端到端加密
- **用途**：生产环境，安全要求高
- **配置**：`config/anp.yaml`

---

**框架版本**：0.1.0  
**Python要求**：≥ 3.11  
**许可证**：MIT License 