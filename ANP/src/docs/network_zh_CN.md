# AgentNetwork - 网络架构

## 概述

`AgentNetwork`类作为多智能体系统的中央协调器和拓扑管理器。它维护所有智能体的注册表，管理它们之间的互连，处理消息路由，并提供全面的监控功能。

## 核心职责

1. **智能体生命周期管理**: 注册、注销和跟踪智能体实例
2. **拓扑管理**: 定义和维护网络拓扑（星型、网状、自定义）
3. **消息路由**: 基于拓扑规则在智能体之间路由消息
4. **健康监控**: 持续健康检查和故障检测
5. **指标收集**: 性能监控和可观测性
6. **故障恢复**: 自动检测和从智能体故障中恢复

## 类结构

```python
class AgentNetwork:
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}           # 智能体注册表
        self._graph: Dict[str, Set[str]] = defaultdict(set)  # 拓扑图
        self._metrics: Dict[str, Any] = {}                # 运行时指标
        self._lock = asyncio.Lock()                       # 线程安全
```

## 关键方法

### 智能体管理

```python
async def register_agent(self, agent: BaseAgent) -> None:
    """注册新的智能体实例（线程安全）"""
    
async def unregister_agent(self, agent_id: str) -> None:
    """移除智能体及其所有连接"""
```

### 连接管理

```python
async def connect_agents(self, src_id: str, dst_id: str) -> None:
    """创建有向边 src → dst（幂等操作）"""
    
async def disconnect_agents(self, src_id: str, dst_id: str) -> None:
    """移除有向边 src → dst"""
```

### 拓扑设置

```python
def setup_star_topology(self, center_id: str) -> None:
    """设置以center_id为中心的星型拓扑"""
    
def setup_mesh_topology(self) -> None:
    """设置全网状拓扑（全连接）"""
```

### 消息路由

```python
async def route_message(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
    """如果边存在则转发消息"""
    
async def broadcast_message(self, src_id: str, payload: Dict[str, Any], exclude: Optional[Set[str]] = None) -> Dict[str, Any]:
    """向所有连接的智能体广播消息"""
```

### 监控

```python
async def health_check(self) -> Dict[str, bool]:
    """检查所有注册智能体的健康状况"""
    
def snapshot_metrics(self) -> Dict[str, Any]:
    """返回当前指标字典"""
```

## 拓扑类型

### 星型拓扑
- **描述**: 一个中央枢纽连接到所有其他智能体
- **用例**: 集中协调，中心辐射模式
- **特点**: 
  - 低复杂度
  - 单点故障
  - 高效的广播操作

### 网状拓扑
- **描述**: 每个智能体都连接到其他所有智能体
- **用例**: 高可用性，点对点通信
- **特点**:
  - 高冗余性
  - 复杂路由
  - 无单点故障

## 连接过程

1. **智能体注册**: 智能体在网络中注册
2. **适配器创建**: 为每个连接创建协议特定的适配器
3. **拓扑应用**: 应用网络拓扑创建连接
4. **健康监控**: 开始持续监控

## 错误处理

- **连接失败**: 指数退避自动重试
- **智能体故障**: 立即检测和隔离
- **网络分区**: 优雅降级和恢复
- **协议错误**: 错误传播和日志记录

## 性能考虑

- **并发操作**: 使用asyncio锁的线程安全操作
- **连接池**: 重用HTTP连接以提高效率
- **批量操作**: 组合操作以获得更好的吞吐量
- **内存管理**: 自动清理断开连接的智能体

## 使用示例

```python
import asyncio
from agent_network import AgentNetwork, BaseAgent

async def example():
    # 创建网络
    network = AgentNetwork()
    
    # 创建并注册智能体
    agent1 = await BaseAgent.create_a2a("agent-1", port=8001, executor=executor1)
    agent2 = await BaseAgent.create_a2a("agent-2", port=8002, executor=executor2)
    
    await network.register_agent(agent1)
    await network.register_agent(agent2)
    
    # 设置星型拓扑
    network.setup_star_topology("agent-1")
    
    # 发送消息
    response = await network.route_message(
        "agent-1", "agent-2", 
        {"message": "你好"}
    )
    
    # 健康检查
    health = await network.health_check()
    print(f"健康状态: {health}")
    
    # 获取指标
    metrics = network.snapshot_metrics()
    print(f"网络指标: {metrics}")

asyncio.run(example())
```

## 性能指标

- **延迟**: 消息路由延迟
- **吞吐量**: 每秒消息数
- **可用性**: 智能体正常运行时间百分比
- **资源使用**: 内存和CPU使用率 