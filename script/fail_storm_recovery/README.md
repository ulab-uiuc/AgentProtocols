# Fail-Storm Recovery Scenario

## 🌩️ 场景概述

Fail-Storm Recovery 是 Agent Protocol Evaluation 项目中的测试场景，专门用于测试协议在节点突发故障时的恢复与稳定性能。

### 📋 测试目标

在运行中的 Gaia Mesh 网络中，于 t=60s 随机杀死 30% Agent，随后继续观察：
1. **恢复时间** - 剩余节点多久检测到断连并恢复
2. **任务成功率** - 任务完成率是否骤降 
3. **系统稳定性** - 重连后系统如何重新稳定

### 🎯 价值意义

不同通信协议（JSON / ANP / A2A / ACP）在心跳设计、重连握手、任务重发布方面实现差异大，本场景可提供量化依据。

## 🏗️ 架构设计

### 核心组件

```
MeshNetwork (扩展)
├── Heartbeat Monitor     ← 心跳检测机制
├── Reconnection Handler  ← 自动重连逻辑  
├── Metrics Collector     ← 故障恢复指标
└── Fail-Storm Injector   ← 故障注入脚本

GaiaAgent (基于shard_qa)
├── GaiaWorker           ← 直接继承ShardWorker
├── GaiaWorkerExecutor   ← 直接继承ShardWorkerExecutor
└── lookup_fragment      ← 复用shard_qa的成熟工具
    send_message
```

## ⏱️ 时序流程

```
t = 0s    MeshNetwork.broadcast_init(GaiaDoc)
t = 30s   正常Gaia文档处理 (lookup_fragment → send_message → collaborate)
t = 60s   故障注入: kill 30% Agent (fail_storm.py)
t = 60-120s 剩余节点检测断连 → 触发重连 → 任务继续
t = 120s  Network 触发评估 & 输出 failstorm_metrics.json
```

## 📊 新增指标

所有指标写入 `failstorm_metrics.json`：

### 恢复性能指标
- **`recovery_ms`** - 首次重连消息时间 - 60000ms
- **`steady_state_ms`** - 系统重新稳定时间
- **`bytes_reconnect`** - 故障后30s内通信字节

### 任务质量指标  
- **`success_rate_drop`** - (故障前成功率 - 故障后成功率) / 故障前成功率
- **`duplicate_work_ratio`** - 重复执行同task_id次数 / 总执行

## 🚀 快速开始

### 1. 环境准备

```bash
cd script/fail_storm_recovery
pip install -r ../../requirements.txt
```

### 2. 配置修改

编辑 `config.yaml`：

```yaml
scenario:
  agent_count: 8        # Agent数量
  protocol: "a2a"       # 协议选择
  kill_fraction: 0.3    # 故障比例
  duration: 120         # 测试时长

llm:
  model:
    type: "openai"      # 或 "local"
    name: "gpt-4"
    openai_api_key: "your-key-here"
```

### 3. 运行测试

```bash
python fail_storm_runner.py
```

### 4. 查看结果

```bash
# 核心指标
cat failstorm_metrics.json

# 详细日志
tail -f logs/fail_storm_*.log

# 工作空间
ls workspaces/agent_*/
```

## 📁 文件结构

```
script/fail_storm_recovery/
├── README.md                    # 本文档
├── config.yaml                  # 配置文件
├── fail_storm_runner.py         # 主运行器
├── fail_storm.py               # 故障注入脚本
├── docs/
│   └── gaia_document.txt       # 示例Gaia文档
├── core/
│   ├── mesh_network.py         # 扩展网络层(心跳+重连)
│   └── failstorm_metrics.py    # 故障恢复指标收集
└── gaia_agents/
    ├── __init__.py
    └── gaia_shard_adapter.py   # Gaia-shard_qa适配器
```

## 🔬 设计约束

### 硬约束
1. **MeshNetwork 广播** - `t=0` 广播 Gaia 文档
2. **异步点对点** - Agent 异步、点对点通信
3. **工具专门化** - 每 Agent 仅 1 Tool（search/extract/triple/reason）
4. **独占端口** - 本地独占端口 127.0.0.1:9xxx
5. **协议可插拔** - ProtocolAdapter 架构
6. **私有工作空间** - workspaces/<node_id>/
7. **故障注入** - fail_storm.py 在 60s 发送 SIGKILL
8. **指标归档** - 新增恢复指标写入 failstorm_metrics.json

### 软约束
- **鲁棒性** - 内置错误处理和重试机制
- **可观测性** - 完整的日志和指标收集
- **可扩展性** - 支持不同协议和拓扑

## 🧪 协议比较

### 测试不同协议

```bash
# 测试 A2A 协议
python fail_storm_runner.py --protocol a2a

# 测试 ANP 协议  
python fail_storm_runner.py --protocol anp

# 测试 ACP 协议
python fail_storm_runner.py --protocol acp
```

### 关键差异指标

| 协议 | recovery_ms | bytes_reconnect | steady_state_ms |
|------|-------------|-----------------|-----------------|
| A2A  | ?           | ?               | ?               |
| ANP  | ?           | ?               | ?               |
| ACP  | ?           | ?               | ?               |

### 协议特性分析

- **A2A**: 基于 EventQueue，异步消息处理
- **ANP**: Agent Network Protocol，包含 CRC 校验
- **ACP**: Agent Communication Protocol，流式处理

## 🛠️ 开发指南

### 扩展新协议

1. **实现 ProtocolAdapter**
```python
class YourProtocolAdapter(BaseProtocolAdapter):
    async def send_message(self, dest, content): ...
    async def receive_message(self): ...
```

2. **适配 GaiaAgent**
```python
class YourProtocolGaiaExecutor:
    def __init__(self, gaia_worker): ...
    async def execute(self, ...): ...
```

### 添加新指标

1. **扩展 FailStormMetricsCollector**
```python
def record_your_metric(self, value):
    self.custom_metrics["your_metric"] = value
```

2. **在 MeshNetwork 中调用**
```python
self.metrics_collector.record_your_metric(measured_value)
```

## 🔧 故障排除

### 调试模式

```bash
# 详细日志
python fail_storm_runner.py --debug

# 单Agent测试
python fail_storm_runner.py --agent-count 1 --duration 30

# 禁用故障注入
python fail_storm_runner.py --kill-fraction 0
```

## 📈 未来扩展

### 计划功能

1. **数据生成器** - 自动生成测试用的 Gaia 文档
2. **评分器模块** - 任务完成质量自动评估  
3. **可插拔拓扑** - 支持 Star、Ring、Full-Mesh 等拓扑
4. **更多协议** - IoA、AIOS 等新兴协议支持

### 性能优化

1. **批量重连** - 优化大规模故障后的重连效率
2. **预测性恢复** - 基于历史数据预测故障模式
3. **自适应心跳** - 根据网络状态动态调整心跳间隔
