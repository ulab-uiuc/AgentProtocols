# Fail-Storm Recovery 场景

## 概述

Fail-Storm Recovery 是一个用于测试多智能体系统容错能力的基准测试场景。该场景专门设计用于评估不同通信协议在节点突发故障时的恢复与稳定性能。

## 🎯 测试目标

在运行中的 Gaia Mesh 网络中，于指定时间点随机杀死一定比例的 Agent，随后继续观察：

1. **故障检测时间**：剩余节点多久检测到断连并开始恢复
2. **任务成功率变化**：任务成功率是否骤降，恢复后是否回升
3. **系统稳定时间**：重连后系统何时重新稳定
4. **恢复机制效率**：不同协议的重连握手、任务重发布效率对比

## 💡 测试价值

不同通信协议（Simple JSON、ANP、A2A、ACP等）在以下方面存在显著差异：
- **心跳设计**：心跳间隔、超时检测机制
- **重连握手**：身份重认证、连接重建流程
- **任务重发布**：失败任务的重新分配策略

本场景可为这些差异提供量化评估依据，帮助选择最适合特定需求的协议。

## 🏗️ 场景设计

### 执行流程

1. **Phase 0: 初始化** (0-3s)
   - 创建指定数量的 Agent
   - 建立 Mesh 网络拓扑
   - 广播 Gaia 文档到所有节点

2. **Phase 1: 正常阶段** (3s - 故障注入时间)
   - 运行常规 Shard QA 任务
   - 记录基线性能指标
   - 建立正常运行基准

3. **Phase 2: 故障注入** (故障注入时间点)
   - 随机选择指定比例的 Agent
   - 发送 SIGKILL 信号终止进程
   - 记录故障注入时间

4. **Phase 3: 恢复监控** (故障注入后 - 结束)
   - 监控剩余节点的故障检测
   - 执行协议特定的重连流程
   - 继续运行 QA 任务直到系统稳定

5. **Phase 4: 结果评估**
   - 收集所有性能指标
   - 生成详细的故障恢复报告

### 核心约束

| # | 约束 | 说明 |
|---|------|------|
| 1 | MeshNetwork 广播 Gaia 文档 | `MeshNetwork.broadcast_init()` |
| 2 | Agent 异步、点对点通信 | 与主场景保持一致 |
| 3 | 每 Agent 仅 1 Tool | 简化测试复杂度 |
| 4 | 本地独占端口 | `127.0.0.1:9xxx` |
| 5 | 协议可插拔 | 通过 `ProtocolAdapter` 实现 |
| 6 | 私有 Workspace | `workspaces/<agent_id>/` |
| 7 | Fail-Storm 注入 | 在指定时间发送 SIGKILL |
| 8 | 指标归档 | 写入 `failstorm_metrics.json` |

## 🚀 使用方法

### 基本用法

```bash
# 激活虚拟环境
source ../../agent/bin/activate

# 设置 API 密钥（必需）
export NVIDIA_API_KEY="your_nvidia_api_key_here"

# 运行基本测试
python fail_storm_runner.py --protocol anp --agents 4
```

### 命令行参数

运行 `python fail_storm_runner.py --help` 查看所有可用参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--protocol` | str | 必需 | 通信协议 (anp, simple_json) |
| `--agents` | int | 3 | Agent 数量 |
| `--runtime` | float | 120.0 | 总运行时间（秒）|
| `--fault-time` | float | 60.0 | 故障注入时间点（秒）|
| `--config` | str | config.yaml | 配置文件路径 |

### 参数设置建议

#### 快速测试 (开发调试)
```bash
python fail_storm_runner.py --protocol anp --agents 3 --runtime 30 --fault-time 15
```

#### 标准测试 (基准评估)
```bash
python fail_storm_runner.py --protocol anp --agents 8 --runtime 120 --fault-time 60
```

#### 压力测试 (性能极限)
```bash
python fail_storm_runner.py --protocol anp --agents 16 --runtime 300 --fault-time 120
```

### 环境变量配置

在运行前，必须设置以下环境变量：

```bash
# NVIDIA API 密钥 (必需)
export NVIDIA_API_KEY="nvapi-xxxxx"
```

## 📊 输出指标

测试完成后，会生成两个结果文件：

### 1. 简化指标 (`results/failstorm_metrics.json`)
- 恢复时间 (Recovery Time)
- 稳定状态时间 (Steady State)
- 成功率下降 (Success Rate Drop)
- 重复工作量 (Duplicate Work)
- 重连字节数 (Reconnect Bytes)

### 2. 详细指标 (`results/detailed_failstorm_metrics.json`)
- 任务执行时间线
- 网络事件记录
- 协议特定指标
- 错误和重试统计

### 关键指标解释

- **Recovery Time**: 从故障注入到第一个 Agent 重连成功的时间
- **Steady State**: 系统达到稳定状态的总时间
- **Success Rate Drop**: 故障前后任务成功率的差异
- **Answer Found Rate**: 总体问题解答成功率
- **Answer Sources**: 本地vs邻居搜索的统计

## 🔧 目前支持的协议

### ANP (Agent Network Protocol)
- **特点**: DID 身份认证、E2E 加密、混合通信 (HTTP + WebSocket)
- **配置**: `protocol_backends/anp/config.yaml`
- **适用场景**: 高安全性要求的分布式系统

### Simple JSON
- **特点**: 简单的 JSON 消息传递、直接 HTTP 通信
- **配置**: `protocol_backends/simple_json/config.yaml`
- **适用场景**: 快速原型开发、轻量级测试

## 📁 项目结构

```
fail_storm_recovery/
├── README.md                          # 本文档
├── fail_storm_runner.py              # 主入口脚本
├── config.yaml                       # 默认配置
├── protocol_backends/                 # 协议后端实现
│   ├── README.md                     # 协议扩展指南
│   ├── base_runner.py               # 抽象基类
│   ├── anp/                         # ANP 协议实现
│   └── simple_json/                 # Simple JSON 协议实现
├── core/                            # 核心组件
│   ├── mesh_network.py             # 网络管理
│   ├── failstorm_metrics.py        # 指标收集
│   └── protocol_factory.py         # 协议工厂
├── shard_qa/                        # Shard QA 任务
├── utils/                           # 工具函数
├── local_deps/                      # 本地化依赖
└── results/                         # 测试结果输出
```

## 🐛 故障排除

### 常见问题

1. **API 密钥错误**
   ```
   错误: Missing required environment variables: NVIDIA_API_KEY
   解决: 正确设置环境变量 export NVIDIA_API_KEY="your_key"
   ```

2. **端口冲突**
   ```
   错误: Address already in use
   解决: 等待进程完全终止，或使用不同的端口范围
   ```

3. **虚拟环境未激活**
   ```
   错误: ModuleNotFoundError
   解决: source ../../agent/bin/activate
   ```

### 调试模式

启用详细日志输出：
```bash
# 查看协议特定的详细日志
python fail_storm_runner.py --protocol anp --agents 3 --runtime 30 --fault-time 15
```

### 性能优化

- **减少 Agent 数量**: 对于快速测试，使用 3-4 个 Agent
- **缩短运行时间**: 开发阶段使用 30-60 秒的运行时间
- **选择合适的协议**: Simple JSON 比 ANP 更快，适合功能测试

## 📈 结果分析

优秀的测试结果应该显示：
- **Recovery Time < 15s**: 快速故障检测和重连
- **Success Rate Drop = 0%**: 无性能损失
- **Answer Found Rate > 85%**: 高任务成功率
- **Final Survivors = Initial Agents**: 完全恢复