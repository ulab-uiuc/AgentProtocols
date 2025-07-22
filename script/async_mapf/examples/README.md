# MAPF Framework Examples

本目录包含异步多智能体路径规划（MAPF）框架的各种示例和演示。

## 示例列表

### 1. 单节点演示 (`single_node_demo.py`)

基本的MAPF执行演示，展示框架的核心功能。

```bash
# 基本演示
python single_node_demo.py basic

# 带指标记录的演示
python single_node_demo.py metrics

# 带实时仪表板的演示
python single_node_demo.py dashboard

# 交互式逐步演示
python single_node_demo.py interactive
```

**功能特性：**
- 4个智能体在10x10网格中的路径规划
- 实时性能监控和指标记录
- 交互式控制和状态查看
- Web仪表板生成

### 2. 协议对比 (`protocol_comparison.py`)

不同通信协议的性能对比基准测试。

```bash
python protocol_comparison.py
```

**对比维度：**
- 执行时间
- 成功率
- 消息吞吐量
- 冲突解决效率

**支持的协议：**
- Dummy Protocol（本地测试）
- A2A Protocol（智能体间通信）
- ANP Protocol（带身份验证的网络协议）

## 配置文件

### Dummy协议配置 (`../config/dummy.yaml`)
- 适用于本地测试和开发
- 无外部依赖
- 可配置的网络延迟和消息丢失模拟

### A2A协议配置 (`../config/a2a.yaml`)
- 智能体间直接通信
- HTTP/WebSocket连接
- 适用于中等规模部署

### ANP协议配置 (`../config/anp.yaml`)
- 支持DID身份验证
- 端到端加密
- 适用于安全要求高的环境

### 分布式配置 (`../config/distributed.yaml`)
- 多节点部署
- 负载均衡
- 故障容错

## 运行要求

### 基本要求
```bash
pip install asyncio pyyaml
```

### 可选依赖（用于高级功能）
```bash
pip install matplotlib pandas numpy seaborn
```

### 协议特定依赖
```bash
# 用于A2A协议
pip install requests websockets

# 用于ANP协议
pip install cryptography did-sdk

# 用于分布式部署
pip install docker kubernetes
```

## 快速开始

1. **简单测试**
```bash
cd script/async_mapf/examples
python single_node_demo.py basic
```

2. **性能基准测试**
```bash
python protocol_comparison.py
```

3. **实时监控**
```bash
python single_node_demo.py dashboard
# 然后在浏览器中打开生成的 demo_dashboard.html
```

## 示例场景说明

### 场景1：基本路径规划
- 4个智能体从角落出发到对角位置
- 中央有少量障碍物
- 测试基本的路径规划和冲突避免

### 场景2：密集环境
- 更多智能体和障碍物
- 测试高冲突场景下的性能
- 评估协调算法效率

### 场景3：动态障碍物
- 临时障碍物的生成和消失
- 测试动态重规划能力
- 适应性路径调整

## 指标和分析

### 性能指标
- **成功率**：到达目标的智能体比例
- **执行时间**：完成任务的总时间
- **路径效率**：实际路径与最优路径的比率
- **消息开销**：通信消息数量和频率
- **冲突数量**：检测和解决的冲突事件

### 分析工具
- **MetricsRecorder**：记录详细的执行指标
- **PerformanceAnalyzer**：生成统计分析和可视化
- **RealtimeDashboard**：实时监控和告警

## 自定义示例

### 创建新示例
1. 复制现有示例作为模板
2. 修改配置文件（world大小、智能体数量、目标位置）
3. 添加自定义的指标收集逻辑
4. 实现特定的分析和可视化

### 示例模板
```python
import asyncio
from pathlib import Path
from script.async_mapf.runners.local_runner import LocalRunner

async def custom_demo():
    config_path = "path/to/your/config.yaml"
    runner = LocalRunner(config_path)
    
    # 自定义设置
    await runner.setup()
    
    # 运行场景
    results = await runner.run()
    
    # 自定义分析
    print(f"Custom metric: {results['custom_value']}")

if __name__ == "__main__":
    asyncio.run(custom_demo())
```

## 故障排除

### 常见问题

1. **导入错误**
   - 确保Python路径包含项目根目录
   - 检查所有依赖是否安装

2. **协议连接失败**
   - 验证网络配置和端口
   - 检查防火墙设置
   - 确认协议服务是否运行

3. **性能问题**
   - 调整tick_ms参数降低CPU使用
   - 减少智能体数量或世界大小
   - 检查内存使用情况

4. **配置错误**
   - 验证YAML语法
   - 检查类路径是否正确
   - 确认所有必需字段都已设置

### 调试技巧

1. **启用详细日志**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **使用交互式模式**
```bash
python single_node_demo.py interactive
```

3. **检查指标输出**
查看生成的指标文件了解详细的执行信息。

## 贡献

欢迎贡献新的示例和改进现有示例！请遵循以下指南：

1. 保持代码简洁和文档完整
2. 添加适当的错误处理
3. 包含使用说明和期望输出
4. 测试所有支持的协议
5. 更新此README文档 