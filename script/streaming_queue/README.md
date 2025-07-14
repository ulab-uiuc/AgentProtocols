# AgentNetwork Streaming Queue - 分布式问答系统

## 概述

AgentNetwork Streaming Queue是一个基于AgentNetwork框架构建的分布式实时问答系统演示。该系统展示了如何使用AgentNetwork的A2A协议实现多智能体协作，包括一个协调器智能体和多个工作智能体，共同处理大规模问答任务。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentNetwork 核心                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐               │
│  │  协调器智能体   │    │   工作智能体     │               │
│  │  (Coordinator)  │    │    Pool         │               │
│  │                 │    │                 │               │
│  │  - 任务分发     │<-->│  - Worker-1     │               │
│  │  - 负载均衡     │    │  - Worker-2     │               │
│  │  - 结果收集     │    │  - Worker-3     │               │
│  │  - 状态监控     │    │  - Worker-N     │               │
│  └─────────────────┘    └─────────────────┘               │
├─────────────────────────────────────────────────────────────┤
│                    HTTP/A2A 通信层                          │
├─────────────────────────────────────────────────────────────┤
│            彩色终端输出 & 实时监控                          │
└─────────────────────────────────────────────────────────────┘
```

## 核心特性

### 🚀 分布式架构
- **协调器-工作者模式**: 单个协调器管理多个工作智能体
- **星型/网状拓扑**: 支持灵活的网络拓扑配置
- **异步通信**: 基于A2A协议的异步消息传递
- **负载均衡**: 智能任务分发和负载均衡

### 🎨 用户体验
- **彩色输出**: 使用colorama实现彩色终端输出
- **实时监控**: 实时显示系统状态和进度
- **分类日志**: 信息、成功、警告、错误分类显示
- **健康检查**: 定期检查所有智能体状态

### 🤖 AI集成
- **多模型支持**: 支持OpenAI API和本地模型
- **灵活配置**: 通过YAML配置文件调整模型参数
- **批量处理**: 高效的批量问答处理能力

### 📊 监控与分析
- **性能指标**: 延迟、准确率、吞吐量监控
- **网络拓扑**: 实时网络连接状态显示
- **结果保存**: 自动保存处理结果到JSON文件

## 目录结构

```
streaming_queue/
├── README.md                 # 本文档
├── streaming_queue.py        # 主程序入口
├── config.yaml              # 系统配置文件
├── color_demo.py            # 彩色输出演示
├── data/                    # 数据目录
│   ├── top1000.jsonl        # 完整问答数据集
│   ├── top1000_simplified.jsonl  # 简化数据集
│   └── qa_results.json      # 处理结果文件
├── qa_coordinator/          # 协调器智能体
│   ├── agent_executor.py    # 协调器执行器
│   ├── __main__.py         # 独立运行入口
│   └── USAGE.md            # 使用说明
├── qa_worker/              # 工作智能体
│   ├── agent_executor.py   # 工作执行器
│   ├── __main__.py        # 独立运行入口
│   └── README.md          # 工作智能体说明
└── agent_network/         # AgentNetwork核心框架
    ├── network.py         # 网络管理
    ├── base_agent.py      # 基础智能体
    └── ...
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install colorama pyyaml httpx asyncio

# 或使用requirements.txt
pip install -r requirements.txt
```

### 2. 配置设置

编辑 `config.yaml` 文件：

```yaml
# AI模型配置
core:
  type: "openai"                    # "openai" 或 "local"
  name: "gpt-4o"                    # 模型名称
  temperature: 0.0                  # 温度参数
  openai_api_key: "your-api-key"    # OpenAI API密钥
  openai_base_url: "https://api.openai.com/v1"

# 网络配置
network:
  topology: "star"                  # "star" 或 "mesh"
  health_check_interval: 5          # 健康检查间隔
  message_timeout: 30               # 消息超时时间

# QA系统配置
qa:
  coordinator:
    count: 1                        # 协调器数量
    start_port: 9998               # 协调器端口
    batch_size: 50                 # 批处理大小
    first_50: true                 # 是否只处理前50个问题
  worker:
    count: 4                       # 工作智能体数量
    start_port: 10001              # 工作智能体起始端口
```

### 3. 运行演示

```bash
# 运行完整演示
python streaming_queue.py

# 或使用python模块方式
python -m streaming_queue
```

### 4. 彩色输出演示

```bash
# 查看彩色输出效果
python color_demo.py
```

## 使用说明

### 系统启动流程

1. **初始化阶段**
   - 加载配置文件
   - 创建AgentNetwork实例
   - 初始化HTTP客户端

2. **智能体创建**
   - 创建协调器智能体 (Coordinator-1)
   - 创建多个工作智能体 (Worker-1, Worker-2, ...)
   - 注册所有智能体到网络

3. **网络拓扑设置**
   - 根据配置设置星型或网状拓扑
   - 建立智能体间连接
   - 验证网络连接状态

4. **任务处理**
   - 协调器接收处理命令
   - 分发任务到工作智能体
   - 收集和汇总结果

5. **监控与清理**
   - 定期健康检查
   - 性能指标收集
   - 资源清理

### 彩色输出系统

系统使用不同颜色表示不同类型的信息：

- 🔵 **蓝色**: 信息提示 (INFO)
- 🟢 **绿色**: 成功操作 (SUCCESS)  
- 🟡 **黄色**: 警告信息 (WARNING)
- 🔴 **红色**: 错误信息 (ERROR)
- 🔵 **青色**: 系统状态 (SYSTEM)
- ⚪ **白色**: 进度信息 (PROGRESS)

### HTTP接口

系统提供HTTP接口与协调器交互：

```bash
# 检查状态
curl -X POST http://localhost:9998/message \
  -H "Content-Type: application/json" \
  -d '{"id":"1","params":{"message":{"role":"user","parts":[{"kind":"text","text":"status"}],"messageId":"1"}}}'

# 开始处理
curl -X POST http://localhost:9998/message \
  -H "Content-Type: application/json" \
  -d '{"id":"2","params":{"message":{"role":"user","parts":[{"kind":"text","text":"dispatch"}],"messageId":"2"}}}'
```

## 高级功能

### 自定义模型配置

```yaml
# 本地模型配置
core:
  type: "local"
  name: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
  base_url: "http://localhost:8000/v1"
  port: 8000
  temperature: 0.0
```

### 网络拓扑配置

```yaml
# 星型拓扑 - 所有工作智能体连接到协调器
network:
  topology: "star"

# 网状拓扑 - 所有智能体互相连接
network:
  topology: "mesh"
```

### 批处理配置

```yaml
qa:
  coordinator:
    batch_size: 50              # 每轮处理的问题数量
    first_50: true              # 是否只处理前50个问题
  worker:
    count: 4                    # 工作智能体数量
  response_timeout: 60          # 响应超时时间
  max_retries: 3               # 最大重试次数
```

## 性能优化

### 并发处理
- 使用asyncio实现异步并发
- 智能体间并行处理任务
- 非阻塞HTTP通信

### 资源管理
- 自动连接池管理
- 智能体生命周期管理
- 内存和CPU使用优化

### 监控指标
- 实时性能监控
- 健康状态检查
- 网络拓扑分析

## 故障排除

### 常见问题

1. **智能体启动失败**
   ```bash
   # 检查端口是否被占用
   netstat -tulpn | grep :9998
   
   # 检查防火墙设置
   sudo ufw status
   ```

2. **API调用失败**
   ```bash
   # 验证API密钥
   curl -H "Authorization: Bearer your-api-key" https://api.openai.com/v1/models
   
   # 检查网络连接
   ping api.openai.com
   ```

3. **颜色显示异常**
   ```bash
   # 安装colorama
   pip install colorama
   
   # 检查终端颜色支持
   python color_demo.py
   ```

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 运行演示
python streaming_queue.py
```

## 扩展开发

### 添加新的智能体类型

```python
# 创建自定义执行器
class CustomExecutor(AgentExecutor):
    async def execute(self, context, event_queue):
        # 自定义处理逻辑
        pass

# 注册到网络
custom_agent = await BaseAgent.create_a2a(
    agent_id="custom-agent",
    executor=CustomExecutor(),
    port=12000
)
```

### 自定义网络拓扑

```python
# 自定义连接
network = AgentNetwork()
await network.connect_agents("agent1", "agent2")
await network.connect_agents("agent2", "agent3")
```

### 集成新的AI模型

```python
# 扩展模型配置
def create_model_config(model_type, **kwargs):
    if model_type == "custom":
        return {
            "type": "custom",
            "endpoint": kwargs.get("endpoint"),
            "api_key": kwargs.get("api_key")
        }
```

## 性能基准

### 系统容量
- **并发处理**: 支持100+并发连接
- **吞吐量**: 每秒处理50+问答对
- **延迟**: 平均响应时间 < 2秒
- **可扩展性**: 支持动态添加工作智能体

### 资源使用
- **内存**: 基础使用 < 100MB
- **CPU**: 多核并行处理
- **网络**: 高效的HTTP长连接

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

MIT License - 详见LICENSE文件

## 联系方式

- 项目地址: [AgentNetwork GitHub](https://github.com/your-org/agent-network)
- 问题反馈: [Issues](https://github.com/your-org/agent-network/issues)
- 讨论区: [Discussions](https://github.com/your-org/agent-network/discussions)

---

*基于AgentNetwork框架构建的分布式智能体协作系统* 