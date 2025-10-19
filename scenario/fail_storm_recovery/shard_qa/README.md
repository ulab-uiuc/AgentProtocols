# Shard QA - 分片式多智能体协作问答系统

## 🌟 项目概述

Shard QA 是一个基于环形拓扑的分布式多智能体协作问答系统，专为多跳推理任务设计。系统通过将知识分片存储在不同的智能体节点上，利用智能体间的协作机制来解决复杂的多步推理问题。

### 核心特性

- 🔄 **环形拓扑架构**：8个工作节点组成环形网络，支持高效的分布式协作
- 🧠 **多跳推理**：基于2WikiMultiHopQA数据集，支持复杂的多步推理任务
- 🔍 **智能分片检索**：知识片段分布存储，通过Function Calling动态路由查询
- 📊 **实时监控**：集成Prometheus监控指标，支持性能分析和调优
- 🎯 **协调器模式**：中心化协调器负责任务分发和结果聚合
- ⚡ **异步处理**：完全异步架构，支持高并发查询处理

## 🏗️ 系统架构

```
         ┌─────────────────┐
         │   Coordinator   │ ← 任务分发与结果聚合
         │   (Port 9998)   │
         └─────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │    Question     │
         │   Distribution  │
         └─────────────────┘
                   │
                   ▼
    ┌───────┬───────┬───────┬───────┐
    │Shard0 │Shard1 │Shard2 │Shard3 │ ← 环形工作节点
    │10001  │10002  │10003  │10004  │
    └───┬───┴───┬───┴───┬───┴───┬───┘
        │       │       │       │
    ┌───┴───┬───┴───┬───┴───┬───┴───┐
    │Shard7 │Shard6 │Shard5 │Shard4 │
    │10008  │10007  │10006  │10005  │
    └───────┴───────┴───────┴───────┘
```

### 组件说明

1. **Coordinator (协调器)**
   - 负责任务分发和结果收集
   - 监控系统性能指标
   - 处理查询路由和超时管理

2. **Shard Workers (分片工作节点)**
   - 存储和检索知识片段
   - 执行Function Calling查询
   - 环形网络中的消息传递

3. **Ring Network (环形网络)**
   - 8个节点的双向环形拓扑
   - 支持TTL控制的消息传播
   - 动态路由和负载均衡

## 📁 目录结构

```
script/shard_qa/
├── config.yaml                # 系统配置文件
├── shard_qa.py                # 主程序入口
├── shard_coordinator/          # 协调器组件
│   ├── __init__.py
│   ├── __main__.py
│   └── agent_executor.py      # 协调器执行逻辑
├── shard_worker/              # 工作节点组件
│   ├── __init__.py
│   ├── __main__.py
│   └── agent_executor.py      # 工作节点执行逻辑
├── data/                      # 数据存储目录
│   └── v1.1_2wiki/           # 2WikiMultiHopQA数据集
├── logs/                      # 日志文件目录
├── test_output.txt           # 测试输出文件
└── test_error.txt            # 测试错误文件
```

## ⚙️ 配置说明

### 核心配置 (config.yaml)

```yaml
# LLM配置
core:
  name: gpt-4o
  base_url: http://localhost:8000/v1
  max_tokens: 4096
  temperature: 0.0

# 数据配置
data:
  base_dir: data/shards
  version: v2.0_shuffled
  manifest_file: data/shards/manifest.json

# 网络配置
network:
  topology: ring
  health_check_interval: 5
  message_timeout: 30

# 分片QA配置
shard_qa:
  workers:
    count: 8
    start_port: 10001
    max_pending: 16
  coordinator:
    count: 1
    start_port: 9998
    total_groups: 24075
  ring_config:
    # 环形拓扑配置
    shard0: {next_id: shard1, prev_id: shard7}
    shard1: {next_id: shard2, prev_id: shard0}
    # ... 其他节点配置
```

### 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `workers.count` | 工作节点数量 | 8 |
| `workers.start_port` | 工作节点起始端口 | 10001 |
| `tool_schema.max_ttl` | 消息最大TTL | 7 |
| `timeouts.response_timeout` | 响应超时时间 | 30s |
| `history.max_len` | 历史记录最大长度 | 20 |

## 🚀 快速开始

### 环境要求

- Python 3.8+
- asyncio, httpx, yaml, colorama
- A2A SDK (可选，用于智能体通信)
- OpenAI API 或兼容的LLM服务

### 安装依赖

```bash
pip install asyncio httpx pyyaml colorama
# 可选：A2A SDK
pip install a2a-sdk
```

### 数据准备

1. 准备2WikiMultiHopQA数据集
2. 将数据分片存储到 `data/shards/` 目录
3. 更新 `config.yaml` 中的数据路径配置

### 启动系统

```bash
# 启动完整系统（协调器 + 8个工作节点）
python script/shard_qa/shard_qa.py

# 或者分别启动组件
python -m script.shard_qa.shard_coordinator  # 启动协调器
python -m script.shard_qa.shard_worker       # 启动工作节点
```

### 系统验证

```bash
# 检查服务状态
curl http://localhost:9998/health    # 协调器健康检查
curl http://localhost:10001/health   # 工作节点健康检查

# 监控指标
curl http://localhost:8000/metrics   # Prometheus指标
```

## 🔧 核心功能

### 1. Function Calling 工具

系统使用OpenAI Function Calling实现智能查询路由：

```python
TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "lookup_fragment",
            "description": "检查本地snippet是否包含答案",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要查询的问题或关键词"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "query_ring",
            "description": "向环形网络中其他节点查询信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要查询的问题"
                    },
                    "target_shard": {
                        "type": "string", 
                        "description": "目标分片ID"
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

### 2. 环形消息传递

系统实现了TTL控制的环形消息传递机制：

- **消息路由**：支持顺时针和逆时针传播
- **TTL控制**：防止消息无限循环
- **重复检测**：避免重复处理相同查询
- **优先级队列**：保证消息处理顺序

### 3. 监控指标

集成Prometheus监控，支持以下指标：

```python
# 性能指标
metrics = {
    "avg_hop": True,                # 平均跳数
    "first_answer_latency": True,   # 首次响应延迟
    "msg_bytes_total": True,        # 消息字节总数
    "ttl_exhausted_total": True     # TTL耗尽次数
}
```

## 🔍 工作流程

### 查询处理流程

1. **任务接收**：协调器接收外部查询请求
2. **任务分发**：根据负载均衡策略分发到工作节点
3. **本地检索**：工作节点首先查询本地知识片段
4. **环形查询**：如本地无答案，通过环形网络查询其他节点
5. **结果聚合**：协调器收集所有响应并生成最终答案
6. **指标记录**：记录性能指标用于系统优化

### 消息传递机制

```python
# 消息格式
message = {
    "messageId": "unique_id",
    "query": "用户查询",
    "ttl": 7,
    "path": ["shard0", "shard1"],
    "source_shard": "shard0",
    "meta": {
        "timestamp": 1234567890,
        "priority": "high"
    }
}
```

## 📊 性能优化

### 系统调优建议

1. **并发控制**
   ```yaml
   workers:
     max_pending: 16  # 调整最大并发数
   ```

2. **超时设置**
   ```yaml
   timeouts:
     response_timeout: 30
     task_timeout: 60
     max_retries: 3
   ```

3. **缓存策略**
   ```yaml
   history:
     max_len: 20  # 调整历史缓存大小
   ```

### 监控和调试

- **日志文件**：`logs/shard_qa_YYYYMMDD_HHMMSS.log`
- **彩色输出**：支持终端彩色日志输出
- **健康检查**：定期检查各组件运行状态
- **指标导出**：Prometheus格式的性能指标

## 🧪 测试和验证

### 功能测试

```bash
# 运行功能测试
python test_shard_qa.py

# 性能基准测试
python benchmark_shard_qa.py
```

### 集成测试

```bash
# 启动测试环境
docker-compose up -d

# 运行集成测试套件
pytest tests/integration/
```

## 🐛 故障排除

### 常见问题

1. **端口冲突**
   ```bash
   # 检查端口占用
   netstat -tlnp | grep :10001
   ```

2. **A2A SDK问题**
   - 确保正确安装A2A SDK
   - 检查API密钥配置
   - 验证网络连接

3. **内存不足**
   - 调整`max_pending`参数
   - 减少`history.max_len`
   - 增加系统内存

4. **响应超时**
   - 检查网络延迟
   - 调整`response_timeout`
   - 优化查询复杂度

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 监控系统资源
import psutil
print(f"CPU: {psutil.cpu_percent()}%")
print(f"Memory: {psutil.virtual_memory().percent}%")
```

## 📈 扩展开发

### 添加新的工具函数

```python
def new_tool_function(self, query: str, **kwargs) -> dict:
    """新的工具函数实现"""
    try:
        # 实现查询逻辑
        result = self.process_query(query)
        return {
            "success": True,
            "data": result,
            "source": self.shard_id
        }
    except Exception as e:
        return {
            "success": False, 
            "error": str(e)
        }
```

### 自定义拓扑结构

```yaml
# 修改网络拓扑
network:
  topology: custom
  custom_config:
    # 定义自定义连接
    connections:
      shard0: [shard1, shard3]
      shard1: [shard2, shard4]
```

## 📚 参考资料

- [2WikiMultiHopQA数据集](https://github.com/Alab-NII/2wikimultihop)
- [OpenAI Function Calling文档](https://platform.openai.com/docs/guides/function-calling)
- [A2A智能体协议](https://github.com/a2a-protocol/a2a-sdk)
- [Prometheus监控指南](https://prometheus.io/docs/)

## 🤝 贡献指南

1. Fork项目仓库
2. 创建功能分支：`git checkout -b feature/new-feature`
3. 提交更改：`git commit -am 'Add new feature'`
4. 推送分支：`git push origin feature/new-feature`
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

---

**版本**：v2.0  
**最后更新**：2024年1月  
**维护者**：Agent Network Team 