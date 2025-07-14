# QA Coordinator 使用指南

## 快速启动

### 方法一：直接运行演示

```bash
cd agent_network/streaming_queue/qa_coordinator
python demo.py
```

选择运行模式：
- **模式 1**: 完整演示 - 启动真实的 A2A agents 和网络
- **模式 2**: 快速测试 - 使用 Mock 网络，快速验证逻辑
- **模式 3**: 自动选择 - 先运行快速测试，可选择继续完整演示

### 方法二：独立启动 Coordinator

```bash
# 启动 Coordinator Agent 服务器
python __main__.py
```

这将在 `http://localhost:9998` 启动 A2A 兼容的 Coordinator 服务。

### 方法三：编程方式集成

```python
import asyncio
from agent_network.network import AgentNetwork
from agent_network.base_agent import BaseAgent
from qa_coordinator.agent_executor import QACoordinatorExecutor

async def setup_system():
    # 1. 创建网络
    network = AgentNetwork()
    
    # 2. 创建 Coordinator
    coordinator_executor = QACoordinatorExecutor()
    coordinator = await BaseAgent.create_a2a(
        agent_id="Coordinator",
        port=9998,
        executor=coordinator_executor
    )
    
    # 3. 创建 Workers (示例)
    workers = []
    for i in range(4):
        worker = await BaseAgent.create_a2a(
            agent_id=f"Worker-{i+1}",
            port=10001+i,
            executor=YourWorkerExecutor()
        )
        workers.append(worker)
    
    # 4. 注册到网络
    await network.register_agent(coordinator)
    for worker in workers:
        await network.register_agent(worker)
    
    # 5. 配置拓扑
    network.setup_star_topology(center_id="Coordinator")
    
    # 6. 配置 Coordinator
    worker_ids = [f"Worker-{i+1}" for i in range(4)]
    coordinator_executor.coordinator.set_network(network, worker_ids)
    
    # 7. 开始派发任务
    result = await coordinator_executor.coordinator.dispatch_round("marco_1000.jsonl")
    print(result)

# 运行
asyncio.run(setup_system())
```

## 配置选项

### Coordinator 配置

```python
coordinator = QACoordinatorExecutor()

# 修改批次大小
coordinator.coordinator.batch_size = 100  # 默认 50

# 设置 Coordinator ID
coordinator.coordinator.coordinator_id = "MyCoordinator"  # 默认 "Coordinator"
```

### 数据文件格式

确保您的 JSONL 文件格式正确：

```json
{"id":"1","q":"What is AI?","a":"AI is..."}
{"id":"2","q":"What is ML?","a":"ML is..."}
```

- `id`: 消息 ID（必需）
- `q`: 问题文本（必需）
- `a`: 答案文本（可选，Coordinator 不使用）

### 网络拓扑选项

```python
# 星型拓扑（推荐用于 Coordinator）
network.setup_star_topology(center_id="Coordinator")

# 或者手动连接
for worker_id in worker_ids:
    await network.connect_agents("Coordinator", worker_id)
```

## 监控和调试

### 查看网络状态

```python
metrics = network.snapshot_metrics()
print(f"Agents: {metrics['agent_count']}")
print(f"Connections: {metrics['edge_count']}")
print(f"Topology: {metrics['topology']}")
```

### 健康检查

```python
health = await network.health_check()
for agent_id, is_healthy in health.items():
    print(f"{agent_id}: {'OK' if is_healthy else 'FAIL'}")
```

### 性能指标

```python
# Coordinator 会自动打印性能统计
result = await coordinator.dispatch_round("data.jsonl")
# 输出包含：
# - 总耗时
# - 成功/失败计数
# - 使用的 Worker 数量
```

## 故障排除

### 常见问题

1. **端口冲突**
   ```bash
   # 检查端口占用
   netstat -tulpn | grep :9998
   # 或修改端口
   coordinator = await BaseAgent.create_a2a(agent_id="Coordinator", port=9999, ...)
   ```

2. **文件未找到**
   ```python
   # 使用绝对路径
   result = await coordinator.dispatch_round("/absolute/path/to/marco_1000.jsonl")
   
   # 或确保文件在正确位置
   # qa_coordinator/marco_1000.jsonl
   ```

3. **Worker 连接失败**
   ```python
   # 确保 Workers 已注册且连接
   print(network.get_topology())
   
   # 检查 Worker 健康状态
   health = await network.health_check()
   ```

### 调试模式

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 或在 Coordinator 中添加打印
coordinator.coordinator.debug = True  # 如果支持
```

## 扩展配置

### 自定义消息格式

如果需要修改 A2A 消息格式：

```python
# 在 agent_executor.py 中修改 payload 构造
payload = {
    "params": {
        "message": {
            "text": question,
            "messageId": message_id,
            # 添加自定义字段
            "priority": "high",
            "category": "qa"
        }
    }
}
```

### 批量大小优化

根据您的需求调整：

```python
# 小批量，低延迟
coordinator.coordinator.batch_size = 10

# 大批量，高吞吐
coordinator.coordinator.batch_size = 200
```

### Worker 选择策略

当前使用 round-robin，可扩展为其他策略：

```python
# 在 dispatch_round 中修改
# worker_cycle = itertools.cycle(self.worker_ids)  # Round-robin
# 可改为负载均衡、随机选择等
```

## 生产环境建议

1. **容错设置**
   - 监控 Worker 健康状态
   - 实现 Worker 自动恢复
   - 添加重试机制

2. **性能优化**
   - 调整批次大小
   - 使用连接池
   - 启用请求缓存

3. **监控集成**
   - 添加 Prometheus 指标
   - 集成日志系统
   - 设置告警规则

4. **安全配置**
   - 启用认证
   - 使用 HTTPS
   - 限制网络访问

## 示例项目结构

```
your_project/
├── coordinator/
│   ├── config.yaml          # 配置文件
│   ├── start.py             # 启动脚本
│   └── data/
│       └── questions.jsonl  # 数据文件
├── workers/
│   ├── worker1.py
│   ├── worker2.py
│   └── ...
└── monitoring/
    ├── metrics.py
    └── health_check.py
```

这样的设置能够确保系统的可维护性和可扩展性。 