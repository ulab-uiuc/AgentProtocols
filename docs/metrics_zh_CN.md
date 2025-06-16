# 指标系统 - 性能监控与可观测性

## 概述

AgentNetwork指标系统使用基于Prometheus的指标提供全面的性能监控和可观测性功能。它跟踪关键性能指标，包括延迟、吞吐量、错误率和整个智能体网络基础设施的资源利用率。

## 关键特性

- **Prometheus集成**: 用于行业标准监控的原生Prometheus指标
- **多维度指标**: 用于详细分解和分析的标签
- **实时监控**: 实时指标收集和报告
- **性能跟踪**: 延迟、吞吐量和错误率监控
- **资源利用率**: 内存、CPU和网络使用情况跟踪
- **自定义指标**: 应用程序特定指标的可扩展框架

## 指标类型

### 计数器
**REQUEST_LATENCY** - 请求处理延迟直方图
```python
REQUEST_LATENCY = Histogram(
    'agent_request_latency_seconds',
    'Latency of agent requests in seconds',
    ['source_agent', 'dest_agent', 'protocol']
)
```

**REQUEST_FAILURES** - 失败请求计数器
```python
REQUEST_FAILURES = Counter(
    'agent_request_failures_total',
    'Total number of failed agent requests',
    ['source_agent', 'dest_agent']
)
```

**MSG_BYTES** - 消息字节传输计数器
```python
MSG_BYTES = Counter(
    'agent_message_bytes_total',
    'Total bytes in agent messages',
    ['direction', 'agent_id']
)
```

**RECOVERY_TIME** - 故障恢复时间直方图
```python
RECOVERY_TIME = Histogram(
    'agent_recovery_time_seconds',
    'Time taken to recover from failures in seconds'
)
```

## MetricsTimer实用工具

`MetricsTimer`类为测量操作持续时间提供了方便的上下文管理器：

```python
class MetricsTimer:
    """用于计时操作并记录到Prometheus直方图的上下文管理器"""
    
    def __init__(self, histogram: Histogram, labels: Tuple[str, ...] = ()):
        self.histogram = histogram
        self.labels = labels
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if self.labels:
            self.histogram.labels(*self.labels).observe(duration)
        else:
            self.histogram.observe(duration)
```

## 使用示例

### 测量请求延迟

```python
# 基本用法
with MetricsTimer(REQUEST_LATENCY, ("agent-1", "agent-2", "a2a")):
    response = await agent.send_message(destination, payload)

# 手动计时
start_time = time.time()
try:
    response = await process_request()
    REQUEST_LATENCY.labels("source", "dest", "protocol").observe(time.time() - start_time)
except Exception as e:
    REQUEST_FAILURES.labels("source", "dest").inc()
    raise
```

### 记录消息统计

```python
# 跟踪出站消息字节
message_size = len(json.dumps(payload))
MSG_BYTES.labels("out", agent_id).inc(message_size)

# 跟踪入站消息字节
MSG_BYTES.labels("in", agent_id).inc(received_bytes)
```

### 恢复时间跟踪

```python
# 记录从故障中恢复的时间
failure_start = time.time()
try:
    await recover_from_failure()
    recovery_duration = time.time() - failure_start
    RECOVERY_TIME.observe(recovery_duration)
except Exception:
    # 恢复失败
    pass
```

## 指标标签和维度

### 智能体标识
- `source_agent`: 发送请求的智能体ID
- `dest_agent`: 接收请求的智能体ID
- `agent_id`: 通用智能体标识符

### 协议信息
- `protocol`: 使用的通信协议(a2a、ioa等)
- `direction`: 消息方向(in/out)

### 操作类型
- `operation`: 执行的操作类型
- `status`: 成功/失败状态
- `error_type`: 错误分类

## 与BaseAgent的集成

BaseAgent类自动与指标系统集成：

```python
async def send(self, dst_id: str, payload: Dict[str, Any]) -> Any:
    # ... 验证代码 ...
    
    protocol_name = type(adapter).__name__.replace("Adapter", "").lower()
    
    with MetricsTimer(REQUEST_LATENCY, (self.agent_id, dst_id, protocol_name)):
        try:
            response = await adapter.send_message(dst_id, enriched_payload)
            
            # 记录成功的消息字节
            msg_size = len(json.dumps(enriched_payload).encode('utf-8'))
            MSG_BYTES.labels("out", self.agent_id).inc(msg_size)
            
            return response
            
        except Exception as e:
            REQUEST_FAILURES.labels(self.agent_id, dst_id).inc()
            raise RuntimeError(f"Failed to send message: {e}") from e
```

## AgentNetwork指标集成

AgentNetwork类提供指标聚合和报告：

```python
def snapshot_metrics(self) -> Dict[str, Any]:
    """返回带有网络统计的当前指标字典"""
    base_metrics = {
        "agent_count": len(self._agents),
        "edge_count": sum(len(edges) for edges in self._graph.values()),
        "topology": self.get_topology()
    }
    base_metrics.update(self._metrics)
    return base_metrics

def record_recovery(self) -> None:
    """记录故障后的成功恢复"""
    t0 = self._metrics.pop("failstorm_t0", None)
    if t0 is not None:
        recovery_time = time.time() - t0
        RECOVERY_TIME.observe(recovery_time)
```

## 监控和告警

### Prometheus查询

**按协议的平均请求延迟:**
```promql
rate(agent_request_latency_seconds_sum[5m]) / rate(agent_request_latency_seconds_count[5m])
```

**请求失败率:**
```promql
rate(agent_request_failures_total[5m]) / rate(agent_request_latency_seconds_count[5m])
```

**消息吞吐量:**
```promql
rate(agent_message_bytes_total[5m])
```

### 常见告警

**高错误率:**
```yaml
- alert: HighAgentErrorRate
  expr: rate(agent_request_failures_total[5m]) > 0.1
  for: 2m
  annotations:
    summary: "智能体网络中检测到高错误率"
```

**高延迟:**
```yaml
- alert: HighAgentLatency
  expr: histogram_quantile(0.95, rate(agent_request_latency_seconds_bucket[5m])) > 1.0
  for: 5m
  annotations:
    summary: "95百分位延迟超过1秒"
```

## 配置和设置

### Prometheus配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'agent-network'
    static_configs:
      - targets: ['localhost:8000']  # 指标端点
    scrape_interval: 5s
    metrics_path: /metrics
```

### Grafana仪表板示例

```json
{
  "dashboard": {
    "title": "AgentNetwork指标",
    "panels": [
      {
        "title": "请求延迟",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(agent_request_latency_seconds_bucket[5m]))",
            "legendFormat": "95百分位"
          }
        ]
      },
      {
        "title": "错误率",
        "type": "graph", 
        "targets": [
          {
            "expr": "rate(agent_request_failures_total[5m])",
            "legendFormat": "每秒错误数"
          }
        ]
      }
    ]
  }
}
```

## 最佳实践

1. **标签基数**: 保持标签基数较低以避免内存问题
2. **采样**: 对高频指标使用适当的采样
3. **聚合**: 尽可能预聚合指标
4. **保留**: 为不同指标类型设置适当的保留策略
5. **告警**: 基于业务影响创建有意义的告警 