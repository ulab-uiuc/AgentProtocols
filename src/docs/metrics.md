# Metrics System - Performance Monitoring and Observability

### Overview

The AgentNetwork metrics system provides comprehensive performance monitoring and observability capabilities using Prometheus-based metrics. It tracks key performance indicators including latency, throughput, error rates, and resource utilization across the entire agent network infrastructure.

### Key Features

- **Prometheus Integration**: Native Prometheus metrics for industry-standard monitoring
- **Multi-dimensional Metrics**: Labels for detailed breakdown and analysis
- **Real-time Monitoring**: Live metrics collection and reporting
- **Performance Tracking**: Latency, throughput, and error rate monitoring
- **Resource Utilization**: Memory, CPU, and network usage tracking
- **Custom Metrics**: Extensible framework for application-specific metrics

### Metric Types

#### Counters
**REQUEST_LATENCY** - Histogram of request processing latencies
```python
REQUEST_LATENCY = Histogram(
    'agent_request_latency_seconds',
    'Latency of agent requests in seconds',
    ['source_agent', 'dest_agent', 'protocol']
)
```

**REQUEST_FAILURES** - Counter of failed requests
```python
REQUEST_FAILURES = Counter(
    'agent_request_failures_total',
    'Total number of failed agent requests',
    ['source_agent', 'dest_agent']
)
```

**MSG_BYTES** - Counter of message bytes transferred
```python
MSG_BYTES = Counter(
    'agent_message_bytes_total',
    'Total bytes in agent messages',
    ['direction', 'agent_id']
)
```

**RECOVERY_TIME** - Histogram of failure recovery times
```python
RECOVERY_TIME = Histogram(
    'agent_recovery_time_seconds',
    'Time taken to recover from failures in seconds'
)
```

### MetricsTimer Utility

The `MetricsTimer` class provides a convenient context manager for measuring operation durations:

```python
class MetricsTimer:
    """Context manager for timing operations and recording to Prometheus histograms."""
    
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

### Usage Examples

#### Measuring Request Latency

```python
# Basic usage
with MetricsTimer(REQUEST_LATENCY, ("agent-1", "agent-2", "a2a")):
    response = await agent.send_message(destination, payload)

# Manual timing
start_time = time.time()
try:
    response = await process_request()
    REQUEST_LATENCY.labels("source", "dest", "protocol").observe(time.time() - start_time)
except Exception as e:
    REQUEST_FAILURES.labels("source", "dest").inc()
    raise
```

#### Recording Message Statistics

```python
# Track outgoing message bytes
message_size = len(json.dumps(payload))
MSG_BYTES.labels("out", agent_id).inc(message_size)

# Track incoming message bytes  
MSG_BYTES.labels("in", agent_id).inc(received_bytes)
```

#### Recovery Time Tracking

```python
# Record time to recover from failure
failure_start = time.time()
try:
    await recover_from_failure()
    recovery_duration = time.time() - failure_start
    RECOVERY_TIME.observe(recovery_duration)
except Exception:
    # Recovery failed
    pass
```

### Metric Labels and Dimensions

#### Agent Identification
- `source_agent`: ID of the agent sending the request
- `dest_agent`: ID of the agent receiving the request
- `agent_id`: General agent identifier

#### Protocol Information
- `protocol`: Communication protocol used (a2a, ioa, etc.)
- `direction`: Message direction (in/out)

#### Operation Types
- `operation`: Type of operation being performed
- `status`: Success/failure status
- `error_type`: Classification of errors

### Integration with BaseAgent

The BaseAgent class automatically integrates with the metrics system:

```python
async def send(self, dst_id: str, payload: Dict[str, Any]) -> Any:
    # ... validation code ...
    
    protocol_name = type(adapter).__name__.replace("Adapter", "").lower()
    
    with MetricsTimer(REQUEST_LATENCY, (self.agent_id, dst_id, protocol_name)):
        try:
            response = await adapter.send_message(dst_id, enriched_payload)
            
            # Record successful message bytes
            msg_size = len(json.dumps(enriched_payload).encode('utf-8'))
            MSG_BYTES.labels("out", self.agent_id).inc(msg_size)
            
            return response
            
        except Exception as e:
            REQUEST_FAILURES.labels(self.agent_id, dst_id).inc()
            raise RuntimeError(f"Failed to send message: {e}") from e
```

### AgentNetwork Metrics Integration

The AgentNetwork class provides metrics aggregation and reporting:

```python
def snapshot_metrics(self) -> Dict[str, Any]:
    """Return current metrics dict with network statistics."""
    base_metrics = {
        "agent_count": len(self._agents),
        "edge_count": sum(len(edges) for edges in self._graph.values()),
        "topology": self.get_topology()
    }
    base_metrics.update(self._metrics)
    return base_metrics

def record_recovery(self) -> None:
    """Record successful recovery after failure."""
    t0 = self._metrics.pop("failstorm_t0", None)
    if t0 is not None:
        recovery_time = time.time() - t0
        RECOVERY_TIME.observe(recovery_time)
```

### Monitoring and Alerting

#### Prometheus Queries

**Average Request Latency by Protocol:**
```promql
rate(agent_request_latency_seconds_sum[5m]) / rate(agent_request_latency_seconds_count[5m])
```

**Request Failure Rate:**
```promql
rate(agent_request_failures_total[5m]) / rate(agent_request_latency_seconds_count[5m])
```

**Message Throughput:**
```promql
rate(agent_message_bytes_total[5m])
```

#### Common Alerts

**High Error Rate:**
```yaml
- alert: HighAgentErrorRate
  expr: rate(agent_request_failures_total[5m]) > 0.1
  for: 2m
  annotations:
    summary: "High error rate detected in agent network"
```

**High Latency:**
```yaml
- alert: HighAgentLatency
  expr: histogram_quantile(0.95, rate(agent_request_latency_seconds_bucket[5m])) > 1.0
  for: 5m
  annotations:
    summary: "95th percentile latency exceeds 1 second"
```

## Configuration and Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'agent-network'
    static_configs:
      - targets: ['localhost:8000']  # Metrics endpoint
    scrape_interval: 5s
    metrics_path: /metrics
```

### Grafana Dashboard Example

```json
{
  "dashboard": {
    "title": "AgentNetwork Metrics",
    "panels": [
      {
        "title": "Request Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(agent_request_latency_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph", 
        "targets": [
          {
            "expr": "rate(agent_request_failures_total[5m])",
            "legendFormat": "Errors per second"
          }
        ]
      }
    ]
  }
}
```

## Best Practices

1. **Label Cardinality**: Keep label cardinality low to avoid memory issues
2. **Sampling**: Use appropriate sampling for high-frequency metrics
3. **Aggregation**: Pre-aggregate metrics when possible
4. **Retention**: Set appropriate retention policies for different metric types
5. **Alerting**: Create meaningful alerts based on business impact 