"""
Monitoring metrics module - Prometheus-based metrics collection and exposure
"""

import asyncio
import time
from typing import Optional

try:
    import psutil
except ImportError as e:
    raise ImportError(f"[METRICS]: Failed to import psutil for system monitoring - {e}")

try:
    import pynvml
except ImportError as e:
    raise ImportError(f"[METRICS]: Failed to import pynvml for GPU monitoring - {e}")

from prometheus_client import Counter, Histogram, Gauge, start_http_server


# --------------------------- Core Metrics ---------------------------
REQUEST_LATENCY = Histogram(
    "agent_request_latency_seconds",
    "Latency for agent→agent messages",
    ["src", "dst", "protocol"]
)

MSG_BYTES = Counter(
    "agent_message_bytes_total",
    "Bytes transferred through AgentNetwork",
    ["direction", "agent_id"]
)

REQUEST_FAILURES = Counter(
    "agent_request_failures_total",
    "Number of failed messages",
    ["src", "dst"]
)

RECOVERY_TIME = Histogram(
    "agent_recovery_time_seconds",
    "Time to recover from failures",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# --------------------------- System Metrics ---------------------------
CPU_PERCENT = Gauge("cpu_percent", "CPU usage (%)", ["host"])
MEMORY_BYTES = Gauge("memory_bytes", "Memory usage in bytes", ["host", "type"])
GPU_IDLE_RATIO = Gauge("gpu_idle_ratio", "Fraction of time GPU is idle", ["gpu_index"])
GPU_MEMORY_BYTES = Gauge("gpu_memory_bytes", "GPU memory usage", ["gpu_index", "type"])


def setup_prometheus_metrics(port: int = 8000) -> None:
    """Start Prometheus metrics server."""
    start_http_server(port)
    print(f"Prometheus metrics server started on port {port}")
    print(f"Metrics available at: http://localhost:{port}/metrics")


async def sample_system_metrics(host_name: str = "localhost") -> None:
    """Continuously sample system metrics."""
    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
    except Exception:
        gpu_count = 0

    while True:
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            CPU_PERCENT.labels(host_name).set(cpu_percent)

            # Memory metrics
            memory = psutil.virtual_memory()
            MEMORY_BYTES.labels(host_name, "total").set(memory.total)
            MEMORY_BYTES.labels(host_name, "used").set(memory.used)
            MEMORY_BYTES.labels(host_name, "available").set(memory.available)

            # GPU metrics
            if gpu_count > 0:
                for i in range(gpu_count):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        GPU_IDLE_RATIO.labels(str(i)).set((100 - util.gpu) / 100)

                        # GPU memory
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        GPU_MEMORY_BYTES.labels(str(i), "total").set(mem_info.total)
                        GPU_MEMORY_BYTES.labels(str(i), "used").set(mem_info.used)
                        GPU_MEMORY_BYTES.labels(str(i), "free").set(mem_info.free)
                    except Exception as e:
                        print(f"Failed to get GPU {i} metrics: {e}")

        except Exception as e:
            print(f"Error sampling system metrics: {e}")

        await asyncio.sleep(1.0)


class MetricsTimer:
    """
    Context manager for timing operations.
    
    Alternative to Histogram.time() with labeled metrics support.
    Usage:
        with MetricsTimer(REQUEST_LATENCY, ("src", "dst", "protocol")):
            # ... timed operation ...
    
    Or use Prometheus native timer (for unlabeled metrics):
        with REQUEST_LATENCY.time():
            # ... timed operation ...
    """

    def __init__(self, metric: Histogram, labels: tuple = ()):
        self.metric = metric
        self.labels = labels
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            if self.labels:
                self.metric.labels(*self.labels).observe(duration)
            else:
                self.metric.observe(duration)


def get_metrics_summary() -> dict:
    """Get a summary of current metrics values."""
    summary = {}
    
    # This is a simplified version - in production you'd use
    # prometheus_client.REGISTRY to get actual values
    summary["metrics_configured"] = {
        "latency_buckets": len(REQUEST_LATENCY._upper_bounds),
        "system_metrics_enabled": True,
        "gpu_metrics_enabled": True
    }
    
    return summary 