# -*- coding: utf-8 -*-
"""
Performance Metrics Collector for Streaming Queue
收集和分析详细的性能指标，包括响应时间分析、连接稳定性、网络错误等
"""

from __future__ import annotations

import time
import statistics
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class WorkerMetrics:
    """单个Worker的性能指标"""
    worker_id: str
    protocol: str
    
    # 任务完成统计
    completed_tasks: int = 0
    failed_tasks: int = 0
    timeout_tasks: int = 0
    
    # 响应时间统计
    response_times: List[float] = field(default_factory=list)
    
    # 连接统计
    connection_retries: int = 0
    network_errors: int = 0
    total_requests: int = 0
    
    # 最近响应时间窗口（用于实时监控）
    recent_response_times: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass  
class ProtocolMetrics:
    """协议级别的聚合指标"""
    protocol_name: str
    
    # 聚合统计
    total_completed: int = 0
    total_failed: int = 0
    total_timeout: int = 0
    total_retries: int = 0
    total_network_errors: int = 0
    
    # 响应时间统计
    all_response_times: List[float] = field(default_factory=list)
    
    # Worker分布
    worker_completion_counts: Dict[str, int] = field(default_factory=dict)


class PerformanceMetricsCollector:
    """性能指标收集器"""
    
    def __init__(self):
        self.worker_metrics: Dict[str, WorkerMetrics] = {}
        self.protocol_metrics: Dict[str, ProtocolMetrics] = {}
        self.test_start_time: Optional[float] = None
        self.test_end_time: Optional[float] = None
        
        # 全局设置
        self.response_timeout: float = 60.0  # 默认超时时间
        
    def register_worker(self, worker_id: str, protocol: str) -> None:
        """注册一个Worker"""
        self.worker_metrics[worker_id] = WorkerMetrics(
            worker_id=worker_id,
            protocol=protocol
        )
        
        if protocol not in self.protocol_metrics:
            self.protocol_metrics[protocol] = ProtocolMetrics(protocol_name=protocol)
    
    def start_test(self) -> None:
        """开始测试计时"""
        self.test_start_time = time.time()
    
    def end_test(self) -> None:
        """结束测试计时"""
        self.test_end_time = time.time()
    
    def record_task_start(self, worker_id: str) -> float:
        """记录任务开始时间，返回开始时间戳"""
        if worker_id in self.worker_metrics:
            self.worker_metrics[worker_id].total_requests += 1
        return time.time()
    
    def record_task_completion(self, worker_id: str, start_time: float, 
                             success: bool, error: Optional[str] = None) -> None:
        """记录任务完成"""
        if worker_id not in self.worker_metrics:
            return
            
        worker = self.worker_metrics[worker_id]
        protocol = self.protocol_metrics[worker.protocol]
        
        response_time = time.time() - start_time
        
        if success:
            worker.completed_tasks += 1
            worker.response_times.append(response_time)
            worker.recent_response_times.append(response_time)
            
            protocol.total_completed += 1
            protocol.all_response_times.append(response_time)
            protocol.worker_completion_counts[worker_id] = worker.completed_tasks
        else:
            # 检查是否是超时
            if response_time >= self.response_timeout:
                worker.timeout_tasks += 1
                protocol.total_timeout += 1
            else:
                worker.failed_tasks += 1
                protocol.total_failed += 1
    
    def record_connection_retry(self, worker_id: str) -> None:
        """记录连接重试"""
        if worker_id in self.worker_metrics:
            self.worker_metrics[worker_id].connection_retries += 1
            protocol = self.worker_metrics[worker_id].protocol
            self.protocol_metrics[protocol].total_retries += 1
    
    def record_network_error(self, worker_id: str, error_type: str) -> None:
        """记录网络错误"""
        if worker_id in self.worker_metrics:
            self.worker_metrics[worker_id].network_errors += 1
            protocol = self.worker_metrics[worker_id].protocol
            self.protocol_metrics[protocol].total_network_errors += 1
    
    def get_worker_statistics(self, worker_id: str) -> Dict[str, Any]:
        """获取单个Worker的统计信息"""
        if worker_id not in self.worker_metrics:
            return {}
            
        worker = self.worker_metrics[worker_id]
        
        # 计算响应时间统计
        response_stats = {}
        if worker.response_times:
            response_stats = {
                "average_response_time": statistics.mean(worker.response_times),
                "min_response_time": min(worker.response_times),
                "max_response_time": max(worker.response_times),
                "response_time_std": statistics.stdev(worker.response_times) if len(worker.response_times) > 1 else 0.0,
                "median_response_time": statistics.median(worker.response_times)
            }
        
        # 计算成功率
        total_tasks = worker.completed_tasks + worker.failed_tasks + worker.timeout_tasks
        success_rate = (worker.completed_tasks / total_tasks) if total_tasks > 0 else 0.0
        
        # 计算网络错误率
        network_error_rate = (worker.network_errors / worker.total_requests) if worker.total_requests > 0 else 0.0
        
        return {
            "worker_id": worker_id,
            "protocol": worker.protocol,
            "completed_tasks": worker.completed_tasks,
            "failed_tasks": worker.failed_tasks,
            "timeout_tasks": worker.timeout_tasks,
            "total_tasks": total_tasks,
            "success_rate": success_rate,
            "connection_retries": worker.connection_retries,
            "network_errors": worker.network_errors,
            "network_error_rate": network_error_rate,
            "total_requests": worker.total_requests,
            **response_stats
        }
    
    def get_protocol_statistics(self, protocol: str) -> Dict[str, Any]:
        """获取协议级别的统计信息"""
        if protocol not in self.protocol_metrics:
            return {}
            
        proto = self.protocol_metrics[protocol]
        
        # 计算协议级响应时间统计
        response_stats = {}
        if proto.all_response_times:
            response_stats = {
                "average_response_time": statistics.mean(proto.all_response_times),
                "min_response_time": min(proto.all_response_times),
                "max_response_time": max(proto.all_response_times),
                "response_time_std": statistics.stdev(proto.all_response_times) if len(proto.all_response_times) > 1 else 0.0,
                "median_response_time": statistics.median(proto.all_response_times)
            }
        
        # 计算Worker负载均衡方差
        completion_counts = list(proto.worker_completion_counts.values())
        load_balance_variance = statistics.variance(completion_counts) if len(completion_counts) > 1 else 0.0
        
        # 计算总体统计
        total_tasks = proto.total_completed + proto.total_failed + proto.total_timeout
        success_rate = (proto.total_completed / total_tasks) if total_tasks > 0 else 0.0
        
        return {
            "protocol": protocol,
            "total_completed": proto.total_completed,
            "total_failed": proto.total_failed,
            "total_timeout": proto.total_timeout,
            "total_tasks": total_tasks,
            "success_rate": success_rate,
            "total_retries": proto.total_retries,
            "total_network_errors": proto.total_network_errors,
            "worker_completion_counts": proto.worker_completion_counts,
            "load_balance_variance": load_balance_variance,
            **response_stats
        }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """获取完整的性能报告"""
        report = {
            "test_duration": (self.test_end_time - self.test_start_time) if (self.test_start_time and self.test_end_time) else None,
            "test_start_time": self.test_start_time,
            "test_end_time": self.test_end_time,
            "protocols": {},
            "workers": {},
            "summary": {}
        }
        
        # 收集协议统计
        for protocol in self.protocol_metrics:
            report["protocols"][protocol] = self.get_protocol_statistics(protocol)
        
        # 收集Worker统计
        for worker_id in self.worker_metrics:
            report["workers"][worker_id] = self.get_worker_statistics(worker_id)
        
        # 生成总结
        total_completed = sum(p["total_completed"] for p in report["protocols"].values())
        total_failed = sum(p["total_failed"] for p in report["protocols"].values())
        total_timeout = sum(p["total_timeout"] for p in report["protocols"].values())
        total_retries = sum(p["total_retries"] for p in report["protocols"].values())
        total_network_errors = sum(p["total_network_errors"] for p in report["protocols"].values())
        
        # 计算所有响应时间的全局统计
        all_response_times = []
        for proto in self.protocol_metrics.values():
            all_response_times.extend(proto.all_response_times)
        
        global_response_stats = {}
        if all_response_times:
            global_response_stats = {
                "global_average_response_time": statistics.mean(all_response_times),
                "global_min_response_time": min(all_response_times),
                "global_max_response_time": max(all_response_times),
                "global_response_time_std": statistics.stdev(all_response_times) if len(all_response_times) > 1 else 0.0,
                "global_median_response_time": statistics.median(all_response_times)
            }
        
        report["summary"] = {
            "total_completed": total_completed,
            "total_failed": total_failed,
            "total_timeout": total_timeout,
            "total_tasks": total_completed + total_failed + total_timeout,
            "overall_success_rate": (total_completed / (total_completed + total_failed + total_timeout)) if (total_completed + total_failed + total_timeout) > 0 else 0.0,
            "total_retries": total_retries,
            "total_network_errors": total_network_errors,
            "protocol_count": len(self.protocol_metrics),
            "worker_count": len(self.worker_metrics),
            **global_response_stats
        }
        
        return report
    
    def print_real_time_stats(self) -> None:
        """打印实时统计信息"""
        print("\n" + "="*60)
        print("🔍 Real-time Performance Statistics")
        print("="*60)
        
        for protocol in self.protocol_metrics:
            proto_stats = self.get_protocol_statistics(protocol)
            print(f"\n📊 Protocol: {protocol}")
            print(f"   Completed: {proto_stats.get('total_completed', 0)}")
            print(f"   Failed: {proto_stats.get('total_failed', 0)}")
            print(f"   Timeout: {proto_stats.get('total_timeout', 0)}")
            print(f"   Success Rate: {proto_stats.get('success_rate', 0):.2%}")
            print(f"   Avg Response Time: {proto_stats.get('average_response_time', 0):.2f}s")
            print(f"   Load Balance Variance: {proto_stats.get('load_balance_variance', 0):.2f}")
            print(f"   Network Errors: {proto_stats.get('total_network_errors', 0)}")
            print(f"   Connection Retries: {proto_stats.get('total_retries', 0)}")
