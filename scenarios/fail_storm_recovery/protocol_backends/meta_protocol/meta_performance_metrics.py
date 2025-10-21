# -*- coding: utf-8 -*-
"""
Meta Protocol Specific Performance Metrics
Performance metrics collection and analysis system specifically designed for the Meta protocol.
"""

from __future__ import annotations

import time
import statistics
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class MetaWorkerMetrics:
    """Performance metrics for a Meta protocol worker"""
    worker_id: str
    protocol: str
    
    # Task completion statistics
    completed_tasks: int = 0
    failed_tasks: int = 0
    timeout_tasks: int = 0
    
    # Response time statistics
    response_times: List[float] = field(default_factory=list)
    
    # Connection statistics
    connection_retries: int = 0
    network_errors: int = 0
    total_requests: int = 0
    
    # Meta-protocol specific metrics
    cross_protocol_calls: int = 0
    llm_routing_decisions: int = 0


@dataclass  
class MetaProtocolMetrics:
    """Aggregated metrics at the Meta protocol level"""
    protocol_name: str
    
    # Aggregated statistics
    total_completed: int = 0
    total_failed: int = 0
    total_timeout: int = 0
    total_retries: int = 0
    total_network_errors: int = 0
    
    # Response time statistics
    all_response_times: List[float] = field(default_factory=list)
    
    # Worker distribution
    worker_completion_counts: Dict[str, int] = field(default_factory=dict)
    
    # Meta-protocol specific statistics
    cross_protocol_efficiency: float = 0.0
    llm_routing_accuracy: float = 0.0


class MetaPerformanceMetricsCollector:
    """Performance metrics collector specialized for the Meta protocol"""
    
    def __init__(self):
        self.worker_metrics: Dict[str, MetaWorkerMetrics] = {}
        self.protocol_metrics: Dict[str, MetaProtocolMetrics] = {}
        self.test_start_time: Optional[float] = None
        self.test_end_time: Optional[float] = None
        
        # Meta-protocol specific metrics
        self.protocol_mix_stats: Dict[str, int] = defaultdict(int)
        self.routing_decisions: List[Dict[str, Any]] = []
        
        # Global setup
        self.response_timeout: float = 60.0
        
    def register_worker(self, worker_id: str, protocol: str) -> None:
        """Register a Meta worker"""
        self.worker_metrics[worker_id] = MetaWorkerMetrics(
            worker_id=worker_id,
            protocol=protocol
        )
        
        if protocol not in self.protocol_metrics:
            self.protocol_metrics[protocol] = MetaProtocolMetrics(protocol_name=protocol)
            
        # Record protocol mix statistics
        self.protocol_mix_stats[protocol] += 1
    
    def start_test(self) -> None:
        """Start test timer"""
        self.test_start_time = time.time()
    
    def end_test(self) -> None:
        """End test timer"""
        self.test_end_time = time.time()
    
    def record_task_start(self, worker_id: str) -> float:
        """Record task start time and return the start timestamp"""
        if worker_id in self.worker_metrics:
            self.worker_metrics[worker_id].total_requests += 1
        return time.time()
    
    def record_task_completion(self, worker_id: str, start_time: float, 
                             success: bool, error: Optional[str] = None) -> None:
        """Record task completion"""
        if worker_id not in self.worker_metrics:
            return
            
        worker = self.worker_metrics[worker_id]
        protocol = self.protocol_metrics[worker.protocol]
        
        response_time = time.time() - start_time
        
        if success:
            worker.completed_tasks += 1
            worker.response_times.append(response_time)
            
            protocol.total_completed += 1
            protocol.all_response_times.append(response_time)
            protocol.worker_completion_counts[worker_id] = worker.completed_tasks
        else:
            # Check whether this was a timeout
            if response_time >= self.response_timeout:
                worker.timeout_tasks += 1
                protocol.total_timeout += 1
            else:
                worker.failed_tasks += 1
                protocol.total_failed += 1
    
    def record_cross_protocol_call(self, src_worker: str, dst_worker: str) -> None:
        """Record a cross-protocol call"""
        if src_worker in self.worker_metrics:
            self.worker_metrics[src_worker].cross_protocol_calls += 1
    
    def record_llm_routing_decision(self, decision: Dict[str, Any]) -> None:
        """Record an LLM routing decision"""
        self.routing_decisions.append({
            **decision,
            "timestamp": time.time()
        })
        
        # Update routing decision counts for all workers
        for worker in self.worker_metrics.values():
            worker.llm_routing_decisions += 1
    
    def record_connection_retry(self, worker_id: str) -> None:
        """Record a connection retry"""
        if worker_id in self.worker_metrics:
            self.worker_metrics[worker_id].connection_retries += 1
            protocol = self.worker_metrics[worker_id].protocol
            self.protocol_metrics[protocol].total_retries += 1
    
    def record_network_error(self, worker_id: str, error_type: str) -> None:
        """Record a network error"""
        if worker_id in self.worker_metrics:
            self.worker_metrics[worker_id].network_errors += 1
            protocol = self.worker_metrics[worker_id].protocol
            self.protocol_metrics[protocol].total_network_errors += 1
    
    def get_worker_statistics(self, worker_id: str) -> Dict[str, Any]:
        """Get statistics for a single worker"""
        if worker_id not in self.worker_metrics:
            return {}
            
        worker = self.worker_metrics[worker_id]
        
        # Calculate response time statistics
        response_stats = {}
        if worker.response_times:
            response_stats = {
                "average_response_time": statistics.mean(worker.response_times),
                "min_response_time": min(worker.response_times),
                "max_response_time": max(worker.response_times),
                "response_time_std": statistics.stdev(worker.response_times) if len(worker.response_times) > 1 else 0.0,
                "median_response_time": statistics.median(worker.response_times)
            }
        
        # Calculate success rate
        total_tasks = worker.completed_tasks + worker.failed_tasks + worker.timeout_tasks
        success_rate = (worker.completed_tasks / total_tasks) if total_tasks > 0 else 0.0
        
        # Calculate network error rate
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
            "cross_protocol_calls": worker.cross_protocol_calls,
            "llm_routing_decisions": worker.llm_routing_decisions,
            **response_stats
        }
    
    def get_protocol_statistics(self, protocol: str) -> Dict[str, Any]:
        """Get statistics at the protocol level"""
        if protocol not in self.protocol_metrics:
            return {}
            
        proto = self.protocol_metrics[protocol]
        
        # Calculate protocol-level response time statistics
        response_stats = {}
        if proto.all_response_times:
            response_stats = {
                "average_response_time": statistics.mean(proto.all_response_times),
                "min_response_time": min(proto.all_response_times),
                "max_response_time": max(proto.all_response_times),
                "response_time_std": statistics.stdev(proto.all_response_times) if len(proto.all_response_times) > 1 else 0.0,
                "median_response_time": statistics.median(proto.all_response_times)
            }
        
        # Calculate worker load-balance variance
        completion_counts = list(proto.worker_completion_counts.values())
        load_balance_variance = statistics.variance(completion_counts) if len(completion_counts) > 1 else 0.0
        
        # Calculate overall statistics
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
    
    def get_meta_specific_report(self) -> Dict[str, Any]:
        """Get a Meta-protocol-specific performance report"""
        return {
            "protocol_mix": dict(self.protocol_mix_stats),
            "total_routing_decisions": len(self.routing_decisions),
            "routing_decisions": self.routing_decisions,
            "cross_protocol_efficiency": self._calculate_cross_protocol_efficiency(),
            "protocol_distribution": self._get_protocol_distribution()
        }
    
    def _calculate_cross_protocol_efficiency(self) -> float:
        """Calculate cross-protocol communication efficiency"""
        total_cross_calls = sum(w.cross_protocol_calls for w in self.worker_metrics.values())
        total_tasks = sum(w.completed_tasks + w.failed_tasks for w in self.worker_metrics.values())
        return (total_cross_calls / total_tasks) if total_tasks > 0 else 0.0
    
    def _get_protocol_distribution(self) -> Dict[str, float]:
        """Get protocol distribution percentages"""
        total_workers = len(self.worker_metrics)
        if total_workers == 0:
            return {}
        
        distribution = {}
        for protocol, count in self.protocol_mix_stats.items():
            distribution[protocol] = (count / total_workers) * 100.0
        return distribution
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get a comprehensive Meta protocol performance report"""
        report = {
            "test_duration": (self.test_end_time - self.test_start_time) if (self.test_start_time and self.test_end_time) else None,
            "test_start_time": self.test_start_time,
            "test_end_time": self.test_end_time,
            "protocols": {},
            "workers": {},
            "meta_specific": self.get_meta_specific_report(),
            "summary": {}
        }
        
        # Collect protocol statistics
        for protocol in self.protocol_metrics:
            report["protocols"][protocol] = self.get_protocol_statistics(protocol)
        
        # Collect worker statistics
        for worker_id in self.worker_metrics:
            report["workers"][worker_id] = self.get_worker_statistics(worker_id)
        
        # Produce summary
        total_completed = sum(p["total_completed"] for p in report["protocols"].values())
        total_failed = sum(p["total_failed"] for p in report["protocols"].values())
        total_timeout = sum(p["total_timeout"] for p in report["protocols"].values())
        total_retries = sum(p["total_retries"] for p in report["protocols"].values())
        total_network_errors = sum(p["total_network_errors"] for p in report["protocols"].values())
        
        # Calculate global statistics for all response times
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
    
    def print_meta_performance_summary(self, output=None) -> None:
        """Print the Meta-protocol specific performance summary"""
        report = self.get_comprehensive_report()
        
        def _print(level: str, msg: str):
            if output and hasattr(output, level):
                getattr(output, level)(msg)
            else:
                print(f"[{level.upper()}] {msg}")
        
        _print("info", "=== Meta Protocol Performance Summary ===")
        
        summary = report.get("summary", {})
        _print("system", f"Test Duration: {report.get('test_duration', 0):.2f} seconds")
        _print("system", f"Total Tasks: {summary.get('total_tasks', 0)}")
        _print("system", f"Success Rate: {summary.get('overall_success_rate', 0):.2%}")
        _print("system", f"Total Retries: {summary.get('total_retries', 0)}")
        _print("system", f"Network Errors: {summary.get('total_network_errors', 0)}")
        
        # Meta-protocol specific statistics
        meta_stats = report.get("meta_specific", {})
        if meta_stats:
            _print("info", "=== Meta Protocol Specific ===")
            _print("progress", f"Protocol Mix: {meta_stats.get('protocol_mix', {})}")
            _print("progress", f"Protocol Distribution: {meta_stats.get('protocol_distribution', {})}")
            _print("progress", f"LLM Routing Decisions: {meta_stats.get('total_routing_decisions', 0)}")
            _print("progress", f"Cross-Protocol Efficiency: {meta_stats.get('cross_protocol_efficiency', 0):.2%}")
        
        # Response time statistics
        if "global_average_response_time" in summary:
            _print("info", "=== Response Time Analysis ===")
            _print("progress", f"Average: {summary.get('global_average_response_time', 0):.2f}s")
            _print("progress", f"Min: {summary.get('global_min_response_time', 0):.2f}s")
            _print("progress", f"Max: {summary.get('global_max_response_time', 0):.2f}s")
            _print("progress", f"Std Dev: {summary.get('global_response_time_std', 0):.2f}s")
            _print("progress", f"Median: {summary.get('global_median_response_time', 0):.2f}s")
        
        # Protocol-level statistics
        protocols = report.get("protocols", {})
        if protocols:
            _print("info", "=== Protocol Performance ===")
            for protocol, stats in protocols.items():
                _print("progress", f"{protocol}:")
                _print("progress", f"  Completed: {stats.get('total_completed', 0)}")
                _print("progress", f"  Success Rate: {stats.get('success_rate', 0):.2%}")
                _print("progress", f"  Avg Response: {stats.get('average_response_time', 0):.2f}s")
                _print("progress", f"  Load Balance Variance: {stats.get('load_balance_variance', 0):.2f}")
                _print("progress", f"  Network Errors: {stats.get('total_network_errors', 0)}")
                _print("progress", f"  Retries: {stats.get('total_retries', 0)}")
        
        # Worker-level statistics
        workers = report.get("workers", {})
        if workers:
            _print("info", "=== Worker Performance ===")
            for worker_id, stats in workers.items():
                _print("progress", f"{worker_id} ({stats.get('protocol', 'unknown')}):")
                _print("progress", f"  Completed: {stats.get('completed_tasks', 0)}")
                _print("progress", f"  Failed: {stats.get('failed_tasks', 0)}")
                _print("progress", f"  Timeout: {stats.get('timeout_tasks', 0)}")
                _print("progress", f"  Network Error Rate: {stats.get('network_error_rate', 0):.2%}")
                _print("progress", f"  Avg Response: {stats.get('average_response_time', 0):.2f}s")
                _print("progress", f"  Cross-Protocol Calls: {stats.get('cross_protocol_calls', 0)}")
