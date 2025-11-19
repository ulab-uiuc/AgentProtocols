# -*- coding: utf-8 -*-
"""
QA Coordinator Base:
- Responsible for: configuration, loading questions, dynamic scheduling, results aggregation and persistence
- Not concerned with: specific communication protocols
- Provides abstract method send_to_worker() to be implemented by concrete protocols (e.g., A2A, HTTP, gRPC)

Usage:
    from qa_coordinator_base import QACoordinatorBase

    class MyCoordinator(QACoordinatorBase):
        async def send_to_worker(self, worker_id: str, question: str) -> dict:
            ...
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

# Import performance metrics collector
try:
    from .performance_metrics import PerformanceMetricsCollector
except ImportError:
    # Fallback if import fails
    class PerformanceMetricsCollector:
        def __init__(self): 
            self.worker_metrics = {}
            self.protocol_metrics = {}
            self.test_start_time = None
            self.test_end_time = None
            self.response_timeout = 60.0
        def register_worker(self, worker_id: str, protocol: str): pass
        def start_test(self): 
            self.test_start_time = time.time()
        def end_test(self): 
            self.test_end_time = time.time()
        def record_task_start(self, worker_id: str): return time.time()
        def record_task_completion(self, worker_id: str, start_time: float, success: bool, error=None): pass
        def record_connection_retry(self, worker_id: str): pass
        def record_network_error(self, worker_id: str, error_type: str): pass
        def get_comprehensive_report(self): 
            return {
                "test_duration": (self.test_end_time - self.test_start_time) if (self.test_start_time and self.test_end_time) else 0.0,
                "summary": {"total_tasks": 0, "overall_success_rate": 0.0, "total_retries": 0, "total_network_errors": 0},
                "protocols": {},
                "workers": {}
            }


class QACoordinatorBase(ABC):
    """QA Coordinator abstract base class: scheduling logic is complete here; communication is implemented by subclasses."""

    def __init__(self, config: Optional[dict] = None, output=None):
        self.config = config or {}
        qa_cfg = self.config.get("qa", {}) or {}
        coordinator_cfg = qa_cfg.get("coordinator", {}) or {}
        network_cfg = qa_cfg.get("network", {}) or {}

        self.batch_size: int = int(coordinator_cfg.get("batch_size", 50))
        self.first_50: bool = bool(coordinator_cfg.get("first_50", True))
        self.data_path: str = coordinator_cfg.get("data_file", "data/top1000_simplified.jsonl")
        self.result_file: str = coordinator_cfg.get("result_file", "data/qa_results_anp_0801.json")

        self.coordinator_id: str = coordinator_cfg.get("coordinator_id", "Coordinator-1")
        self.worker_ids: List[str] = []
        self.agent_network: Any = None  # Container for "network/router/client"; semantics defined by subclasses
        self.output = output
        
        # Initialize performance metrics collector
        self.metrics_collector = PerformanceMetricsCollector()
        self.response_timeout = float(network_cfg.get("response_timeout", 60))
        self.metrics_collector.response_timeout = self.response_timeout

    # --------------- Basic setup ---------------
    def set_network(self, network: Any, worker_ids: List[str], protocol_name: str = "unknown") -> None:
        """Setup network/router and the available worker list. The concrete type of 'network' is interpreted by subclasses."""
        self.agent_network = network
        self.worker_ids = list(worker_ids)
        
        # Register workers in metrics collector
        for worker_id in worker_ids:
            self.metrics_collector.register_worker(worker_id, protocol_name)

    # --------------- I/O helpers ---------------
    def _o(self, level: str, msg: str) -> None:
        if not self.output:
            return
        try:
            fn = getattr(self.output, level, None)
            if callable(fn):
                fn(msg)
        except Exception:
            pass

    # --------------- Question loading ---------------
    async def load_questions(self) -> List[Dict[str, str]]:
        """Load questions from a JSONL file; returns a list like [{'id':..., 'question':...}, ...]"""
        questions: List[Dict[str, str]] = []
        
        # If path is relative, resolve relative to the streaming_queue directory
        if not Path(self.data_path).is_absolute():
            # Locate the streaming_queue directory
            current_file = Path(__file__).resolve()
            streaming_queue_dir = current_file.parent.parent  # From core dir back to streaming_queue
            p = streaming_queue_dir / self.data_path
        else:
            p = Path(self.data_path)

        if not p.exists():
            self._o("error", f"Question file does not exist: {p}")
            return questions

        with p.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    q = item.get("q", "")
                    qid = item.get("id", str(len(questions) + 1))
                    if q:
                        questions.append({"id": qid, "question": q})
                    if self.first_50 and len(questions) >= 100:
                        break
                except json.JSONDecodeError as e:
                    self._o("error", f"JSON parsing failed: {line[:50]}... Error: {e}")

        self._o("system", f"Loaded {len(questions)} questions")
        return questions

    # --------------- Abstract communication ---------------
    @abstractmethod
    async def send_to_worker(self, worker_id: str, question: str) -> Dict[str, Any]:
        """
        Abstract: send the question to the specified worker and return a standardized result dict:
            {
                "answer": <str or None>,
                "raw": <protocol-native response, optional>,
            }
        Must be implemented by subclasses (e.g., A2A, HTTP, gRPC).
        """
        raise NotImplementedError

    # --------------- Scheduling: dynamic load ---------------
    async def dispatch_questions_dynamically(self, questions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        if not self.worker_ids:
            self._o("warning", "No workers configured.")
            self._o("warning", f"Debug info - agent_network: {self.agent_network is not None}, worker_ids: {self.worker_ids}")
            return []
        
        self._o("info", f"Starting dynamic dispatch: {len(questions)} questions, {len(self.worker_ids)} workers")
        self._o("info", f"Worker IDs: {self.worker_ids}")
        self._o("info", f"Network available: {self.agent_network is not None}")

        # Start performance tracking
        self.metrics_collector.start_test()

        q_queue: asyncio.Queue = asyncio.Queue()
        for q in questions:
            await q_queue.put(q)
        r_queue: asyncio.Queue = asyncio.Queue()

        workers = [
            asyncio.create_task(self._worker_loop(worker_id, q_queue, r_queue))
            for worker_id in self.worker_ids
        ]
        collector = asyncio.create_task(self._collect_results(r_queue, len(questions)))

        results = await collector
        await asyncio.gather(*workers, return_exceptions=True)
        
    # End performance tracking
        self.metrics_collector.end_test()
        
        return results

    async def _worker_loop(self, worker_id: str, q_queue: asyncio.Queue, r_queue: asyncio.Queue) -> None:
        processed = 0
        while True:
            try:
                item = q_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            processed += 1
            qid = item["id"]
            qtext = item["question"]
            self._o("progress", f"{worker_id} processing {qid}: {qtext[:50]}...")

            # Record task start with metrics collector
            start = self.metrics_collector.record_task_start(worker_id)
            
            try:
                resp = await self.send_to_worker(worker_id, qtext)
                answer = (resp or {}).get("answer") or "No answer received"
                status = "success" if answer and answer != "No answer received" else "failed"
                err = None
                
                # Extract detailed timing information
                timing_info = (resp or {}).get("timing", {})
                request_timing = timing_info.get("request_timing", {})
                llm_timing = timing_info.get("llm_timing", {})
                adapter_time = timing_info.get("adapter_time")
                
                # Record successful completion
                self.metrics_collector.record_task_completion(worker_id, start, status == "success", err)
                
            except Exception as e:
                answer, status, err = None, "failed", str(e)
                resp = None
                timing_info = {}
                request_timing = {}
                llm_timing = {}
                adapter_time = None
                
                # Record failed completion and check for network errors
                self.metrics_collector.record_task_completion(worker_id, start, False, str(e))
                
                # Check if it's a network-related error
                if any(keyword in str(e).lower() for keyword in ["connection", "timeout", "network", "refused", "unreachable"]):
                    self.metrics_collector.record_network_error(worker_id, "network_error")
                    
            end = time.time()

            await r_queue.put({
                "question_id": qid,
                "question": qtext,
                "worker": worker_id,
                "answer": answer,
                "response_time": (end - start) if status == "success" else None,
                "timestamp": time.time(),
                "status": status,
                "error": err,
                "raw": resp,
                "detailed_timing": {
                    "total_time": end - start,
                    "request_time": request_timing.get("total_request_time"),
                    "llm_time": llm_timing.get("llm_execution_time"),
                    "adapter_time": adapter_time,
                    "request_start": request_timing.get("request_start"),
                    "request_end": request_timing.get("request_end"),
                    "llm_start": llm_timing.get("llm_start"),
                    "llm_end": llm_timing.get("llm_end")
                }
            })
            q_queue.task_done()

        self._o("system", f"{worker_id} completed, processed {processed} questions")

    async def _collect_results(self, r_queue: asyncio.Queue, total: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        got = 0
        while got < total:
            item = await r_queue.get()
            out.append(item)
            got += 1
            if got % 5 == 0 or got == total:
                self._o("system", f"Collected {got}/{total} results")
            r_queue.task_done()
        self._o("success", f"Result collection completed, collected {got} results")
        return out

    # --------------- One-click execute and persist ---------------
    async def dispatch_round(self) -> str:
        questions = await self.load_questions()
        if not questions:
            return "Error: No questions loaded"

        t0 = time.time()
        results = await self.dispatch_questions_dynamically(questions)
        t1 = time.time()
        
        # Update metrics collector with total time
        if hasattr(self, 'metrics_collector') and self.metrics_collector:
            if not self.metrics_collector.test_start_time:
                self.metrics_collector.test_start_time = t0
            if not self.metrics_collector.test_end_time:
                self.metrics_collector.test_end_time = t1

        await self.save_results(results)

        succ = sum(1 for r in results if r["status"] == "success")
        fail = sum(1 for r in results if r["status"] == "failed")
        summary = (
            f"Dispatch round completed in {t1 - t0:.2f} seconds\n"
            f"Total processed: {len(results)}\n"
            f"Successfully processed: {succ}\n"
            f"Failed: {fail}\n"
            f"Workers used: {len(self.worker_ids)}\n"
            f"Results saved to file"
        )
        return summary

    async def save_results(self, results: List[Dict[str, Any]]) -> None:
        try:
            # If path is relative, resolve relative to the streaming_queue directory
            if not Path(self.result_file).is_absolute():
                current_file = Path(__file__).resolve()
                streaming_queue_dir = current_file.parent.parent  # From core dir back to streaming_queue
                p = streaming_queue_dir / self.result_file
            else:
                p = Path(self.result_file)
            p.parent.mkdir(parents=True, exist_ok=True)

            # Basic statistics
            times = [r["response_time"] for r in results if r.get("response_time")]
            avg_rt = (sum(times) / len(times)) if times else 0.0
            
            # Calculate detailed timing statistics
            request_times = [r.get("detailed_timing", {}).get("request_time") for r in results 
                           if r.get("detailed_timing", {}).get("request_time")]
            llm_times = [r.get("detailed_timing", {}).get("llm_time") for r in results 
                        if r.get("detailed_timing", {}).get("llm_time")]
            adapter_times = [r.get("detailed_timing", {}).get("adapter_time") for r in results 
                           if r.get("detailed_timing", {}).get("adapter_time")]
            
            timing_stats = {
                "average_request_time": (sum(request_times) / len(request_times)) if request_times else 0.0,
                "average_llm_time": (sum(llm_times) / len(llm_times)) if llm_times else 0.0,
                "average_adapter_time": (sum(adapter_times) / len(adapter_times)) if adapter_times else 0.0,
                "max_request_time": max(request_times) if request_times else 0.0,
                "max_llm_time": max(llm_times) if llm_times else 0.0,
                "max_adapter_time": max(adapter_times) if adapter_times else 0.0,
                "min_request_time": min(request_times) if request_times else 0.0,
                "min_llm_time": min(llm_times) if llm_times else 0.0,
                "min_adapter_time": min(adapter_times) if adapter_times else 0.0,
            }
            
            # Get comprehensive performance metrics
            performance_report = self.metrics_collector.get_comprehensive_report()
            
            payload = {
                "metadata": {
                    "total_questions": len(results),
                    "successful_questions": sum(1 for r in results if r["status"] == "success"),
                    "failed_questions": sum(1 for r in results if r["status"] == "failed"),
                    "average_response_time": avg_rt,
                    "timestamp": time.time(),
                    "network_type": self.__class__.__name__,
                },
                "timing_breakdown": timing_stats,
                "detailed_performance_metrics": performance_report,
                "results": results,
            }
            p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            self._o("success", f"Results saved to: {p}")
            
            # Print performance summary including timing breakdown
            self._print_performance_summary(performance_report, timing_stats)
            
        except Exception as e:
            self._o("error", f"Failed to save results: {e}")
    
    def _print_performance_summary(self, performance_report: Dict[str, Any], timing_stats: Dict[str, Any] = None) -> None:
        """Print a detailed performance summary"""
        self._o("info", "=== Detailed Performance Summary ===")
        
        summary = performance_report.get("summary", {})
        self._o("system", f"Test Duration: {summary.get('test_duration', 0):.2f} seconds")
        self._o("system", f"Total Tasks: {summary.get('total_tasks', 0)}")
        self._o("system", f"Success Rate: {summary.get('overall_success_rate', 0):.2%}")
        self._o("system", f"Total Retries: {summary.get('total_retries', 0)}")
        self._o("system", f"Network Errors: {summary.get('total_network_errors', 0)}")
        
        # Timing breakdown analysis
        if timing_stats:
            self._o("info", "=== Timing Breakdown Analysis (API Bottleneck Detection) ===")
            avg_req = timing_stats.get('average_request_time', 0)
            avg_llm = timing_stats.get('average_llm_time', 0)
            avg_adapter = timing_stats.get('average_adapter_time', 0)
            
            self._o("progress", f"Average Request Time: {avg_req:.3f}s")
            self._o("progress", f"  - LLM Time: {avg_llm:.3f}s ({avg_llm / avg_req * 100 if avg_req > 0 else 0:.1f}%)")
            self._o("progress", f"  - Adapter Overhead: {avg_adapter:.3f}s ({avg_adapter / avg_req * 100 if avg_req > 0 else 0:.1f}%)")
            self._o("progress", f"Max Times - Request: {timing_stats.get('max_request_time', 0):.3f}s, LLM: {timing_stats.get('max_llm_time', 0):.3f}s, Adapter: {timing_stats.get('max_adapter_time', 0):.3f}s")
            self._o("progress", f"Min Times - Request: {timing_stats.get('min_request_time', 0):.3f}s, LLM: {timing_stats.get('min_llm_time', 0):.3f}s, Adapter: {timing_stats.get('min_adapter_time', 0):.3f}s")
            
            # 检测API速率瓶颈的指标
            if avg_adapter > avg_llm * 0.5:  # 如果adapter时间超过LLM时间的50%
                self._o("warning", f"⚠️  High adapter overhead detected! Adapter time ({avg_adapter:.3f}s) is significant compared to LLM time ({avg_llm:.3f}s)")
                self._o("warning", "   This may indicate API rate limiting or network bottlenecks.")
        
        # Response time statistics
        if "global_average_response_time" in summary:
            self._o("info", "=== Response Time Analysis ===")
            self._o("progress", f"Average: {summary.get('global_average_response_time', 0):.2f}s")
            self._o("progress", f"Min: {summary.get('global_min_response_time', 0):.2f}s")
            self._o("progress", f"Max: {summary.get('global_max_response_time', 0):.2f}s")
            self._o("progress", f"Std Dev: {summary.get('global_response_time_std', 0):.2f}s")
            self._o("progress", f"Median: {summary.get('global_median_response_time', 0):.2f}s")
        
        # Protocol-level statistics
        protocols = performance_report.get("protocols", {})
        if protocols:
            self._o("info", "=== Protocol Performance ===")
            for protocol, stats in protocols.items():
                self._o("progress", f"{protocol}:")
                self._o("progress", f"  Completed: {stats.get('total_completed', 0)}")
                self._o("progress", f"  Success Rate: {stats.get('success_rate', 0):.2%}")
                self._o("progress", f"  Avg Response: {stats.get('average_response_time', 0):.2f}s")
                self._o("progress", f"  Load Balance Variance: {stats.get('load_balance_variance', 0):.2f}")
                self._o("progress", f"  Network Errors: {stats.get('total_network_errors', 0)}")
                self._o("progress", f"  Retries: {stats.get('total_retries', 0)}")
        
        # Worker-level statistics
        workers = performance_report.get("workers", {})
        if workers:
            self._o("info", "=== Worker Performance ===")
            for worker_id, stats in workers.items():
                self._o("progress", f"{worker_id} ({stats.get('protocol', 'unknown')}):")
                self._o("progress", f"  Completed: {stats.get('completed_tasks', 0)}")
                self._o("progress", f"  Failed: {stats.get('failed_tasks', 0)}")
                self._o("progress", f"  Timeout: {stats.get('timeout_tasks', 0)}")
                self._o("progress", f"  Network Error Rate: {stats.get('network_error_rate', 0):.2%}")
                self._o("progress", f"  Avg Response: {stats.get('average_response_time', 0):.2f}s")

    # --------------- Status query ---------------
    async def get_status(self) -> str:
        net = "Connected" if self.agent_network else "Not connected"
        return (
            "QA Coordinator Status:\n"
            f"Configuration: batch_size={self.batch_size}, first_50={self.first_50}\n"
            f"Data path: {self.data_path}\n"
            f"Network status: {net}\n"
            f"Worker count: {len(self.worker_ids)}\n"
            f"Available commands: dispatch, status"
        )
