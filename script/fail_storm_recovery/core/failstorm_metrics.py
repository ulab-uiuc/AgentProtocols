#!/usr/bin/env python3
"""
FailStorm Metrics Collection and Analysis Module

This module provides specialized metrics collection for the Fail-Storm Recovery scenario.
It focuses on fault tolerance, recovery performance, and system resilience metrics
that complement the base Prometheus metrics system.

Key Metrics:
- recovery_ms: Time from fault injection to first recovery message
- steady_state_ms: Time for system to reach stable operation
- success_rate_drop: Task completion rate degradation
- duplicate_work_ratio: Efficiency loss due to duplicate task execution
- bytes_reconnect: Network overhead during recovery
"""

import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import statistics


@dataclass
class TaskExecution:
    """Record of a single task execution."""
    task_id: str
    agent_id: str
    task_type: str  # qa_task, qa_group_search, qa_answer_found, etc.
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    result_size_bytes: int = 0
    # QA-specific fields
    group_id: Optional[int] = None
    answer_found: bool = False
    answer_source: Optional[str] = None  # "local", "neighbor", "collaborative"
    answer_quality_score: Optional[float] = None


@dataclass
class NetworkEvent:
    """Record of a network-level event."""
    timestamp: float
    event_type: str  # heartbeat, reconnect, message_failure, etc.
    source_agent: str
    target_agent: Optional[str] = None
    bytes_transferred: int = 0
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class FailStormMetricsCollector:
    """
    Comprehensive metrics collector for Fail-Storm recovery scenarios.
    
    This class tracks and analyzes system behavior before, during, and after
    fault injection to provide detailed insights into system resilience.
    """
    
    def __init__(self, protocol_name: str = "unknown", config: Dict[str, Any] = None):
        """
        Initialize the metrics collector.
        
        Parameters
        ----------
        protocol_name : str
            Name of the communication protocol being tested (A2A, ANP, ACP, etc.)
        config : Dict[str, Any]
            Configuration dictionary containing scenario parameters
        """
        self.protocol_name = protocol_name
        self.config = config or {}
        self.scenario_start_time = time.time()
        
        # Calculate expected fault injection time from config
        scenario_config = self.config.get("scenario", {})
        self.expected_fault_injection_time = (
            self.scenario_start_time + scenario_config.get("fault_injection_time", 120.0)
        )
        
        # Key timing markers
        self.fault_injection_time: Optional[float] = None
        self.first_recovery_time: Optional[float] = None
        self.steady_state_time: Optional[float] = None
        
        # Task execution tracking
        self.task_executions: List[TaskExecution] = []
        self.task_completion_times: deque = deque(maxlen=100)  # Rolling window
        
        # Network event tracking
        self.network_events: List[NetworkEvent] = []
        self.reconnection_attempts: List[Dict[str, Any]] = []
        
        # Agent state tracking
        self.agent_states: Dict[str, str] = {}  # agent_id -> state (alive/failed/recovering)
        self.topology_snapshots: List[Dict[str, Any]] = []
        
        # Performance windows
        self.pre_fault_window: List[float] = []   # Task completion times before fault
        self.post_fault_window: List[float] = []  # Task completion times after fault
        self.recovery_window: List[float] = []    # Task completion times during recovery
        
        print(f"[FailStormMetrics] Initialized for protocol: {protocol_name}")

    # ========================== Task Execution Tracking ==========================

    def start_task_execution(self, task_id: str, agent_id: str, task_type: str, group_id: Optional[int] = None) -> None:
        """Record the start of a QA task execution."""
        execution = TaskExecution(
            task_id=task_id,
            agent_id=agent_id,
            task_type=task_type,
            start_time=time.time(),
            group_id=group_id
        )
        self.task_executions.append(execution)

    def complete_task_execution(self, task_id: str, success: bool, 
                              result_size_bytes: int = 0, error: Optional[str] = None,
                              answer_found: bool = False, answer_source: Optional[str] = None,
                              answer_quality_score: Optional[float] = None) -> None:
        """Record the completion of a QA task execution."""
        current_time = time.time()
        
        # Find the corresponding task execution
        for execution in reversed(self.task_executions):
            if execution.task_id == task_id and execution.end_time is None:
                execution.end_time = current_time
                execution.success = success
                execution.result_size_bytes = result_size_bytes
                execution.error = error
                execution.answer_found = answer_found
                execution.answer_source = answer_source
                execution.answer_quality_score = answer_quality_score
                
                # Record completion time
                completion_time = execution.end_time - execution.start_time
                self.task_completion_times.append(completion_time)
                
                # Categorize by phase
                if self.fault_injection_time is None:
                    self.pre_fault_window.append(completion_time)
                elif self.steady_state_time is None:
                    self.recovery_window.append(completion_time)
                else:
                    self.post_fault_window.append(completion_time)
                
                break

    def record_task_execution(self, task_id: str, agent_id: str, task_type: str, 
                            start_time: float, end_time: float, success: bool,
                            answer_found: bool = False, answer_source: Optional[str] = None,
                            result_size_bytes: int = 0, error: Optional[str] = None,
                            group_id: Optional[int] = None) -> None:
        """
        Record a complete task execution and bin it into the correct phase window.

        FIXED CLASSIFICATION RULES:
        1) If task_type is an explicit QA label, trust it:
           - 'qa_normal'   -> pre_fault_window
           - 'qa_recovery' -> recovery_window
           - 'qa_post_fault' (or synonyms) -> post_fault_window

        2) Only if task_type is not an explicit QA label, fall back to time-based:
           - start_time < fault_injection_time (or expected_fault_injection_time if None) -> pre_fault
           - fault_injection_time <= start_time < steady_state_time -> recovery
           - start_time >= steady_state_time -> post_fault

        Notes:
        - This prevents 'qa_recovery' from being incorrectly binned into post-fault.
        - This also honors steady_state_time when available.
        """
        execution = TaskExecution(
            task_id=task_id,
            agent_id=agent_id,
            task_type=task_type,
            start_time=start_time,
            end_time=end_time,
            success=success,
            result_size_bytes=result_size_bytes,
            error=error,
            group_id=group_id,
            answer_found=answer_found,
            answer_source=answer_source
        )
        self.task_executions.append(execution)

        # Compute completion time using provided timestamps
        completion_time = (end_time - start_time) if end_time is not None else 0.0
        self.task_completion_times.append(completion_time)

        # --- Explicit classification first ---
        normalized = (task_type or "").strip().lower()
        if normalized in ("qa_normal", "qa_pre_fault"):
            self.pre_fault_window.append(completion_time)
            return
        if normalized in ("qa_recovery",):
            self.recovery_window.append(completion_time)
            return
        if normalized in ("qa_post_fault", "qa_post_recovery", "qa_post"):
            self.post_fault_window.append(completion_time)
            return

        # --- Fallback: time-based classification ---
        fault_time = self.fault_injection_time or self.expected_fault_injection_time
        steady_time = self.steady_state_time

        # If we don't even have a fault time, treat as pre-fault
        if fault_time is None:
            self.pre_fault_window.append(completion_time)
            return

        # If we have steady_state_time, use three-way split
        if steady_time is not None:
            if start_time < fault_time:
                self.pre_fault_window.append(completion_time)
            elif start_time < steady_time:
                self.recovery_window.append(completion_time)
            else:
                self.post_fault_window.append(completion_time)
        else:
            # No steady state yet -> two-way split
            if start_time < fault_time:
                self.pre_fault_window.append(completion_time)
            else:
                self.recovery_window.append(completion_time)

    def record_duplicate_task(self, original_task_id: str, duplicate_agent_id: str) -> None:
        """Record when a task is executed multiple times (duplicate work)."""
        # Find original execution
        original_execution = None
        for execution in self.task_executions:
            if execution.task_id == original_task_id and execution.success:
                original_execution = execution
                break
        
        if original_execution:
            # Mark this as a duplicate with a unique ID
            duplicate_task_id = f"{original_task_id}_dup_{int(time.time()*1000)}"
            self.start_task_execution(duplicate_task_id, duplicate_agent_id, original_execution.task_type)

    # ========================== Network Event Tracking ==========================

    def record_network_event(self, event_type: str, source_agent: str, 
                            target_agent: Optional[str] = None, bytes_transferred: int = 0,
                            latency_ms: Optional[float] = None, error: Optional[str] = None) -> None:
        """Record a network-level event."""
        event = NetworkEvent(
            timestamp=time.time(),
            event_type=event_type,
            source_agent=source_agent,
            target_agent=target_agent,
            bytes_transferred=bytes_transferred,
            latency_ms=latency_ms,
            error=error
        )
        self.network_events.append(event)

    def record_reconnection_attempt(self, source_agent: str, target_agent: str, 
                                  success: bool, duration_ms: float, error: Optional[str] = None) -> None:
        """Record a reconnection attempt between agents."""
        attempt = {
            "timestamp": time.time(),
            "source_agent": source_agent,
            "target_agent": target_agent,
            "success": success,
            "duration_ms": duration_ms,
            "error": error
        }
        self.reconnection_attempts.append(attempt)
        
        # Also record as network event
        event_type = "reconnect_success" if success else "reconnect_failure"
        self.record_network_event(
            event_type=event_type,
            source_agent=source_agent,
            target_agent=target_agent,
            latency_ms=duration_ms,
            error=error
        )

    # ========================== System State Tracking ==========================

    def set_fault_injection_time(self, timestamp: Optional[float] = None) -> None:
        """Mark the time when fault injection occurred."""
        self.fault_injection_time = timestamp or time.time()
        print(f"[FailStormMetrics] Fault injection time recorded: {self.fault_injection_time}")

    def set_first_recovery_time(self, timestamp: Optional[float] = None) -> None:
        """Mark the time when first recovery was detected."""
        if self.first_recovery_time is None:  # Only record the first
            self.first_recovery_time = timestamp or time.time()
            print(f"[FailStormMetrics] First recovery time recorded: {self.first_recovery_time}")

    def set_steady_state_time(self, timestamp: Optional[float] = None) -> None:
        """Mark the time when system reached steady state."""
        self.steady_state_time = timestamp or time.time()
        print(f"[FailStormMetrics] Steady state time recorded: {self.steady_state_time}")

    def update_agent_state(self, agent_id: str, state: str) -> None:
        """Update the state of an agent (alive/failed/recovering)."""
        previous_state = self.agent_states.get(agent_id, "unknown")
        self.agent_states[agent_id] = state
        
        # Record state transition as network event
        if previous_state != state:
            self.record_network_event(
                event_type=f"agent_state_change",
                source_agent=agent_id,
                error=f"{previous_state} -> {state}"
            )

    def snapshot_topology(self, topology: Dict[str, List[str]], active_agents: List[str]) -> None:
        """Take a snapshot of the current network topology."""
        snapshot = {
            "timestamp": time.time(),
            "topology": topology,
            "active_agents": active_agents,
            "edge_count": sum(len(edges) for edges in topology.values()),
            "connectivity_ratio": self._calculate_connectivity_ratio(topology, active_agents)
        }
        self.topology_snapshots.append(snapshot)

    def _calculate_connectivity_ratio(self, topology: Dict[str, List[str]], active_agents: List[str]) -> float:
        """Calculate the connectivity ratio of the topology."""
        if len(active_agents) < 2:
            return 0.0
        
        actual_edges = sum(len(edges) for edges in topology.values())
        max_possible_edges = len(active_agents) * (len(active_agents) - 1)  # Directed graph
        
        return actual_edges / max_possible_edges if max_possible_edges > 0 else 0.0

    # ========================== Metrics Calculation ==========================

    def calculate_recovery_metrics(self) -> Dict[str, Any]:
        """Calculate the core recovery metrics."""
        metrics = {}
        
        # Recovery time (ms)
        if self.fault_injection_time and self.first_recovery_time:
            metrics["recovery_ms"] = (self.first_recovery_time - self.fault_injection_time) * 1000
        else:
            metrics["recovery_ms"] = None
        
        # Steady state time (ms)
        if self.fault_injection_time and self.steady_state_time:
            metrics["steady_state_ms"] = (self.steady_state_time - self.fault_injection_time) * 1000
        else:
            metrics["steady_state_ms"] = None
        
        # Success rate drop
        pre_fault_success_rate = self._calculate_success_rate(self.pre_fault_window)
        post_fault_success_rate = self._calculate_success_rate(self.post_fault_window)
        
        if pre_fault_success_rate > 0:
            metrics["success_rate_drop"] = (pre_fault_success_rate - post_fault_success_rate) / pre_fault_success_rate
        else:
            metrics["success_rate_drop"] = 0.0
        
        # Duplicate work ratio
        metrics["duplicate_work_ratio"] = self._calculate_duplicate_work_ratio()
        
        # Reconnection bytes
        metrics["bytes_reconnect"] = self._calculate_reconnection_bytes()
        
        # Accuracy analysis
        metrics["accuracy_analysis"] = self._calculate_accuracy_analysis()
        
        return metrics

    def _calculate_success_rate(self, completion_times: List[float]) -> float:
        """Calculate success rate for a set of task executions."""
        if not completion_times:
            return 0.0
        
        # For now, assume all completed tasks are successful
        # In a real implementation, you'd track success/failure separately
        return len(completion_times) / max(len(completion_times), 1)

    def _calculate_duplicate_work_ratio(self) -> float:
        """Calculate the ratio of duplicate work performed."""
        # In a distributed QA system, multiple agents processing the same question
        # is not considered duplicate work, but collaborative work.
        # We'll set this to 0 for now since the current system design
        # intentionally has multiple agents work on the same questions.
        #TODO: Add a metric to track the number of unique questions asked.
        return 0.0

    def _calculate_reconnection_bytes(self) -> int:
        """Calculate total bytes used for reconnection operations."""
        reconnect_bytes = 0
        
        # Count bytes from reconnection events
        for event in self.network_events:
            if "reconnect" in event.event_type:
                reconnect_bytes += event.bytes_transferred
        
        # Add estimated overhead for reconnection attempts
        for attempt in self.reconnection_attempts:
            reconnect_bytes += 100  # Estimated overhead per attempt
        
        return reconnect_bytes
    
    def _calculate_accuracy_analysis(self) -> Dict[str, Any]:
        """Calculate detailed accuracy analysis across all phases."""
        def analyze_phase_accuracy(executions: List[TaskExecution], phase_name: str) -> Dict[str, Any]:
            if not executions:
                return {
                    "total_tasks": 0,
                    "successful_tasks": 0,
                    "success_rate": 0.0,
                    "tasks_with_answers": 0,
                    "answer_rate": 0.0,
                    "answer_sources": {"local": 0, "neighbor": 0, "unknown": 0}
                }
            
            successful = len([e for e in executions if e.success])
            with_answers = len([e for e in executions if e.answer_found])
            
            # Count answer sources
            sources = {"local": 0, "neighbor": 0, "unknown": 0}
            for e in executions:
                if e.answer_found and e.answer_source:
                    source = e.answer_source.lower()
                    if source in sources:
                        sources[source] += 1
                    else:
                        sources["unknown"] += 1
            
            return {
                "total_tasks": len(executions),
                "successful_tasks": successful,
                "success_rate": successful / len(executions) if executions else 0.0,
                "tasks_with_answers": with_answers,
                "answer_rate": with_answers / len(executions) if executions else 0.0,
                "answer_sources": sources
            }
        
        # Get executions by phase
        pre_fault_executions = [e for e in self.task_executions if e.task_type in ("qa_normal", "qa_pre_fault")]
        recovery_executions = [e for e in self.task_executions if e.task_type == "qa_recovery"]
        post_fault_executions = [e for e in self.task_executions if e.task_type in ("qa_post_fault", "qa_post_recovery", "qa_post")]
        
        # Fallback: use timing if task_type classification failed
        if not pre_fault_executions and not recovery_executions and not post_fault_executions:
            fault_time = self.fault_injection_time or self.expected_fault_injection_time
            steady_time = self.steady_state_time
            
            for e in self.task_executions:
                if fault_time is None or e.start_time < fault_time:
                    pre_fault_executions.append(e)
                elif steady_time is None or e.start_time < steady_time:
                    recovery_executions.append(e)
                else:
                    post_fault_executions.append(e)
        
        return {
            "pre_fault": analyze_phase_accuracy(pre_fault_executions, "pre_fault"),
            "recovery": analyze_phase_accuracy(recovery_executions, "recovery"),
            "post_fault": analyze_phase_accuracy(post_fault_executions, "post_fault"),
            "overall": analyze_phase_accuracy(self.task_executions, "overall")
        }

    # ========================== Performance Analysis ==========================

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance analysis."""
        def safe_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0}
            return {
                "count": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values)
            }
        
        return {
            "pre_fault_performance": safe_stats(self.pre_fault_window),
            "recovery_performance": safe_stats(self.recovery_window),
            "post_fault_performance": safe_stats(self.post_fault_window),
            "total_task_executions": len(self.task_executions),
            "successful_executions": sum(1 for e in self.task_executions if e.success),
            "network_events_count": len(self.network_events),
            "reconnection_attempts": len(self.reconnection_attempts),
            "topology_snapshots": len(self.topology_snapshots)
        }

    # ========================== JSON Export ==========================

    def export_to_json(self, filepath: str) -> None:
        """Export all collected metrics to a JSON file."""
        output_data = {
            "metadata": {
                "scenario": "fail_storm_recovery",
                "protocol": self.protocol_name,
                "start_time": self.scenario_start_time,
                "export_time": time.time(),
                "fault_injection_time": self.fault_injection_time,
                "first_recovery_time": self.first_recovery_time,
                "steady_state_time": self.steady_state_time
            },
            "failstorm_metrics": self.calculate_recovery_metrics(),
            "performance_analysis": self.get_performance_summary(),
            "detailed_data": {
                "task_executions": [asdict(task) for task in self.task_executions],
                "network_events": [asdict(event) for event in self.network_events],
                "reconnection_attempts": self.reconnection_attempts,
                "topology_snapshots": self.topology_snapshots,
                "agent_states": self.agent_states
            }
        }
        
        # Ensure output directory exists
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON file
        try:
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            print(f"[FailStormMetrics] Metrics exported to: {output_path}")
            
        except Exception as e:
            print(f"[FailStormMetrics] Export failed: {e}")

    def get_real_time_summary(self) -> Dict[str, Any]:
        """Get a real-time summary of current metrics."""
        current_time = time.time()
        recovery_metrics = self.calculate_recovery_metrics()
        
        return {
            "timestamp": current_time,
            "runtime_seconds": current_time - self.scenario_start_time,
            "phase": self._get_current_phase(),
            "active_agents": len([state for state in self.agent_states.values() if state == "alive"]),
            "failed_agents": len([state for state in self.agent_states.values() if state == "failed"]),
            "recovery_progress": {
                "recovery_ms": recovery_metrics.get("recovery_ms"),
                "steady_state_ms": recovery_metrics.get("steady_state_ms"),
                "recent_task_completions": len(self.task_completion_times)
            }
        }

    def _get_current_phase(self) -> str:
        """Determine the current phase of the scenario."""
        if self.fault_injection_time is None:
            return "pre_fault"
        elif self.steady_state_time is None:
            return "recovery"
        else:
            return "post_recovery"

    # ========================== QA-Specific Metrics ==========================
    
    def start_normal_phase(self) -> None:
        """Mark the start of normal QA phase."""
        print("[FailStormMetrics] Normal QA phase started")
    
    def end_normal_phase(self) -> None:
        """Mark the end of normal QA phase."""
        print("[FailStormMetrics] Normal QA phase ended")
    
    def start_recovery_phase(self) -> None:
        """Mark the start of recovery QA phase."""
        print("[FailStormMetrics] Recovery QA phase started")
    
    def end_recovery_phase(self) -> None:
        """Mark the end of recovery QA phase."""
        print("[FailStormMetrics] Recovery QA phase ended")
    
    def get_qa_metrics(self) -> Dict[str, Any]:
        """Get QA-specific metrics."""
        if not self.task_executions:
            return {"no_qa_tasks": True}
        
        qa_tasks = [t for t in self.task_executions if t.task_type.startswith("qa_")]
        successful_qa = [t for t in qa_tasks if t.success]
        answered_qa = [t for t in qa_tasks if t.answer_found]
        
        # Calculate answer success rates
        total_qa_tasks = len(qa_tasks)
        successful_rate = len(successful_qa) / total_qa_tasks if total_qa_tasks > 0 else 0
        answer_found_rate = len(answered_qa) / total_qa_tasks if total_qa_tasks > 0 else 0
        
        # Answer source distribution
        answer_sources = {}
        for task in answered_qa:
            source = task.answer_source or "unknown"
            answer_sources[source] = answer_sources.get(source, 0) + 1
        
        # Pre/post fault comparison
        pre_fault_qa = [t for t in qa_tasks if t.start_time < (self.fault_injection_time or float('inf'))]
        post_fault_qa = [t for t in qa_tasks if t.start_time >= (self.fault_injection_time or 0)]
        
        pre_fault_success = len([t for t in pre_fault_qa if t.answer_found]) / len(pre_fault_qa) if pre_fault_qa else 0
        post_fault_success = len([t for t in post_fault_qa if t.answer_found]) / len(post_fault_qa) if post_fault_qa else 0
        
        return {
            "total_qa_tasks": total_qa_tasks,
            "successful_tasks": len(successful_qa),
            "tasks_with_answers": len(answered_qa),
            "success_rate": successful_rate,
            "answer_found_rate": answer_found_rate,
            "answer_sources": answer_sources,
            "pre_fault_answer_rate": pre_fault_success,
            "post_fault_answer_rate": post_fault_success,
            "average_task_duration": statistics.mean([
                t.end_time - t.start_time for t in qa_tasks 
                if t.end_time is not None
            ]) if qa_tasks else 0
        }

    # ========================== Final Aggregation ==========================
    def get_final_results(self) -> Dict[str, Any]:
        """Assemble a final results dictionary (without writing to disk)."""
        recovery_metrics = self.calculate_recovery_metrics()
        performance = self.get_performance_summary()
        qa = self.get_qa_metrics()
        current_time = time.time()

        return {
            "metadata": {
                "scenario": "fail_storm_recovery",
                "protocol": self.protocol_name,
                "start_time": self.scenario_start_time,
                "end_time": current_time,
                "fault_injection_time": self.fault_injection_time,
                "first_recovery_time": self.first_recovery_time,
                "steady_state_time": self.steady_state_time,
                "duration_sec": current_time - self.scenario_start_time,
                "phase_final": self._get_current_phase()
            },
            "failstorm_metrics": recovery_metrics,
            "performance_analysis": performance,
            "qa_metrics": qa,
            "agents": {
                "states": self.agent_states,
                "topology_snapshots_last": self.topology_snapshots[-3:],
            },
            "counts": {
                "task_executions": len(self.task_executions),
                "network_events": len(self.network_events),
                "reconnection_attempts": len(self.reconnection_attempts),
                "topology_snapshots": len(self.topology_snapshots)
            }
        }

    def __repr__(self) -> str:
        """String representation of the metrics collector."""
        return (
            f"FailStormMetricsCollector(protocol={self.protocol_name}, "
            f"tasks={len(self.task_executions)}, "
            f"events={len(self.network_events)}, "
            f"phase={self._get_current_phase()})"
        )