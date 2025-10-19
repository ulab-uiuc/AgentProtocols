"""
Result Manager for Fail Storm Recovery Scenario

Manages answer collection, validation, and result transmission back to external systems.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from .answer_validator import FailStormAnswerValidator, AnswerValidationResult, AnswerQuality


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ANSWER_FOUND = "answer_found"
    NO_ANSWER = "no_answer"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class TaskResult:
    """Complete task result with validation."""
    task_id: str
    group_id: int
    question: str
    status: TaskStatus
    answer: Optional[str]
    validation_result: Optional[AnswerValidationResult]
    worker_responses: List[Dict[str, Any]]
    execution_time: float
    first_answer_time: Optional[float]
    completion_time: float
    protocol_used: str
    error_message: Optional[str] = None


class FailStormResultManager:
    """
    Enhanced result manager for fail storm recovery scenario.
    
    Provides:
    - Answer validation and quality assessment
    - Result aggregation and transmission
    - Callback mechanisms for external systems
    - Performance metrics collection
    """
    
    def __init__(self, validator: FailStormAnswerValidator = None):
        """
        Initialize result manager.
        
        Args:
            validator: Answer validator instance
        """
        self.validator = validator or FailStormAnswerValidator()
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.result_callbacks: List[Callable] = []
        self.metrics: Dict[str, Any] = {}
        
        print("📊 Fail Storm Result Manager initialized")
    
    def register_result_callback(self, callback: Callable[[TaskResult], None]):
        """Register a callback to be called when tasks complete."""
        self.result_callbacks.append(callback)
        print(f"📞 Registered result callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
    
    def start_task(self, task_id: str, group_id: int, question: str, 
                  protocol: str = "unknown") -> None:
        """Start tracking a new task."""
        self.active_tasks[task_id] = {
            "task_id": task_id,
            "group_id": group_id,
            "question": question,
            "protocol": protocol,
            "status": TaskStatus.PENDING,
            "start_time": time.time(),
            "worker_responses": [],
            "answer_candidates": [],
            "first_answer_time": None,
            "completion_time": None,
            "error_message": None
        }
        
        print(f"📝 Started tracking task {task_id} (Group {group_id})")
    
    async def process_worker_response(self, task_id: str, worker_id: str, 
                                    content: str, meta: Dict[str, Any] = None) -> bool:
        """
        Process a worker response and determine if task is complete.
        
        Args:
            task_id: Task identifier
            worker_id: Worker that sent the response
            content: Response content
            meta: Additional metadata
            
        Returns:
            True if task is complete, False otherwise
        """
        if task_id not in self.active_tasks:
            print(f"⚠️ Received response for unknown task: {task_id}")
            return False
        
        task_info = self.active_tasks[task_id]
        current_time = time.time()
        
        # Record worker response
        response_record = {
            "worker_id": worker_id,
            "content": content,
            "meta": meta or {},
            "timestamp": current_time
        }
        task_info["worker_responses"].append(response_record)
        
        # Update task status
        if task_info["status"] == TaskStatus.PENDING:
            task_info["status"] = TaskStatus.IN_PROGRESS
        
        # Check for answer found
        if "ANSWER_FOUND:" in content:
            answer_text = content.split("ANSWER_FOUND:")[1].strip()
            if "(" in answer_text:  # Remove hop count if present
                answer_text = answer_text.split("(")[0].strip()
            
            # Record first answer time
            if not task_info["first_answer_time"]:
                task_info["first_answer_time"] = current_time
            
            # Add to answer candidates
            task_info["answer_candidates"].append({
                "answer": answer_text,
                "worker_id": worker_id,
                "timestamp": current_time,
                "source_context": meta.get("source_context", "")
            })
            
            # Validate answer
            try:
                validation_result = await self.validator.validate_answer(
                    question=task_info["question"],
                    answer=answer_text,
                    source_context=meta.get("source_context", ""),
                    worker_id=worker_id
                )
                
                # If answer is valid and good quality, complete the task
                if (validation_result.is_valid and 
                    validation_result.quality in [AnswerQuality.EXCELLENT, AnswerQuality.GOOD]):
                    
                    await self._complete_task(task_id, TaskStatus.ANSWER_FOUND, 
                                            answer_text, validation_result)
                    return True
                
                elif validation_result.quality == AnswerQuality.FAIR:
                    # Fair quality - wait a bit for potentially better answers
                    print(f"📋 Task {task_id}: Fair quality answer received, waiting for better answers...")
                    # Don't complete yet, but record as potential answer
                    
                else:
                    print(f"⚠️ Task {task_id}: Poor quality answer rejected")
                
            except Exception as e:
                print(f"❌ Task {task_id}: Answer validation failed: {e}")
        
        # Check for no answer
        elif "NO_ANSWER" in content or "TTL_EXHAUSTED" in content:
            task_info["status"] = TaskStatus.NO_ANSWER
            
            # If we have no good answers and this is a definitive "no answer", complete
            if not task_info["answer_candidates"]:
                await self._complete_task(task_id, TaskStatus.NO_ANSWER, None, None)
                return True
        
        # Check for errors
        elif "SEARCH_ERROR" in content or "ERROR:" in content:
            error_msg = content.split("ERROR:")[-1].strip() if "ERROR:" in content else content
            task_info["error_message"] = error_msg
            task_info["status"] = TaskStatus.ERROR
        
        return False
    
    async def _complete_task(self, task_id: str, status: TaskStatus, 
                           final_answer: Optional[str], 
                           validation_result: Optional[AnswerValidationResult]) -> None:
        """Complete a task and trigger callbacks."""
        if task_id not in self.active_tasks:
            return
        
        task_info = self.active_tasks[task_id]
        completion_time = time.time()
        
        # Create final task result
        task_result = TaskResult(
            task_id=task_id,
            group_id=task_info["group_id"],
            question=task_info["question"],
            status=status,
            answer=final_answer,
            validation_result=validation_result,
            worker_responses=task_info["worker_responses"],
            execution_time=completion_time - task_info["start_time"],
            first_answer_time=task_info["first_answer_time"],
            completion_time=completion_time,
            protocol_used=task_info["protocol"],
            error_message=task_info.get("error_message")
        )
        
        # Move to completed tasks
        self.completed_tasks[task_id] = task_result
        del self.active_tasks[task_id]
        
        # Update metrics
        self._update_metrics(task_result)
        
        # Trigger callbacks
        for callback in self.result_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task_result)
                else:
                    callback(task_result)
            except Exception as e:
                print(f"❌ Result callback failed: {e}")
        
        print(f"✅ Task {task_id} completed with status: {status.value}")
    
    def _update_metrics(self, task_result: TaskResult):
        """Update performance metrics."""
        protocol = task_result.protocol_used
        
        if protocol not in self.metrics:
            self.metrics[protocol] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "answer_found_tasks": 0,
                "total_execution_time": 0.0,
                "total_answer_time": 0.0,
                "quality_distribution": {}
            }
        
        metrics = self.metrics[protocol]
        metrics["total_tasks"] += 1
        metrics["total_execution_time"] += task_result.execution_time
        
        if task_result.status == TaskStatus.ANSWER_FOUND:
            metrics["successful_tasks"] += 1
            metrics["answer_found_tasks"] += 1
            
            if task_result.first_answer_time:
                answer_time = task_result.first_answer_time - (task_result.completion_time - task_result.execution_time)
                metrics["total_answer_time"] += answer_time
            
            if task_result.validation_result:
                quality = task_result.validation_result.quality.value
                metrics["quality_distribution"][quality] = metrics["quality_distribution"].get(quality, 0) + 1
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a task."""
        if task_id in self.active_tasks:
            task_info = self.active_tasks[task_id]
            return {
                "task_id": task_id,
                "status": task_info["status"].value if isinstance(task_info["status"], TaskStatus) else task_info["status"],
                "elapsed_time": time.time() - task_info["start_time"],
                "worker_responses": len(task_info["worker_responses"]),
                "answer_candidates": len(task_info["answer_candidates"])
            }
        elif task_id in self.completed_tasks:
            result = self.completed_tasks[task_id]
            return asdict(result)
        
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "protocol_metrics": self.metrics,
            "validation_stats": self.validator.get_validation_statistics(),
            "callback_count": len(self.result_callbacks)
        }


# Global instance
result_manager = FailStormResultManager()


# Convenience functions
def register_task_callback(callback: Callable[[TaskResult], None]):
    """Register a callback for task completion."""
    result_manager.register_result_callback(callback)


async def process_shard_response(task_id: str, worker_id: str, content: str, 
                               meta: Dict[str, Any] = None) -> bool:
    """Process a shard worker response."""
    return await result_manager.process_worker_response(task_id, worker_id, content, meta)


def get_task_metrics() -> Dict[str, Any]:
    """Get task performance metrics."""
    return result_manager.get_performance_metrics()
