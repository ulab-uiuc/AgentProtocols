"""
Utility functions and data structures for MAPF algorithms.

Contains helper classes and functions used across the MAPF framework.
"""

from __future__ import annotations
import time
import asyncio
from typing import Dict, Any, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum


class PathfindingResult(NamedTuple):
    """Result of pathfinding operation."""
    path: List[Tuple[int, int]]
    cost: int
    found: bool
    computation_time: float


class MessageType(Enum):
    """Standard message types for agent communication."""
    PATH_CONFLICT = "path_conflict"
    MOVE_REQUEST = "move_request"
    MOVE_APPROVED = "move_approved"
    MOVE_DENIED = "move_denied"
    COORDINATION_REQUEST = "coordination_request"
    PRIORITY_UPDATE = "priority_update"
    GOAL_REACHED = "goal_reached"
    STATUS_UPDATE = "status_update"
    TIME_SYNC = "time_sync"
    AGENT_COMPLETED = "agent_completed"


@dataclass
class AgentState:
    """Represents the current state of an agent."""
    agent_id: int
    position: Tuple[int, int]
    goal: Tuple[int, int]
    path: List[Tuple[int, int]]
    path_index: int
    is_active: bool
    last_update: float


@dataclass
class ConflictInfo:
    """Information about a detected conflict."""
    timestamp: float
    agent_ids: List[int]
    conflict_type: str
    position: Tuple[int, int]
    resolution_strategy: Optional[str] = None


class Timer:
    """Simple timer utility for performance measurement."""
    
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self) -> float:
        """Get elapsed time since start."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class MovingAverage:
    """Calculate moving average for performance metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values: List[float] = []
        self.sum = 0.0
    
    def add(self, value: float) -> None:
        """Add a new value to the moving average."""
        self.values.append(value)
        self.sum += value
        
        if len(self.values) > self.window_size:
            removed = self.values.pop(0)
            self.sum -= removed
    
    def average(self) -> float:
        """Get the current moving average."""
        if not self.values:
            return 0.0
        return self.sum / len(self.values)
    
    def count(self) -> int:
        """Get the number of values in the window."""
        return len(self.values)


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two positions."""
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def get_neighbors_4connected(x: int, y: int) -> List[Tuple[int, int]]:
    """Get 4-connected neighbors of a position."""
    return [(x, y+1), (x+1, y), (x, y-1), (x-1, y)]


def get_neighbors_8connected(x: int, y: int) -> List[Tuple[int, int]]:
    """Get 8-connected neighbors of a position."""
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            neighbors.append((x + dx, y + dy))
    return neighbors


def create_message(msg_type: MessageType, sender_id: int, **kwargs) -> Dict[str, Any]:
    """Create a standardized message."""
    return {
        "type": msg_type.value,
        "sender_id": sender_id,
        "timestamp": time.time(),
        **kwargs
    }


def validate_position(pos: Tuple[int, int], grid_size: int) -> bool:
    """Validate if a position is within grid bounds."""
    x, y = pos
    return 0 <= x < grid_size and 0 <= y < grid_size


def path_length(path: List[Tuple[int, int]]) -> int:
    """Calculate the length of a path."""
    return len(path) - 1 if len(path) > 1 else 0


def path_cost(path: List[Tuple[int, int]]) -> float:
    """Calculate the total cost of a path."""
    if len(path) < 2:
        return 0.0
    
    total_cost = 0.0
    for i in range(1, len(path)):
        total_cost += euclidean_distance(path[i-1], path[i])
    
    return total_cost


async def timeout_wrapper(coro, timeout_seconds: float, default_value=None):
    """Wrap a coroutine with a timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return default_value


def format_position(pos: Tuple[int, int]) -> str:
    """Format position as string."""
    return f"({pos[0]}, {pos[1]})"


def format_path(path: List[Tuple[int, int]]) -> str:
    """Format path as string."""
    if not path:
        return "[]"
    
    formatted = " -> ".join(format_position(pos) for pos in path)
    return f"[{formatted}]"


class PerformanceMonitor:
    """Monitor performance metrics for MAPF scenarios."""
    
    def __init__(self):
        self.metrics = {
            "message_count": 0,
            "conflict_count": 0,
            "resolution_count": 0,
            "path_computations": 0,
            "successful_moves": 0,
            "failed_moves": 0
        }
        
        self.timing_metrics = {
            "avg_path_computation_time": MovingAverage(),
            "avg_message_processing_time": MovingAverage(),
            "avg_conflict_resolution_time": MovingAverage()
        }
        
        self.start_time = time.time()
    
    def increment(self, metric_name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        if metric_name in self.metrics:
            self.metrics[metric_name] += value
    
    def add_timing(self, metric_name: str, duration: float) -> None:
        """Add a timing measurement."""
        if metric_name in self.timing_metrics:
            self.timing_metrics[metric_name].add(duration)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        elapsed_time = time.time() - self.start_time
        
        summary = {
            "elapsed_time": elapsed_time,
            **self.metrics
        }
        
        # Add timing averages
        for name, moving_avg in self.timing_metrics.items():
            summary[name] = moving_avg.average()
        
        # Calculate rates
        if elapsed_time > 0:
            summary["messages_per_second"] = self.metrics["message_count"] / elapsed_time
            summary["conflicts_per_second"] = self.metrics["conflict_count"] / elapsed_time
        
        return summary
    
    def reset(self) -> None:
        """Reset all metrics."""
        for key in self.metrics:
            self.metrics[key] = 0
        
        for moving_avg in self.timing_metrics.values():
            moving_avg.values.clear()
            moving_avg.sum = 0.0
        
        self.start_time = time.time()


def log_message(level: str, message: str, agent_id: Optional[int] = None) -> None:
    """Simple logging function."""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    agent_prefix = f"[Agent-{agent_id}] " if agent_id is not None else ""
    print(f"[{timestamp}] {level.upper()}: {agent_prefix}{message}")


def debug_log(message: str, agent_id: Optional[int] = None) -> None:
    """Debug level logging."""
    log_message("debug", message, agent_id)


def info_log(message: str, agent_id: Optional[int] = None) -> None:
    """Info level logging."""
    log_message("info", message, agent_id)


def error_log(message: str, agent_id: Optional[int] = None) -> None:
    """Error level logging."""
    log_message("error", message, agent_id) 