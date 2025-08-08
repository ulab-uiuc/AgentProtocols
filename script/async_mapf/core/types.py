"""
Core data structures for MAPF network coordination.
"""

from dataclasses import dataclass, field
from typing import Tuple, List
from enum import Enum
import uuid
import time

# Type alias for coordinates
Coord = Tuple[int, int]  # (x, y)


class EventType(Enum):
    """Event types for the priority queue."""
    MOVE = "move"
    FEEDBACK = "feedback"

class MoveStatus(Enum):
    """Move command response status."""
    OK = "OK"
    REJECT = "REJECT" 
    CONFLICT = "CONFLICT"


@dataclass
class MoveCmd:
    """Agent -> Network move command with concurrent execution support."""
    agent_id: int
    action: str                    # 'U' 'D' 'L' 'R' 'S' (Up/Down/Left/Right/Stay)
    client_ts: int                 # generation time (ms)
    exec_ts: int | None = None     # planned time; None => ASAP
    eta_ms: int = 100              # Expected execution time (ms from now)
    time_window_ms: int = 50       # Tolerance window for conflicts
    move_id: str = ""              # Unique move identifier for tracking
    new_pos: tuple = None          # Target position (x, y) - derived from action


@dataclass
class StepRecord:
    """Historical record of a single movement step."""
    step_ts: int                   # exec_ts when action was applied
    agent_id: int
    from_pos: Coord
    to_pos: Coord
    latency_ms: int               # step_ts - client_ts  
    collision: bool               # whether collision occurred


@dataclass
class Event:
    """Event in the priority queue."""
    exec_ts: int                  # execution timestamp
    etype: EventType              # event type enum
    payload: any                  # event-specific data
    
    def __lt__(self, other):
        """For heapq comparison."""
        return self.exec_ts < other.exec_ts


@dataclass
class MoveFeedback:
    """Feedback sent back to agent after move execution."""
    agent_id: int
    success: bool                 # whether move was successful
    actual_pos: Coord            # actual position after move
    collision: bool              # whether collision occurred
    latency_ms: int              # actual latency
    step_ts: int 