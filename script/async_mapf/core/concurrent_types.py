"""
Enhanced data structures for concurrent MAPF coordination.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from enum import Enum
import uuid
import time

# Type alias for coordinates
Coord = Tuple[int, int]  # (x, y)

class MoveStatus(Enum):
    """Move command response status."""
    OK = "OK"
    REJECT = "REJECT" 
    CONFLICT = "CONFLICT"
    PENDING = "PENDING"

@dataclass
class ConcurrentMoveCmd:
    """Enhanced move command for concurrent execution"""
    agent_id: int
    new_pos: Coord                     # Target position (x, y)
    eta_ms: int = 100                  # Expected time to arrive (ms from now)
    time_window_ms: int = 50           # Tolerance window for conflicts  
    move_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0                  # Higher = more important
    max_wait_ms: int = 1000           # Max time to wait for resolution

@dataclass
class MoveResponse:
    """Immediate response to move command"""
    agent_id: int
    move_id: str
    status: MoveStatus
    reason: str = ""
    conflicting_agents: List[int] = field(default_factory=list)
    suggested_eta_ms: Optional[int] = None
    reservation_id: Optional[str] = None

@dataclass 
class ConflictNotification:
    """Notification of conflict to involved agents"""
    conflict_id: str
    cell: Coord
    time_window: Tuple[int, int]      # (start_ms, end_ms)
    conflicting_agents: List[int]
    conflict_priority: int = 0

@dataclass
class NegotiationMessage:
    """Agent-to-agent negotiation message"""
    from_agent: int
    to_agent: int
    conflict_id: str
    message_type: str                 # "YIELD", "PROPOSE_TIME", "ACCEPT", "REJECT"
    proposed_eta_ms: Optional[int] = None
    reason: str = ""

@dataclass
class ReservationRecord:
    """Time-space reservation record"""
    agent_id: int
    cell: Coord
    start_time_ms: int
    end_time_ms: int
    move_id: str
    priority: int = 0

@dataclass
class MoveFeedback:
    """Final execution result"""
    agent_id: int
    move_id: str
    success: bool
    actual_pos: Coord
    actual_time_ms: int
    reason: str = ""