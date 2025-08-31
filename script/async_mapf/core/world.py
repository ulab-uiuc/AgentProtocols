"""
GridWorld environment for Multi-Agent Path Finding.

Handles:
- Grid-based world representation
- Collision detection
- Obstacle management
- Goal assignment
"""

from __future__ import annotations
import time
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from .utils import validate_position, manhattan_distance


@dataclass
class Position:
    """Represents a position in the grid."""
    x: int
    y: int
    
    def __iter__(self):
        return iter((self.x, self.y))
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def to_tuple(self) -> Tuple[int, int]:
        """Convert to tuple representation."""
        return (self.x, self.y)


class GridWorld:
    """Grid-based world for MAPF scenarios."""
    
    def __init__(self, size: int = 10, goals: Optional[List[Tuple[int, int]]] = None):
        """
        Initialize the grid world.
        
        Args:
            size: Grid size (size x size)
            goals: List of goal positions for agents
        """
        self.size = size
        self.goals = goals or [(size-1, size-1), (0, size-1), (size-1, 0), (0, 0)]
        self.obstacles: Set[Tuple[int, int]] = set()
        self.agent_positions: Dict[int, Tuple[int, int]] = {}
        self.timestamp = 0.0
        
        # Track movement history for analysis
        self.movement_history: List[Dict[str, Any]] = []
        
        # Static obstacles (walls, barriers)
        self.static_obstacles: Set[Tuple[int, int]] = set()
        
        # Dynamic obstacles (temporary blocks)
        self.dynamic_obstacles: Dict[Tuple[int, int], float] = {}  # position -> expiry_time
        
    def add_obstacle(self, x: int, y: int, is_static: bool = True) -> None:
        """Add an obstacle at the given position."""
        if self.is_valid_position(x, y):
            pos = (x, y)
            self.obstacles.add(pos)
            if is_static:
                self.static_obstacles.add(pos)
    
    def add_temporary_obstacle(self, x: int, y: int, duration: float) -> None:
        """Add a temporary obstacle that expires after duration seconds."""
        if self.is_valid_position(x, y):
            pos = (x, y)
            self.obstacles.add(pos)
            self.dynamic_obstacles[pos] = self.timestamp + duration
    
    def remove_obstacle(self, x: int, y: int) -> None:
        """Remove an obstacle from the given position."""
        pos = (x, y)
        self.obstacles.discard(pos)
        self.static_obstacles.discard(pos)
        self.dynamic_obstacles.pop(pos, None)
    
    def update_dynamic_obstacles(self) -> None:
        """Remove expired dynamic obstacles."""
        expired = []
        for pos, expiry_time in self.dynamic_obstacles.items():
            if self.timestamp >= expiry_time:
                expired.append(pos)
        
        for pos in expired:
            self.obstacles.discard(pos)
            del self.dynamic_obstacles[pos]
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is within grid bounds."""
        return validate_position((x, y), self.size)
    
    def is_obstacle(self, x: int, y: int) -> bool:
        """Check if position contains an obstacle."""
        return (x, y) in self.obstacles
    
    def is_occupied_by_agent(self, x: int, y: int, exclude_agent: Optional[int] = None) -> bool:
        """Check if position is occupied by any agent."""
        pos = (x, y)
        for aid, agent_pos in self.agent_positions.items():
            if aid != exclude_agent and agent_pos == pos:
                return True
        return False
    
    def is_free(self, x: int, y: int, exclude_agent: Optional[int] = None) -> bool:
        """
        Check if position is free (no obstacles or other agents).
        
        Args:
            x, y: Position coordinates
            exclude_agent: Agent ID to exclude from collision check
        """
        if not self.is_valid_position(x, y):
            return False
        
        if self.is_obstacle(x, y):
            return False
        
        if self.is_occupied_by_agent(x, y, exclude_agent):
            return False
        
        return True
    
    def move_agent(self, agent_id: int, new_x: int, new_y: int) -> bool:
        """
        Attempt to move an agent to a new position.
        
        Returns:
            True if move was successful, False otherwise
        """
        old_pos = self.agent_positions.get(agent_id)
        new_pos = (new_x, new_y)
        
        if self.is_free(new_x, new_y, exclude_agent=agent_id):
            self.agent_positions[agent_id] = new_pos
            
            # Record movement in history
            self.movement_history.append({
                "timestamp": self.timestamp,
                "agent_id": agent_id,
                "from": old_pos,
                "to": new_pos,
                "success": True
            })
            
            return True
        else:
            # Record failed move attempt
            self.movement_history.append({
                "timestamp": self.timestamp,
                "agent_id": agent_id,
                "from": old_pos,
                "to": new_pos,
                "success": False,
                "reason": self._get_block_reason(new_x, new_y, agent_id)
            })
            
            return False
    
    def _get_block_reason(self, x: int, y: int, agent_id: int) -> str:
        """Get reason why a position is blocked."""
        if not self.is_valid_position(x, y):
            return "out_of_bounds"
        if self.is_obstacle(x, y):
            return "obstacle"
        if self.is_occupied_by_agent(x, y, agent_id):
            return "agent_collision"
        return "unknown"
    
    def get_neighbors(self, x: int, y: int, connectivity: int = 4) -> List[Tuple[int, int]]:
        """
        Get valid neighboring positions.
        
        Args:
            x, y: Current position
            connectivity: 4 for 4-connected, 8 for 8-connected
        """
        neighbors = []
        
        if connectivity == 4:
            # 4-connected neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if self.is_valid_position(nx, ny):
                    neighbors.append((nx, ny))
        elif connectivity == 8:
            # 8-connected neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if self.is_valid_position(nx, ny):
                        neighbors.append((nx, ny))
        
        return neighbors
    
    def get_free_neighbors(self, x: int, y: int, exclude_agent: Optional[int] = None, 
                          connectivity: int = 4) -> List[Tuple[int, int]]:
        """Get free neighboring positions."""
        neighbors = self.get_neighbors(x, y, connectivity)
        return [pos for pos in neighbors if self.is_free(*pos, exclude_agent)]
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return manhattan_distance(pos1, pos2)
    
    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    
    def get_agent_at_position(self, x: int, y: int) -> Optional[int]:
        """Get the ID of the agent at the given position."""
        pos = (x, y)
        for aid, agent_pos in self.agent_positions.items():
            if agent_pos == pos:
                return aid
        return None
    
    def get_distance_to_goal(self, agent_id: int) -> int:
        """Get Manhattan distance from agent to its goal."""
        if agent_id not in self.agent_positions or agent_id >= len(self.goals):
            return float('inf')
        
        agent_pos = self.agent_positions[agent_id]
        goal_pos = self.goals[agent_id]
        return self.manhattan_distance(agent_pos, goal_pos)
    
    def is_agent_at_goal(self, agent_id: int) -> bool:
        """Check if an agent has reached its goal."""
        if agent_id not in self.agent_positions or agent_id >= len(self.goals):
            return False
        
        return self.agent_positions[agent_id] == self.goals[agent_id]
    
    def get_all_goals_status(self) -> Dict[int, bool]:
        """Get goal completion status for all agents."""
        status = {}
        for agent_id in self.agent_positions:
            status[agent_id] = self.is_agent_at_goal(agent_id)
        return status
    
    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """Detect current position conflicts between agents."""
        conflicts = []
        positions = {}
        
        # Group agents by position
        for agent_id, pos in self.agent_positions.items():
            if pos not in positions:
                positions[pos] = []
            positions[pos].append(agent_id)
        
        # Find conflicts (multiple agents at same position)
        for pos, agents in positions.items():
            if len(agents) > 1:
                conflicts.append({
                    "type": "position_conflict",
                    "position": pos,
                    "agents": agents,
                    "timestamp": self.timestamp
                })
        
        return conflicts
    
    def update_timestamp(self) -> None:
        """Update world timestamp."""
        self.timestamp = time.time()
        self.update_dynamic_obstacles()
    
    def get_world_state(self) -> Dict[str, Any]:
        """Get current world state snapshot."""
        return {
            "timestamp": self.timestamp,
            "size": self.size,
            "agent_positions": self.agent_positions.copy(),
            "static_obstacles": list(self.static_obstacles),
            "dynamic_obstacles": dict(self.dynamic_obstacles),
            "goals": self.goals.copy(),
            "conflicts": self.detect_conflicts()
        }
    
    def get_occupancy_grid(self) -> List[List[str]]:
        """
        Get a visual representation of the grid.
        
        Returns:
            2D list where each cell contains:
            - '.' for empty space
            - '#' for obstacle  
            - str(agent_id) for agent position
            - 'G' for goal position (if no agent there)
        """
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        
        # Mark obstacles
        for x, y in self.obstacles:
            if self.is_valid_position(x, y):
                grid[y][x] = '#'
        
        # Mark goals
        for i, (x, y) in enumerate(self.goals):
            if self.is_valid_position(x, y) and grid[y][x] == '.':
                grid[y][x] = f'G{i}'
        
        # Mark agents (agents override goals)
        for agent_id, (x, y) in self.agent_positions.items():
            if self.is_valid_position(x, y):
                grid[y][x] = str(agent_id)
        
        return grid
    
    def print_grid(self) -> None:
        """Print a visual representation of the current grid state."""
        grid = self.get_occupancy_grid()
        
        print(f"\nGrid State (timestamp: {self.timestamp:.2f}):")
        print("  " + "".join(f"{i%10}" for i in range(self.size)))
        
        for y, row in enumerate(grid):
            print(f"{y%10} " + "".join(f"{cell:1}" for cell in row))
        
        print(f"\nAgent positions: {self.agent_positions}")
        print(f"Goals: {self.goals}")
        print(f"Obstacles: {len(self.obstacles)} total")
    
    def clear_agents(self) -> None:
        """Remove all agents from the world."""
        self.agent_positions.clear()
    
    def reset(self) -> None:
        """Reset world to initial state."""
        self.clear_agents()
        self.movement_history.clear()
        self.dynamic_obstacles.clear()
        self.obstacles = self.static_obstacles.copy()
        self.timestamp = 0.0 