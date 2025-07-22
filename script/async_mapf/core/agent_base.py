"""
Base robot class for MAPF agents.

Contains complete pathfinding and coordination algorithms,
with abstract communication methods to be implemented by protocol backends.
"""

from __future__ import annotations
import abc
import asyncio
import heapq
from typing import Dict, Any, Optional, Tuple, List
from .world import GridWorld
from .utils import PathfindingResult


class BaseRobot(abc.ABC):
    """
    Algorithm-complete robot with abstract communication hooks.
    
    Implements:
    - A* pathfinding
    - Conflict resolution
    - Coordination protocols
    
    Abstract methods (to be implemented by protocol backends):
    - send_msg: Send message to another agent
    - recv_msg: Receive message with timeout
    """
    
    def __init__(self, aid: int, world: GridWorld, goal: Tuple[int, int]) -> None:
        """
        Initialize robot.
        
        Args:
            aid: Agent ID
            world: Shared world reference
            goal: Target position (x, y)
        """
        self.aid = aid
        self.world = world
        self.goal = goal
        self.current_pos: Optional[Tuple[int, int]] = None
        self.path: List[Tuple[int, int]] = []
        self.path_index = 0
        self.is_active = True
        self.last_replan_time = 0.0
        
        # Communication state
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._coordination_state = {}
        
    # ==================== Abstract Communication Methods ====================
    
    @abc.abstractmethod
    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """
        Send message to another agent.
        
        Args:
            dst: Destination agent ID
            payload: Message content
        """
        pass
    
    @abc.abstractmethod
    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Receive message with optional timeout.
        
        Args:
            timeout: Maximum wait time in seconds (0 = non-blocking)
            
        Returns:
            Received message or None if timeout
        """
        pass
    
    # ==================== Algorithmic Implementation ====================
    
    def initialize_position(self, start_pos: Tuple[int, int]) -> None:
        """Set initial position and update world state."""
        self.current_pos = start_pos
        self.world.agent_positions[self.aid] = start_pos
    
    def compute_path(self) -> List[Tuple[int, int]]:
        """
        Compute A* path from current position to goal.
        
        Returns:
            List of positions forming the path
        """
        if not self.current_pos:
            return []
        
        return self._astar_search(self.current_pos, self.goal)
    
    def _astar_search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding implementation."""
        # Priority queue: (f_score, g_score, position, path)
        open_set = [(0, 0, start, [start])]
        closed_set = set()
        
        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            if current == goal:
                return path
            
            closed_set.add(current)
            
            for neighbor in self.world.get_neighbors(*current):
                if neighbor in closed_set or not self.world.is_free(*neighbor, exclude_agent=self.aid):
                    continue
                
                new_g_score = g_score + 1
                h_score = self.world.manhattan_distance(neighbor, goal)
                new_f_score = new_g_score + h_score
                new_path = path + [neighbor]
                
                heapq.heappush(open_set, (new_f_score, new_g_score, neighbor, new_path))
        
        return []  # No path found
    
    async def move_next(self) -> bool:
        """
        Execute next move in the planned path.
        
        Returns:
            True if moved successfully, False if blocked or reached goal
        """
        if not self.path or self.path_index >= len(self.path):
            return False
        
        next_pos = self.path[self.path_index]
        
        # Check if next position is still free
        if not self.world.is_free(*next_pos, exclude_agent=self.aid):
            # Path blocked, request replan
            await self._handle_path_conflict(next_pos)
            return False
        
        # Execute move
        if self.world.move_agent(self.aid, *next_pos):
            self.current_pos = next_pos
            self.path_index += 1
            return True
        
        return False
    
    async def _handle_path_conflict(self, blocked_pos: Tuple[int, int]) -> None:
        """Handle path conflicts with other agents."""
        # Send conflict notification to nearby agents
        conflict_msg = {
            "type": "path_conflict",
            "agent_id": self.aid,
            "blocked_position": blocked_pos,
            "timestamp": self.world.timestamp
        }
        
        # Broadcast to all agents (in real implementation, would be nearby agents)
        for other_aid in self.world.agent_positions:
            if other_aid != self.aid:
                await self.send_msg(other_aid, conflict_msg)
        
        # Replan path
        self.path = self.compute_path()
        self.path_index = 0
    
    async def process_messages(self) -> None:
        """Process incoming messages and update coordination state."""
        while True:
            msg = await self.recv_msg(timeout=0.0)  # Non-blocking
            if not msg:
                break
            
            await self._handle_message(msg)
    
    async def _handle_message(self, msg: Dict[str, Any]) -> None:
        """Handle different types of coordination messages."""
        msg_type = msg.get("type")
        
        if msg_type == "path_conflict":
            await self._handle_conflict_message(msg)
        elif msg_type == "coordination_request":
            await self._handle_coordination_request(msg)
        elif msg_type == "priority_update":
            await self._handle_priority_update(msg)
    
    async def _handle_conflict_message(self, msg: Dict[str, Any]) -> None:
        """Handle path conflict notifications."""
        # Simple strategy: agents with higher ID yield
        other_aid = msg["agent_id"]
        if self.aid > other_aid:
            # Wait a bit before replanning
            await asyncio.sleep(0.1)
            self.path = self.compute_path()
            self.path_index = 0
    
    async def _handle_coordination_request(self, msg: Dict[str, Any]) -> None:
        """Handle coordination requests from other agents."""
        # Implement coordination protocol here
        pass
    
    async def _handle_priority_update(self, msg: Dict[str, Any]) -> None:
        """Handle priority updates from coordinator."""
        # Update agent priority for conflict resolution
        pass
    
    def is_at_goal(self) -> bool:
        """Check if agent has reached its goal."""
        return self.current_pos == self.goal
    
    async def run(self) -> None:
        """
        Main agent execution loop.
        
        Handles:
        - Path planning
        - Movement execution
        - Message processing
        - Conflict resolution
        """
        # Initialize path
        if self.current_pos:
            self.path = self.compute_path()
        
        while self.is_active and not self.is_at_goal():
            # Process incoming messages
            await self.process_messages()
            
            # Attempt to move
            if self.path and self.path_index < len(self.path):
                moved = await self.move_next()
                if not moved:
                    # Wait a bit before retrying
                    await asyncio.sleep(0.1)
            else:
                # Replan if no valid path
                self.path = self.compute_path()
                self.path_index = 0
                if not self.path:
                    # No path available, wait
                    await asyncio.sleep(0.5)
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.05)
        
        # Notify completion
        completion_msg = {
            "type": "goal_reached",
            "agent_id": self.aid,
            "final_position": self.current_pos
        }
        
        # Broadcast completion (implementation depends on protocol)
        for other_aid in self.world.agent_positions:
            if other_aid != self.aid:
                await self.send_msg(other_aid, completion_msg) 