"""
Base network coordinator for MAPF scenarios.

Provides global clock synchronization and conflict arbitration,
with abstract I/O methods for different communication protocols.
"""

from __future__ import annotations
import abc
import time
import asyncio
import heapq
from typing import Dict, Any, Tuple, List, Optional, Set
from .world import GridWorld
from .utils import (
    ConflictInfo, AgentState, MessageType, PerformanceMonitor,
    create_message, info_log, error_log
)


class BaseNet(abc.ABC):
    """
    Global coordinator with abstract protocol I/O hooks.
    
    Implements:
    - Global time synchronization
    - Conflict detection and arbitration
    - Performance monitoring
    
    Abstract methods (to be implemented by protocol backends):
    - deliver: Send message to specific agent
    - poll: Check for incoming messages
    """
    
    def __init__(self, world: GridWorld, tick_ms: int = 10) -> None:
        """
        Initialize network coordinator.
        
        Args:
            world: Shared world reference
            tick_ms: Time step duration in milliseconds
        """
        self.world = world
        self.tick_ms = tick_ms
        self.tick_duration = tick_ms / 1000.0  # Convert to seconds
        
        # Global state
        self.current_tick = 0
        self.start_time = 0.0
        self.is_running = False
        
        # Agent tracking
        self.active_agents: Set[int] = set()
        self.completed_agents: Set[int] = set()
        self.agent_states: Dict[int, AgentState] = {}
        
        # Conflict resolution
        self.conflict_queue: List[Tuple[float, ConflictInfo]] = []
        self.pending_moves: Dict[int, Tuple[int, int]] = {}
        self.move_priorities: Dict[int, int] = {}  # agent_id -> priority
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Message handling
        self._message_buffer: List[Tuple[int, Dict[str, Any]]] = []
        
        # Coordination policies
        self.conflict_resolution_strategy = "priority_based"  # priority_based, fifo, random
        self.max_resolution_attempts = 3
        
    # ==================== Abstract Protocol Methods ====================
    
    @abc.abstractmethod
    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """
        Deliver message to specific agent.
        
        Args:
            dst: Destination agent ID
            msg: Message payload
        """
        pass
    
    @abc.abstractmethod
    async def poll(self) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Poll for incoming messages from agents.
        
        Returns:
            List of (sender_id, message) tuples
        """
        pass
    
    # ==================== Agent Management ====================
    
    def register_agent(self, agent_id: int, start_pos: Tuple[int, int], 
                      goal_pos: Tuple[int, int], priority: int = 0) -> None:
        """Register a new agent in the system."""
        self.active_agents.add(agent_id)
        self.world.agent_positions[agent_id] = start_pos
        self.move_priorities[agent_id] = priority
        
        # Initialize agent state
        self.agent_states[agent_id] = AgentState(
            agent_id=agent_id,
            position=start_pos,
            goal=goal_pos,
            path=[],
            path_index=0,
            is_active=True,
            last_update=self.world.timestamp
        )
        
        info_log(f"Registered agent {agent_id} at {start_pos} with goal {goal_pos}")
    
    def unregister_agent(self, agent_id: int) -> None:
        """Remove agent from active tracking."""
        self.active_agents.discard(agent_id)
        self.completed_agents.add(agent_id)
        
        if agent_id in self.world.agent_positions:
            del self.world.agent_positions[agent_id]
        
        if agent_id in self.agent_states:
            self.agent_states[agent_id].is_active = False
        
        self.move_priorities.pop(agent_id, None)
        self.pending_moves.pop(agent_id, None)
        
        info_log(f"Unregistered agent {agent_id}")
    
    def get_agent_state(self, agent_id: int) -> Optional[AgentState]:
        """Get current state of an agent."""
        return self.agent_states.get(agent_id)
    
    def update_agent_state(self, agent_id: int, **kwargs) -> None:
        """Update agent state with provided fields."""
        if agent_id in self.agent_states:
            state = self.agent_states[agent_id]
            for key, value in kwargs.items():
                if hasattr(state, key):
                    setattr(state, key, value)
            state.last_update = self.world.timestamp
    
    # ==================== Message Processing ====================
    
    async def process_messages(self) -> None:
        """Process all incoming messages from agents."""
        start_time = time.time()
        
        messages = await self.poll()
        self._message_buffer.extend(messages)
        
        # Process buffered messages
        for sender_id, msg in self._message_buffer:
            await self._handle_message(sender_id, msg)
            self.performance_monitor.increment("message_count")
        
        self._message_buffer.clear()
        
        # Record processing time
        processing_time = time.time() - start_time
        self.performance_monitor.add_timing("avg_message_processing_time", processing_time)
    
    async def _handle_message(self, sender_id: int, msg: Dict[str, Any]) -> None:
        """Handle different types of messages from agents."""
        msg_type = msg.get("type")
        
        if msg_type == MessageType.MOVE_REQUEST.value:
            await self._handle_move_request(sender_id, msg)
        elif msg_type == MessageType.PATH_CONFLICT.value:
            await self._handle_conflict_report(sender_id, msg)
        elif msg_type == MessageType.GOAL_REACHED.value:
            await self._handle_goal_reached(sender_id, msg)
        elif msg_type == MessageType.STATUS_UPDATE.value:
            await self._handle_status_update(sender_id, msg)
        else:
            error_log(f"Unknown message type: {msg_type} from agent {sender_id}")
    
    async def _handle_move_request(self, agent_id: int, msg: Dict[str, Any]) -> None:
        """Process movement request from agent."""
        target_pos = tuple(msg["target_position"])
        
        # Update pending moves
        self.pending_moves[agent_id] = target_pos
        
        # Check for conflicts
        conflicts = self._detect_move_conflicts(agent_id, target_pos)
        
        if conflicts:
            await self._resolve_move_conflicts(conflicts)
        else:
            # Approve move immediately
            await self._approve_move(agent_id, target_pos)
    
    def _detect_move_conflicts(self, agent_id: int, target_pos: Tuple[int, int]) -> List[ConflictInfo]:
        """Detect all conflicts involving a move request."""
        conflicts = []
        
        # Check position conflicts
        conflicting_agents = []
        
        # Check if position is occupied by another agent
        for other_id, other_pos in self.world.agent_positions.items():
            if other_id != agent_id and other_pos == target_pos:
                conflicting_agents.append(other_id)
        
        # Check if multiple agents are trying to move to same position
        for other_id, pending_pos in self.pending_moves.items():
            if other_id != agent_id and pending_pos == target_pos:
                conflicting_agents.append(other_id)
        
        if conflicting_agents:
            conflict = ConflictInfo(
                timestamp=self.world.timestamp,
                agent_ids=[agent_id] + conflicting_agents,
                conflict_type="position_conflict",
                position=target_pos,
                resolution_strategy=self.conflict_resolution_strategy
            )
            conflicts.append(conflict)
        
        # Check for swap conflicts (agents swapping positions)
        current_pos = self.world.agent_positions.get(agent_id)
        if current_pos:
            for other_id, pending_pos in self.pending_moves.items():
                if (other_id != agent_id and 
                    pending_pos == current_pos and 
                    self.world.agent_positions.get(other_id) == target_pos):
                    
                    swap_conflict = ConflictInfo(
                        timestamp=self.world.timestamp,
                        agent_ids=[agent_id, other_id],
                        conflict_type="swap_conflict",
                        position=target_pos,
                        resolution_strategy=self.conflict_resolution_strategy
                    )
                    conflicts.append(swap_conflict)
        
        return conflicts
    
    async def _resolve_move_conflicts(self, conflicts: List[ConflictInfo]) -> None:
        """Resolve movement conflicts using configured strategy."""
        start_time = time.time()
        
        for conflict in conflicts:
            self.performance_monitor.increment("conflict_count")
            
            if self.conflict_resolution_strategy == "priority_based":
                await self._resolve_by_priority(conflict)
            elif self.conflict_resolution_strategy == "fifo":
                await self._resolve_by_fifo(conflict)
            else:
                await self._resolve_by_priority(conflict)  # Default fallback
            
            self.performance_monitor.increment("resolution_count")
        
        # Record resolution time
        resolution_time = time.time() - start_time
        self.performance_monitor.add_timing("avg_conflict_resolution_time", resolution_time)
    
    async def _resolve_by_priority(self, conflict: ConflictInfo) -> None:
        """Resolve conflict using agent priorities."""
        agents = conflict.agent_ids
        
        # Sort by priority (lower value = higher priority)
        agents.sort(key=lambda aid: self.move_priorities.get(aid, float('inf')))
        
        winner = agents[0]
        losers = agents[1:]
        
        # Approve winner's move
        target_pos = self.pending_moves.get(winner)
        if target_pos:
            await self._approve_move(winner, target_pos)
        
        # Deny losers' moves
        for loser_id in losers:
            await self._deny_move(loser_id, "priority_conflict", winner)
    
    async def _resolve_by_fifo(self, conflict: ConflictInfo) -> None:
        """Resolve conflict using first-come-first-served."""
        # Use agent state update time as FIFO criterion
        agents = conflict.agent_ids
        agents.sort(key=lambda aid: self.agent_states.get(aid, AgentState(0, (0,0), (0,0), [], 0, True, 0)).last_update)
        
        winner = agents[0]
        losers = agents[1:]
        
        # Approve winner's move
        target_pos = self.pending_moves.get(winner)
        if target_pos:
            await self._approve_move(winner, target_pos)
        
        # Deny losers' moves
        for loser_id in losers:
            await self._deny_move(loser_id, "fifo_conflict", winner)
    
    async def _approve_move(self, agent_id: int, target_pos: Tuple[int, int]) -> None:
        """Approve an agent's move request."""
        approval_msg = create_message(
            MessageType.MOVE_APPROVED,
            sender_id=-1,  # Network coordinator
            target_position=target_pos,
            agent_id=agent_id
        )
        
        await self.deliver(agent_id, approval_msg)
        
        # Update world state
        self.world.move_agent(agent_id, *target_pos)
        self.update_agent_state(agent_id, position=target_pos)
        
        # Remove from pending moves
        self.pending_moves.pop(agent_id, None)
        
        self.performance_monitor.increment("successful_moves")
    
    async def _deny_move(self, agent_id: int, reason: str, winning_agent: Optional[int] = None) -> None:
        """Deny an agent's move request."""
        denial_msg = create_message(
            MessageType.MOVE_DENIED,
            sender_id=-1,  # Network coordinator
            reason=reason,
            winning_agent=winning_agent,
            agent_id=agent_id
        )
        
        await self.deliver(agent_id, denial_msg)
        
        # Remove from pending moves
        self.pending_moves.pop(agent_id, None)
        
        self.performance_monitor.increment("failed_moves")
    
    async def _handle_conflict_report(self, agent_id: int, msg: Dict[str, Any]) -> None:
        """Handle conflict reports from agents."""
        conflict_info = ConflictInfo(
            timestamp=self.world.timestamp,
            agent_ids=[agent_id] + msg.get("involved_agents", []),
            conflict_type=msg.get("conflict_type", "reported_conflict"),
            position=tuple(msg.get("position", (0, 0))),
            resolution_strategy=self.conflict_resolution_strategy
        )
        
        heapq.heappush(self.conflict_queue, (self.world.timestamp, conflict_info))
    
    async def _handle_goal_reached(self, agent_id: int, msg: Dict[str, Any]) -> None:
        """Handle goal completion notifications."""
        if agent_id in self.active_agents:
            info_log(f"Agent {agent_id} reached goal at {msg.get('final_position')}")
            
            self.unregister_agent(agent_id)
            
            # Broadcast completion to other agents
            completion_broadcast = create_message(
                MessageType.AGENT_COMPLETED,
                sender_id=-1,  # Network coordinator
                completed_agent=agent_id,
                final_position=msg.get("final_position")
            )
            
            for other_id in self.active_agents:
                await self.deliver(other_id, completion_broadcast)
    
    async def _handle_status_update(self, agent_id: int, msg: Dict[str, Any]) -> None:
        """Handle status updates from agents."""
        # Update agent state with provided information
        update_fields = {}
        
        if "position" in msg:
            update_fields["position"] = tuple(msg["position"])
        if "path" in msg:
            update_fields["path"] = msg["path"]
        if "path_index" in msg:
            update_fields["path_index"] = msg["path_index"]
        
        if update_fields:
            self.update_agent_state(agent_id, **update_fields)
    
    # ==================== Global Coordination ====================
    
    def process_pending_conflicts(self) -> None:
        """Process accumulated conflicts in priority order."""
        current_time = self.world.timestamp
        
        while self.conflict_queue:
            timestamp, conflict_info = self.conflict_queue[0]
            
            # Process conflicts within current time window
            if timestamp <= current_time:
                heapq.heappop(self.conflict_queue)
                # Additional conflict processing if needed
            else:
                break
    
    def update_global_clock(self) -> None:
        """Update global time and synchronize with agents."""
        self.current_tick += 1
        self.world.update_timestamp()
    
    async def broadcast_time_sync(self) -> None:
        """Send time synchronization to all active agents."""
        sync_msg = create_message(
            MessageType.TIME_SYNC,
            sender_id=-1,  # Network coordinator
            tick=self.current_tick,
            global_timestamp=self.world.timestamp
        )
        
        for agent_id in self.active_agents:
            await self.deliver(agent_id, sync_msg)
    
    def is_scenario_complete(self) -> bool:
        """Check if all agents have completed their tasks."""
        return len(self.active_agents) == 0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        base_metrics = self.performance_monitor.get_summary()
        
        return {
            **base_metrics,
            "current_tick": self.current_tick,
            "active_agents": len(self.active_agents),
            "completed_agents": len(self.completed_agents),
            "pending_moves": len(self.pending_moves),
            "pending_conflicts": len(self.conflict_queue)
        }
    
    # ==================== Main Execution Loop ====================
    
    async def run(self) -> None:
        """
        Main network coordinator loop.
        
        Handles:
        - Global time synchronization
        - Message processing
        - Conflict resolution
        - Performance monitoring
        """
        self.start_time = time.time()
        self.is_running = True
        
        info_log(f"Starting MAPF coordinator with {len(self.active_agents)} agents")
        
        while self.is_running and not self.is_scenario_complete():
            tick_start = time.time()
            
            try:
                # Process incoming messages
                await self.process_messages()
                
                # Update global clock
                self.update_global_clock()
                
                # Broadcast time sync periodically
                if self.current_tick % 10 == 0:
                    await self.broadcast_time_sync()
                
                # Process pending conflicts
                self.process_pending_conflicts()
                
                # Log progress periodically
                if self.current_tick % 100 == 0:
                    metrics = self.get_performance_metrics()
                    info_log(f"Tick {self.current_tick}: {metrics['active_agents']} active, "
                           f"{metrics['conflict_count']} conflicts, "
                           f"{metrics['message_count']} messages")
                
                # Sleep to maintain tick rate
                tick_end = time.time()
                tick_duration = tick_end - tick_start
                sleep_time = max(0, self.tick_duration - tick_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            except Exception as e:
                error_log(f"Error in coordinator loop: {e}")
                break
        
        # Final metrics
        final_metrics = self.get_performance_metrics()
        info_log(f"MAPF scenario completed in {final_metrics['elapsed_time']:.2f}s")
        info_log(f"Final metrics: {final_metrics}")
        
        self.is_running = False 