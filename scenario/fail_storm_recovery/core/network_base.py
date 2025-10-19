"""
Fail-Storm Network Base - Abstract network coordinator for fail-storm scenarios.

Provides protocol-agnostic network coordination with fault detection and recovery,
while delegating actual communication to protocol-specific adapters.
"""

import asyncio
import time
import json
import random
from abc import ABC, abstractmethod
from typing import Dict, Set, List, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

from core.comm import AbstractCommAdapter


@dataclass
class AgentState:
    """State information for a registered agent."""
    agent_id: str
    position: Optional[Tuple[int, int]] = None
    last_heartbeat: float = 0.0
    is_active: bool = True
    last_update: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FaultEvent:
    """Record of a fault detection event."""
    timestamp: float
    agent_id: str
    event_type: str  # "heartbeat_timeout", "connection_lost", "manual_injection"
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class FailStormNetworkBase(ABC):
    """
    Abstract network coordinator for fail-storm recovery scenarios.
    
    Implements:
    - Agent registration and management
    - Fault detection via heartbeat monitoring
    - Recovery coordination and topology healing
    - Metrics collection for fail-storm analysis
    
    Delegates actual communication to protocol-specific adapters.
    """
    
    def __init__(self, 
                 comm_adapter: AbstractCommAdapter,
                 heartbeat_interval: float = 5.0,
                 heartbeat_timeout: float = 15.0,
                 debug_mode: bool = False):
        """
        Initialize fail-storm network coordinator.
        
        Parameters
        ----------
        comm_adapter : AbstractCommAdapter
            Protocol-specific communication adapter
        heartbeat_interval : float
            Interval between heartbeat messages (seconds)
        heartbeat_timeout : float
            Time to wait before considering a node failed (seconds)
        debug_mode : bool
            Enable verbose debug output
        """
        self.comm_adapter = comm_adapter
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.debug_mode = debug_mode
        
        # Agent management
        self.agents: Dict[str, AgentState] = {}
        self.active_agents: Set[str] = set()
        self.failed_agents: Set[str] = set()
        
        # Fault detection state
        self._heartbeat_tasks: Dict[str, asyncio.Task] = {}
        self._fault_events: List[FaultEvent] = []
        self._recovery_events: List[Dict[str, Any]] = []
        
        # Network state
        self._topology_changes: List[Dict[str, Any]] = []
        self._gaia_document: Optional[Dict[str, Any]] = None
        self._is_running = False
        
        # Metrics for fail-storm analysis
        self._fault_injection_time: Optional[float] = None
        self._first_recovery_time: Optional[float] = None
        self._message_stats = {
            "heartbeats_sent": 0,
            "heartbeats_received": 0,
            "reconnection_attempts": 0,
            "successful_reconnections": 0,
            "failed_reconnections": 0,
            "fault_events": 0,
            "recovery_events": 0
        }
    
    # ==================== Abstract Protocol Methods ====================
    
    @abstractmethod
    async def deliver(self, dst: str, msg: Dict[str, Any]) -> None:
        """
        Deliver message to specific agent via protocol adapter.
        
        Args:
            dst: Destination agent ID
            msg: Message payload
        """
        pass
    
    @abstractmethod
    async def poll(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Poll for incoming messages from protocol adapter.
        
        Returns:
            List of (sender_id, message) tuples
        """
        pass
    
    # ==================== Agent Management ====================
    
    async def register_agent(self, agent_id: str, **kwargs) -> None:
        """Register a new agent in the network."""
        if agent_id in self.agents:
            print(f"[NetworkBase] Agent {agent_id} already registered")
            return
        
        # Create agent state
        agent_state = AgentState(
            agent_id=agent_id,
            last_heartbeat=time.time(),
            metadata=kwargs
        )
        
        self.agents[agent_id] = agent_state
        self.active_agents.add(agent_id)
        self.failed_agents.discard(agent_id)
        
        # Start heartbeat monitoring
        await self._start_heartbeat_monitoring(agent_id)
        
        # Record topology change
        self._record_topology_change("agent_registered", agent_id)
        
        print(f"[NetworkBase] Agent {agent_id} registered")
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the network."""
        if agent_id not in self.agents:
            return
        
        # Stop heartbeat monitoring
        await self._stop_heartbeat_monitoring(agent_id)
        
        # Remove from tracking
        self.agents.pop(agent_id, None)
        self.active_agents.discard(agent_id)
        self.failed_agents.discard(agent_id)
        
        # Record topology change
        self._record_topology_change("agent_unregistered", agent_id)
        
        print(f"[NetworkBase] Agent {agent_id} unregistered")
    
    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Get current state of an agent."""
        return self.agents.get(agent_id)
    
    def get_active_agents(self) -> Set[str]:
        """Get set of currently active agents."""
        return self.active_agents.copy()
    
    def get_failed_agents(self) -> Set[str]:
        """Get set of currently failed agents."""
        return self.failed_agents.copy()
    
    # ==================== Fault Detection ====================
    
    async def _start_heartbeat_monitoring(self, agent_id: str) -> None:
        """Start heartbeat monitoring for a specific agent."""
        if agent_id in self._heartbeat_tasks:
            return  # Already monitoring
        
        # Skip heartbeat monitoring if there's only one agent
        if len(self.agents) <= 1:
            if self.debug_mode:
                print(f"[NetworkBase] Skipping heartbeat monitoring for {agent_id} (single agent)")
            return
        
        async def heartbeat_loop():
            """Heartbeat loop for monitoring agent health."""
            while agent_id in self.agents and agent_id not in self.failed_agents:
                try:
                    # Send heartbeat
                    await self.comm_adapter.send_heartbeat(agent_id)
                    self._message_stats["heartbeats_sent"] += 1
                    
                    # Check for missed heartbeats from others
                    await self._check_heartbeat_timeouts()
                    
                    # Wait for next heartbeat
                    await asyncio.sleep(self.heartbeat_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"[NetworkBase] Heartbeat error for {agent_id}: {e}")
                    await asyncio.sleep(1.0)  # Brief pause before retry
        
        # Start heartbeat task
        task = asyncio.create_task(heartbeat_loop())
        self._heartbeat_tasks[agent_id] = task
        
        if self.debug_mode:
            print(f"[NetworkBase] Started heartbeat monitoring for {agent_id}")
    
    async def _stop_heartbeat_monitoring(self, agent_id: str) -> None:
        """Stop heartbeat monitoring for a specific agent."""
        if agent_id in self._heartbeat_tasks:
            task = self._heartbeat_tasks.pop(agent_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    async def _check_heartbeat_timeouts(self) -> None:
        """Check for agents that have missed heartbeats."""
        current_time = time.time()
        
        for agent_id, agent_state in self.agents.items():
            if agent_id in self.failed_agents:
                continue
            
            time_since_heartbeat = current_time - agent_state.last_heartbeat
            
            if time_since_heartbeat > self.heartbeat_timeout:
                await self._handle_agent_failure(agent_id, "heartbeat_timeout", {
                    "time_since_heartbeat": time_since_heartbeat,
                    "timeout_threshold": self.heartbeat_timeout
                })
    
    async def _handle_agent_failure(self, agent_id: str, failure_type: str, details: Dict[str, Any]) -> None:
        """Handle detection of agent failure."""
        if agent_id in self.failed_agents:
            return  # Already marked as failed
        
        # Mark as failed
        self.active_agents.discard(agent_id)
        self.failed_agents.add(agent_id)
        
        if agent_id in self.agents:
            self.agents[agent_id].is_active = False
        
        # Record fault event
        fault_event = FaultEvent(
            timestamp=time.time(),
            agent_id=agent_id,
            event_type=failure_type,
            details=details
        )
        self._fault_events.append(fault_event)
        self._message_stats["fault_events"] += 1
        
        # Record topology change
        self._record_topology_change("agent_failed", agent_id, details)
        
        print(f"[NetworkBase] Agent {agent_id} failed: {failure_type}")
        
        # Trigger recovery process
        await self._trigger_recovery(agent_id, failure_type)
    
    async def _trigger_recovery(self, failed_agent_id: str, failure_type: str) -> None:
        """Trigger recovery process for failed agent."""
        recovery_start = time.time()
        
        if self._first_recovery_time is None:
            self._first_recovery_time = recovery_start
        
        # Attempt reconnection
        self._message_stats["reconnection_attempts"] += 1
        
        try:
            # Notify other agents about the failure
            failure_notification = {
                "type": "agent_failure",
                "failed_agent": failed_agent_id,
                "failure_type": failure_type,
                "timestamp": recovery_start
            }
            await self.comm_adapter.broadcast(failure_notification)
            
            # Record recovery event
            recovery_event = {
                "timestamp": recovery_start,
                "failed_agent": failed_agent_id,
                "failure_type": failure_type,
                "recovery_attempted": True
            }
            self._recovery_events.append(recovery_event)
            self._message_stats["recovery_events"] += 1
            
            print(f"[NetworkBase] Recovery triggered for {failed_agent_id}")
            
        except Exception as e:
            print(f"[NetworkBase] Recovery failed for {failed_agent_id}: {e}")
            self._message_stats["failed_reconnections"] += 1
    
    # ==================== Message Processing ====================
    
    async def process_messages(self) -> None:
        """Process all incoming messages from agents."""
        messages = await self.poll()
        
        for sender_id, message in messages:
            await self._handle_message(sender_id, message)
    
    async def _handle_message(self, sender_id: str, message: Dict[str, Any]) -> None:
        """Handle different types of messages from agents."""
        msg_type = message.get("type")
        
        if msg_type == "heartbeat":
            await self._handle_heartbeat(sender_id, message)
        elif msg_type == "agent_failure":
            await self._handle_failure_notification(sender_id, message)
        elif msg_type == "agent_recovery":
            await self._handle_recovery_notification(sender_id, message)
        elif msg_type == "topology_update":
            await self._handle_topology_update(sender_id, message)
        else:
            # Handle protocol-specific messages
            await self._handle_protocol_message(sender_id, message)
    
    async def _handle_heartbeat(self, sender_id: str, message: Dict[str, Any]) -> None:
        """Handle heartbeat message from agent."""
        if sender_id in self.agents:
            self.agents[sender_id].last_heartbeat = time.time()
            self._message_stats["heartbeats_received"] += 1
            
            # If agent was previously failed, mark as recovered
            if sender_id in self.failed_agents:
                await self._handle_agent_recovery(sender_id)
    
    async def _handle_failure_notification(self, sender_id: str, message: Dict[str, Any]) -> None:
        """Handle failure notification from another agent."""
        failed_agent = message.get("failed_agent")
        if failed_agent and failed_agent not in self.failed_agents:
            await self._handle_agent_failure(failed_agent, "remote_notification", message)
    
    async def _handle_recovery_notification(self, sender_id: str, message: Dict[str, Any]) -> None:
        """Handle recovery notification from agent."""
        recovered_agent = message.get("recovered_agent")
        if recovered_agent:
            await self._handle_agent_recovery(recovered_agent)
    
    async def _handle_topology_update(self, sender_id: str, message: Dict[str, Any]) -> None:
        """Handle topology update from agent."""
        # Update local topology information
        topology_data = message.get("topology", {})
        self._topology_changes.append({
            "timestamp": time.time(),
            "source": sender_id,
            "data": topology_data
        })
    
    async def _handle_protocol_message(self, sender_id: str, message: Dict[str, Any]) -> None:
        """Handle protocol-specific messages (to be overridden by subclasses)."""
        if self.debug_mode:
            print(f"[NetworkBase] Protocol message from {sender_id}: {message.get('type', 'unknown')}")
    
    async def _handle_agent_recovery(self, agent_id: str) -> None:
        """Handle recovery of a previously failed agent."""
        if agent_id not in self.failed_agents:
            return
        
        # Mark as active again
        self.failed_agents.discard(agent_id)
        self.active_agents.add(agent_id)
        
        if agent_id in self.agents:
            self.agents[agent_id].is_active = True
            self.agents[agent_id].last_heartbeat = time.time()
        
        # Restart heartbeat monitoring
        await self._start_heartbeat_monitoring(agent_id)
        
        # Record recovery
        recovery_event = {
            "timestamp": time.time(),
            "recovered_agent": agent_id,
            "recovery_successful": True
        }
        self._recovery_events.append(recovery_event)
        self._message_stats["successful_reconnections"] += 1
        
        # Record topology change
        self._record_topology_change("agent_recovered", agent_id)
        
        print(f"[NetworkBase] Agent {agent_id} recovered")
    
    # ==================== Utility Methods ====================
    
    def _record_topology_change(self, change_type: str, agent_id: str, details: Dict[str, Any] = None) -> None:
        """Record a topology change event."""
        if details is None:
            details = {}
        
        self._topology_changes.append({
            "timestamp": time.time(),
            "type": change_type,
            "agent_id": agent_id,
            "details": details
        })
    
    def set_gaia_document(self, document: Dict[str, Any]) -> None:
        """Set the Gaia document for broadcast."""
        self._gaia_document = document
    
    def get_gaia_document(self) -> Optional[Dict[str, Any]]:
        """Get the current Gaia document."""
        return self._gaia_document
    
    def set_fault_injection_time(self, timestamp: float) -> None:
        """Set the time when fault injection occurred."""
        self._fault_injection_time = timestamp
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current network metrics."""
        return {
            "active_agents": len(self.active_agents),
            "failed_agents": len(self.failed_agents),
            "total_agents": len(self.agents),
            "fault_events": len(self._fault_events),
            "recovery_events": len(self._recovery_events),
            "topology_changes": len(self._topology_changes),
            "message_stats": self._message_stats.copy(),
            "fault_injection_time": self._fault_injection_time,
            "first_recovery_time": self._first_recovery_time
        }
    
    def get_fault_events(self) -> List[FaultEvent]:
        """Get list of fault events."""
        return self._fault_events.copy()
    
    def get_recovery_events(self) -> List[Dict[str, Any]]:
        """Get list of recovery events."""
        return self._recovery_events.copy()
    
    def get_topology_changes(self) -> List[Dict[str, Any]]:
        """Get list of topology changes."""
        return self._topology_changes.copy()
    
    # ==================== Lifecycle Management ====================
    
    async def start(self) -> None:
        """Start the network coordinator."""
        if self._is_running:
            return
        
        self._is_running = True
        await self.comm_adapter.connect()
        print(f"[NetworkBase] Network coordinator started")
    
    async def stop(self) -> None:
        """Stop the network coordinator."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Stop all heartbeat tasks
        for agent_id in list(self._heartbeat_tasks.keys()):
            await self._stop_heartbeat_monitoring(agent_id)
        
        await self.comm_adapter.disconnect()
        print(f"[NetworkBase] Network coordinator stopped")
    
    def is_running(self) -> bool:
        """Check if the network coordinator is running."""
        return self._is_running 