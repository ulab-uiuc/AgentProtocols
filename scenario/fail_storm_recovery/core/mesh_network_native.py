#!/usr/bin/env python3
"""
Native MeshNetwork implementation for Fail-Storm Recovery scenario.

This module provides a self-contained mesh network implementation that doesn't
depend on src/ Meta Protocol components, enabling pure protocol-native communication.
"""

import asyncio
import time
import logging
from typing import Dict, Set, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class AgentStatus(Enum):
    """Agent status in the mesh network."""
    ACTIVE = "active"
    DISCONNECTED = "disconnected" 
    FAILED = "failed"
    RECONNECTING = "reconnecting"


@dataclass
class AgentInfo:
    """Information about an agent in the mesh network."""
    agent_id: str
    status: AgentStatus
    last_heartbeat: float
    host: str
    port: int
    protocol: str
    connection_count: int = 0
    fail_count: int = 0


class NativeMeshNetwork:
    """
    Native mesh network implementation for fail-storm scenarios.
    
    This implementation focuses on:
    - Heartbeat-based fault detection
    - Automatic topology recovery after failures  
    - Protocol-agnostic agent management
    - Real-time metrics collection
    """
    
    def __init__(self, heartbeat_interval: float = 5.0, heartbeat_timeout: float = 15.0, 
                 debug_mode: bool = False):
        """
        Initialize native mesh network.
        
        Args:
            heartbeat_interval: Interval between heartbeat checks (seconds)
            heartbeat_timeout: Time to consider agent failed (seconds)
            debug_mode: Enable debug logging
        """
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.debug_mode = debug_mode
        
        # Agent management
        self.agents: Dict[str, AgentInfo] = {}
        self.connections: Dict[str, Set[str]] = {}  # agent_id -> set of connected agents
        
        # Fault detection and recovery
        self.fault_injection_time: Optional[float] = None
        self.failed_agents: Set[str] = set()
        self.recovering_agents: Set[str] = set()
        
        # Metrics and callbacks
        self.metrics = {
            "total_agents": 0,
            "active_agents": 0,
            "failed_agents": 0,
            "total_connections": 0,
            "heartbeat_failures": 0,
            "recovery_attempts": 0
        }
        
        # Event callbacks
        self.on_agent_failed: Optional[Callable[[str], None]] = None
        self.on_agent_recovered: Optional[Callable[[str], None]] = None
        
        # Internal state
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False
        
        self.logger = logging.getLogger("NativeMeshNetwork")
        if debug_mode:
            self.logger.setLevel(logging.DEBUG)
    
    async def start(self) -> None:
        """Start the mesh network monitoring."""
        if self._running:
            return
            
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.logger.info("Native mesh network started")
    
    async def stop(self) -> None:
        """Stop the mesh network monitoring."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Native mesh network stopped")
    
    def register_agent(self, agent_id: str, host: str, port: int, protocol: str) -> None:
        """Register a new agent in the mesh network."""
        agent_info = AgentInfo(
            agent_id=agent_id,
            status=AgentStatus.ACTIVE,
            last_heartbeat=time.time(),
            host=host,
            port=port,
            protocol=protocol
        )
        
        self.agents[agent_id] = agent_info
        self.connections[agent_id] = set()
        
        self.metrics["total_agents"] += 1
        self.metrics["active_agents"] += 1
        
        self.logger.info(f"Registered agent: {agent_id} ({protocol}://{host}:{port})")
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the mesh network."""
        if agent_id not in self.agents:
            return
            
        # Remove all connections to this agent
        for other_id in self.connections:
            self.connections[other_id].discard(agent_id)
        
        # Remove agent
        agent_info = self.agents.pop(agent_id)
        self.connections.pop(agent_id, None)
        
        # Update metrics
        if agent_info.status == AgentStatus.ACTIVE:
            self.metrics["active_agents"] -= 1
        elif agent_info.status == AgentStatus.FAILED:
            self.metrics["failed_agents"] -= 1
            
        self.failed_agents.discard(agent_id)
        self.recovering_agents.discard(agent_id)
        
        self.logger.info(f"Unregistered agent: {agent_id}")
    
    async def connect_agents(self, agent1_id: str, agent2_id: str) -> bool:
        """Establish a bidirectional connection between two agents."""
        if agent1_id not in self.agents or agent2_id not in self.agents:
            return False
            
        # Add bidirectional connection
        self.connections[agent1_id].add(agent2_id)
        self.connections[agent2_id].add(agent1_id)
        
        # Update connection counts
        self.agents[agent1_id].connection_count += 1
        self.agents[agent2_id].connection_count += 1
        
        self.metrics["total_connections"] += 1
        
        self.logger.debug(f"Connected: {agent1_id} ↔ {agent2_id}")
        return True
    
    def disconnect_agents(self, agent1_id: str, agent2_id: str) -> bool:
        """Remove connection between two agents."""
        if agent1_id not in self.connections or agent2_id not in self.connections:
            return False
            
        # Remove bidirectional connection
        self.connections[agent1_id].discard(agent2_id)
        self.connections[agent2_id].discard(agent1_id)
        
        # Update connection counts
        if agent1_id in self.agents:
            self.agents[agent1_id].connection_count = max(0, self.agents[agent1_id].connection_count - 1)
        if agent2_id in self.agents:
            self.agents[agent2_id].connection_count = max(0, self.agents[agent2_id].connection_count - 1)
            
        self.metrics["total_connections"] = max(0, self.metrics["total_connections"] - 1)
        
        self.logger.debug(f"Disconnected: {agent1_id} ↔ {agent2_id}")
        return True
    
    def heartbeat(self, agent_id: str) -> bool:
        """Record a heartbeat from an agent."""
        if agent_id not in self.agents:
            return False
            
        self.agents[agent_id].last_heartbeat = time.time()
        
        # If agent was failed/recovering, mark as recovered
        if self.agents[agent_id].status in [AgentStatus.FAILED, AgentStatus.RECONNECTING]:
            self._mark_agent_recovered(agent_id)
            
        return True
    
    def set_fault_injection_time(self, fault_time: float) -> None:
        """Set the fault injection timestamp for metrics."""
        self.fault_injection_time = fault_time
        self.logger.info(f"Fault injection time set: {fault_time}")
    
    def mark_agent_failed(self, agent_id: str) -> None:
        """Manually mark an agent as failed (for fault injection)."""
        if agent_id not in self.agents:
            return
            
        self._mark_agent_failed(agent_id)
        self.logger.warning(f"Agent manually marked as failed: {agent_id}")
    
    def get_active_agents(self) -> Set[str]:
        """Get set of currently active agent IDs."""
        return {aid for aid, info in self.agents.items() 
                if info.status == AgentStatus.ACTIVE}
    
    def get_failed_agents(self) -> Set[str]:
        """Get set of currently failed agent IDs."""
        return self.failed_agents.copy()
    
    def get_agent_info(self, agent_id: str) -> Optional[AgentInfo]:
        """Get information about a specific agent."""
        return self.agents.get(agent_id)
    
    def get_connections(self, agent_id: str) -> Set[str]:
        """Get set of agents connected to the specified agent."""
        return self.connections.get(agent_id, set()).copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current mesh network metrics."""
        # Update real-time metrics
        active_count = len(self.get_active_agents())
        failed_count = len(self.failed_agents)
        
        return {
            **self.metrics,
            "active_agents": active_count,
            "failed_agents": failed_count,
            "recovering_agents": len(self.recovering_agents),
            "total_connections": sum(len(conns) for conns in self.connections.values()) // 2
        }
    
    async def broadcast_message(self, message: Dict[str, Any], 
                              sender_id: Optional[str] = None) -> Dict[str, bool]:
        """
        Broadcast a message to all active agents.
        
        Args:
            message: Message to broadcast
            sender_id: ID of sending agent (excluded from broadcast)
            
        Returns:
            Dict mapping agent_id to success status
        """
        results = {}
        active_agents = self.get_active_agents()
        
        if sender_id:
            active_agents.discard(sender_id)
            
        for agent_id in active_agents:
            try:
                # In a real implementation, this would send the actual message
                # For now, we just simulate success
                results[agent_id] = True
                self.logger.debug(f"Broadcast message to {agent_id}: success")
            except Exception as e:
                results[agent_id] = False
                self.logger.error(f"Broadcast message to {agent_id}: failed - {e}")
                
        return results
    
    # ========================================
    # Compatibility Methods for fail_storm_runner
    # ========================================
    
    async def register_agent_async(self, agent) -> None:
        """Register agent with compatibility for BaseAgent objects (async version)."""
        if hasattr(agent, 'agent_id'):
            agent_id = agent.agent_id
            host = getattr(agent, 'host', '127.0.0.1')
            port = getattr(agent, 'port', 9000)
            protocol = getattr(agent, 'protocol', 'unknown')
            self.register_agent(agent_id, host, port, protocol)
        else:
            raise ValueError("Agent must have agent_id attribute")
    
    async def setup_mesh_topology(self) -> None:
        """Setup full mesh topology between all agents."""
        agent_ids = list(self.agents.keys())
        for i, agent1_id in enumerate(agent_ids):
            for j, agent2_id in enumerate(agent_ids):
                if i != j:
                    await self.connect_agents(agent1_id, agent2_id)
        self.logger.info(f"Mesh topology established: {len(agent_ids)} agents")
    
    def get_topology(self) -> Dict[str, Set[str]]:
        """Get current network topology as adjacency list."""
        return self.connections.copy()
    
    async def broadcast_init(self, document: Any, broadcaster_id: str) -> Dict[str, bool]:
        """Broadcast initialization document to all agents."""
        message = {
            "type": "document_broadcast",
            "document": document,
            "broadcaster": broadcaster_id
        }
        return await self.broadcast_message(message, broadcaster_id)
    
    async def _handle_node_failure(self, agent_id: str) -> None:
        """Handle detected node failure."""
        self._mark_agent_failed(agent_id)
        self.logger.info(f"Handled failure of agent: {agent_id}")
    
    def get_topology_health(self) -> Dict[str, Any]:
        """Get topology health information."""
        total_agents = len(self.agents)
        active_agents = len(self.get_active_agents())
        failed_agents = len(self.failed_agents)
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "failed_agents": failed_agents,
            "health_percentage": (active_agents / max(1, total_agents)) * 100,
            "connectivity": "full" if active_agents == total_agents else "partial"
        }
    
    async def wait_for_steady_state(self, min_stability_time: float = 5.0) -> float:
        """Wait for network to reach steady state."""
        start_time = time.time()
        last_change_time = start_time
        
        while time.time() - last_change_time < min_stability_time:
            await asyncio.sleep(1.0)
            # In a real implementation, we'd check for topology changes
            # For now, we just wait for the minimum time
            
        steady_time = time.time() - start_time
        self.logger.info(f"Steady state reached after {steady_time:.2f}s")
        return steady_time
    
    def get_failure_metrics(self) -> Dict[str, Any]:
        """Get failure-related metrics."""
        return {
            "total_failures": len(self.failed_agents),
            "recovered_agents": self.metrics["recovery_attempts"],
            "heartbeat_failures": self.metrics["heartbeat_failures"],
            "topology_health": self.get_topology_health()
        }
    
    async def cleanup(self) -> None:
        """Cleanup network resources."""
        await self.stop()
        self.logger.info("Network cleanup completed")
    
    # ========================================
    # Private Methods
    # ========================================
    
    async def _heartbeat_loop(self) -> None:
        """Main heartbeat monitoring loop."""
        while self._running:
            try:
                await self._check_heartbeats()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(1)  # Brief pause before retry
    
    async def _check_heartbeats(self) -> None:
        """Check all agents for heartbeat timeout."""
        current_time = time.time()
        
        for agent_id, agent_info in self.agents.items():
            if agent_info.status != AgentStatus.ACTIVE:
                continue
                
            time_since_heartbeat = current_time - agent_info.last_heartbeat
            
            if time_since_heartbeat > self.heartbeat_timeout:
                self._mark_agent_failed(agent_id)
                self.metrics["heartbeat_failures"] += 1
                self.logger.warning(
                    f"Agent {agent_id} failed heartbeat check "
                    f"(last seen: {time_since_heartbeat:.1f}s ago)"
                )
    
    def _mark_agent_failed(self, agent_id: str) -> None:
        """Mark an agent as failed and trigger callbacks."""
        if agent_id not in self.agents:
            return
            
        old_status = self.agents[agent_id].status
        self.agents[agent_id].status = AgentStatus.FAILED
        self.agents[agent_id].fail_count += 1
        
        # Update sets
        self.failed_agents.add(agent_id)
        self.recovering_agents.discard(agent_id)
        
        # Update metrics
        if old_status == AgentStatus.ACTIVE:
            self.metrics["active_agents"] -= 1
        self.metrics["failed_agents"] += 1
        
        # Trigger callback
        if self.on_agent_failed:
            try:
                self.on_agent_failed(agent_id)
            except Exception as e:
                self.logger.error(f"Error in agent_failed callback: {e}")
    
    def _mark_agent_recovered(self, agent_id: str) -> None:
        """Mark an agent as recovered and trigger callbacks."""
        if agent_id not in self.agents:
            return
            
        old_status = self.agents[agent_id].status
        self.agents[agent_id].status = AgentStatus.ACTIVE
        
        # Update sets
        self.failed_agents.discard(agent_id)
        self.recovering_agents.discard(agent_id)
        
        # Update metrics
        if old_status == AgentStatus.FAILED:
            self.metrics["failed_agents"] -= 1
        self.metrics["active_agents"] += 1
        self.metrics["recovery_attempts"] += 1
        
        # Trigger callback
        if self.on_agent_recovered:
            try:
                self.on_agent_recovered(agent_id)
            except Exception as e:
                self.logger.error(f"Error in agent_recovered callback: {e}")
                
        self.logger.info(f"Agent {agent_id} recovered successfully")


# Compatibility function for existing code
def create_mesh_network(heartbeat_interval: float = 5.0, 
                       heartbeat_timeout: float = 15.0,
                       debug_mode: bool = False) -> NativeMeshNetwork:
    """Create a native mesh network instance."""
    return NativeMeshNetwork(
        heartbeat_interval=heartbeat_interval,
        heartbeat_timeout=heartbeat_timeout, 
        debug_mode=debug_mode
    )
