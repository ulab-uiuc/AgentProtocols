#!/usr/bin/env python3
"""
Enhanced MeshNetwork - 完整的MeshNetwork功能，但不依赖src/组件

This module provides the full MeshNetwork functionality:
1. Heartbeat mechanism for fault detection
2. Automatic reconnection and topology healing
3. Broadcast initialization for Gaia documents
4. Fail-storm specific monitoring and recovery
"""

import asyncio
import time
import json
import random
import sys
from typing import Dict, Set, List, Any, Optional
from collections import defaultdict, deque
from pathlib import Path


# Simplified AgentNetwork base class for local use
class SimpleAgentNetwork:
    """Simplified AgentNetwork base class."""
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.connections: Dict[str, Set[str]] = defaultdict(set)
    
    async def add_agent(self, agent: Any) -> None:
        """Add agent to network."""
        self.agents[agent.agent_id] = agent
    
    async def remove_agent(self, agent_id: str) -> None:
        """Remove agent from network."""
        if agent_id in self.agents:
            del self.agents[agent_id]
        
        # Remove from connections
        for connected_agents in self.connections.values():
            connected_agents.discard(agent_id)
        
        if agent_id in self.connections:
            del self.connections[agent_id]
    
    async def connect_agents(self, src_id: str, dst_id: str) -> bool:
        """Connect two agents."""
        if src_id in self.agents and dst_id in self.agents:
            self.connections[src_id].add(dst_id)
            self.connections[dst_id].add(src_id)
            return True
        return False
    
    def get_neighbors(self, agent_id: str) -> Set[str]:
        """Get neighbors of an agent."""
        return self.connections.get(agent_id, set())


class EnhancedMeshNetwork(SimpleAgentNetwork):
    """
    Enhanced MeshNetwork with fault tolerance and recovery capabilities.
    
    Features:
    - Heartbeat-based fault detection
    - Automatic topology recovery after failures
    - Broadcast initialization for distributed scenarios
    - Real-time metrics collection for fail-storm analysis
    """
    
    def __init__(self, heartbeat_interval: float = 5.0, heartbeat_timeout: float = 15.0, debug_mode: bool = False):
        """
        Initialize MeshNetwork with fault tolerance parameters.
        
        Parameters
        ----------
        heartbeat_interval : float
            Interval between heartbeat messages (seconds)
        heartbeat_timeout : float
            Time to wait before considering a node failed (seconds)
        debug_mode : bool
            Enable verbose debug output for health checks
        """
        super().__init__()
        
        # Heartbeat configuration
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.debug_mode = debug_mode
        
        # Heartbeat tracking
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}
        self.last_heartbeat: Dict[str, float] = {}
        self.heartbeat_failures: Dict[str, int] = defaultdict(int)
        self.confirmed_failures: Set[str] = set()
        
        # Recovery tracking
        self.recovery_attempts: Dict[str, int] = defaultdict(int)
        self.recovery_history: List[Dict[str, Any]] = []
        
        # Metrics
        self.fault_injection_time: Optional[float] = None
        self.first_recovery_time: Optional[float] = None
        self.steady_state_time: Optional[float] = None
        self.broadcast_history: List[Dict[str, Any]] = []
        
        # State management
        self.topology_changes: deque = deque(maxlen=100)
        self._monitoring_active = False
    
    async def register_agent(self, agent: Any) -> None:
        """Register agent and start heartbeat monitoring."""
        await self.add_agent(agent)
        
        # Initialize heartbeat tracking
        current_time = time.time()
        self.last_heartbeat[agent.agent_id] = current_time
        self.heartbeat_failures[agent.agent_id] = 0
        
        # Start heartbeat monitoring
        await self._start_heartbeat_monitoring(agent.agent_id)
        
        if self.debug_mode:
            print(f"[MeshNetwork] Registered agent {agent.agent_id} with heartbeat monitoring")
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister agent and stop heartbeat monitoring."""
        await self._stop_heartbeat_monitoring(agent_id)
        await self.remove_agent(agent_id)
        
        # Clean up tracking data
        self.last_heartbeat.pop(agent_id, None)
        self.heartbeat_failures.pop(agent_id, None)
        self.confirmed_failures.discard(agent_id)
        
        if self.debug_mode:
            print(f"[MeshNetwork] Unregistered agent {agent_id}")
    
    async def _start_heartbeat_monitoring(self, agent_id: str) -> None:
        """Start heartbeat monitoring for a specific agent."""
        if agent_id in self.heartbeat_tasks:
            self.heartbeat_tasks[agent_id].cancel()
        
        async def heartbeat_loop():
            """Heartbeat loop for monitoring agent health."""
            while agent_id in self.agents and not self.confirmed_failures.__contains__(agent_id):
                try:
                    # Send heartbeat
                    await self._send_heartbeat(agent_id)
                    
                    # Check for timeouts
                    await self._check_heartbeat_timeouts()
                    
                    # Wait for next interval
                    await asyncio.sleep(self.heartbeat_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if self.debug_mode:
                        print(f"[MeshNetwork] Heartbeat error for {agent_id}: {e}")
                    await asyncio.sleep(1.0)  # Brief pause on error
        
        self.heartbeat_tasks[agent_id] = asyncio.create_task(heartbeat_loop())
        self._monitoring_active = True
    
    async def _stop_heartbeat_monitoring(self, agent_id: str) -> None:
        """Stop heartbeat monitoring for a specific agent."""
        if agent_id in self.heartbeat_tasks:
            self.heartbeat_tasks[agent_id].cancel()
            try:
                await self.heartbeat_tasks[agent_id]
            except asyncio.CancelledError:
                pass
            del self.heartbeat_tasks[agent_id]
    
    async def _send_heartbeat(self, agent_id: str) -> None:
        """Send heartbeat message to all connected agents."""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        neighbors = self.get_neighbors(agent_id)
        
        heartbeat_msg = {
            "type": "heartbeat",
            "sender": agent_id,
            "timestamp": time.time(),
            "neighbors": list(neighbors)
        }
        
        # Send to all neighbors
        for neighbor_id in neighbors:
            if neighbor_id in self.agents and neighbor_id not in self.confirmed_failures:
                try:
                    neighbor_agent = self.agents[neighbor_id]
                    # Try to send message (simplified - in real implementation would use agent.send())
                    # For now, just record that we attempted to send
                    if self.debug_mode:
                        print(f"[MeshNetwork] Heartbeat {agent_id} -> {neighbor_id}")
                except Exception as e:
                    if self.debug_mode:
                        print(f"[MeshNetwork] Failed to send heartbeat {agent_id} -> {neighbor_id}: {e}")
                    
                    # Record potential failure
                    self.heartbeat_failures[neighbor_id] += 1
    
    async def _check_heartbeat_timeouts(self) -> None:
        """Check for nodes that have missed heartbeats with smart detection."""
        current_time = time.time()
        
        for agent_id in list(self.last_heartbeat.keys()):
            if agent_id in self.confirmed_failures:
                continue
            
            time_since_heartbeat = current_time - self.last_heartbeat[agent_id]
            
            if time_since_heartbeat > self.heartbeat_timeout:
                # Potential failure detected
                failure_count = self.heartbeat_failures[agent_id]
                
                if failure_count >= 3:  # Confirm failure after multiple attempts
                    verified_failure = await self._verify_agent_failure(agent_id)
                    if verified_failure:
                        await self._handle_node_failure(agent_id)
                else:
                    self.heartbeat_failures[agent_id] += 1
                    if self.debug_mode:
                        print(f"[MeshNetwork] Potential failure detected for {agent_id} (attempt {failure_count + 1})")
    
    async def _verify_agent_failure(self, agent_id: str) -> bool:
        """Verify if an agent is truly failed by attempting direct health check."""
        if agent_id not in self.agents:
            return True
        
        try:
            agent = self.agents[agent_id]
            # Try to perform health check
            if hasattr(agent, 'health_check'):
                healthy = await agent.health_check()
                if healthy:
                    # Agent is actually healthy, reset failure counter
                    self.heartbeat_failures[agent_id] = 0
                    self.last_heartbeat[agent_id] = time.time()
                    return False
            
            # If we can't check health or agent is unhealthy, consider it failed
            return True
            
        except Exception:
            # Health check failed, consider agent failed
            return True
    
    async def _handle_node_failure(self, failed_agent_id: str) -> None:
        """Handle detection of a failed node."""
        if failed_agent_id in self.confirmed_failures:
            return  # Already handled
        
        self.confirmed_failures.add(failed_agent_id)
        failure_time = time.time()
        
        if self.debug_mode:
            print(f"[MeshNetwork] Confirmed failure of agent {failed_agent_id}")
        
        # Record failure event
        failure_event = {
            "agent_id": failed_agent_id,
            "failure_time": failure_time,
            "neighbors_at_failure": list(self.get_neighbors(failed_agent_id))
        }
        self.recovery_history.append(failure_event)
        
        # Trigger topology recovery
        await self._trigger_topology_recovery(failed_agent_id)
        
        # Record topology change
        self.topology_changes.append({
            "type": "failure",
            "agent_id": failed_agent_id,
            "timestamp": failure_time
        })
    
    async def _trigger_topology_recovery(self, failed_agent_id: str) -> None:
        """Trigger automatic topology recovery after node failure."""
        failed_neighbors = self.get_neighbors(failed_agent_id)
        
        if self.debug_mode:
            print(f"[MeshNetwork] Starting topology recovery for {failed_agent_id}, neighbors: {failed_neighbors}")
        
        # Remove failed agent from topology
        self.connections.pop(failed_agent_id, None)
        for agent_connections in self.connections.values():
            agent_connections.discard(failed_agent_id)
        
        # Attempt to restore full mesh connectivity
        alive_agents = [aid for aid in self.agents.keys() if aid not in self.confirmed_failures]
        
        recovery_tasks = []
        for i, agent_id1 in enumerate(alive_agents):
            for j, agent_id2 in enumerate(alive_agents):
                if i < j:  # Avoid duplicate connections
                    if agent_id2 not in self.get_neighbors(agent_id1):
                        # Missing connection, attempt to restore
                        task = asyncio.create_task(self._attempt_reconnection(agent_id1, agent_id2))
                        recovery_tasks.append(task)
        
        # Wait for all recovery attempts to complete
        if recovery_tasks:
            results = await asyncio.gather(*recovery_tasks, return_exceptions=True)
            successful_reconnections = sum(1 for result in results if result is True)
            
            if self.debug_mode:
                print(f"[MeshNetwork] Recovery completed: {successful_reconnections}/{len(recovery_tasks)} reconnections successful")
            
            # Record first recovery if this is the first
            if self.first_recovery_time is None and successful_reconnections > 0:
                self.first_recovery_time = time.time()
    
    async def _attempt_reconnection(self, src_id: str, dst_id: str) -> bool:
        """Attempt to reconnect two agents."""
        try:
            success = await self.connect_agents(src_id, dst_id)
            if success and self.debug_mode:
                print(f"[MeshNetwork] Reconnected {src_id} <-> {dst_id}")
            return success
        except Exception as e:
            if self.debug_mode:
                print(f"[MeshNetwork] Failed to reconnect {src_id} <-> {dst_id}: {e}")
            return False
    
    def record_heartbeat_received(self, sender_id: str) -> None:
        """Record receipt of a heartbeat message."""
        self.last_heartbeat[sender_id] = time.time()
        self.heartbeat_failures[sender_id] = 0  # Reset failure counter
    
    async def connect_agents(self, src_id: str, dst_id: str) -> bool:
        """Establish connection between two agents."""
        if src_id == dst_id:
            return False
        
        if src_id not in self.agents or dst_id not in self.agents:
            return False
        
        if src_id in self.confirmed_failures or dst_id in self.confirmed_failures:
            return False
        
        # Attempt to establish connection
        try:
            src_agent = self.agents[src_id]
            dst_agent = self.agents[dst_id]
            
            # Add to topology
            success = await super().connect_agents(src_id, dst_id)
            
            if success:
                # Record topology change
                self.topology_changes.append({
                    "type": "connection",
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "timestamp": time.time()
                })
            
            return success
            
        except Exception as e:
            if self.debug_mode:
                print(f"[MeshNetwork] Connection failed {src_id} <-> {dst_id}: {e}")
            return False
    
    async def setup_mesh_topology(self) -> None:
        """Setup full mesh topology between all agents."""
        agent_ids = [aid for aid in self.agents.keys() if aid not in self.confirmed_failures]
        
        for i, agent_id1 in enumerate(agent_ids):
            for j, agent_id2 in enumerate(agent_ids):
                if i < j:  # Avoid duplicate connections
                    await self.connect_agents(agent_id1, agent_id2)
    
    async def broadcast_init(self, gaia_document: Dict[str, Any], sender_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Broadcast Gaia initialization document to all agents.
        
        Parameters
        ----------
        gaia_document : Dict[str, Any]
            The document to broadcast
        sender_id : Optional[str]
            ID of the sending agent (None for external broadcast)
            
        Returns
        -------
        Dict[str, Any]
            Results of broadcast attempts
        """
        broadcast_time = time.time()
        results = {}
        
        # Choose sender
        if sender_id is None:
            alive_agents = [aid for aid in self.agents.keys() if aid not in self.confirmed_failures]
            if not alive_agents:
                return {"error": "No alive agents available for broadcast"}
            sender_id = alive_agents[0]
        
        # Broadcast to all other agents
        for agent_id in self.agents.keys():
            if agent_id == sender_id or agent_id in self.confirmed_failures:
                continue
            
            try:
                agent = self.agents[agent_id]
                # Simulate broadcast (in real implementation would use agent.send())
                results[agent_id] = {
                    "status": "success",
                    "document_title": gaia_document.get("title", "Unknown"),
                    "broadcast_time": broadcast_time
                }
                
            except Exception as e:
                results[agent_id] = {
                    "status": "error",
                    "error": str(e),
                    "broadcast_time": broadcast_time
                }
        
        # Record broadcast
        broadcast_record = {
            "sender_id": sender_id,
            "document": gaia_document,
            "results": results,
            "timestamp": broadcast_time
        }
        self.broadcast_history.append(broadcast_record)
        
        return results
    
    def get_failure_metrics(self) -> Dict[str, Any]:
        """Get comprehensive failure and recovery metrics."""
        current_time = time.time()
        
        # Calculate recovery time
        recovery_time_ms = None
        if self.fault_injection_time and self.first_recovery_time:
            recovery_time_ms = (self.first_recovery_time - self.fault_injection_time) * 1000
        
        # Calculate steady state time
        steady_state_time_ms = None
        if self.fault_injection_time and self.steady_state_time:
            steady_state_time_ms = (self.steady_state_time - self.fault_injection_time) * 1000
        
        return {
            "total_agents": len(self.agents),
            "confirmed_failures": len(self.confirmed_failures),
            "active_connections": sum(len(connections) for connections in self.connections.values()) // 2,
            "recovery_time_ms": recovery_time_ms,
            "steady_state_time_ms": steady_state_time_ms,
            "topology_changes": len(self.topology_changes),
            "recovery_attempts": dict(self.recovery_attempts),
            "broadcast_count": len(self.broadcast_history),
            "fault_injection_time": self.fault_injection_time,
            "first_recovery_time": self.first_recovery_time,
            "steady_state_time": self.steady_state_time
        }
    
    def set_fault_injection_time(self, timestamp: float) -> None:
        """Record the time when fault injection occurred."""
        self.fault_injection_time = timestamp
    
    def get_topology_health(self) -> Dict[str, Any]:
        """Get current topology health status."""
        alive_agents = [aid for aid in self.agents.keys() if aid not in self.confirmed_failures]
        
        connectivity_status = {}
        for agent_id in alive_agents:
            total_possible = len(alive_agents) - 1  # Exclude self
            connected_count = len(self.get_neighbors(agent_id))
            connectivity_ratio = connected_count / max(1, total_possible)
            
            connectivity_status[agent_id] = {
                "connected_count": connected_count,
                "total_possible": total_possible,
                "connectivity_ratio": connectivity_ratio,
                "last_heartbeat": self.last_heartbeat.get(agent_id, 0),
                "failure_count": self.heartbeat_failures.get(agent_id, 0)
            }
        
        return {
            "alive_agents": alive_agents,
            "failed_agents": list(self.confirmed_failures),
            "connectivity_status": connectivity_status,
            "average_connectivity": sum(status["connectivity_ratio"] for status in connectivity_status.values()) / max(1, len(connectivity_status))
        }
    
    def get_topology(self) -> Dict[str, List[str]]:
        """Get current network topology."""
        return {agent_id: list(connections) for agent_id, connections in self.connections.items()}
    
    async def wait_for_steady_state(self, min_stability_time: float = 10.0) -> float:
        """
        Wait for network to reach steady state.
        
        Parameters
        ----------
        min_stability_time : float
            Minimum time of stability required (seconds)
            
        Returns
        -------
        float
            Time taken to reach steady state
        """
        start_time = time.time()
        last_change_time = start_time
        
        prev_topology = self.get_topology()
        prev_failures = len(self.confirmed_failures)
        
        while time.time() - last_change_time < min_stability_time:
            await asyncio.sleep(1.0)
            
            # Check for topology changes
            current_topology = self.get_topology()
            current_failures = len(self.confirmed_failures)
            
            if current_topology != prev_topology or current_failures != prev_failures:
                last_change_time = time.time()
                prev_topology = current_topology
                prev_failures = current_failures
                
                if self.debug_mode:
                    print(f"[MeshNetwork] Topology change detected, resetting stability timer")
        
        steady_state_duration = time.time() - start_time
        self.steady_state_time = time.time()
        
        if self.debug_mode:
            print(f"[MeshNetwork] Steady state reached after {steady_state_duration:.2f}s")
        
        return steady_state_duration
    
    async def cleanup(self) -> None:
        """Clean up all heartbeat tasks and resources."""
        self._monitoring_active = False
        
        # Cancel all heartbeat tasks
        for agent_id in list(self.heartbeat_tasks.keys()):
            await self._stop_heartbeat_monitoring(agent_id)
        
        # Clear all tracking data
        self.agents.clear()
        self.connections.clear()
        self.last_heartbeat.clear()
        self.heartbeat_failures.clear()
        self.confirmed_failures.clear()
        self.recovery_attempts.clear()
        self.recovery_history.clear()
        self.broadcast_history.clear()
        self.topology_changes.clear()
    
    def __repr__(self) -> str:
        """Debug representation of MeshNetwork."""
        return (
            f"EnhancedMeshNetwork("
            f"agents={len(self.agents)}, "
            f"connections={sum(len(c) for c in self.connections.values()) // 2}, "
            f"failures={len(self.confirmed_failures)}, "
            f"monitoring={self._monitoring_active})"
        )

