#!/usr/bin/env python3
"""
MeshNetwork - 扩展的AgentNetwork，支持故障恢复和心跳检测

This module extends the base AgentNetwork to add:
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

# Add paths to import base components
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from src.core.network import AgentNetwork
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.core.base_agent import BaseAgent


class MeshNetwork(AgentNetwork):
    """
    Enhanced AgentNetwork with fault tolerance and recovery capabilities.
    
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
        self._debug_mode = debug_mode
        
        # Failure detection state
        self._heartbeat_tasks: Dict[str, asyncio.Task] = {}
        self._last_heartbeat: Dict[str, float] = {}
        self._failed_nodes: Set[str] = set()
        self._recovery_events: List[Dict[str, Any]] = []
        
        # Broadcast state
        self._broadcast_history: deque = deque(maxlen=100)
        self._gaia_document: Optional[Dict[str, Any]] = None
        
        # Metrics for fail-storm analysis
        self._fault_injection_time: Optional[float] = None
        self._first_recovery_time: Optional[float] = None
        self._topology_changes: List[Dict[str, Any]] = []
        self._message_stats = {
            "heartbeats_sent": 0,
            "heartbeats_received": 0,
            "reconnection_attempts": 0,
            "successful_reconnections": 0,
            "failed_reconnections": 0
        }

    async def register_agent(self, agent: BaseAgent) -> None:
        """Register agent and start heartbeat monitoring."""
        await super().register_agent(agent)
        
        # Start heartbeat for this agent
        await self._start_heartbeat_monitoring(agent.agent_id)
        
        # Record as alive
        self._last_heartbeat[agent.agent_id] = time.time()
        
        print(f"[MeshNetwork] Agent {agent.agent_id} registered with heartbeat monitoring")
        
        # If this is the second agent, start heartbeat monitoring for all agents
        if len(self._agents) == 2:
            for existing_agent_id in self._agents:
                if existing_agent_id not in self._heartbeat_tasks:
                    await self._start_heartbeat_monitoring(existing_agent_id)

    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister agent and stop heartbeat monitoring."""
        await self._stop_heartbeat_monitoring(agent_id)
        await super().unregister_agent(agent_id)
        
        # Clean up heartbeat state
        self._last_heartbeat.pop(agent_id, None)
        self._failed_nodes.discard(agent_id)
        
        print(f"[MeshNetwork] Agent {agent_id} unregistered and heartbeat stopped")

    async def _start_heartbeat_monitoring(self, agent_id: str) -> None:
        """Start heartbeat monitoring for a specific agent."""
        if agent_id in self._heartbeat_tasks:
            return  # Already monitoring
        
        # Skip heartbeat monitoring if there's only one agent
        if len(self._agents) <= 1:
            print(f"[MeshNetwork] Skipping heartbeat monitoring for {agent_id} (single agent)")
            return
        
        async def heartbeat_loop():
            """Heartbeat loop for monitoring agent health."""
            while agent_id in self._agents and agent_id not in self._failed_nodes:
                try:
                    # Send heartbeat to all neighbors
                    await self._send_heartbeat(agent_id)
                    
                    # Check for missed heartbeats from others
                    await self._check_heartbeat_timeouts()
                    
                    # Wait for next heartbeat
                    await asyncio.sleep(self.heartbeat_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"[MeshNetwork] Heartbeat error for {agent_id}: {e}")
                    await asyncio.sleep(1.0)  # Brief pause before retry
        
        # Start heartbeat task
        task = asyncio.create_task(heartbeat_loop())
        self._heartbeat_tasks[agent_id] = task

    async def _stop_heartbeat_monitoring(self, agent_id: str) -> None:
        """Stop heartbeat monitoring for a specific agent."""
        if agent_id in self._heartbeat_tasks:
            task = self._heartbeat_tasks.pop(agent_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _send_heartbeat(self, agent_id: str) -> None:
        """Send heartbeat message to all connected agents."""
        if agent_id not in self._agents:
            return
            
        # Create simple heartbeat payload - auto-detect protocol from agent
        current_time = time.time()
        
        # Try to detect if agent uses simple_json protocol
        agent = self._agents.get(agent_id)
        uses_simple_json = (agent and 
                           hasattr(agent, '_self_agent_card') and 
                           agent._self_agent_card and 
                           agent._self_agent_card.get('protocol') == 'simple_json')
        
        if uses_simple_json:
            # Simple JSON format
            heartbeat_payload = {
                "type": "heartbeat",
                "sender": agent_id,
                "timestamp": current_time,
                "topology_version": len(self._topology_changes),
                "text": f"Heartbeat from {agent_id} at {current_time}"
            }
        else:
            # A2A format for compatibility
            import uuid
            heartbeat_payload = {
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "system",
                    "parts": [
                        {
                            "type": "text",
                            "text": f"Heartbeat from {agent_id} at {current_time}"
                        }
                    ]
                },
                "context": {
                    "type": "heartbeat",
                    "sender": agent_id,
                    "timestamp": current_time,
                    "topology_version": len(self._topology_changes)
                },
                "source": agent_id
            }
        
        # Send to all neighbors in the graph
        neighbors = self._graph.get(agent_id, set())
        failed_neighbors = set()
        
        # Create a copy to avoid concurrent modification
        for neighbor_id in list(neighbors):
            if neighbor_id in self._failed_nodes:
                continue
                
            # Check if agent still exists and has outbound connections
            if agent_id not in self._agents:
                continue
                
            agent = self._agents[agent_id]
            if not hasattr(agent, '_outbound') or not agent._outbound:
                continue
                
            try:
                await self.route_message(agent_id, neighbor_id, heartbeat_payload)
                self._message_stats["heartbeats_sent"] += 1
                
            except Exception as e:
                # Only log as debug if it's a "No outbound adapter" error
                if "No outbound adapter found" in str(e):
                    # This is expected during disconnections, don't spam logs
                    pass
                else:
                    print(f"[MeshNetwork] Heartbeat failed {agent_id} -> {neighbor_id}: {e}")
                failed_neighbors.add(neighbor_id)
        
        # Mark failed neighbors for recovery
        for failed_id in failed_neighbors:
            await self._handle_node_failure(failed_id)

    async def _check_heartbeat_timeouts(self) -> None:
        """Check for nodes that have missed heartbeats with smart detection."""
        current_time = time.time()
        
        # Create a copy to avoid concurrent modification
        for agent_id, last_heartbeat in list(self._last_heartbeat.items()):
            if agent_id in self._failed_nodes:
                continue
                
            time_since_heartbeat = current_time - last_heartbeat
            
            # Use graduated failure detection instead of immediate timeout
            if time_since_heartbeat > self.heartbeat_timeout:
                # First attempt: try to verify failure with direct health check
                if await self._verify_agent_failure(agent_id):
                    await self._handle_node_failure(agent_id)
                else:
                    # Agent responded to health check, update heartbeat time
                    self._last_heartbeat[agent_id] = current_time
                    # 减少false alarm的噪音输出
                    if self._debug_mode:
                        print(f"[MeshNetwork] Agent {agent_id} recovered from timeout (false alarm)")
    
    async def _verify_agent_failure(self, agent_id: str) -> bool:
        """Verify if an agent is truly failed by attempting direct health check."""
        if agent_id not in self._agents:
            return True  # Agent not registered, consider failed
        
        agent = self._agents[agent_id]
        try:
            # Try direct health check
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://{agent.host}:{agent.port}/health")
                if response.status_code == 200:
                    # 减少健康检查的噪音输出，只在调试模式下显示
                    if self._debug_mode:
                        print(f"[MeshNetwork] Health check passed for {agent_id}")
                    return False  # Agent is alive
        except Exception as e:
            print(f"[MeshNetwork] Health check failed for {agent_id}: {e}")
        
        return True  # Confirmed failure

    async def _handle_node_failure(self, failed_agent_id: str) -> None:
        """Handle detection of a failed node."""
        if failed_agent_id in self._failed_nodes:
            return  # Already marked as failed
        
        print(f"[MeshNetwork] Detected failure of agent: {failed_agent_id}")
        
        # Mark as failed
        self._failed_nodes.add(failed_agent_id)
        
        # Record failure event
        failure_event = {
            "type": "node_failure",
            "agent_id": failed_agent_id,
            "timestamp": time.time(),
            "detected_by": "heartbeat_timeout"
        }
        self._recovery_events.append(failure_event)
        
        # Trigger topology recovery
        await self._trigger_topology_recovery(failed_agent_id)

    async def _trigger_topology_recovery(self, failed_agent_id: str) -> None:
        """Trigger automatic topology recovery after node failure."""
        print(f"[MeshNetwork] Starting topology recovery for failed agent: {failed_agent_id}")
        
        # Record start of recovery
        if self._first_recovery_time is None:
            self._first_recovery_time = time.time()
        
        # Get all remaining alive agents
        alive_agents = set(self._agents.keys()) - self._failed_nodes
        
        if len(alive_agents) < 2:
            print("[MeshNetwork] Not enough alive agents for mesh recovery")
            return
        
        # Rebuild mesh topology among surviving nodes
        recovery_tasks = []
        for src_id in alive_agents:
            for dst_id in alive_agents:
                if src_id != dst_id:
                    # Check if connection exists, if not, create it
                    if dst_id not in self._graph.get(src_id, set()):
                        recovery_tasks.append(self._attempt_reconnection(src_id, dst_id))
        
        if recovery_tasks:
            print(f"[MeshNetwork] Attempting {len(recovery_tasks)} reconnections...")
            self._message_stats["reconnection_attempts"] += len(recovery_tasks)
            
            # Execute reconnections with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*recovery_tasks, return_exceptions=True),
                    timeout=30.0
                )
                
                # Count successful reconnections
                successful = sum(1 for r in results if r is True)
                self._message_stats["successful_reconnections"] += successful
                self._message_stats["failed_reconnections"] += len(recovery_tasks) - successful
                
                print(f"[MeshNetwork] Recovery completed: {successful}/{len(recovery_tasks)} reconnections successful")
                
            except asyncio.TimeoutError:
                print("[MeshNetwork] Topology recovery timed out")
                self._message_stats["failed_reconnections"] += len(recovery_tasks)
        
        # Record topology change
        topology_change = {
            "type": "topology_recovery",
            "timestamp": time.time(),
            "failed_agent": failed_agent_id,
            "remaining_agents": len(alive_agents),
            "recovery_attempts": len(recovery_tasks)
        }
        self._topology_changes.append(topology_change)

    async def _attempt_reconnection(self, src_id: str, dst_id: str) -> bool:
        """Attempt to reconnect two agents."""
        try:
            await self.connect_agents(src_id, dst_id)
            print(f"[MeshNetwork] Reconnected: {src_id} -> {dst_id}")
            return True
        except Exception as e:
            print(f"[MeshNetwork] Reconnection failed {src_id} -> {dst_id}: {e}")
            return False

    def record_heartbeat_received(self, sender_id: str) -> None:
        """Record receipt of a heartbeat message."""
        self._last_heartbeat[sender_id] = time.time()
        self._message_stats["heartbeats_received"] += 1
    
    async def connect_agents(self, src_id: str, dst_id: str) -> bool:
        """Establish connection between two agents."""
        try:
            if src_id not in self._agents or dst_id not in self._agents:
                return False
            
            # Add to topology graph
            if src_id not in self._graph:
                self._graph[src_id] = set()
            self._graph[src_id].add(dst_id)
            
            print(f"[MeshNetwork] Connected: {src_id} → {dst_id}")
            return True
            
        except Exception as e:
            print(f"[MeshNetwork] Connection failed {src_id} → {dst_id}: {e}")
            return False
        
        # Remove from failed nodes if it was marked as failed
        if sender_id in self._failed_nodes:
            print(f"[MeshNetwork] Agent {sender_id} recovered from failure")
            self._failed_nodes.discard(sender_id)
            
            # Record recovery event
            recovery_event = {
                "type": "node_recovery",
                "agent_id": sender_id,
                "timestamp": time.time()
            }
            self._recovery_events.append(recovery_event)

    # ========================== Broadcast Functionality ==========================

    async def broadcast_init(self, gaia_document: Dict[str, Any], sender_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Broadcast Gaia document to all agents in the mesh network.
        
        Parameters
        ----------
        gaia_document : Dict[str, Any]
            The Gaia document to broadcast to all agents
        sender_id : Optional[str]
            Agent ID that initiates the broadcast (if None, uses first available agent)
            
        Returns
        -------
        Dict[str, Any]
            Results of broadcast delivery to each agent
        """
        # Store the Gaia document
        self._gaia_document = gaia_document
        
        # Choose sender if not specified
        if sender_id is None:
            available_agents = set(self._agents.keys()) - self._failed_nodes
            if not available_agents:
                raise RuntimeError("No available agents for broadcast")
            sender_id = list(available_agents)[0]
        
        if sender_id not in self._agents or sender_id in self._failed_nodes:
            raise ValueError(f"Sender agent {sender_id} is not available")
        
        # Create broadcast payload
        broadcast_payload = {
            "type": "gaia_document_init",
            "sender": sender_id,
            "timestamp": time.time(),
            "document": gaia_document,
            "broadcast_id": f"gaia_init_{int(time.time() * 1000)}"
        }
        
        # Record broadcast
        self._broadcast_history.append({
            "timestamp": time.time(),
            "sender": sender_id,
            "type": "gaia_document_init",
            "broadcast_id": broadcast_payload["broadcast_id"]
        })
        
        print(f"[MeshNetwork] Broadcasting Gaia document from {sender_id} to all agents...")
        
        # Broadcast to all agents except sender
        results = await self.broadcast_message(
            sender_id, 
            broadcast_payload, 
            exclude={sender_id}
        )
        
        print(f"[MeshNetwork] Gaia document broadcast completed. Results: {len(results)} deliveries")
        return results

    # ========================== Monitoring and Metrics ==========================

    def get_failure_metrics(self) -> Dict[str, Any]:
        """Get comprehensive failure and recovery metrics."""
        current_time = time.time()
        
        # Calculate recovery time
        recovery_time_ms = None
        if self._fault_injection_time and self._first_recovery_time:
            recovery_time_ms = (self._first_recovery_time - self._fault_injection_time) * 1000
        
        # Calculate current health
        alive_agents = set(self._agents.keys()) - self._failed_nodes
        total_possible_connections = len(alive_agents) * (len(alive_agents) - 1)
        actual_connections = sum(len(edges) for edges in self._graph.values())
        
        return {
            "recovery_time_ms": recovery_time_ms,
            "failed_nodes_count": len(self._failed_nodes),
            "alive_nodes_count": len(alive_agents),
            "topology_connectivity": actual_connections / max(total_possible_connections, 1),
            "recovery_events_count": len(self._recovery_events),
            "topology_changes_count": len(self._topology_changes),
            "message_stats": self._message_stats.copy(),
            "last_topology_change": self._topology_changes[-1] if self._topology_changes else None
        }

    def set_fault_injection_time(self, timestamp: float) -> None:
        """Record the time when fault injection occurred."""
        self._fault_injection_time = timestamp
        print(f"[MeshNetwork] Fault injection time recorded: {timestamp}")

    def get_topology_health(self) -> Dict[str, Any]:
        """Get current topology health status."""
        alive_agents = set(self._agents.keys()) - self._failed_nodes
        
        # Check connectivity for each alive agent
        connectivity_status = {}
        for agent_id in alive_agents:
            neighbors = self._graph.get(agent_id, set()) - self._failed_nodes
            expected_neighbors = len(alive_agents) - 1  # All others in mesh
            connectivity_status[agent_id] = {
                "connected_neighbors": len(neighbors),
                "expected_neighbors": expected_neighbors,
                "connectivity_ratio": len(neighbors) / max(expected_neighbors, 1)
            }
        
        return {
            "alive_agents": list(alive_agents),
            "failed_agents": list(self._failed_nodes),
            "connectivity_status": connectivity_status,
            "overall_health": len(alive_agents) / max(len(self._agents), 1)
        }

    async def wait_for_steady_state(self, min_stability_time: float = 10.0) -> float:
        """
        Wait for network to reach steady state after failures.
        
        Parameters
        ----------
        min_stability_time : float
            Minimum time (seconds) of stability required
            
        Returns
        -------
        float
            Time taken to reach steady state (seconds)
        """
        start_time = time.time()
        last_change_time = start_time
        
        print(f"[MeshNetwork] Waiting for steady state (min {min_stability_time}s stability)...")
        
        while True:
            await asyncio.sleep(1.0)
            current_time = time.time()
            
            # Check if there have been recent topology changes
            recent_changes = [
                change for change in self._topology_changes
                if current_time - change["timestamp"] < min_stability_time
            ]
            
            if not recent_changes:
                # No recent changes, check if we've waited long enough
                if current_time - last_change_time >= min_stability_time:
                    steady_time = current_time - start_time
                    print(f"[MeshNetwork] Steady state reached after {steady_time:.2f}s")
                    return steady_time
            else:
                # Reset timer due to recent changes
                last_change_time = recent_changes[-1]["timestamp"]

    async def cleanup(self) -> None:
        """Clean up all heartbeat tasks and resources."""
        print("[MeshNetwork] Cleaning up heartbeat monitoring...")
        
        # Cancel all heartbeat tasks
        cleanup_tasks = []
        for agent_id in list(self._heartbeat_tasks.keys()):
            cleanup_tasks.append(self._stop_heartbeat_monitoring(agent_id))
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Clear state
        self._heartbeat_tasks.clear()
        self._last_heartbeat.clear()
        self._failed_nodes.clear()
        
        print("[MeshNetwork] Cleanup completed")

    def __repr__(self) -> str:
        """Debug representation of MeshNetwork."""
        alive_count = len(set(self._agents.keys()) - self._failed_nodes)
        failed_count = len(self._failed_nodes)
        
        return (
            f"MeshNetwork(agents={len(self._agents)}, "
            f"alive={alive_count}, failed={failed_count}, "
            f"heartbeat_interval={self.heartbeat_interval}s)"
        )