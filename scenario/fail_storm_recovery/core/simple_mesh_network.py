#!/usr/bin/env python3
"""
Simplified MeshNetwork for Fail-Storm Recovery scenarios.

This is a minimal implementation that doesn't depend on src/ components.
"""

import asyncio
import time
import json
import random
from typing import Dict, Set, List, Any, Optional
from collections import defaultdict


class SimpleMeshNetwork:
    """
    Simplified mesh network for fail-storm testing scenarios.
    
    This implementation focuses only on the basic network functionality
    needed for fail-storm recovery testing without complex dependencies.
    """
    
    def __init__(self, heartbeat_interval: float = 5.0):
        self.heartbeat_interval = heartbeat_interval
        self.agents: Dict[str, Any] = {}  # agent_id -> agent
        self.connections: Dict[str, Set[str]] = defaultdict(set)  # agent_id -> set of connected agents
        self.heartbeat_times: Dict[str, float] = {}  # agent_id -> last heartbeat time
        self.fault_injection_time: Optional[float] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def register_agent(self, agent: Any) -> None:
        """Register an agent with the network."""
        self.agents[agent.agent_id] = agent
        self.heartbeat_times[agent.agent_id] = time.time()
        
        if not self._running:
            self._running = True
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the network."""
        if agent_id in self.agents:
            del self.agents[agent_id]
        
        if agent_id in self.heartbeat_times:
            del self.heartbeat_times[agent_id]
        
        # Remove from all connections
        for connected_agents in self.connections.values():
            connected_agents.discard(agent_id)
        
        if agent_id in self.connections:
            del self.connections[agent_id]
    
    async def connect_agents(self, agent_id1: str, agent_id2: str) -> None:
        """Create a connection between two agents."""
        if agent_id1 in self.agents and agent_id2 in self.agents:
            self.connections[agent_id1].add(agent_id2)
            self.connections[agent_id2].add(agent_id1)
    
    async def disconnect_agents(self, agent_id1: str, agent_id2: str) -> None:
        """Remove connection between two agents."""
        self.connections[agent_id1].discard(agent_id2)
        self.connections[agent_id2].discard(agent_id1)
    
    async def setup_mesh_topology(self) -> None:
        """Setup full mesh topology between all agents."""
        agent_ids = list(self.agents.keys())
        
        for i, agent_id1 in enumerate(agent_ids):
            for j, agent_id2 in enumerate(agent_ids):
                if i != j:
                    await self.connect_agents(agent_id1, agent_id2)
    
    def get_topology(self) -> Dict[str, List[str]]:
        """Get current network topology."""
        return {agent_id: list(connections) for agent_id, connections in self.connections.items()}
    
    def get_topology_health(self) -> Dict[str, Any]:
        """Get topology health information."""
        alive_agents = []
        connectivity_status = {}
        
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'health_check'):
                # We can't easily do async health check here, so assume alive if registered
                alive_agents.append(agent_id)
                
                total_agents = len(self.agents) - 1  # Exclude self
                connected_count = len(self.connections.get(agent_id, set()))
                connectivity_ratio = connected_count / max(1, total_agents)
                
                connectivity_status[agent_id] = {
                    "connected_count": connected_count,
                    "total_possible": total_agents,
                    "connectivity_ratio": connectivity_ratio
                }
        
        return {
            "alive_agents": alive_agents,
            "connectivity_status": connectivity_status
        }
    
    async def broadcast_init(self, document: Dict[str, Any], broadcaster_id: str) -> Dict[str, Any]:
        """Broadcast initialization document to all agents."""
        results = {}
        
        for agent_id, agent in self.agents.items():
            if agent_id == broadcaster_id:
                continue
            
            try:
                # Simple broadcast - just record that we attempted it
                results[agent_id] = {"status": "broadcasted", "document_title": document.get("title", "Unknown")}
            except Exception as e:
                results[agent_id] = {"error": str(e)}
        
        return results
    
    def set_fault_injection_time(self, fault_time: float) -> None:
        """Set the fault injection timestamp."""
        self.fault_injection_time = fault_time
    
    async def _handle_node_failure(self, agent_id: str) -> None:
        """Handle node failure by updating connections."""
        # Remove failed agent from all connections
        for connected_agents in self.connections.values():
            connected_agents.discard(agent_id)
        
        if agent_id in self.connections:
            del self.connections[agent_id]
        
        if agent_id in self.heartbeat_times:
            del self.heartbeat_times[agent_id]
    
    async def _heartbeat_monitor(self) -> None:
        """Monitor agent heartbeats."""
        while self._running:
            try:
                current_time = time.time()
                
                for agent_id in list(self.heartbeat_times.keys()):
                    # Update heartbeat time for alive agents
                    if agent_id in self.agents:
                        agent = self.agents[agent_id]
                        if hasattr(agent, 'health_check'):
                            try:
                                # We'll assume the agent is alive if it's still registered
                                self.heartbeat_times[agent_id] = current_time
                            except Exception:
                                # Agent seems to be down
                                await self._handle_node_failure(agent_id)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue monitoring
                print(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(1.0)
    
    async def wait_for_steady_state(self, min_stability_time: float = 5.0) -> float:
        """Wait for network to reach steady state."""
        start_time = time.time()
        last_change_time = start_time
        
        prev_topology = self.get_topology()
        
        while time.time() - last_change_time < min_stability_time:
            await asyncio.sleep(1.0)
            
            current_topology = self.get_topology()
            if current_topology != prev_topology:
                last_change_time = time.time()
                prev_topology = current_topology
        
        return time.time() - start_time
    
    def get_failure_metrics(self) -> Dict[str, Any]:
        """Get failure and recovery metrics."""
        current_time = time.time()
        
        metrics = {
            "total_agents": len(self.agents),
            "active_connections": sum(len(connections) for connections in self.connections.values()) // 2,
            "fault_injection_time": self.fault_injection_time,
            "network_uptime": current_time - (self.fault_injection_time or current_time)
        }
        
        return metrics
    
    async def cleanup(self) -> None:
        """Cleanup network resources."""
        self._running = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        self.agents.clear()
        self.connections.clear()
        self.heartbeat_times.clear()

