# -*- coding: utf-8 -*-
"""
NetworkBase - Protocol-agnostic network manager for privacy testing
Manages agent topology and routing without protocol-specific logic.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any, Dict, Set, List, Optional

# Import communication backend interface
try:
    from ..comm.base import BaseCommBackend
except ImportError:
    try:
        from comm.base import BaseCommBackend
    except ImportError:
        from script.safety_tech.comm.base import BaseCommBackend


class NetworkBase:
    """Protocol-agnostic network manager for privacy testing agents."""

    def __init__(self, comm_backend: BaseCommBackend):
        if comm_backend is None:
            raise ValueError("NetworkBase requires a concrete comm_backend.")
        
        # Only maintain id -> endpoint mapping, no Agent instances
        self._endpoints: Dict[str, str] = {}
        self._graph: Dict[str, Set[str]] = defaultdict(set)
        self._metrics: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._comm: BaseCommBackend = comm_backend

    # --------------------------- Agent Management ---------------------------
    async def register_agent(self, agent_id: str, address: str) -> None:
        """Register agent's reachable endpoint (thread-safe)."""
        async with self._lock:
            if agent_id in self._endpoints:
                raise ValueError(f"Agent {agent_id} already exists.")
            self._endpoints[agent_id] = address
            self._graph[agent_id]  # ensure vertex exists
            await self._comm.register_endpoint(agent_id, address)
            print(f"[NetworkBase] Registered agent: {agent_id} @ {address}")

    async def unregister_agent(self, agent_id: str) -> None:
        """Remove agent and related edges."""
        async with self._lock:
            if agent_id not in self._endpoints:
                raise KeyError(f"Agent {agent_id} not found.")

            del self._endpoints[agent_id]
            if agent_id in self._graph:
                del self._graph[agent_id]

            # Remove incoming edges
            for src_id in list(self._graph.keys()):
                self._graph[src_id].discard(agent_id)

    async def connect_agents(self, src_id: str, dst_id: str) -> None:
        """Create directed edge src → dst (idempotent)."""
        async with self._lock:
            if src_id not in self._endpoints or dst_id not in self._endpoints:
                raise KeyError("Both agents must be registered first.")

            if dst_id not in self._graph[src_id]:
                self._graph[src_id].add(dst_id)
                await self._comm.connect(src_id, dst_id)

    # --------------------------- Topology Management ---------------------------
    def setup_star_topology(self, center_id: str) -> None:
        """Create star topology with center_id as hub."""
        if center_id not in self._endpoints:
            raise KeyError(f"Center agent {center_id} not registered.")
        
        # Connect center to all other agents (bidirectional)
        for agent_id in self._endpoints:
            if agent_id != center_id:
                self._graph[center_id].add(agent_id)
                self._graph[agent_id].add(center_id)
        
        print(f"[NetworkBase] Star topology created with center: {center_id}")

    def setup_mesh_topology(self) -> None:
        """Create full mesh topology (all agents connected to all)."""
        agent_ids = list(self._endpoints.keys())
        for src_id in agent_ids:
            for dst_id in agent_ids:
                if src_id != dst_id:
                    self._graph[src_id].add(dst_id)
        
        print(f"[NetworkBase] Mesh topology created with {len(agent_ids)} agents")

    # --------------------------- Communication ---------------------------
    async def route_message(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Route message from src to dst through communication backend."""
        if src_id not in self._endpoints:
            raise KeyError(f"Source agent {src_id} not registered.")
        if dst_id not in self._endpoints:
            raise KeyError(f"Destination agent {dst_id} not registered.")

        try:
            response = await self._comm.send(src_id, dst_id, payload)
            return response
        except Exception as e:
            print(f"[NetworkBase] Message routing failed {src_id} -> {dst_id}: {e}")
            return None

    async def broadcast_message(self, src_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Broadcast message from src to all connected agents."""
        if src_id not in self._endpoints:
            raise KeyError(f"Source agent {src_id} not registered.")

        results = {}
        neighbors = self._graph.get(src_id, set())
        
        for dst_id in neighbors:
            try:
                response = await self._comm.send(src_id, dst_id, payload)
                results[dst_id] = response
            except Exception as e:
                print(f"[NetworkBase] Broadcast failed {src_id} -> {dst_id}: {e}")
                results[dst_id] = None

        return results

    # --------------------------- Health & Monitoring ---------------------------
    async def health_check(self, agent_id: str) -> bool:
        """Check if specific agent is healthy."""
        if agent_id not in self._endpoints:
            return False
        
        try:
            return await self._comm.health_check(agent_id)
        except Exception as e:
            print(f"[NetworkBase] Health check failed for {agent_id}: {e}")
            return False

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all registered agents."""
        results = {}
        for agent_id in self._endpoints:
            results[agent_id] = await self.health_check(agent_id)
        return results

    # --------------------------- Information ---------------------------
    def get_agents(self) -> List[str]:
        """Get list of all registered agent IDs."""
        return list(self._endpoints.keys())

    def get_neighbors(self, agent_id: str) -> Set[str]:
        """Get neighbors of specified agent."""
        return self._graph.get(agent_id, set()).copy()

    def get_topology_info(self) -> Dict[str, Any]:
        """Get current topology information."""
        return {
            "agents": list(self._endpoints.keys()),
            "edges": {src: list(dsts) for src, dsts in self._graph.items()},
            "total_agents": len(self._endpoints),
            "total_edges": sum(len(dsts) for dsts in self._graph.values())
        }

    # --------------------------- Cleanup ---------------------------
    async def close(self) -> None:
        """Close network and underlying communication backend."""
        await self._comm.close()
        self._endpoints.clear()
        self._graph.clear()
        print("[NetworkBase] Network closed")

