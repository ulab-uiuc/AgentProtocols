"""
AgentNetwork - Global scheduler and topology manager
"""

import asyncio
import time
from collections import defaultdict
from typing import Any, Dict, Set, List, Optional

try:
    from src.core.base_agent import BaseAgent
    from src.core.metrics import RECOVERY_TIME
except ImportError:
    from .base_agent import BaseAgent
    from .metrics import RECOVERY_TIME


class AgentNetwork:
    """Holds agents, their topology, and orchestrates traffic."""

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._graph: Dict[str, Set[str]] = defaultdict(set)
        self._metrics: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    # --------------------------- CRUD ---------------------------
    async def register_agent(self, agent: BaseAgent) -> None:
        """Register a new agent instance (thread-safe)."""
        async with self._lock:
            if agent.agent_id in self._agents:
                raise ValueError(f"Agent {agent.agent_id} already exists.")
            self._agents[agent.agent_id] = agent
            self._graph[agent.agent_id]  # ensure vertex exists
            print(f"Registered agent: {agent.agent_id}")

    async def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent and all its connections."""
        async with self._lock:
            if agent_id not in self._agents:
                raise KeyError(f"Agent {agent_id} not found.")

            # Remove from agents dict
            del self._agents[agent_id]

            # Remove from graph
            del self._graph[agent_id]

            # Remove incoming edges
            for src_id in self._graph:
                self._graph[src_id].discard(agent_id)
                if agent_id in self._agents.get(src_id, {}).outgoing_edges:
                    self._agents[src_id].outgoing_edges.discard(agent_id)

    async def connect_agents(self, src_id: str, dst_id: str) -> None:
        """Create a directed edge src → dst (idempotent)."""
        async with self._lock:
            if src_id not in self._agents or dst_id not in self._agents:
                raise KeyError("Both agents must be registered first.")

            # Add to graph
            self._graph[src_id].add(dst_id)
            self._agents[src_id].outgoing_edges.add(dst_id)

            # DO NOT import or create protocol adapters here.
            # Outbound adapters are installed by the runner/coordinator.
            # This method only manages topology.
            
            # print(f"Connected: {src_id} → {dst_id}")

    async def disconnect_agents(self, src_id: str, dst_id: str) -> None:
        """Remove a directed edge src → dst."""
        async with self._lock:
            if src_id in self._graph:
                self._graph[src_id].discard(dst_id)

            if src_id in self._agents:
                src_agent = self._agents[src_id]
                src_agent.outgoing_edges.discard(dst_id)
                # Remove outbound adapter
                src_agent.remove_outbound_adapter(dst_id)

    async def kill_agents(self, agent_ids: Set[str]) -> None:
        """Simulate failure by removing agents & edges."""
        self._metrics["failstorm_t0"] = time.time()

        async with self._lock:
            for aid in agent_ids:
                if aid in self._agents:
                    del self._agents[aid]

                # Remove from graph
                if aid in self._graph:
                    del self._graph[aid]

                # Remove incoming edges
                for nbrs in self._graph.values():
                    nbrs.discard(aid)

                print(f"Killed agent: {aid}")

    # --------------------------- topology management ---------------------------
    def get_topology(self) -> Dict[str, List[str]]:
        """Return current network topology."""
        return {src: list(dsts) for src, dsts in self._graph.items()}

    def get_agents(self) -> Dict[str, BaseAgent]:
        """Return all registered agents."""
        return self._agents.copy()

    async def setup_star_topology(self, center_id: str) -> None:
        """Setup star topology with center_id as hub."""
        if center_id not in self._agents:
            raise KeyError(f"Center agent {center_id} not found.")

        tasks = []
        for agent_id in self._agents:
            if agent_id != center_id:
                # Connect periphery to center
                tasks.append(self.connect_agents(agent_id, center_id))
                # Connect center to periphery
                tasks.append(self.connect_agents(center_id, agent_id))

        # Wait for all connections to complete
        await asyncio.gather(*tasks)

    async def setup_mesh_topology(self) -> None:
        """Setup full mesh topology (all-to-all connections)."""
        agent_ids = list(self._agents.keys())
        tasks = []
        for src_id in agent_ids:
            for dst_id in agent_ids:
                if src_id != dst_id:
                    tasks.append(self.connect_agents(src_id, dst_id))

        # Wait for all connections to complete
        await asyncio.gather(*tasks)

    # --------------------------- messaging ---------------------------
    async def route_message(
        self,
        src_id: str,
        dst_id: str,
        payload: Dict[str, Any]
    ) -> Any:
        """Forward message if edge exists."""
        if dst_id not in self._graph.get(src_id, set()):
            raise PermissionError(f"{src_id} cannot reach {dst_id}.")

        src_agent = self._agents.get(src_id)
        if not src_agent:
            raise KeyError(f"Source agent {src_id} not found.")

        return await src_agent.send(dst_id, payload)

    async def broadcast_message(
        self,
        src_id: str,
        payload: Dict[str, Any],
        exclude: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """Broadcast message to all connected agents."""
        if src_id not in self._agents:
            raise KeyError(f"Source agent {src_id} not found.")

        exclude = exclude or set()
        targets = self._graph.get(src_id, set()) - exclude
        results = {}

        for dst_id in targets:
            try:
                result = await self.route_message(src_id, dst_id, payload)
                results[dst_id] = result
            except Exception as e:
                results[dst_id] = {"error": str(e)}

        return results

    # --------------------------- monitoring ---------------------------
    def snapshot_metrics(self) -> Dict[str, Any]:
        """Return current metrics dict."""
        base_metrics = {
            "agent_count": len(self._agents),
            "edge_count": sum(len(edges) for edges in self._graph.values()),
            "topology": self.get_topology()
        }
        base_metrics.update(self._metrics)
        return base_metrics

    def record_recovery(self) -> None:
        """Record successful recovery after failure."""
        t0 = self._metrics.pop("failstorm_t0", None)
        if t0 is not None:
            recovery_time = time.time() - t0
            RECOVERY_TIME.observe(recovery_time)
            print(f"Recovery completed in {recovery_time:.2f}s")

    # --------------------------- health checks ---------------------------
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all registered agents."""
        health_status = {}

        for agent_id, agent in self._agents.items():
            try:
                # Use agent's built-in health check method
                is_healthy = await agent.health_check()
                health_status[agent_id] = is_healthy
            except Exception:
                health_status[agent_id] = False

        return health_status

    def __repr__(self) -> str:
        return f"AgentNetwork(agents={len(self._agents)}, edges={sum(len(e) for e in self._graph.values())})"