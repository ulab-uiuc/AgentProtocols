# -*- coding: utf-8 -*-
"""
NetworkBase - Global scheduler and topology manager
现在的实现：
  * 仅登记 (agent_id -> endpoint)
  * 通信全部通过 BaseCommBackend（由子类/外部注入）
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any, Dict, Set, List, Optional

# 指标（如无可选依赖，可自行降级为 no-op）
try:
    from metrics import RECOVERY_TIME  # type: ignore
except Exception:
    class _Dummy:
        def observe(self, *_args, **_kwargs): ...
    RECOVERY_TIME = _Dummy()

# 仅依赖抽象通信后端
import sys
from pathlib import Path

# Add streaming_queue to path for imports
streaming_queue_path = Path(__file__).parent.parent
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))

try:
    from comm.base import BaseCommBackend  # 直接从 streaming_queue 根目录导入
except ImportError:
    # 兜底：使用绝对导入路径
    try:
        current_file = Path(__file__).resolve()
        comm_path = current_file.parent.parent / "comm"
        sys.path.insert(0, str(comm_path.parent))
        from comm.base import BaseCommBackend
    except ImportError as e:
        raise ImportError(f"Cannot import BaseCommBackend: {e}")


class NetworkBase:
    """Holds agents, their topology, and orchestrates traffic (protocol-agnostic)."""

    def __init__(self, comm_backend: BaseCommBackend):
        if comm_backend is None:
            raise ValueError("NetworkBase requires a concrete comm_backend.")
        # 仅维护 id -> endpoint，不保存 Agent 实例
        self._endpoints: Dict[str, str] = {}
        self._graph: Dict[str, Set[str]] = defaultdict(set)
        self._metrics: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._comm: BaseCommBackend = comm_backend

    # --------------------------- CRUD ---------------------------
    async def register_agent(self, agent_id: str, address: str) -> None:
        """注册 agent 的可达 endpoint（线程安全）。"""
        async with self._lock:
            if agent_id in self._endpoints:
                raise ValueError(f"Agent {agent_id} already exists.")
            self._endpoints[agent_id] = address
            self._graph[agent_id]  # ensure vertex exists
            await self._comm.register_endpoint(agent_id, address)
            print(f"Registered agent: {agent_id} @ {address}")

    async def unregister_agent(self, agent_id: str) -> None:
        """移除 agent 以及相关边。"""
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
        """创建一条有向边 src → dst（幂等）。"""
        async with self._lock:
            if src_id not in self._endpoints or dst_id not in self._endpoints:
                raise KeyError("Both agents must be registered first.")

            if dst_id not in self._graph[src_id]:
                self._graph[src_id].add(dst_id)

            # 可选：让通信后端做建连/预热
            await self._comm.connect(src_id, dst_id)
            print(f"Connected: {src_id} → {dst_id}")

    async def disconnect_agents(self, src_id: str, dst_id: str) -> None:
        """移除一条有向边 src → dst。"""
        async with self._lock:
            if src_id in self._graph:
                self._graph[src_id].discard(dst_id)

    async def kill_agents(self, agent_ids: Set[str]) -> None:
        """模拟故障，直接移除节点与边。"""
        self._metrics["failstorm_t0"] = time.time()
        async with self._lock:
            for aid in agent_ids:
                if aid in self._endpoints:
                    del self._endpoints[aid]
                if aid in self._graph:
                    del self._graph[aid]
                for nbrs in self._graph.values():
                    nbrs.discard(aid)
                print(f"Killed agent: {aid}")

    # --------------------------- topology management ---------------------------
    def get_topology(self) -> Dict[str, List[str]]:
        """返回当前网络拓扑。"""
        return {src: list(dsts) for src, dsts in self._graph.items()}

    def get_agents(self) -> List[str]:
        """返回所有已注册的 agent_id。"""
        return list(self._endpoints.keys())

    def setup_star_topology(self, center_id: str) -> None:
        """以 center_id 为中心建立星型拓扑。"""
        if center_id not in self._endpoints:
            raise KeyError(f"Center agent {center_id} not found.")

        for agent_id in self._endpoints:
            if agent_id != center_id:
                asyncio.create_task(self.connect_agents(agent_id, center_id))
                asyncio.create_task(self.connect_agents(center_id, agent_id))

    def setup_mesh_topology(self) -> None:
        """全互联拓扑。"""
        agent_ids = list(self._endpoints.keys())
        for src_id in agent_ids:
            for dst_id in agent_ids:
                if src_id != dst_id:
                    asyncio.create_task(self.connect_agents(src_id, dst_id))

    # --------------------------- messaging ---------------------------
    async def route_message(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """若拓扑允许，则通过通信后端转发。"""
        if dst_id not in self._graph.get(src_id, set()):
            raise PermissionError(f"{src_id} cannot reach {dst_id}.")
        return await self._comm.send(src_id, dst_id, payload)

    async def broadcast_message(
        self, src_id: str, payload: Dict[str, Any], exclude: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """对外广播（仅对有边的邻居）。"""
        if src_id not in self._endpoints:
            raise KeyError(f"Source agent {src_id} not found.")

        exclude = exclude or set()
        targets = self._graph.get(src_id, set()) - exclude
        results: Dict[str, Any] = {}
        for dst_id in targets:
            try:
                results[dst_id] = await self.route_message(src_id, dst_id, payload)
            except Exception as e:
                results[dst_id] = {"error": str(e)}
        return results

    # --------------------------- monitoring ---------------------------
    def snapshot_metrics(self) -> Dict[str, Any]:
        base_metrics = {
            "agent_count": len(self._endpoints),
            "edge_count": sum(len(edges) for edges in self._graph.values()),
            "topology": self.get_topology(),
        }
        base_metrics.update(self._metrics)
        return base_metrics

    def record_recovery(self) -> None:
        t0 = self._metrics.pop("failstorm_t0", None)
        if t0 is not None:
            recovery_time = time.time() - t0
            try:
                RECOVERY_TIME.observe(recovery_time)
            except Exception:
                pass
            print(f"Recovery completed in {recovery_time:.2f}s")

    # --------------------------- health checks ---------------------------
    async def health_check(self) -> Dict[str, bool]:
        status: Dict[str, bool] = {}
        for agent_id in list(self._endpoints.keys()):
            try:
                ok = await self._comm.health_check(agent_id)
                status[agent_id] = ok
            except Exception:
                status[agent_id] = False
        return status

    async def close(self) -> None:
        await self._comm.close()

    def __repr__(self) -> str:
        return f"NetworkBase(agents={len(self._endpoints)}, edges={sum(len(e) for e in self._graph.values())})"
