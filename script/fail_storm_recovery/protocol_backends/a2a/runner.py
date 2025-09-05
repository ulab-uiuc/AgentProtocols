#!/usr/bin/env python3
"""
A2A protocol runner for Fail-Storm Recovery scenario.

This module implements the A2A (Agent-to-Agent) protocol specific functionality
while inheriting all core logic from the base runner.
"""

import asyncio
import json
import os
import random
import signal
import time
from typing import Dict, Any, Optional, Set, List
from pathlib import Path
from dataclasses import dataclass
import sys

# Add paths for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.simple_base_agent import SimpleBaseAgent as BaseAgent
from protocol_backends.base_runner import FailStormRunnerBase

# Import shard_qa components dynamically to avoid circular imports
shard_qa_path = Path(__file__).parent.parent.parent / "shard_qa"
sys.path.insert(0, str(shard_qa_path))
import importlib.util
spec = importlib.util.spec_from_file_location("agent_executor", shard_qa_path / "shard_worker" / "agent_executor.py")
agent_executor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_executor_module)
ShardWorkerExecutor = agent_executor_module.ShardWorkerExecutor


@dataclass
class A2AAgentProc:
    """Track a single agent process and its runtime info."""
    agent_id: str
    port: int
    ws_port: int
    workspace: Path
    process: Optional[asyncio.subprocess.Process] = None
    alive: bool = False
    start_ts: float = 0.0


class A2ARunner(FailStormRunnerBase):
    """
    A2A protocol runner.
    
    Implements protocol-specific agent creation and management for A2A protocol
    while inheriting all core Fail-Storm functionality from FailStormRunnerBase.
    
    Responsibilities:
    1) Spawn N A2A agents as local processes (single host, unique ports)
    2) Build mesh topology via A2A peer handshake
    3) Broadcast initial Gaia doc
    4) Execute baseline QA workload
    5) Inject a fail-storm by SIGKILL subset of agents
    6) Monitor recovery: re-handshake, task re-publish, success-rate trend
    7) Persist metrics to results JSONs
    """

    def __init__(self, config_path: str = "config.yaml"):
        # If using default config, try protocol-specific config first
        if config_path == "config.yaml":
            protocol_config = Path(__file__).parent / "config.yaml"
            if protocol_config.exists():
                config_path = str(protocol_config)
        
        super().__init__(config_path)
        
        # Ensure protocol is set correctly in config
        if "scenario" not in self.config:
            self.config["scenario"] = {}
        self.config["scenario"]["protocol"] = "a2a"

        # Read network and protocol params
        net = self.config.get("network", {})
        self.base_port = int(net.get("base_port", 9000))
        self.base_ws_port = int(net.get("base_ws_port", 29000))
        self.heartbeat_interval = float(net.get("heartbeat_interval", 5.0))
        self.heartbeat_timeout = float(net.get("heartbeat_timeout", 12.0))
        self.connect_timeout = float(net.get("connection_timeout", 10.0))

        self.a2a_conf = self.config.get("a2a", {})
        self.agent_start_cmd = self.a2a_conf.get(
            "agent_start_cmd",
            # === 替换点 #1：填入你实际的 A2A Agent 启动脚本 ===
            # e.g. ["python", "script/a2a/agent.py", "--port", "{port}", "--ws-port", "{ws_port}", "--id", "{agent_id}", "--workspace", "{ws}"]
            ["python", "local_deps/a2a_agent.py", "--port", "{port}", "--ws-port", "{ws_port}", "--id", "{agent_id}", "--workspace", "{ws}"]
        )
        # === 替换点 #2：填入存活检查的 HTTP/WS 端点路径 ===
        self.health_path = self.a2a_conf.get("health_path", "/healthz")
        # === 替换点 #3：填入"建立邻居/握手"的 API 路径 ===
        self.peer_add_path = self.a2a_conf.get("peer_add_path", "/mesh/add_peer")
        self.broadcast_path = self.a2a_conf.get("broadcast_path", "/mesh/broadcast")  # for Gaia doc
        self.qa_path = self.a2a_conf.get("qa_path", "/qa/submit")  # shard QA submit

        # Book-keeping for subprocess mode
        self.agent_processes: Dict[str, A2AAgentProc] = {}
        self.killed_agents: Set[str] = set()
        self.mesh_built: bool = False

        # Result fields (example)
        self.recovery_first_ok_ts: Optional[float] = None
        self.steady_state_ts: Optional[float] = None
        
        self.output.info("Initialized A2A protocol runner")

    # ========================================
    # Protocol-Specific Implementation
    # ========================================

    async def create_agent(self, agent_id: str, host: str, port: int, executor: ShardWorkerExecutor) -> BaseAgent:
        """Create agent using A2A protocol."""
        try:
            # A2A uses subprocess-based agents
            self.output.progress(f"Setting up A2A agent {agent_id}...")
            
            # Create agent using A2A protocol
            agent = await BaseAgent.create_a2a(
                agent_id=agent_id,
                host=host,
                port=port,
                executor=executor
            )
            
            self.output.success(f"A2A agent {agent_id} created successfully")
            return agent
            
        except Exception as e:
            self.output.error(f"Failed to create A2A agent {agent_id}: {e}")
            raise

    def get_protocol_info(self, agent_id: str, port: int, data_file: str) -> str:
        """Get A2A protocol display information."""
        return f"🚀 [A2A] Created {agent_id} - HTTP: {port}, WebSocket: {self.base_ws_port}, Data: {data_file}"

    def get_reconnection_info(self, agent_id: str, port: int) -> List[str]:
        """Get A2A protocol reconnection information."""
        return [
            f"🔗 [A2A] Agent {agent_id} RECONNECTED on port {port}",
            f"📡 [A2A] WebSocket endpoint: ws://127.0.0.1:{self.base_ws_port}",
            f"🌐 [A2A] HTTP REST API: http://127.0.0.1:{port}",
            f"✅ [A2A] A2A protocol active"
        ]

    # ========================================
    # A2A-Specific Methods (继承的原有实现)
    # ========================================
    
    async def _old_create_agent(self, agent_id: str, host: str, port: int, executor: ShardWorkerExecutor) -> BaseAgent:
        """Legacy A2A agent creation method."""
        # For compatibility with base runner, we use BaseAgent.create_a2a
        # The subprocess management is handled separately in create_agent_subprocess
        return await BaseAgent.create_a2a(
            agent_id=agent_id,
            host=host,
            port=port,
            executor=executor
        )
    
    def get_protocol_info(self, agent_id: str, port: int, data_file: str) -> str:
        """Get A2A protocol display information."""
        ws_port = self.base_ws_port + int(agent_id.split("_")[-1]) if "_" in agent_id else self.base_ws_port
        return f"🚀 [A2A] Created {agent_id} - HTTP: {port}, WebSocket: {ws_port}, Data: {data_file}"
    
    def get_reconnection_info(self, agent_id: str, port: int) -> List[str]:
        """Get A2A protocol reconnection information."""
        return [
            f"🔄 [A2A] Reconnecting {agent_id} via peer re-registration",
            f"🔗 [A2A] Re-establishing mesh connections on port {port}",
            f"⚡ [A2A] Using missed-heartbeat detector for fault detection"
        ]

    # ========================================
    # A2A-Specific Subprocess Management
    # ========================================

    async def create_agent_subprocess(self, agent_id: str, port: int, data_file: str) -> bool:
        """
        Spawn an A2A agent as a local subprocess with dedicated ports and workspace.

        Args:
            agent_id: unique id
            port: HTTP port for REST (liveness, control)
            data_file: path to agent's local QA shard dataset
        """
        try:
            workspace = Path(self.workspace_dir) / agent_id
            workspace.mkdir(parents=True, exist_ok=True)

            ws_port = self.base_ws_port + int(agent_id.split("_")[-1]) if "_" in agent_id else self.base_ws_port
            cmd = [s.format(
                port=port,
                ws_port=ws_port,
                agent_id=agent_id,
                ws=str(workspace),
                data=str(data_file)
            ) for s in self.agent_start_cmd]

            self.output.info(f"🚀 [A2A] Spawning {agent_id} @ http://127.0.0.1:{port} (ws:{ws_port})")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )

            self.agent_processes[agent_id] = A2AAgentProc(
                agent_id=agent_id, port=port, ws_port=ws_port,
                workspace=workspace, process=proc, alive=True, start_ts=time.time()
            )

            # Wait for health ready
            ok = await self._wait_agent_healthy(port, timeout=self.connect_timeout)
            if not ok:
                self.output.error(f"❌ [A2A] {agent_id} not healthy in {self.connect_timeout}s")
                return False

            self.output.success(f"✅ [A2A] Created {agent_id} on port {port}")
            return True

        except Exception as e:
            self.output.error(f"❌ [A2A] Failed to create {agent_id}: {e}")
            return False

    # ========================================
    # A2A Scenario Phases
    # ========================================

    async def _setup_mesh_topology(self) -> bool:
        """Build full mesh by calling /mesh/add_peer on each node to each other."""
        try:
            agent_list = list(self.agent_processes.values())
            for src in agent_list:
                for dst in agent_list:
                    if src.agent_id == dst.agent_id:
                        continue
                    await self._a2a_add_peer(src.port, dst.port)

            self.mesh_built = True
            self.output.success("🔗 [A2A] Mesh topology established")
            return True

        except Exception as e:
            self.output.error(f"❌ [A2A] Mesh setup failed: {e}")
            return False

    async def _broadcast_document(self) -> bool:
        """Broadcast Gaia doc to all agents using /mesh/broadcast."""
        try:
            doc = await self._load_gaia_document()
            tasks = [self._http_post_json(a.port, self.broadcast_path, {"doc": doc}) 
                    for a in self.agent_processes.values()]
            await asyncio.gather(*tasks, return_exceptions=True)
            self.output.success("📡 [A2A] Gaia document broadcasted")
            return True
        except Exception as e:
            self.output.error(f"❌ [A2A] Broadcast failed: {e}")
            return False

    async def _execute_normal_phase(self) -> None:
        """Execute normal Shard QA collaborative retrieval task with A2A."""
        import asyncio
        import time
        
        normal_phase_duration = self.config.get("shard_qa", {}).get("normal_phase_duration", 30.0)
        
        self.output.progress(f"🔍 [A2A] Running Shard QA collaborative retrieval for {normal_phase_duration}s...")
        
        # Start metrics collection for normal phase
        if self.metrics_collector:
            self.metrics_collector.start_normal_phase()
        
        start_time = time.time()
        qa_tasks = []
        
        # Start QA task execution on all agents simultaneously
        for agent_id, worker in self.shard_workers.items():
            task = asyncio.create_task(self._run_qa_task_for_agent(agent_id, worker, normal_phase_duration))
            qa_tasks.append(task)
        
        # Wait for normal phase duration with A2A status updates
        elapsed = 0
        while elapsed < normal_phase_duration:
            await asyncio.sleep(10)  # Check every 10 seconds
            elapsed = time.time() - start_time
            remaining = normal_phase_duration - elapsed
            if remaining > 0:
                self.output.progress(f"   Normal phase: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
        
        # Cancel remaining tasks
        for task in qa_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete cancellation
        await asyncio.gather(*qa_tasks, return_exceptions=True)
        
        # End metrics collection for normal phase
        if self.metrics_collector:
            self.metrics_collector.end_normal_phase()
        
        # Collect final task counts for normal phase
        for agent_id, worker in self.shard_workers.items():
            task_count = getattr(worker, 'completed_tasks', 0)
            self.output.progress(f"   {agent_id}: Normal phase completed with {task_count} QA tasks")
        
        elapsed = time.time() - start_time
        self.output.success(f"[A2A] Normal phase completed in {elapsed:.2f}s")

    async def _run_qa_task_for_agent(self, agent_id: str, worker, duration: float):
        """Run QA task for a specific A2A agent."""
        import asyncio
        import time
        
        start_time = time.time()
        task_count = 0
        
        try:
            while time.time() - start_time < duration and not self.shutdown_event.is_set():
                try:
                    # Execute QA task for group 0 (standard test case)
                    task_start_time = time.time()
                    result = await worker.worker.start_task(0)
                    task_end_time = time.time()
                    task_count += 1
                    
                    # Record task execution in metrics
                    if self.metrics_collector:
                        answer_found = result and "answer found" in result.lower()
                        answer_source = "local" if "LOCAL" in str(result) else "neighbor" if "NEIGHBOR" in str(result) else "unknown"
                        self.metrics_collector.record_task_execution(
                            task_id=f"{agent_id}_normal_{task_count}",
                            agent_id=agent_id,
                            task_type="qa_normal",
                            start_time=task_start_time,
                            end_time=task_end_time,
                            success=True,  # Task completed successfully
                            answer_found=answer_found,
                            answer_source=answer_source,
                            group_id=0
                        )
                    
                    if result and "answer found" in result.lower():
                        # Show minimal search result from agent
                        if "DOCUMENT SEARCH SUCCESS" in result:
                            self.output.progress(f"🔍 [{agent_id}] Found answer")
                        else:
                            self.output.progress(f"{agent_id}: Found answer (task #{task_count})")
                    
                    # Track task completion
                    worker.completed_tasks = getattr(worker, 'completed_tasks', 0) + 1
                    
                    # Brief pause between tasks
                    await asyncio.sleep(2.0)
                    
                except Exception as e:
                    self.output.warning(f"{agent_id}: QA task failed: {e}")
                    await asyncio.sleep(1.0)  # Brief pause on error
                    
        except asyncio.CancelledError:
            self.output.progress(f"{agent_id}: QA task cancelled (completed {task_count} tasks)")
            raise
        except Exception as e:
            self.output.error(f"{agent_id}: QA task error: {e}")
        
        self.output.progress(f"{agent_id}: Normal phase completed with {task_count} QA tasks")

    async def _execute_fault_injection(self) -> None:
        """Execute fault injection by stopping agents."""
        kill_count = max(1, int(len(self.agents) * self.config["scenario"]["kill_fraction"]))
        if kill_count >= len(self.agents):
            kill_count = len(self.agents) - 1  # Keep at least one agent alive
            
        # Pick victims from active agents
        agent_ids = list(self.agents.keys())
        import random
        random.shuffle(agent_ids)
        victims = agent_ids[:kill_count]
        
        self.output.warning(f"💥 Killing {len(victims)} agents: {', '.join(victims)}")
        
        # Mark fault injection start time
        if self.metrics_collector:
            self.metrics_collector.set_fault_injection_time()
        
        # Stop the victim agents
        for agent_id in victims:
            agent = self.agents.get(agent_id)
            if agent:
                try:
                    self.output.warning(f"   ✗ Killed agent: {agent_id} (will attempt reconnection later)")
                    await agent.stop()
                    self.killed_agents.add(agent_id)
                    
                    # Update metrics
                    if self.metrics_collector:
                        self.metrics_collector.update_agent_state(agent_id, "killed")
                        
                except Exception as e:
                    self.output.error(f"Failed to stop agent {agent_id}: {e}")
        
        fault_elapsed = time.time() - self.scenario_start_time if hasattr(self, 'scenario_start_time') else 0
        self.output.warning(f"⚠️  Fault injection completed at t={fault_elapsed:.1f}s")

    async def _monitor_recovery(self) -> None:
        """
        After fault injection, monitor surviving nodes' view of the mesh,
        attempt reconnections, and detect when system returns to steady-state.
        """
        recovery_budget = float(self.config["scenario"].get("recovery_duration", 60))
        start = time.time()
        first_ok_captured = False

        survivors = [a for a in self.agent_processes.values() if a.agent_id not in self.killed_agents]

        # Keep checking liveness + peer tables
        while time.time() - start < recovery_budget:
            await asyncio.sleep(self.heartbeat_interval)

            # 1) Check survivor health
            health = await asyncio.gather(*[self._is_agent_healthy(a.port) for a in survivors], return_exceptions=True)
            alive_ratio = sum(1 for h in health if h is True) / max(1, len(survivors))

            # 2) Try to reconnect killed ones (if your design can auto-restart them, add that here)
            #    Here we demonstrate "no auto respawn", only peer mesh stabilization.
            mesh_ok = await self._check_mesh_consistency()

            # 3) Detect first_ok and steady-state
            if mesh_ok and not first_ok_captured:
                self.recovery_first_ok_ts = time.time()
                first_ok_captured = True
                if self.metrics_collector:
                    self.metrics_collector.set_first_recovery_time(self.recovery_first_ok_ts)

            if mesh_ok and alive_ratio >= 1.0:
                # Optional: define "steady" as some consecutive healthy checks
                self.steady_state_ts = time.time()
                if self.metrics_collector:
                    self.metrics_collector.set_steady_state_time(self.steady_state_ts)
                break

            self.output.info(
                f"🔄 [A2A] Recovery tick: alive={alive_ratio:.2f}, mesh_ok={mesh_ok}, elapsed={int(time.time()-start)}s"
            )

        self.output.success("🔄 [A2A] Recovery monitoring finished")

    # ========================================
    # Metrics finalize
    # ========================================

    async def _finalize_scenario(self) -> Dict[str, Any]:
        """
        Add A2A-specific metrics and return final dictionary that FailStorm base will persist.
        """
        end_time = time.time()
        total_runtime = end_time - self.scenario_start_time if hasattr(self, 'scenario_start_time') else 0
        
        # Compute derived metrics
        fault_ts = None
        if self.metrics_collector:
            fault_ts = self.metrics_collector.fault_injection_time
        rec_ts = self.recovery_first_ok_ts
        steady_ts = self.steady_state_ts

        final: Dict[str, Any] = {
            "metadata": {
                "scenario": "fail_storm_recovery", 
                "protocol": "a2a",
                "start_time": getattr(self, 'scenario_start_time', end_time - total_runtime),
                "end_time": end_time,
                "total_runtime": total_runtime,
                "config": self.config
            },
            "agent_summary": {
                "initial_count": len(self.agents),
                "temporarily_killed_count": len(self.killed_agents),
                "currently_killed_count": len(self.killed_agents),
                "permanently_failed_count": 0,
                "surviving_count": len(self.agents) - len(self.killed_agents),
                "reconnected_count": 0,
                "temporarily_killed_agents": list(self.killed_agents),
                "killed_agents": list(self.killed_agents),
                "permanently_failed_agents": []
            },
            "a2a_specific": {
                "mesh_built": self.mesh_built,
                "killed_agents": sorted(list(self.killed_agents)),
                "recovery_time": (rec_ts - fault_ts) if (rec_ts and fault_ts) else None,
                "steady_state_time": (steady_ts - fault_ts) if (steady_ts and fault_ts) else None,
                "http_endpoints": len(self.agents),
                "websocket_endpoints": 1 if self.base_ws_port else 0,
                "a2a_protocol_active": True
            },
            "timing": {
                "total_runtime": total_runtime,
                "fault_time": fault_ts,
                "recovery_end_time": end_time,
                "setup_time": getattr(self, 'setup_time', 0),
                "normal_phase_duration": self.config.get("shard_qa", {}).get("normal_phase_duration", 30),
                "recovery_phase_duration": self.config.get("scenario", {}).get("recovery_duration", 60)
            }
        }
        
        # Add comprehensive metrics if available (like ANP does)
        if self.metrics_collector:
            try:
                # Get performance metrics
                metrics_summary = self.metrics_collector.calculate_recovery_metrics()
                final["failstorm_metrics"] = metrics_summary
                
                # Get QA metrics
                qa_metrics = self.metrics_collector.get_qa_metrics()
                final["qa_metrics"] = qa_metrics
                
                # Add LLM outputs info
                final["llm_outputs"] = {
                    "saved": False,  # A2A doesn't save LLM outputs by default
                    "directory": None
                }
                
            except Exception as e:
                self.output.error(f"Failed to collect comprehensive metrics: {e}")
                # Don't use fallback - let the error be visible
                raise
        
        return final

    # ========================================
    # Helper methods
    # ========================================

    def _pick_victims(self, kill_count: int) -> List[str]:
        alive_ids = [aid for aid, ap in self.agent_processes.items() if ap.alive]
        random.shuffle(alive_ids)
        return alive_ids[:max(0, kill_count)]

    async def _wait_agent_healthy(self, port: int, timeout: float) -> bool:
        """
        Poll /healthz until agent reports ready.
        """
        start = time.time()
        while time.time() - start < timeout:
            if await self._is_agent_healthy(port):
                return True
            await asyncio.sleep(0.3)
        return False

    async def _is_agent_healthy(self, port: int) -> bool:
        """
        Hit health endpoint. Replace with your real probe if needed.
        """
        try:
            resp = await self._http_get(f"http://127.0.0.1:{port}{self.health_path}")
            return (resp.get("status") == "ok")
        except Exception:
            return False

    async def _a2a_add_peer(self, src_port: int, dst_port: int) -> None:
        """
        Call /mesh/add_peer on src, telling it about dst.
        """
        payload = {"peer": f"http://127.0.0.1:{dst_port}"}
        await self._http_post_json(src_port, self.peer_add_path, payload)

    async def _check_mesh_consistency(self) -> bool:
        """
        Optional: ask each agent for its peer-table and verify that all survivors see each other.
        If your agent exposes /mesh/peers, consult it here.
        """
        # For minimal viable, just rely on health ratio.
        return True

    async def _load_gaia_document(self) -> Dict[str, Any]:
        """
        Load Gaia doc from config or file. Here we just return a stub.
        """
        return {
            "title": "Gaia Init",
            "version": "v1",
            "ts": time.time(),
            "notes": "Replace this with your real Gaia content"
        }

    # ========================================
    # HTTP utils (no external deps)
    # ========================================

    async def _http_get(self, url: str) -> Dict[str, Any]:
        """
        Minimal HTTP GET using asyncio + built-in socket is overkill.
        Please replace with your project's HTTP client helper if available.
        Here we use aiohttp if present, otherwise raise.
        """
        try:
            import aiohttp
        except ImportError as e:
            raise RuntimeError("aiohttp is required for A2A runner HTTP calls") from e

        timeout = aiohttp.ClientTimeout(total=self.connect_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(url) as r:
                r.raise_for_status()
                return await r.json()

    async def _http_post_json(self, port_or_url, path_or_payload, payload_opt=None) -> Dict[str, Any]:
        """
        If called with (port:int, path:str, payload:dict) build URL; if (url:str, payload:dict) use URL directly.
        """
        try:
            import aiohttp
        except ImportError as e:
            raise RuntimeError("aiohttp is required for A2A runner HTTP calls") from e

        if isinstance(port_or_url, int):
            url = f"http://127.0.0.1:{port_or_url}{path_or_payload}"
            payload = payload_opt or {}
        else:
            url = port_or_url
            payload = path_or_payload or {}

        timeout = aiohttp.ClientTimeout(total=self.connect_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(url, json=payload) as r:
                r.raise_for_status()
                ctype = r.headers.get("Content-Type", "")
                if "application/json" in ctype:
                    return await r.json()
                text = await r.text()
                return {"text": text}
