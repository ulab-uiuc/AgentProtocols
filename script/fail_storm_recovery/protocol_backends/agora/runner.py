#!/usr/bin/env python3
"""
Agora protocol runner for Fail-Storm Recovery scenario.

This module implements the Agora protocol specific functionality
while inheriting all core logic from the base runner.
"""

from pathlib import Path
from typing import Dict, List, Any
import sys
import time
import asyncio

# Add paths for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.simple_base_agent import SimpleBaseAgent as BaseAgent
from protocol_backends.base_runner import FailStormRunnerBase
from .agent import create_agora_agent

# Import shard_qa components dynamically to avoid circular imports
shard_qa_path = Path(__file__).parent.parent.parent / "shard_qa"
sys.path.insert(0, str(shard_qa_path))
import importlib.util
spec = importlib.util.spec_from_file_location("agent_executor", shard_qa_path / "shard_worker" / "agent_executor.py")
agent_executor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_executor_module)

# Create Agora-specific implementations to avoid coordinator dependency
class AgoraAgentExecutor(agent_executor_module.BaseAgentExecutor):
    """Agora-specific agent executor"""
    async def execute(self, context, event_queue):
        # Agora uses tool-based execution, this is just for compatibility
        pass
    
    async def cancel(self, context, event_queue):
        # Agora uses tool-based execution, this is just for compatibility
        pass

class AgoraRequestContext(agent_executor_module.BaseRequestContext):
    """Agora-specific request context"""
    def __init__(self, input_data):
        self.input_data = input_data
    
    def get_user_input(self):
        return self.input_data

class AgoraEventQueue(agent_executor_module.BaseEventQueue):
    """Agora-specific event queue"""
    def __init__(self):
        self.events = []
    
    async def enqueue_event(self, event):
        self.events.append(event)
        return event

def agora_new_agent_text_message(text, role="user"):
    """Agora-specific text message creation"""
    return {"type": "text", "content": text, "role": str(role)}

# Patch the _send_to_coordinator method to avoid coordinator dependency
original_send_to_coordinator = agent_executor_module.ShardWorker._send_to_coordinator

async def _patched_send_to_coordinator(self, content: str, path: List[str] = None, ttl: int = 0):
    """Patched version that doesn't require coordinator"""
    # For Agora fail-storm testing, we don't need coordinator
    # Just log the message that would have been sent
    if hasattr(self, 'output') and self.output:
        self.output.info(f"[{self.shard_id}] Would send to coordinator: {content}")
    return "Coordinator message skipped (Agora mode)"

# Apply the patch
agent_executor_module.ShardWorker._send_to_coordinator = _patched_send_to_coordinator

# Inject Agora implementations into the agent_executor module
agent_executor_module.AgentExecutor = AgoraAgentExecutor
agent_executor_module.RequestContext = AgoraRequestContext
agent_executor_module.EventQueue = AgoraEventQueue
agent_executor_module.new_agent_text_message = agora_new_agent_text_message

ShardWorkerExecutor = agent_executor_module.ShardWorkerExecutor


class AgoraRunner(FailStormRunnerBase):
    """Agora 协议 runner（精简版，参照 ACP Runner 结构）。

    去掉对 `agent.register_endpoint` / SDK Toolformer 等依赖，统一用 `SimpleBaseAgent`
    与 mesh_network 的通用机制，避免 AttributeError。
    """

    def __init__(self, config_path: str = "config.yaml"):
        if config_path == "config.yaml":
            protocol_config = Path(__file__).parent / "config.yaml"
            if protocol_config.exists():
                config_path = str(protocol_config)
        super().__init__(config_path)
        if "scenario" not in self.config:
            self.config["scenario"] = {}
        self.config["scenario"]["protocol"] = "agora"
        self.agora_sessions: Dict[str, Any] = {}
        self.output.info("Initialized Agora protocol runner (simplified)")

    # -------- Protocol-specific required overrides --------
    async def create_agent(self, agent_id: str, host: str, port: int, executor: ShardWorkerExecutor) -> BaseAgent:
        """创建 Agora Agent（与 ACP 逻辑一致，调用工厂方法）。"""
        try:
            agent = await create_agora_agent(
                agent_id=agent_id,
                host=host,
                port=port,
                executor=executor
            )
            # 存储会话信息（保持与 ACP Runner 类似结构，可扩展）
            self.agora_sessions[agent_id] = {
                "base_url": f"http://{host}:{port}",
                "session_id": f"agora_session_{agent_id}_{int(time.time())}",
                "executor": executor
            }
            self.output.success(f"Agora agent {agent_id} created")
            return agent
        except Exception as e:
            self.output.error(f"Failed to create Agora agent {agent_id}: {e}")
            raise

    def get_protocol_info(self, agent_id: str, port: int, data_file: str) -> str:
        return f"🎵 [Agora] Created {agent_id} - HTTP: {port}, Data: {data_file}"

    def get_reconnection_info(self, agent_id: str, port: int) -> List[str]:
        return [
            f"   ✓ Reconnected {agent_id} via Agora(simple)",
            f"   ✓ Agora endpoint: http://127.0.0.1:{port}",
            f"   ✓ SimpleBaseAgent restored"
        ]

    # -------- Mesh topology --------
    async def _setup_mesh_topology(self) -> None:
        self.output.progress("🔗 [Agora] Setting up mesh topology (simple mode)...")
        await self.mesh_network.setup_mesh_topology()
        await asyncio.sleep(1.0)  # 轻微等待拓扑稳定
        topology = self.mesh_network.get_topology()
        expected = len(self.agents) * (len(self.agents) - 1)
        actual = sum(len(v) for v in topology.values())
        self.output.success(f"🔗 [Agora] Mesh topology established: {actual}/{expected} connections")

    async def _broadcast_document(self) -> None:
        """简化广播：仿 ACP 将文档放入会话结构。"""
        if not self.agents:
            raise RuntimeError("No Agora agents available for broadcast")
        try:
            success = 0
            for agent_id in self.agents:
                if agent_id in self.agora_sessions:
                    self.agora_sessions[agent_id]["document"] = self.document
                    success += 1
            self.output.success(f"📡 [Agora] Document broadcast to {success}/{len(self.agents)} agents")
        except Exception as e:
            self.output.error(f"📡 [Agora] Broadcast failed: {e}")
            raise

    async def _execute_normal_phase(self) -> None:
        import asyncio, time
        duration = self.config.get("shard_qa", {}).get("normal_phase_duration", 30.0)
        self.output.progress(f"🔍 [Agora] Running Shard QA normal phase for {duration}s...")
        if self.metrics_collector:
            self.metrics_collector.start_normal_phase()
        start = time.time()
        tasks = [asyncio.create_task(self._run_qa_task_for_agent(aid, worker, duration)) for aid, worker in self.shard_workers.items()]
        elapsed = 0
        while elapsed < duration:
            await asyncio.sleep(10)
            elapsed = time.time() - start
            remain = duration - elapsed
            if remain > 0:
                self.output.progress(f"   Normal phase: {elapsed:.0f}s elapsed, {remain:.0f}s remaining")
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.metrics_collector:
            self.metrics_collector.end_normal_phase()
        for aid, worker in self.shard_workers.items():
            cnt = getattr(worker, 'completed_tasks', 0)
            self.output.progress(f"   {aid}: Normal phase completed with {cnt} QA tasks")
        self.output.success(f"[Agora] Normal phase completed in {time.time()-start:.2f}s")

    async def _run_qa_task_for_agent(self, agent_id: str, worker, duration: float):
        import asyncio, time
        start = time.time()
        count = 0
        try:
            while time.time() - start < duration and not self.shutdown_event.is_set():
                try:
                    t0 = time.time()
                    result = await worker.worker.start_task(0)
                    t1 = time.time()
                    count += 1
                    if self.metrics_collector:
                        answer_found = result and "answer found" in result.lower()
                        rs = str(result).upper()
                        if "NEIGHBOR" in rs:
                            source = "neighbor"
                        elif "DOCUMENT SEARCH SUCCESS" in rs or "LOCAL" in rs:
                            source = "local"
                        else:
                            source = "unknown"
                        self.metrics_collector.record_task_execution(
                            task_id=f"{agent_id}_normal_{count}",
                            agent_id=agent_id,
                            task_type="qa_normal",
                            start_time=t0,
                            end_time=t1,
                            success=True,
                            answer_found=answer_found,
                            answer_source=source,
                            group_id=0
                        )
                    if result and "answer found" in result.lower():
                        self.output.progress(f"{agent_id}: Found answer (task #{count})")
                    worker.completed_tasks = getattr(worker, 'completed_tasks', 0) + 1
                    await asyncio.sleep(2.0)
                except Exception as e:
                    self.output.warning(f"{agent_id}: QA task failed: {e}")
                    await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            self.output.progress(f"{agent_id}: QA task cancelled (completed {count} tasks)")
            raise
        self.output.progress(f"{agent_id}: Normal phase completed with {count} QA tasks")

    async def _monitor_recovery(self) -> None:
        import asyncio, time
        recovery_duration = self.config["scenario"].get("recovery_duration", 60.0)
        self.output.info(f"🔄 [Agora] Monitoring recovery for {recovery_duration}s...")
        start = time.time()
        first_recovery = False
        # Post-fault QA tasks for alive agents
        tasks = []
        for aid, worker in self.shard_workers.items():
            if aid not in self.killed_agents:
                tasks.append(asyncio.create_task(self._run_recovery_qa_task(aid, worker, recovery_duration)))
        while time.time() - start < recovery_duration:
            alive = 0
            for aid in list(self.agents.keys()):
                if aid not in self.killed_agents:
                    alive += 1
                elif aid in self.killed_agents:
                    # Try reconnection similar to ACP pattern
                    try:
                        session = self.agora_sessions.get(aid)
                        if session:
                            executor = session['executor']
                            port = int(session['base_url'].split(':')[-1])
                            new_agent = await BaseAgent.create_agora(
                                agent_id=aid,
                                host="127.0.0.1",
                                port=port,
                                executor=executor
                            )
                            self.agents[aid] = new_agent
                            if await new_agent.health_check():
                                self.killed_agents.remove(aid)
                                alive += 1
                                if not first_recovery and self.metrics_collector:
                                    self.metrics_collector.set_first_recovery_time(time.time())
                                    first_recovery = True
                                self.output.success(f"   ✓ Agora agent {aid} reconnected")
                                if self.metrics_collector:
                                    self.metrics_collector.update_agent_state(aid, "recovered")
                    except Exception as e:
                        self.output.warning(f"   ⚠️ Reconnect failed {aid}: {e}")
            ratio = alive / len(self.agents) if self.agents else 0
            elapsed = time.time() - start
            self.output.info(f"🔄 [Agora] Recovery tick: alive={ratio:.2%}, elapsed={elapsed:.0f}s")
            if ratio >= 1.0 and len(self.killed_agents) == 0 and elapsed >= 10.0:
                if self.metrics_collector:
                    self.metrics_collector.set_steady_state_time(time.time())
                self.output.success(f"🔄 [Agora] All agents recovered at t={elapsed:.1f}s")
                break
            await asyncio.sleep(5)
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.output.success("🔄 [Agora] Recovery monitoring finished")

    async def _run_recovery_qa_task(self, agent_id: str, worker, duration: float):
        import asyncio, time
        start = time.time()
        count = 0
        while time.time() - start < duration and not self.shutdown_event.is_set():
            try:
                t0 = time.time()
                result = await worker.worker.start_task(0)
                t1 = time.time()
                count += 1
                if self.metrics_collector:
                    answer_found = result and "answer found" in result.lower()
                    src_upper = str(result).upper()
                    if "NEIGHBOR" in src_upper:
                        src = "neighbor"
                    elif "DOCUMENT SEARCH SUCCESS" in src_upper or "LOCAL" in src_upper:
                        src = "local"
                    else:
                        src = "unknown"
                    self.metrics_collector.record_task_execution(
                        task_id=f"{agent_id}_recovery_{count}",
                        agent_id=agent_id,
                        task_type="qa_recovery",
                        start_time=t0,
                        end_time=t1,
                        success=True,
                        answer_found=answer_found,
                        answer_source=src,
                        group_id=0
                    )
                if result and "answer found" in str(result).lower():
                    self.output.progress(f"{agent_id}: Found answer (recovery task #{count})")
                worker.recovery_completed_tasks = getattr(worker, 'recovery_completed_tasks', 0) + 1
                await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.output.warning(f"{agent_id}: recovery QA task error: {e}")
                await asyncio.sleep(1.0)

    async def _execute_fault_injection(self) -> None:
        import time, asyncio
        fault_time = self.config["scenario"]["fault_injection_time"]
        elapsed = time.time() - self.scenario_start_time
        if elapsed < fault_time:
            wait = fault_time - elapsed
            self.output.progress(f"⏰ [Agora] Waiting {wait:.1f}s until fault injection time...")
            await asyncio.sleep(wait)
        if self.metrics_collector:
            self.metrics_collector.set_fault_injection_time()
        await self._inject_faults(self.config["scenario"]["kill_fraction"])
        self.output.warning(f"💥 [Agora] Fault injection executed at t={fault_time:.1f}s (scenario time)")

    async def _inject_faults(self, kill_fraction: float) -> None:
        import random, asyncio
        ids = list(self.agents.keys())
        num = max(1, int(len(ids) * kill_fraction))
        if num >= len(ids):
            num = len(ids) - 1
        victims = random.sample(ids, num)
        self.output.warning(f"💥 Killing {len(victims)} Agora agents: {', '.join(victims)}")
        self._originally_killed_agents = set(victims)
        for vid in victims:
            agent = self.agents.get(vid)
            if agent:
                try:
                    await agent.stop()
                    self.killed_agents.add(vid)
                    if self.metrics_collector:
                        self.metrics_collector.update_agent_state(vid, "killed")
                    self.output.warning(f"   ✗ Killed Agora agent: {vid}")
                except Exception as e:
                    self.output.error(f"Failed to stop Agora agent {vid}: {e}")

    # Removed: _schedule_agora_reconnection / _reestablish_agent_connections (not needed in simplified mode)

    async def _finalize_scenario(self) -> Dict[str, Any]:
        """收尾：与 ACP Runner 结构类似，提供基础统计。"""
        import time
        fault_ts = self.metrics_collector.fault_injection_time if self.metrics_collector else None
        rec_ts = self.metrics_collector.first_recovery_time if self.metrics_collector else None
        steady_ts = self.metrics_collector.steady_state_time if self.metrics_collector else None
        total_agents = len(self.agents)
        killed = getattr(self, '_originally_killed_agents', self.killed_agents.copy())
        alive_agents = [aid for aid in self.agents.keys() if aid not in self.killed_agents]
        recovered = [aid for aid in killed if aid not in self.killed_agents]
        final = {
            "metadata": {
                "protocol": "agora",
                "scenario": "fail_storm_recovery",
                "agent_count": len(self.agents),
                "kill_fraction": self.config["scenario"]["kill_fraction"],
                "timestamp": time.time(),
                "total_runtime": time.time() - self.scenario_start_time if hasattr(self, 'scenario_start_time') else 0.0
            },
            "agent_summary": {
                "initial_count": total_agents,
                "temporarily_killed_count": len(killed),
                "currently_killed_count": len(self.killed_agents),
                "permanently_failed_count": 0,
                "surviving_count": len(alive_agents),
                "reconnected_count": len(recovered),
                "temporarily_killed_agents": list(killed),
                "currently_killed_agents": list(self.killed_agents),
                "permanently_failed_agents": [],
                "surviving_agents": alive_agents
            },
            "agora_specific": {
                "sessions_created": len(self.agora_sessions),
                "document_broadcast": "success",
                "mesh_connections": len(self.agents) * (len(self.agents) - 1)
            },
            "timing": {
                "fault_injection_time": fault_ts,
                "first_recovery_time": rec_ts,
                "steady_state_time": steady_ts,
                "total_runtime": time.time() - self.scenario_start_time if hasattr(self, 'scenario_start_time') else None,
                "normal_phase_duration": self.config.get("shard_qa", {}).get("normal_phase_duration", 30),
                "recovery_phase_duration": self.config.get("scenario", {}).get("recovery_duration", 60)
            }
        }
        if self.metrics_collector:
            try:
                metrics_summary = self.metrics_collector.calculate_recovery_metrics()
                final["failstorm_metrics"] = metrics_summary
                qa_metrics = self.metrics_collector.get_qa_metrics()
                final["qa_metrics"] = qa_metrics
                final["llm_outputs"] = {"saved": False, "directory": None}
            except Exception as e:
                self.output.error(f"Failed to collect metrics: {e}")
                raise
        return final

    async def _save_results(self, results: Dict[str, Any]) -> None:
        """Save Agora scenario results to files."""
        import json
        
        # Save main results file
        results_file = self.results_dir / self.config["output"]["results_file"]
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.output.success(f"💾 [Agora] Results saved to: {results_file}")
            
        except Exception as e:
            self.output.error(f"❌ [Agora] Failed to save results: {e}")
        
        # Save detailed metrics if available
        if self.metrics_collector:
            detailed_metrics_file = self.results_dir / "detailed_failstorm_metrics.json"
            try:
                self.metrics_collector.export_to_json(str(detailed_metrics_file))
                self.output.success(f"💾 [Agora] Detailed metrics saved to: {detailed_metrics_file}")
            except Exception as e:
                self.output.error(f"❌ [Agora] Failed to save detailed metrics: {e}")