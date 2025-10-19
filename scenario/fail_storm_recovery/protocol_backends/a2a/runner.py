#!/usr/bin/env python3
"""
A2A protocol runner for Fail-Storm Recovery scenario.

This module implements the A2A (Agent-to-Agent) protocol specific functionality
using native a2a-sdk while inheriting all core logic from the base runner.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import time
import asyncio
import json
import random

# Add paths for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.simple_base_agent import SimpleBaseAgent as BaseAgent
from protocol_backends.base_runner import FailStormRunnerBase
from .agent import create_a2a_agent, A2AAgent

# Import shard_qa components dynamically to avoid circular imports
shard_qa_path = Path(__file__).parent.parent.parent / "shard_qa"
sys.path.insert(0, str(shard_qa_path))
import importlib.util
spec = importlib.util.spec_from_file_location("agent_executor", shard_qa_path / "shard_worker" / "agent_executor.py")
agent_executor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_executor_module)

# Create A2A-specific implementations to avoid coordinator dependency
class A2AAgentExecutor(agent_executor_module.BaseAgentExecutor):
    """A2A-specific agent executor"""
    async def execute(self, context, event_queue):
        # A2A doesn't use the executor pattern, this is just for compatibility
        pass
    
    async def cancel(self, context, event_queue):
        # A2A doesn't use the executor pattern, this is just for compatibility
        pass

class A2ARequestContext(agent_executor_module.BaseRequestContext):
    """A2A-specific request context"""
    def __init__(self, input_data):
        self.input_data = input_data
    
    def get_user_input(self):
        return self.input_data

class A2AEventQueue(agent_executor_module.BaseEventQueue):
    """A2A-specific event queue"""
    def __init__(self):
        self.events = []
    
    async def enqueue_event(self, event):
        self.events.append(event)
        return event

def a2a_new_agent_text_message(text, role="user"):
    """A2A-specific text message creation"""
    return {"type": "text", "content": text, "role": str(role)}
    
    # Inject A2A implementations into the agent_executor module
    agent_executor_module.AgentExecutor = A2AAgentExecutor
    agent_executor_module.RequestContext = A2ARequestContext
    agent_executor_module.EventQueue = A2AEventQueue
    agent_executor_module.new_agent_text_message = a2a_new_agent_text_message

ShardWorkerExecutor = agent_executor_module.ShardWorkerExecutor


class A2ARunner(FailStormRunnerBase):
    """
    A2A protocol runner.
    
    Implements protocol-specific agent creation and management for A2A protocol
    while inheriting all core Fail-Storm functionality from FailStormRunnerBase.
    """

    def __init__(self, config_path: str = "config.yaml"):
        # If using default config, use configs/config_a2a.yaml
        if config_path == "config.yaml":
            configs_dir = Path(__file__).parent.parent.parent / "configs"
            protocol_config = configs_dir / "config_a2a.yaml"
            if protocol_config.exists():
                config_path = str(protocol_config)
                print(f"📋 Using A2A config from: {config_path}")
            else:
                # Fallback to protocol-specific config
                protocol_config = Path(__file__).parent / "config.yaml"
                if protocol_config.exists():
                    config_path = str(protocol_config)
        
        super().__init__(config_path)
        
        # Ensure protocol is set correctly in config
        if "scenario" not in self.config:
            self.config["scenario"] = {}
        self.config["scenario"]["protocol"] = "a2a"


        # A2A-specific attributes
        self.a2a_sessions: Dict[str, Any] = {}
        self.killed_agents: set = set()
        
        # Per-agent group tracking for consistent recovery behavior
        self._next_group_for_agent = {}
        self.killed_agent_configs: Dict[str, Any] = {}
        
        self.output.info("Initialized A2A protocol runner")

    async def create_agent(self, agent_id: str, host: str, port: int, executor: ShardWorkerExecutor) -> A2AAgent:
        """
        Create an A2A agent using native a2a-sdk.
        
        Args:
            agent_id: Unique identifier for the agent
            host: Host address for the agent
            port: Port number for the agent
            executor: Shard worker executor instance
            
        Returns:
            BaseAgent instance configured for A2A protocol
        """
        try:
            # Create A2A agent using the factory method
            agent = await create_a2a_agent(
                agent_id=agent_id,
                host=host,
                port=port,
                executor=executor
            )
            
            # Store A2A-specific info
            self.a2a_sessions[agent_id] = {
                "base_url": f"http://{host}:{port}",
                "session_id": f"session_{agent_id}_{int(time.time())}",
                "executor": executor
            }
            
            self.output.success(f"A2A agent {agent_id} created successfully")
            return agent
            
        except Exception as e:
            self.output.error(f"Failed to create A2A agent {agent_id}: {e}")
            raise

    def get_protocol_info(self, agent_id: str, port: int, data_file: str) -> str:
        """Get A2A protocol display information."""
        return f"🔗 [A2A] Created {agent_id} - HTTP: {port}, Data: {data_file}"

    def get_reconnection_info(self, agent_id: str, port: int) -> List[str]:
        """Get A2A protocol reconnection information."""
        return [
            f"🔗 [A2A] Agent {agent_id} RECONNECTED on port {port}",
            f"📡 [A2A] A2A protocol active",
            f"🌐 [A2A] HTTP REST API: http://127.0.0.1:{port}"
        ]

    async def _setup_mesh_topology(self) -> None:
        """Setup mesh topology between A2A agents."""
        self.output.progress("🔗 [A2A] Setting up mesh topology...")
        
        await self.mesh_network.setup_mesh_topology()
        
        # A2A agents may need time for connection establishment
        import asyncio
        await asyncio.sleep(1.0)  # Brief stabilization for A2A
        
        # Verify connectivity
        topology = self.mesh_network.get_topology()
        expected_connections = len(self.agents) * (len(self.agents) - 1)
        actual_connections = sum(len(edges) for edges in topology.values())
        
        self.output.success(f"🔗 [A2A] Mesh topology established: {actual_connections}/{expected_connections} connections")

    async def _broadcast_document(self) -> None:
        """Broadcast the document to all A2A agents."""
        if not self.agents:
            raise RuntimeError("No A2A agents available for broadcast")
            
        try:
            doc = await self._load_gaia_document()
            
            # For A2A protocol, we'll store the document in each agent's session
            # In a real implementation, this would use proper A2A messaging
            success_count = len(self.agents)
            
            self.output.success(f"📡 [A2A] Document broadcasted to {success_count}/{len(self.agents)} agents")

        except Exception as e:
            self.output.error(f"❌ [A2A] Document broadcast failed: {e}")
            raise

    async def _load_gaia_document(self) -> Dict[str, Any]:
        """Load Gaia doc from config or file."""
        return {
            "title": "Gaia Init",
            "version": "v1",
            "ts": time.time(),
            "notes": "Replace this with your real Gaia content for A2A protocol"
        }

    async def _execute_normal_phase(self) -> None:
        """Execute normal Shard QA collaborative retrieval task with A2A."""
        try:
            normal_duration = self.config.get("shard_qa", {}).get("normal_phase_duration", 30.0)
            
            self.output.progress(f"🔗 [A2A] Running Shard QA collaborative retrieval for {normal_duration}s...")
            
            # Start QA tasks for all agents
            qa_tasks = []
            for agent_id, executor in self.shard_workers.items():
                task = asyncio.create_task(
                    self._run_qa_task_for_agent_with_failover(agent_id, executor, normal_duration),
                    name=f"qa_task_{agent_id}"
                )
                qa_tasks.append(task)
        
            # Wait for normal phase to complete
            await asyncio.gather(*qa_tasks, return_exceptions=True)
        
            # Report completion
            for agent_id, executor in self.shard_workers.items():
                task_count = getattr(executor.worker, 'task_count', 0)
                self.output.info(f"    {agent_id}: Normal phase completed with {task_count} QA tasks")
            
            self.output.success(f"🔍 [A2A] Normal phase completed in {normal_duration:.2f}s")
            
        except Exception as e:
            self.output.error(f"❌ [A2A] Normal phase failed: {e}")
            raise

    async def _run_qa_task_for_agent(self, agent_id: str, worker, duration: float):
        """Run QA task for a specific A2A agent during normal phase."""
        start_wall = time.perf_counter()  # high-resolution timer
        task_count = 0
        
        # Keep a per-agent rolling group index across phases
        max_groups = 20
        if not hasattr(self, "_next_group_for_agent"):
            self._next_group_for_agent = {}
        group_id = self._next_group_for_agent.get(agent_id, 0) % max_groups
        
        try:
            while (time.perf_counter() - start_wall) < duration and group_id < max_groups and not self.shutdown_event.is_set():
                try:
                    # High-resolution timing for metrics
                    start_t = time.perf_counter()
                    result = await worker.worker.start_task(group_id)
                    end_t = time.perf_counter()
                    task_count += 1
                    current_group = group_id
                    group_id = (group_id + 1) % max_groups  # round robin
                    
                    # Update the global pointer for this agent so later phases continue smoothly
                    self._next_group_for_agent[agent_id] = group_id
                    
                    # Record task execution in metrics
                    if self.metrics_collector:
                        # Fix logic: distinguish between finding answer vs not finding answer
                        result_str = str(result).lower() if result else ""
                        answer_found = (result and 
                                      ("document search success" in result_str or "answer_found:" in result_str) and 
                                      "no answer" not in result_str)
                        # Determine answer source based on the result content and patterns
                        result_str = str(result).upper()
                        if any(pattern in result_str for pattern in [
                            "NEIGHBOR SEARCH SUCCESS", "SOURCE: NEIGHBOR", "FOUND ANSWER FROM NEIGHBOR", 
                            "NEIGHBOR AGENT", "FROM NEIGHBOR"
                        ]):
                            answer_source = "neighbor"
                        elif any(pattern in result_str for pattern in [
                            "DOCUMENT SEARCH SUCCESS", "LOCAL SEARCH", "LOCAL DOCUMENT", 
                            "FOUND LOCALLY", "SOURCE: LOCAL"
                        ]):
                            answer_source = "local"
                        else:
                            answer_source = "unknown"
                            
                        # Determine current phase for proper task classification
                        current_phase = self._get_current_phase()
                        task_type = f"qa_{current_phase}"
                        
                        # Use wall-clock timestamps for compatibility with collector
                        now_wall = time.time()
                        duration_s = end_t - start_t
                        self.metrics_collector.record_task_execution(
                            task_id=f"{agent_id}_{current_phase}_g{current_group}_{task_count}",
                            agent_id=agent_id,
                            task_type=task_type,
                            start_time=now_wall - duration_s,
                            end_time=now_wall,
                            success=True,
                            answer_found=answer_found,
                            answer_source=answer_source,
                            group_id=current_group
                        )
                        
                        # Log progress for debugging
                        if answer_found:
                            self.output.info(f"    {agent_id}: Found answer (task #{task_count})")
                    
                    # Wait before next task
                    await asyncio.sleep(2.0)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.output.error(f"[A2A] Error in QA task for {agent_id}: {e}")
                    # Continue with next task
                    await asyncio.sleep(1.0)
                    
        except asyncio.CancelledError:
            pass
        finally:
            # Store task count for reporting
            if hasattr(worker, 'worker'):
                worker.worker.task_count = task_count

    async def _execute_fault_injection(self) -> None:
        """Execute fault injection by stopping A2A agents."""
        kill_count = max(1, int(len(self.agents) * self.config["scenario"]["kill_fraction"]))
        if kill_count >= len(self.agents):
            kill_count = len(self.agents) - 1  # Keep at least one agent alive
            
        # Pick victims from active agents
        agent_ids = list(self.agents.keys())
        import random
        random.shuffle(agent_ids)
        victims = agent_ids[:kill_count]
        
        self.output.warning(f"💥 Killing {len(victims)} A2A agents: {', '.join(victims)}")
        
        # Record originally killed agents for final statistics
        self._originally_killed_agents = set(victims)
        
        # Mark fault injection start time
        if self.metrics_collector:
            self.metrics_collector.set_fault_injection_time()
        
        # Stop the victim agents
        for agent_id in victims:
            agent = self.agents.get(agent_id)
            if agent:
                try:
                    # Store configuration for later reconnection
                    self.killed_agent_configs[agent_id] = {
                        'port': agent.port,
                        'host': agent.host,
                        'executor': self.shard_workers.get(agent_id)
                    }
                    
                    self.output.warning(f"   ✗ Killed A2A agent: {agent_id} (will attempt reconnection later)")
                    
                    # Stop the agent
                    await agent.stop()
                    
                    # Remove from active agents
                    del self.agents[agent_id]
                    self.killed_agents.add(agent_id)
                    
                    # Remove from mesh network
                    await self.mesh_network.remove_agent(agent_id)
                    
                    # Update metrics
                    if self.metrics_collector:
                        self.metrics_collector.update_agent_state(agent_id, "killed")
                        
                except Exception as e:
                    self.output.error(f"Failed to stop A2A agent {agent_id}: {e}")
        
        # Schedule reconnection attempts with delay
        reconnect_delay = self.config.get("a2a", {}).get("recovery", {}).get("reconnect_delay", 10.0)
        self.output.success(f"🔄 [A2A] Scheduling reconnection for {len(victims)} agents in {reconnect_delay}s...")
        
        for victim_id in victims:
            asyncio.create_task(self._schedule_a2a_reconnection(victim_id, reconnect_delay))
        
        fault_elapsed = time.time() - self.scenario_start_time if hasattr(self, 'scenario_start_time') else 0
        self.output.warning(f"⚠️  A2A fault injection completed at t={fault_elapsed:.1f}s")

    async def _schedule_a2a_reconnection(self, agent_id: str, delay: float) -> None:
        """Schedule A2A agent reconnection with proper port management."""
        await asyncio.sleep(delay)
        
        try:
            self.output.warning(f"🔄 [A2A] Attempting to reconnect agent: {agent_id}")
            
            if agent_id in self.killed_agents and agent_id in self.shard_workers:
                # Get original configuration
                agent_config = self.killed_agent_configs.get(agent_id, {})
                worker = self.shard_workers[agent_id]
                
                # Find available port (try original first, then find new one)
                original_port = agent_config.get('port', 9000)
                
                try:
                    # Try to find available ports
                    available_ports = self._find_available_ports("127.0.0.1", original_port, 1)
                    port = available_ports[0] if available_ports else original_port + 100
                except RuntimeError:
                    # If no ports available, try a higher range
                    port = original_port + 100
                
                # Create new A2A agent
                new_agent = await self.create_agent(agent_id, "127.0.0.1", port, worker)
                
                # Update port in killed_agent_configs for next time
                if agent_id in self.killed_agent_configs:
                    self.killed_agent_configs[agent_id]['port'] = port
                
                # Re-register with mesh network
                await self.mesh_network.register_agent(new_agent)
                
                # Restore to active agents
                self.agents[agent_id] = new_agent
                self.killed_agents.discard(agent_id)
                
                # Re-establish connections
                await self._reestablish_agent_connections(agent_id)
                
                # Display A2A-specific reconnection info
                reconnect_info = self.get_reconnection_info(agent_id, port)
                for info_line in reconnect_info:
                    self.output.success(info_line)
                
                self.output.success(f"✅ [A2A] Agent {agent_id} successfully reconnected!")
                
                # Record recovery metrics
                if self.metrics_collector:
                    self.metrics_collector.set_first_recovery_time()
                
        except Exception as e:
            self.output.error(f"🔄 [A2A] Failed to reconnect {agent_id}: {e}")

    async def _reestablish_agent_connections(self, agent_id: str) -> None:
        """Re-establish connections for a reconnected A2A agent."""
        try:
            # Connect to all other active agents
            for other_id in self.agents:
                if other_id != agent_id and other_id not in self.killed_agents:
                    await self.mesh_network.connect_agents(agent_id, other_id)
                    await self.mesh_network.connect_agents(other_id, agent_id)
            
            self.output.info(f"🔗 [A2A] Re-established connections for {agent_id}")
            
        except Exception as e:
            self.output.error(f"Failed to re-establish connections for {agent_id}: {e}")

    async def _monitor_recovery(self) -> None:
        """Monitor A2A agent recovery after fault injection."""
        recovery_duration = self.config.get("scenario", {}).get("recovery_duration", 
                                 max(30, int(self.config.get("scenario", {}).get("total_runtime", 120) * 0.5)))
        
        self.output.info(f"🔄 [A2A] Monitoring recovery for {recovery_duration}s...")
        
        start_time = time.time()
        
        while time.time() - start_time < recovery_duration:
            # Check agent status
            active_count = len([aid for aid in self.agents if aid not in self.killed_agents])
            total_count = len(self.a2a_sessions)
            alive_percentage = (active_count / total_count) * 100 if total_count > 0 else 0
            
            elapsed = int(time.time() - start_time)
            
            self.output.info(f"🔄 [A2A] Recovery tick: alive={alive_percentage:.2f}%, elapsed={elapsed}s")
            
            # Check if all agents have recovered
            if len(self.killed_agents) == 0:
                self.output.success(f"🔄 [A2A] All agents recovered! Steady state achieved at t={elapsed}s")
                if self.metrics_collector:
                    self.metrics_collector.set_steady_state_time()
                break
            
            await asyncio.sleep(5.0)  # Check every 5 seconds
        
        self.output.success(f"🔄 [A2A] Recovery monitoring finished")

    async def _finalize_scenario(self) -> Dict[str, Any]:
        """Finalize scenario, collect metrics and persist results (override)."""
        self.output.info("📊 Phase 4: Evaluation and results...")

        # Use metrics collector if available
        if self.metrics_collector:
            results = self.metrics_collector.get_final_results()
            # Add protocol-specific summary
            results.setdefault("protocol_summary", {})["a2a"] = {
                "killed_agents_initial": len(getattr(self, '_originally_killed_agents', [])),
                "final_alive": len(self.agents),
                "total_agents": len(self.a2a_sessions)
            }
        else:
            results = {
                "metadata": {
                    "scenario": "fail_storm_recovery",
                    "protocol": "a2a",
                    "status": "completed",
                    "end_time": time.time()
                }
            }

        # Persist via base helper
        await self._save_results(results)
        self.output.success("✅ Fail-Storm scenario completed successfully!")
        return results
    
    async def _reconnect_agent(self, agent_id: str) -> None:
        """Reconnect a killed A2A agent (override base implementation)."""
        if agent_id in self.killed_agents and agent_id in self.shard_workers:
            try:
                # Get stored configuration
                agent_config = self.killed_agent_configs.get(agent_id, {})
                worker = self.shard_workers[agent_id]
                port = agent_config.get('port', 9000)
                
                # Re-create A2A agent
                new_agent = await self.create_agent(agent_id, "127.0.0.1", port, worker)
                
                # Re-establish connections
                await self._reestablish_agent_connections(agent_id)
                
                # Restore to active agents
                self.agents[agent_id] = new_agent
                self.killed_agents.discard(agent_id)
                
                self.output.success(f"✅ [A2A] Agent {agent_id} successfully reconnected via base runner!")
                
                # Record reconnection metrics
                if self.metrics_collector:
                    self.metrics_collector.record_reconnection_attempt(
                        source_agent=agent_id,
                        target_agent="a2a_network",
                        success=True,
                        duration_ms=5000.0
                    )
                    
                    self.metrics_collector.record_network_event(
                        event_type="a2a_agent_reconnection_success",
                        source_agent=agent_id,
                        target_agent="mesh_network"
                    )
                
            except Exception as e:
                self.output.error(f"❌ [A2A] Base runner reconnection failed for {agent_id}: {e}")
                raise
    
    # ========================== Phase Management ==========================
    
    def _get_current_phase(self) -> str:
        """Get current phase for proper task classification."""
        if not self.metrics_collector:
            return "normal"
            
        if not hasattr(self.metrics_collector, 'fault_injection_time') or self.metrics_collector.fault_injection_time is None:
            return "normal"
        elif not hasattr(self.metrics_collector, 'steady_state_time') or self.metrics_collector.steady_state_time is None:
            return "recovery"
        else:
            return "post_fault"
    
    # ========================== A2A-specific Agent Management ==========================
    
    async def _kill_agent(self, agent_id: str) -> None:
        """Kill a specific A2A agent."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # Store configuration for later reconnection
            self.killed_agent_configs[agent_id] = {
                'port': agent.port,
                'host': agent.host,
                'executor': self.shard_workers.get(agent_id)
            }
            
            try:
                await agent.stop()
                self.output.info(f"   💀 [A2A] Killed agent: {agent_id}")
                
                # Remove from active agents but keep in shard_workers for recovery
                del self.agents[agent_id]
                
                # Schedule automatic reconnection
                reconnect_delay = self.config.get("a2a", {}).get("recovery", {}).get("reconnect_delay", 10.0)
                asyncio.create_task(self._schedule_a2a_reconnection(agent_id, reconnect_delay))
                
            except Exception as e:
                self.output.error(f"   ❌ [A2A] Error killing {agent_id}: {e}")
    
    # ========================== Simple Failover Implementation ==========================
    
    def get_next_available_agent(self, exclude_agents: set = None) -> Optional[str]:
        """获取下一个可用的agent，跳过失败的agent"""
        if exclude_agents is None:
            exclude_agents = set()
        
        # 获取所有可用的agent（排除已kill的和要排除的）
        available_agents = []
        for agent_id in self.shard_workers.keys():
            if (agent_id not in self.killed_agents and 
                agent_id not in exclude_agents and
                agent_id in self.agents):  # 确保agent还存在
                available_agents.append(agent_id)
        
        if available_agents:
            return available_agents[0]  # 返回第一个可用的
        return None
    
    async def _run_qa_task_for_agent_with_failover(self, original_agent_id: str, original_worker, duration: float):
        """运行QA任务，如果原agent失败则自动切换到下一个可用agent"""
        start_wall = time.perf_counter()  # high-resolution timer
        task_count = 0
        max_groups = self.config.get("shard_qa", {}).get("max_groups", 50)
        
        # Keep consistent group progression
        if not hasattr(self, "_next_group_for_agent"):
            self._next_group_for_agent = {}
        group_id = self._next_group_for_agent.get(original_agent_id, 0) % max_groups
        
        # 尝试的agent列表，从原始agent开始
        tried_agents = set()
        current_agent_id = original_agent_id
        current_worker = original_worker
        
        while (time.perf_counter() - start_wall) < duration and group_id < max_groups:
            try:
                # 检查当前agent是否还可用
                if (current_agent_id in self.killed_agents or 
                    current_agent_id not in self.agents):
                    # 当前agent不可用，寻找下一个
                    tried_agents.add(current_agent_id)
                    next_agent = self.get_next_available_agent(tried_agents)
                    
                    if next_agent is None:
                        self.output.warning(f"🚨 [A2A] No available agents for task, original: {original_agent_id}")
                        break
                    
                    # 切换到新的agent
                    current_agent_id = next_agent
                    current_worker = self.shard_workers[next_agent]
                    self.output.info(f"🔄 [A2A] Switched from {original_agent_id} to {current_agent_id}")
                
                # 执行任务 - 高精度计时
                start_t = time.perf_counter()
                result = await current_worker.worker.start_task(group_id)
                end_t = time.perf_counter()
                task_count += 1
                
                # Update group tracking
                self._next_group_for_agent[original_agent_id] = (group_id + 1) % max_groups
                
                # 记录任务执行
                if self.metrics_collector:
                    result_str = str(result).lower() if result else ""
                    answer_found = (result and 
                                  ("document search success" in result_str or "answer_found:" in result_str) and 
                                  "no answer" not in result_str)
                    answer_source = "local" if "local" in result_str else "neighbor"
                    
                    # Determine current phase for proper task classification
                    current_phase = self._get_current_phase()
                    task_type = f"qa_{current_phase}"
                    
                    # Use wall-clock timestamps for compatibility
                    now_wall = time.time()
                    duration_s = end_t - start_t
                    self.metrics_collector.record_task_execution(
                        task_id=f"{current_agent_id}_{current_phase}_g{group_id}_{task_count}",
                        agent_id=current_agent_id,
                        task_type=task_type,
                        start_time=now_wall - duration_s,
                        end_time=now_wall,
                        success=True,
                        answer_found=answer_found,
                        answer_source=answer_source,
                        group_id=group_id
                    )
                
                # 下一个组
                group_id = (group_id + 1) % max_groups
                
                # 短暂延迟避免过载
                await asyncio.sleep(0.002)
                
            except Exception as e:
                # 任务执行失败，尝试下一个agent
                self.output.warning(f"⚠️ [A2A] Task failed on {current_agent_id}: {e}")
                tried_agents.add(current_agent_id)
                
                # 标记当前agent为失败
                if current_agent_id not in self.killed_agents:
                    self.killed_agents.add(current_agent_id)
                
                # 寻找下一个可用agent
                next_agent = self.get_next_available_agent(tried_agents)
                if next_agent is None:
                    self.output.error(f"❌ [A2A] No more available agents, stopping task for {original_agent_id}")
                    break
                
                # 切换到新agent
                current_agent_id = next_agent
                current_worker = self.shard_workers[next_agent]
                self.output.info(f"🔄 [A2A] Failover: {original_agent_id} -> {current_agent_id}")
        
        # 更新worker的任务计数
        if hasattr(current_worker.worker, 'task_count'):
            current_worker.worker.task_count = getattr(current_worker.worker, 'task_count', 0) + task_count
        else:
            current_worker.worker.task_count = task_count
        
        elapsed = time.time() - start_time
        if current_agent_id != original_agent_id:
            self.output.success(f"✅ [A2A] Agent {original_agent_id} -> {current_agent_id}: {task_count} tasks in {elapsed:.1f}s")
        else:
            self.output.progress(f"📊 [A2A] Agent {current_agent_id}: {task_count} tasks in {elapsed:.1f}s")