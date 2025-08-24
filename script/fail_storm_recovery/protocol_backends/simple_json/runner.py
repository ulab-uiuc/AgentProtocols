#!/usr/bin/env python3
"""
Simple JSON protocol runner for Fail-Storm Recovery scenario.

This module implements the simple_json protocol-specific functionality
while inheriting all core logic from the base runner.
"""

from pathlib import Path
from typing import Dict, List, Any
import sys
import time

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


class SimpleJsonRunner(FailStormRunnerBase):
    """
    Simple JSON protocol runner.
    
    Implements protocol-specific agent creation and management for simple_json protocol
    while inheriting all core Fail-Storm functionality from FailStormRunnerBase.
    """

    def __init__(self, config_path: str = "config.yaml"):
        # If using default config, try protocol-specific config first
        if config_path == "config.yaml":
            protocol_config = Path(__file__).parent / "config.yaml"
            if protocol_config.exists():
                config_path = str(protocol_config)
        
        super().__init__(config_path)
        self.output.info("Initialized Simple JSON protocol runner")

    # ========================================
    # Protocol-Specific Implementation
    # ========================================
    
    async def create_agent(self, agent_id: str, host: str, port: int, executor: ShardWorkerExecutor) -> BaseAgent:
        """Create agent using simple_json protocol."""
        return await BaseAgent.create_simple_json(
            agent_id=agent_id,
            host=host,
            port=port,
            executor=executor
        )
    
    def get_protocol_info(self, agent_id: str, port: int, data_file: str) -> str:
        """Get simple_json protocol display information."""
        return f"Created Simple JSON agent: {agent_id} on port {port} with data: {data_file}"
    
    def get_reconnection_info(self, agent_id: str, port: int) -> List[str]:
        """Get simple_json protocol reconnection information."""
        return [f"🔗 [SIMPLE_JSON] Agent {agent_id} RECONNECTED on port {port}"]

    # ========================================
    # Extended Implementation for Complete Functionality
    # ========================================
    
    async def _setup_mesh_topology(self) -> None:
        """Setup full mesh topology between agents."""
        await self.mesh_network.setup_mesh_topology()
        
        # Wait for topology to stabilize
        import asyncio
        await asyncio.sleep(2.0)
        
        # Verify connectivity
        topology = self.mesh_network.get_topology()
        expected_connections = len(self.agents) * (len(self.agents) - 1)
        actual_connections = sum(len(edges) for edges in topology.values())
        
        self.output.success(f"Mesh topology established: {actual_connections}/{expected_connections} connections")

    async def _broadcast_document(self) -> None:
        """Broadcast the document to all agents."""
        if not self.agents:
            raise RuntimeError("No agents available for broadcast")
        
        # Use first agent as broadcaster
        broadcaster_id = list(self.agents.keys())[0]
        
        results = await self.mesh_network.broadcast_init(self.document, broadcaster_id)
        
        successful_deliveries = sum(1 for result in results.values() if "error" not in str(result))
        total_targets = len(results)
        
        self.output.success(f"Document broadcast: {successful_deliveries}/{total_targets} deliveries successful")

    async def _execute_normal_phase(self) -> None:
        """Execute normal Shard QA collaborative retrieval task."""
        import asyncio
        
        normal_phase_duration = self.config.get("shard_qa", {}).get("normal_phase_duration", 30.0)
        
        self.output.progress(f"Running Shard QA collaborative retrieval for {normal_phase_duration}s...")
        
        # Start metrics collection for normal phase
        if self.metrics_collector:
            self.metrics_collector.start_normal_phase()
        
        start_time = time.time()
        qa_tasks = []
        
        # Start QA task execution on all agents simultaneously
        for agent_id, worker in self.shard_workers.items():
            task = asyncio.create_task(self._run_qa_task_for_agent(agent_id, worker, normal_phase_duration))
            qa_tasks.append(task)
        
        # Wait for normal phase duration or until shutdown
        elapsed = 0
        while elapsed < normal_phase_duration and not self.shutdown_event.is_set():
            await asyncio.sleep(1.0)
            elapsed = time.time() - start_time
            
            # Progress indicator every 10 seconds
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                remaining = normal_phase_duration - elapsed
                self.output.progress(f"Normal phase: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
        
        # Stop all QA tasks
        for task in qa_tasks:
            task.cancel()
        
        # Wait for tasks to complete cancellation
        await asyncio.gather(*qa_tasks, return_exceptions=True)
        
        # End metrics collection for normal phase
        if self.metrics_collector:
            self.metrics_collector.end_normal_phase()
        
        self.phase_timers["normal_phase_completed"] = time.time()
        elapsed = self.phase_timers["normal_phase_completed"] - start_time
        self.output.success(f"Normal phase completed in {elapsed:.2f}s")
    
    async def _run_qa_task_for_agent(self, agent_id: str, worker: 'ShardWorkerExecutor', duration: float) -> None:
        """Run continuous QA tasks for a single agent during normal phase."""
        import asyncio
        import time
        
        start_time = time.time()
        task_count = 0
        
        try:
            while (time.time() - start_time) < duration and not self.shutdown_event.is_set():
                try:
                    # Execute QA task for group 0 (standard test case)
                    result = await worker.worker.start_task(0)
                    task_count += 1
                    
                    if result and "answer found" in result.lower():
                        # Show minimal search result from agent
                        if "DOCUMENT SEARCH SUCCESS" in result:
                            self.output.progress(f"🔍 [{agent_id}] Found answer")
                        else:
                            self.output.progress(f"{agent_id}: Found answer (task #{task_count})")
                    
                    # Brief pause between tasks
                    await asyncio.sleep(2.0)
                    
                except Exception as e:
                    self.output.warning(f"{agent_id}: QA task failed: {e}")
                    await asyncio.sleep(1.0)
                    
        except asyncio.CancelledError:
            self.output.progress(f"{agent_id}: QA task cancelled (completed {task_count} tasks)")
            raise
        except Exception as e:
            self.output.error(f"{agent_id}: QA task error: {e}")
        
        self.output.progress(f"{agent_id}: Normal phase completed with {task_count} QA tasks")

    async def _execute_fault_injection(self) -> None:
        """Execute the fault injection."""
        import time
        import asyncio
        
        fault_time = self.config["scenario"]["fault_injection_time"]
        elapsed = time.time() - self.scenario_start_time
        
        if elapsed < fault_time:
            wait_time = fault_time - elapsed
            self.output.progress(f"Waiting {wait_time:.1f}s until fault injection time...")
            await asyncio.sleep(wait_time)
        
        # Record fault injection time
        fault_injection_time = time.time()
        if self.metrics_collector:
            self.metrics_collector.set_fault_injection_time(fault_injection_time)
        
        if self.mesh_network:
            self.mesh_network.set_fault_injection_time(fault_injection_time)
        
        # Execute fault injection
        kill_fraction = self.config["scenario"]["kill_fraction"]
        await self._inject_faults(kill_fraction)
        
        self.phase_timers["fault_injection_completed"] = time.time()
        self.output.warning(f"Fault injection completed at t={fault_injection_time - self.scenario_start_time:.1f}s")

    async def _inject_faults(self, kill_fraction: float) -> None:
        """Inject faults by killing random agents with capability to reconnect."""
        import random
        
        agent_ids = list(self.agents.keys())
        num_victims = max(1, int(len(agent_ids) * kill_fraction))
        
        victims = random.sample(agent_ids, num_victims)
        
        self.output.warning(f"Killing {len(victims)} agents: {', '.join(victims)}")
        
        # Kill agents but save their configs for reconnection
        killed_agents = set()
        for agent_id in victims:
            try:
                agent = self.agents[agent_id]
                
                # Save agent config for reconnection
                agent_config = {
                    "agent_id": agent_id,
                    "host": agent.host,
                    "port": agent.port,
                    "protocol": self.config["scenario"]["protocol"],
                    "shard_worker": self.shard_workers.get(agent_id),
                    "neighbors": {
                        "prev_id": f"agent{(int(agent_id.replace('agent', '')) - 1) % self.config['scenario']['agent_count']}",
                        "next_id": f"agent{(int(agent_id.replace('agent', '')) + 1) % self.config['scenario']['agent_count']}"
                    }
                }
                self.killed_agent_configs[agent_id] = agent_config
                
                # Stop the agent
                await agent.stop()
                
                # Remove from network
                await self.mesh_network.unregister_agent(agent_id)
                
                # Remove from active tracking but keep worker
                del self.agents[agent_id]
                killed_agents.add(agent_id)
                
                # Update agent state in metrics
                if self.metrics_collector:
                    self.metrics_collector.update_agent_state(agent_id, "failed")
                
                self.output.progress(f"✗ Killed agent: {agent_id} (will attempt reconnection later)")
                
            except Exception as e:
                self.output.error(f"Failed to kill agent {agent_id}: {e}")
        
        self.killed_agents = killed_agents
        self.temporarily_killed_agents.update(killed_agents)
        
        # Trigger network recovery
        if self.mesh_network:
            for agent_id in killed_agents:
                await self.mesh_network._handle_node_failure(agent_id)
        
        # Schedule agent reconnections if enabled
        if self.config["scenario"].get("enable_reconnection", True):
            reconnect_delay = self.config["scenario"].get("reconnection_delay", 10.0)
            import asyncio
            asyncio.create_task(self._schedule_agent_reconnections(reconnect_delay))

    async def _monitor_recovery(self) -> None:
        """Monitor recovery and continue QA tasks."""
        import asyncio
        import time
        
        total_runtime = self.config["scenario"]["total_runtime"]
        fault_time = self.config["scenario"]["fault_injection_time"]
        recovery_duration = total_runtime - fault_time
        
        recovery_start = time.time()
        recovery_timeout = recovery_start + recovery_duration
        
        self.output.progress(f"Monitoring recovery and continuing QA tasks for {recovery_duration}s...")
        
        # Start metrics collection for recovery phase
        if self.metrics_collector:
            self.metrics_collector.start_recovery_phase()
        
        # Track recovery state
        first_recovery_detected = False
        qa_tasks = []
        
        # Restart QA tasks on surviving agents
        surviving_workers = {aid: worker for aid, worker in self.shard_workers.items() 
                           if aid in self.agents and aid not in self.killed_agents}
        
        agents_with_qa_tasks = set()
        
        if surviving_workers:
            self.output.progress(f"Restarting QA tasks on {len(surviving_workers)} surviving agents...")
            for agent_id, worker in surviving_workers.items():
                task = asyncio.create_task(self._run_recovery_qa_task(agent_id, worker, recovery_duration))
                qa_tasks.append(task)
                agents_with_qa_tasks.add(agent_id)
        
        # Monitor recovery
        last_agent_count = len(self.agents)
        while time.time() < recovery_timeout and not self.shutdown_event.is_set():
            # Check for recovery indicators
            if not first_recovery_detected and self.mesh_network:
                topology_health = self.mesh_network.get_topology_health()
                alive_agents = topology_health.get("alive_agents", [])
                
                if len(alive_agents) > 0:
                    avg_connectivity = sum(
                        status.get("connectivity_ratio", 0) 
                        for status in topology_health.get("connectivity_status", {}).values()
                    ) / max(len(alive_agents), 1)
                    
                    if avg_connectivity > 0.7:  # 70% connectivity restored
                        if self.metrics_collector:
                            self.metrics_collector.set_first_recovery_time()
                            first_recovery_detected = True
                            self.output.success("🔄 First recovery signs detected!")
            
            # Check if any agents have reconnected
            current_agent_count = len(self.agents)
            if current_agent_count > last_agent_count:
                self.output.success(f"🔄 Agent reconnection detected! Active agents: {current_agent_count}")
                # Start QA tasks for newly reconnected agents
                newly_connected = set(self.agents.keys()) - agents_with_qa_tasks
                for agent_id in newly_connected:
                    if agent_id in self.shard_workers:
                        remaining_time = recovery_timeout - time.time()
                        if remaining_time > 0:
                            task = asyncio.create_task(
                                self._run_recovery_qa_task(agent_id, self.shard_workers[agent_id], remaining_time)
                            )
                            qa_tasks.append(task)
                            agents_with_qa_tasks.add(agent_id)
                            self.output.progress(f"Started QA task for reconnected agent: {agent_id}")
                last_agent_count = current_agent_count
            
            elapsed = time.time() - recovery_start
            remaining = recovery_duration - elapsed
            if int(elapsed) % 15 == 0 and int(elapsed) > 0:
                self.output.progress(f"Recovery phase: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
            
            await asyncio.sleep(2.0)
        
        # Stop all QA tasks
        for task in qa_tasks:
            task.cancel()
        await asyncio.gather(*qa_tasks, return_exceptions=True)
        
        # Wait for steady state
        if self.mesh_network:
            try:
                steady_time = await self.mesh_network.wait_for_steady_state(min_stability_time=5.0)
                if self.metrics_collector:
                    self.metrics_collector.set_steady_state_time()
                self.output.success(f"🎯 Steady state reached in {steady_time:.2f}s")
            except asyncio.TimeoutError:
                self.output.warning("Steady state not reached within timeout")
        
        # End metrics collection for recovery phase
        if self.metrics_collector:
            self.metrics_collector.end_recovery_phase()
        
        self.phase_timers["recovery_completed"] = time.time()
        total_elapsed = self.phase_timers["recovery_completed"] - recovery_start
        self.output.success(f"Recovery phase completed in {total_elapsed:.2f}s")

    async def _schedule_agent_reconnections(self, delay: float) -> None:
        """Schedule reconnection attempts for killed agents."""
        import asyncio
        
        if not self.killed_agent_configs:
            return
        
        self.output.progress(f"Scheduling reconnection for {len(self.killed_agent_configs)} agents in {delay}s...")
        await asyncio.sleep(delay)
        
        for agent_id, config in self.killed_agent_configs.items():
            if self.shutdown_event.is_set():
                break
            
            self.output.warning(f"🔄 Attempting to reconnect agent: {agent_id}")
            success = await self._reconnect_agent(agent_id, config)
            
            if success:
                self.output.success(f"✅ Agent {agent_id} successfully reconnected!")
                self.killed_agents.discard(agent_id)
                
                # Update metrics
                if self.metrics_collector:
                    self.metrics_collector.update_agent_state(agent_id, "recovering")
            else:
                self.output.error(f"❌ Failed to reconnect agent: {agent_id}")
            
            await asyncio.sleep(2.0)

    async def _reconnect_agent(self, agent_id: str, config: Dict[str, Any]) -> bool:
        """Reconnect a previously killed agent."""
        try:
            host = config["host"]
            original_port = config["port"]
            
            # Find new available port
            try:
                available_ports = self._find_available_ports(host, original_port, 1)
                port = available_ports[0]
                if port != original_port:
                    self.output.warning(f"Port {original_port} still in use, using {port} for {agent_id}")
            except RuntimeError:
                self.output.error(f"No available ports found for reconnecting {agent_id}")
                return False
            
            # Get shard worker
            shard_worker = config["shard_worker"]
            if not shard_worker:
                self.output.error(f"No shard worker found for {agent_id}")
                return False
            
            # Recreate agent using protocol-specific method
            agent = await self.create_agent(agent_id, host, port, shard_worker)
            
            # Re-register to mesh network
            await self.mesh_network.register_agent(agent)
            self.agents[agent_id] = agent
            
            # Re-establish connections
            await self._reestablish_agent_connections(agent_id)
            
            # Display protocol-specific reconnection information
            reconnection_messages = self.get_reconnection_info(agent_id, port)
            for message in reconnection_messages:
                self.output.success(message)
            
            return True
            
        except Exception as e:
            self.output.error(f"Reconnection failed for {agent_id}: {e}")
            return False

    async def _reestablish_agent_connections(self, agent_id: str) -> None:
        """Re-establish agent connections with other nodes."""
        try:
            # Connect to all surviving agents
            for other_agent_id in self.agents.keys():
                if other_agent_id != agent_id:
                    try:
                        await self.mesh_network.connect_agents(agent_id, other_agent_id)
                        await self.mesh_network.connect_agents(other_agent_id, agent_id)
                    except Exception as e:
                        self.output.warning(f"Failed to connect {agent_id} ↔ {other_agent_id}: {e}")
                        
        except Exception as e:
            self.output.error(f"Failed to reestablish connections for {agent_id}: {e}")

    async def _finalize_scenario(self) -> Dict[str, Any]:
        """Finalize the scenario and generate results."""
        import time
        
        end_time = time.time()
        total_runtime = end_time - self.scenario_start_time
        
        self.output.progress("Collecting final metrics...")
        
        # Generate comprehensive results
        results = {
            "metadata": {
                "scenario": "fail_storm_recovery",
                "protocol": self.config["scenario"]["protocol"],
                "start_time": self.scenario_start_time,
                "end_time": end_time,
                "total_runtime": total_runtime,
                "config": self.config
            },
            "agent_summary": {
                "initial_count": self.config["scenario"]["agent_count"],
                "temporarily_killed_count": len(self.temporarily_killed_agents),
                "currently_killed_count": len(self.killed_agents),
                "permanently_failed_count": len(self.permanently_failed_agents),
                "surviving_count": len(self.agents),
                "reconnected_count": len(self.temporarily_killed_agents) - len(self.killed_agents),
                "temporarily_killed_agents": list(self.temporarily_killed_agents),
                "currently_killed_agents": list(self.killed_agents),
                "permanently_failed_agents": list(self.permanently_failed_agents),
                "surviving_agents": list(self.agents.keys())
            },
            "timing": {
                "setup_time": self.phase_timers.get("setup_completed", 0) - self.scenario_start_time,
                "normal_phase_time": self.phase_timers.get("normal_phase_completed", 0) - self.scenario_start_time,
                "fault_injection_time": self.phase_timers.get("fault_injection_completed", 0) - self.scenario_start_time,
                "recovery_time": self.phase_timers.get("recovery_completed", 0) - self.scenario_start_time
            },
            "network_metrics": {},
            "failstorm_metrics": {},
            "llm_outputs": {
                "saved": False,
                "directory": "disabled"
            }
        }
        
        # Collect network metrics
        if self.mesh_network:
            results["network_metrics"] = self.mesh_network.get_failure_metrics()
        
        # Collect failstorm metrics
        if self.metrics_collector:
            results["failstorm_metrics"] = self.metrics_collector.calculate_recovery_metrics()
            results["performance_analysis"] = self.metrics_collector.get_performance_summary()
            results["qa_metrics"] = self.metrics_collector.get_qa_metrics()
        
        # Save results
        await self._save_results(results)
        
        return results

    async def _save_results(self, results: Dict[str, Any]) -> None:
        """Save scenario results to files."""
        import json
        
        # Save main results file
        results_file = self.results_dir / self.config["output"]["results_file"]
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.output.success(f"📁 Results saved to: {results_file}")
            
        except Exception as e:
            self.output.error(f"Failed to save results: {e}")
        
        # Save detailed metrics if available
        if self.metrics_collector:
            detailed_metrics_file = self.results_dir / "detailed_failstorm_metrics.json"
            try:
                self.metrics_collector.export_to_json(str(detailed_metrics_file))
            except Exception as e:
                self.output.error(f"Failed to save detailed metrics: {e}")
    
    async def _run_recovery_qa_task(self, agent_id: str, worker: 'ShardWorkerExecutor', duration: float) -> None:
        """Run continuous QA tasks for a single agent during recovery phase."""
        import asyncio
        import time
        
        start_time = time.time()
        task_count = 0
        
        try:
            while (time.time() - start_time) < duration and not self.shutdown_event.is_set():
                try:
                    # Execute QA task for group 0 and optionally group 1
                    for group_id in [0, 1]:
                        result = await worker.worker.start_task(group_id)
                        task_count += 1
                        
                        if result and "answer found" in result.lower():
                            # Show minimal search result from agent
                            if "DOCUMENT SEARCH SUCCESS" in result:
                                self.output.progress(f"🔍 [{agent_id}] Found answer")
                            else:
                                self.output.progress(f"{agent_id}: Found answer (task #{task_count})")
                            
                            # Check if this might be a final answer
                            if "final" in result.lower() or "complete" in result.lower():
                                self.output.success(f"{agent_id}: Potential final answer found!")
                        
                        # Brief pause between tasks
                        await asyncio.sleep(1.5)
                    
                except Exception as e:
                    self.output.warning(f"{agent_id}: Recovery QA task failed: {e}")
                    await asyncio.sleep(2.0)
                    
        except asyncio.CancelledError:
            self.output.progress(f"{agent_id}: Recovery QA task cancelled (completed {task_count} tasks)")
            raise
        except Exception as e:
            self.output.error(f"{agent_id}: Recovery QA task error: {e}")
        
        self.output.progress(f"{agent_id}: Recovery phase completed with {task_count} QA tasks")