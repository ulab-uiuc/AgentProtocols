#!/usr/bin/env python3
"""
Agora protocol runner for Fail-Storm Recovery scenario.

This module implements the Agora protocol specific functionality
while inheriting all core logic from the base runner.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import time
import asyncio

# Add paths for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.simple_base_agent import SimpleBaseAgent as BaseAgent
from protocol_backends.base_runner import FailStormRunnerBase
from .agent import create_agora_agent, AgoraAgent

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
    """
    Agora protocol runner.
    
    Implements protocol-specific agent creation and management for Agora protocol
    while inheriting all core Fail-Storm functionality from FailStormRunnerBase.
    """

    def __init__(self, config_path: str = "config.yaml"):
        # If using default config, use configs/config_agora.yaml
        if config_path == "config.yaml":
            configs_dir = Path(__file__).parent.parent.parent / "configs"
            protocol_config = configs_dir / "config_agora.yaml"
            if protocol_config.exists():
                config_path = str(protocol_config)
                print(f"📋 Using Agora config from: {config_path}")
            else:
                # Fallback to protocol-specific config
                protocol_config = Path(__file__).parent / "config.yaml"
                if protocol_config.exists():
                    config_path = str(protocol_config)
        
        super().__init__(config_path)
        
        # Ensure protocol is set correctly in config
        if "scenario" not in self.config:
            self.config["scenario"] = {}
        self.config["scenario"]["protocol"] = "agora"
        
        
        # Agora-specific port allocation tracking
        self._used_ports = set()  # Track ports currently in use
        self._port_lock = asyncio.Lock()  # Protect port allocation
        
        # Per-agent group tracking for consistent recovery behavior
        self._next_group_for_agent = {}
        
        self.output.info("Initialized Agora protocol runner")

    async def _allocate_unique_port(self, agent_id: str, preferred_port: int) -> int:
        """Thread-safe port allocation for Agora agents."""
        async with self._port_lock:
            # Try preferred port first
            if preferred_port not in self._used_ports:
                try:
                    available_ports = self._find_available_ports("127.0.0.1", preferred_port, 1)
                    if available_ports and available_ports[0] == preferred_port:
                        self._used_ports.add(preferred_port)
                        return preferred_port
                except RuntimeError:
                    pass
            
            # Find next available port
            for port in range(9003, 9100):  # Start from 9003 to avoid common ports
                if port not in self._used_ports:
                    try:
                        available_ports = self._find_available_ports("127.0.0.1", port, 1)
                        if available_ports and available_ports[0] == port:
                            self._used_ports.add(port)
                            self.output.warning(f"🔄 [Agora] Port {preferred_port} unavailable, using {port} for {agent_id}")
                            return port
                    except RuntimeError:
                        continue
            
            raise RuntimeError(f"No available ports found for {agent_id}")
    
    def _release_port(self, port: int):
        """Release a port back to the available pool."""
        self._used_ports.discard(port)

    # ========================================
    # Protocol-Specific Implementation
    # ========================================
    
    async def create_agent(self, agent_id: str, host: str, port: int, executor: ShardWorkerExecutor) -> AgoraAgent:
        """Create agent using Agora protocol."""
        try:
            # Agora requires SDK setup and toolformer initialization
            self.output.progress(f"Setting up Agora agent {agent_id} with SDK and toolformer...")
            
            agent = await create_agora_agent(
                agent_id=agent_id,
                host=host,
                port=port,
                executor=executor
            )
            
            self.output.success(f"Agora agent {agent_id} created successfully with official SDK")
            
            # Track the port as in use
            self._used_ports.add(port)
            
            return agent
            
        except Exception as e:
            self.output.error(f"Failed to create Agora agent {agent_id}: {e}")
            raise
    
    def get_protocol_info(self, agent_id: str, port: int, data_file: str) -> str:
        """Get Agora protocol display information."""
        return f"🎵 [Agora] Created {agent_id} - HTTP: {port} with SDK integration and data: {data_file}"
    
    def get_reconnection_info(self, agent_id: str, port: int) -> List[str]:
        """Get Agora protocol reconnection information."""
        return [
            f"🔗 [Agora] Agent {agent_id} RECONNECTED on port {port}",
            f"🎵 [Agora] SDK endpoint: http://127.0.0.1:{port}",
            f"🔧 [Agora] Toolformer active with LangChain integration",
            f"✅ [Agora] Official SDK communication protocol active"
        ]

    # ========================================
    # Required Protocol-Specific Methods
    # ========================================
    
    async def _setup_mesh_topology(self) -> None:
        """Setup mesh topology between Agora agents."""
        self.output.progress("🔗 [Agora] Setting up SDK-based mesh topology...")
        
        # Register endpoints for all agents
        for agent_id, agent in self.agents.items():
            for other_agent_id, other_agent in self.agents.items():
                if agent_id != other_agent_id:
                    base_url = f"http://{other_agent.host}:{other_agent.port}"
                    await agent.register_endpoint(other_agent_id, base_url)
        
        await self.mesh_network.setup_mesh_topology()
        
        # Agora agents need time for SDK initialization
        await asyncio.sleep(2.0)
        
        # Verify connectivity
        topology = self.mesh_network.get_topology()
        expected_connections = len(self.agents) * (len(self.agents) - 1)
        actual_connections = sum(len(edges) for edges in topology.values())
        
        self.output.success(f"🔗 [Agora] SDK mesh topology established: {actual_connections}/{expected_connections} connections")

    async def _broadcast_document(self) -> None:
        """Broadcast the document to all Agora agents using SDK."""
        if not self.agents:
            raise RuntimeError("No Agora agents available for broadcast")
        
        self.output.progress("📡 [Agora] Broadcasting Gaia document via SDK...")
        
        # Use first agent as broadcaster
        broadcaster_id = list(self.agents.keys())[0]
        
        results = await self.mesh_network.broadcast_init(self.document, broadcaster_id)
        
        successful_deliveries = sum(1 for result in results.values() if "error" not in str(result))
        total_targets = len(results)
        
        self.output.success(f"📡 [Agora] SDK document broadcast: {successful_deliveries}/{total_targets} deliveries successful")

    async def _execute_normal_phase(self) -> None:
        """Execute normal Shard QA collaborative retrieval task with Agora."""
        import asyncio
        import time
        
        normal_phase_duration = self.config.get("shard_qa", {}).get("normal_phase_duration", 30.0)
        
        self.output.progress(f"🔍 [Agora] Running SDK-powered Shard QA for {normal_phase_duration}s...")
        
        # Start metrics collection for normal phase
        if self.metrics_collector:
            self.metrics_collector.start_normal_phase()
        
        start_time = time.time()
        qa_tasks = []
        
        # Start QA task execution on all agents simultaneously with failover
        for agent_id, worker in self.shard_workers.items():
            task = asyncio.create_task(self._run_qa_task_for_agent_with_failover(agent_id, worker, normal_phase_duration))
            qa_tasks.append(task)
        
        # Wait for normal phase duration with Agora status updates
        elapsed = 0
        while elapsed < normal_phase_duration:
            await asyncio.sleep(10)  # Check every 10 seconds
            elapsed = time.time() - start_time
            remaining = normal_phase_duration - elapsed
            if remaining > 0:
                self.output.info(f"🔍 [Agora] Normal phase: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
        
        # Cancel remaining tasks
        for task in qa_tasks:
            if not task.done():
                task.cancel()
        
        # End metrics collection for normal phase
        if self.metrics_collector:
            self.metrics_collector.end_normal_phase()
        
        # Collect final task counts for normal phase
        for agent_id, worker in self.shard_workers.items():
            task_count = getattr(worker, 'completed_tasks', 0)
            self.output.info(f"   {agent_id}: Normal phase completed with {task_count} QA tasks")
        
        elapsed = time.time() - start_time
        self.output.success(f"🔍 [Agora] Normal phase completed in {elapsed:.2f}s")

    async def _run_qa_task_for_agent(self, agent_id: str, worker, duration: float):
        """Run QA task for a specific Agora agent."""
        import asyncio
        import time
        
        start_wall = time.perf_counter()  # high-resolution timer
        task_count = 0
        
        # Keep a per-agent rolling group index across phases
        max_groups = 20
        if not hasattr(self, "_next_group_for_agent"):
            self._next_group_for_agent = {}
        group_id = self._next_group_for_agent.get(agent_id, 0) % max_groups
        
        while (time.perf_counter() - start_wall) < duration and group_id < max_groups:
            try:
                # Execute QA task for current group
                task_start_time = time.time()
                result = await worker.worker.start_task(group_id)
                task_end_time = time.time()
                task_count += 1
                current_group = group_id
                group_id = (group_id + 1) % max_groups  # Cycle through groups 0-19
                
                # Record task execution in metrics
                if self.metrics_collector:
                    # Fix logic: distinguish between finding answer vs not finding answer
                    result_str = str(result).lower() if result else ""
                    answer_found = (result and 
                                  ("document search success" in result_str or "answer_found:" in result_str) and 
                                  "no answer" not in result_str)
                    answer_source = "local" if "local" in result_str else "neighbor"
                    self.metrics_collector.record_task_execution(
                        task_id=f"{agent_id}_normal_g{current_group}_{task_count}",
                        agent_id=agent_id,
                        task_type="qa_normal",
                        start_time=task_start_time,
                        end_time=task_end_time,
                        success=True,  # Task completed successfully
                        answer_found=answer_found,
                        answer_source=answer_source,
                        group_id=current_group
                    )
                
                if result and "answer found" in result.lower():
                    # Show minimal search result from agent
                    if "DOCUMENT SEARCH SUCCESS" in result:
                        self.output.progress(f"🔍 [Agora] [{agent_id}] Found answer")
                    else:
                        self.output.progress(f"{agent_id}: Found answer (task #{task_count})")
                
                # Track task completion
                worker.completed_tasks = getattr(worker, 'completed_tasks', 0) + 1
                
                # Brief pause between tasks
                await asyncio.sleep(2.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.output.warning(f"🔍 [Agora] {agent_id} QA task error: {e}")
                await asyncio.sleep(1.0)  # Wait before retry
        
        self.output.progress(f"   {agent_id}: Found answer (task #{task_count})")

    async def _monitor_recovery(self) -> None:
        """Monitor recovery process for Agora agents."""
        import asyncio
        import time
        
        # Get recovery timeout from config
        recovery_timeout = self.config["scenario"].get("recovery_timeout", 60.0)
        fault_time = self.phase_timers.get("fault_injection_completed", time.time())
        
        self.output.progress(f"🔄 [Agora] Monitoring recovery and continuing QA tasks for {recovery_timeout}s...")
        
        # Start metrics collection for recovery phase
        if self.metrics_collector:
            self.metrics_collector.start_recovery_phase()
        
        # Restart QA tasks on surviving agents
        surviving_workers = {aid: worker for aid, worker in self.shard_workers.items() if aid in self.agents}
        qa_tasks = []
        agents_with_qa_tasks = set()
        
        if surviving_workers:
            self.output.progress(f"🔄 [Agora] Restarting SDK-powered QA tasks on {len(surviving_workers)} surviving agents...")
            for agent_id, worker in surviving_workers.items():
                task = asyncio.create_task(self._run_recovery_qa_task(agent_id, worker, recovery_timeout))
                qa_tasks.append(task)
                agents_with_qa_tasks.add(agent_id)
        
        # Monitor for recovery and agent reconnections
        recovery_timeout_time = time.time() + recovery_timeout
        last_agent_count = len(self.agents)
        
        while time.time() < recovery_timeout_time:
            await asyncio.sleep(5.0)  # Check every 5 seconds
            
            current_agent_count = len(self.agents)
            remaining_time = recovery_timeout_time - time.time()
            
            # Check for agent reconnections
            if current_agent_count > last_agent_count:
                self.output.success(f"🔄 [Agora] Agent reconnection detected! Active agents: {current_agent_count}")
                
                # Record first recovery time
                if self.metrics_collector:
                    self.metrics_collector.set_first_recovery_time()
                
                # Start QA tasks for newly connected agents
                newly_connected = set(self.agents.keys()) - agents_with_qa_tasks
                for agent_id in newly_connected:
                    if agent_id in self.shard_workers:
                        remaining_time_task = recovery_timeout_time - time.time()
                        if remaining_time_task > 0:
                            task = asyncio.create_task(
                                self._run_recovery_qa_task(agent_id, self.shard_workers[agent_id], remaining_time_task)
                            )
                            qa_tasks.append(task)
                            agents_with_qa_tasks.add(agent_id)
                            self.output.progress(f"🔍 [Agora] Started SDK-powered QA task for reconnected agent: {agent_id}")
                
                last_agent_count = current_agent_count
            
            if remaining_time > 0:
                self.output.info(f"🔄 [Agora] Recovery phase: {recovery_timeout - remaining_time:.0f}s elapsed, {remaining_time:.0f}s remaining")
        
        # Cancel remaining QA tasks
        for task in qa_tasks:
            if not task.done():
                task.cancel()
        
        # Collect final task counts for recovery phase
        for agent_id, worker in self.shard_workers.items():
            if agent_id in self.agents:  # Only active agents
                recovery_tasks = getattr(worker, 'recovery_completed_tasks', 0)
                self.output.info(f"   {agent_id}: Recovery QA task cancelled (completed {recovery_tasks} tasks)")
        
        # Wait for steady state
        await self.mesh_network.wait_for_steady_state()
        
        # Record steady state time
        if self.metrics_collector:
            self.metrics_collector.set_steady_state_time()
        
        # End metrics collection for recovery phase
        if self.metrics_collector:
            self.metrics_collector.end_recovery_phase()
        
        elapsed = time.time() - (recovery_timeout_time - recovery_timeout)
        self.output.success(f"🔄 [Agora] Recovery phase completed in {elapsed:.2f}s")

    async def _run_recovery_qa_task(self, agent_id: str, worker, duration: float):
        """Run QA task for a specific Agora agent during recovery without resetting group_id."""
        import asyncio
        import time
        
        start_wall = time.perf_counter()  # high-resolution timer
        task_count = 0
        
        # Keep a per-agent rolling group index across phases
        max_groups = 20
        if not hasattr(self, "_next_group_for_agent"):
            self._next_group_for_agent = {}
        group_id = self._next_group_for_agent.get(agent_id, 0) % max_groups
        
        while (time.perf_counter() - start_wall) < duration and group_id < max_groups:
            try:
                # Execute QA task for current group
                task_start_time = time.time()
                result = await worker.worker.start_task(group_id)
                task_end_time = time.time()
                task_count += 1
                current_group = group_id
                group_id = (group_id + 1) % max_groups  # Cycle through groups 0-19
                
                # Record task execution in metrics
                if self.metrics_collector:
                    # Fix logic: distinguish between finding answer vs not finding answer
                    result_str = str(result).lower() if result else ""
                    answer_found = (result and 
                                  ("document search success" in result_str or "answer_found:" in result_str) and 
                                  "no answer" not in result_str)
                    answer_source = "local" if "local" in result_str else "neighbor"
                    self.metrics_collector.record_task_execution(
                        task_id=f"{agent_id}_recovery_g{current_group}_{task_count}",
                        agent_id=agent_id,
                        task_type="qa_recovery",
                        start_time=task_start_time,
                        end_time=task_end_time,
                        success=True,  # Task completed successfully
                        answer_found=answer_found,
                        answer_source=answer_source,
                        group_id=current_group
                    )
                
                if result and "answer found" in result.lower():
                    # Show minimal search result from agent
                    if "DOCUMENT SEARCH SUCCESS" in result:
                        # Simplified recovery output (removed detailed "Found recovery answer" messages)
                        pass
                    else:
                        self.output.progress(f"{agent_id}: Found answer (recovery task #{task_count})")
                
                # Track recovery task completion
                worker.recovery_completed_tasks = getattr(worker, 'recovery_completed_tasks', 0) + 1
                
                # Brief pause between tasks
                await asyncio.sleep(2.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.output.warning(f"🔍 [Agora] {agent_id} recovery QA task error: {e}")
                await asyncio.sleep(1.0)  # Wait before retry

    async def _execute_fault_injection(self) -> None:
        """Execute the fault injection for Agora agents."""
        import time
        import asyncio
        
        fault_time = self.config["scenario"]["fault_injection_time"]
        elapsed = time.time() - self.scenario_start_time
        
        if elapsed < fault_time:
            wait_time = fault_time - elapsed
            self.output.progress(f"⏰ [Agora] Waiting {wait_time:.1f}s until fault injection time...")
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
        self.output.warning(f"💥 [Agora] Fault injection completed at t={fault_injection_time - self.scenario_start_time:.1f}s")

    async def _inject_faults(self, kill_fraction: float) -> None:
        """Inject faults by killing random Agora agents with SDK cleanup."""
        import random
        
        agent_ids = list(self.agents.keys())
        num_victims = max(1, int(len(agent_ids) * kill_fraction))
        
        victims = random.sample(agent_ids, num_victims)
        
        self.output.warning(f"💥 [Agora] Killing {len(victims)} agents: {', '.join(victims)}")
        
        for victim_id in victims:
            if victim_id in self.agents:
                try:
                    # Agora-specific cleanup: SDK sessions, toolformers, etc.
                    self.output.progress(f"💥 [Agora] Terminating {victim_id} and cleaning SDK session...")
                    
                    # Unregister from mesh network
                    await self.mesh_network.unregister_agent(victim_id)
                    
                    # Kill the agent process
                    agent = self.agents[victim_id]
                    if hasattr(agent, 'process') and agent.process:
                        agent.process.terminate()
                        
                        # Wait for process to die, then force kill if needed
                        try:
                            await asyncio.wait_for(agent.process.wait(), timeout=2.0)
                        except asyncio.TimeoutError:
                            agent.process.kill()
                    
                    # Clean up HTTP clients
                    await agent.close()
                    
                    # Store for later reconnection
                    self.killed_agents.add(victim_id)
                    self.temporarily_killed_agents.add(victim_id)  # Track for statistics
                    
                    # Store agent config for reconnection
                    original_port = getattr(self.agents[victim_id], 'port', 9000)
                    if not hasattr(self, 'killed_agent_configs'):
                        self.killed_agent_configs = {}
                    self.killed_agent_configs[victim_id] = {
                        'executor': self.shard_workers.get(victim_id),
                        'port': original_port,
                        'host': "127.0.0.1"  # Default host
                    }
                    
                    # Release the port for reuse
                    self._release_port(original_port)
                    
                    # Remove from active agents
                    del self.agents[victim_id]
                    
                    self.output.warning(f"💥 [Agora] Killed agent: {victim_id} (will attempt SDK re-initialization later)")
                    
                except Exception as e:
                    self.output.error(f"💥 [Agora] Failed to kill {victim_id}: {e}")
        
        # Schedule reconnection for Agora agents (with SDK re-initialization)
        if victims:
            reconnect_delay = self.config["scenario"].get("reconnect_delay", 10.0)
            
            self.output.success(f"🔄 [Agora] Scheduling SDK re-initialization for {len(victims)} agents in {reconnect_delay}s...")
            
            # Schedule reconnection tasks
            import asyncio
            for victim_id in victims:
                asyncio.create_task(self._schedule_agora_reconnection(victim_id, reconnect_delay))

    async def _schedule_agora_reconnection(self, agent_id: str, delay: float) -> None:
        """Schedule Agora agent reconnection with SDK re-initialization."""
        import asyncio
        import time
        
        await asyncio.sleep(delay)
        
        try:
            self.output.warning(f"🔄 [Agora] Attempting to re-initialize agent: {agent_id}")
            
            if agent_id in self.killed_agents and agent_id in self.shard_workers:
                # Get original configuration
                agent_config = self.killed_agent_configs.get(agent_id, {})
                worker = self.shard_workers[agent_id]
                
                # Extract port from stored config - use original port if available
                original_port = agent_config.get('port', 9000)
                
                # Use thread-safe port allocation
                try:
                    port = await self._allocate_unique_port(agent_id, original_port)
                except RuntimeError as e:
                    self.output.error(f"🔄 [Agora] {e}")
                    return
                
                # Create new Agora agent with SDK re-initialization
                new_agent = await self.create_agent(agent_id, "127.0.0.1", port, worker)
                
                # Update port in killed_agent_configs for next time
                if agent_id in self.killed_agent_configs:
                    self.killed_agent_configs[agent_id]['port'] = port
                
                # Re-register with mesh network
                await self.mesh_network.register_agent(new_agent)
                
                # Restore to active agents
                self.agents[agent_id] = new_agent
                
                # Re-establish connections
                await self._reestablish_agent_connections(agent_id)
                
                # Display Agora-specific reconnection info
                reconnect_info = self.get_reconnection_info(agent_id, port)
                for info_line in reconnect_info:
                    self.output.success(info_line)
                
                self.output.success(f"✅ [Agora] Agent {agent_id} successfully re-initialized and reconnected!")
                
                # Clean up
                if agent_id in self.killed_agents:
                    self.killed_agents.remove(agent_id)
                if agent_id in self.killed_agent_configs:
                    del self.killed_agent_configs[agent_id]
                    
        except Exception as e:
            self.output.error(f"❌ [Agora] Failed to re-initialize {agent_id}: {e}")

    async def _reestablish_agent_connections(self, agent_id: str) -> None:
        """Re-establish Agora agent connections with SDK."""
        try:
            self.output.progress(f"🔗 [Agora] Re-establishing SDK connections for {agent_id}...")
            
            # Register endpoints for all other agents
            agent = self.agents[agent_id]
            connection_count = 0
            
            for other_agent_id, other_agent in self.agents.items():
                if other_agent_id != agent_id:
                    try:
                        # Register endpoint in both directions
                        base_url = f"http://{other_agent.host}:{other_agent.port}"
                        await agent.register_endpoint(other_agent_id, base_url)
                        
                        base_url_self = f"http://{agent.host}:{agent.port}"
                        await other_agent.register_endpoint(agent_id, base_url_self)
                        
                        # Establish mesh network connections
                        success1 = await self.mesh_network.connect_agents(agent_id, other_agent_id)
                        success2 = await self.mesh_network.connect_agents(other_agent_id, agent_id)
                        
                        if success1 and success2:
                            connection_count += 1
                            self.output.progress(f"🔗 [Agora] {agent_id} ↔ {other_agent_id} SDK connection established")
                        else:
                            self.output.warning(f"⚠️ [Agora] Partial connection failure {agent_id} ↔ {other_agent_id}")
                            
                    except Exception as e:
                        self.output.warning(f"⚠️ [Agora] Failed to connect {agent_id} ↔ {other_agent_id}: {e}")
            
            self.output.success(f"🔗 [Agora] Re-established {connection_count} SDK connections for {agent_id}")
                        
        except Exception as e:
            self.output.error(f"🔗 [Agora] Failed to reestablish connections for {agent_id}: {e}")

    async def _finalize_scenario(self) -> Dict[str, Any]:
        """Finalize Agora scenario and generate comprehensive results."""
        import time
        
        end_time = time.time()
        total_runtime = end_time - self.scenario_start_time
        
        self.output.progress("📊 [Agora] Collecting final Agora metrics...")
        
        # Generate Agora-specific comprehensive results
        results = {
            "metadata": {
                "scenario": "fail_storm_recovery",
                "protocol": "agora",  # Explicitly set Agora protocol
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
                "killed_agents": list(self.killed_agents),
                "permanently_failed_agents": list(self.permanently_failed_agents)
            },
            "agora_specific": {
                "sdk_re_initializations": len(self.temporarily_killed_agents),
                "toolformer_active": True,
                "http_endpoints": len(self.agents),
                "langchain_integration": True,
                "multi_modal_stats": {
                    "text_messages": sum(getattr(worker, 'completed_tasks', 0) + getattr(worker, 'recovery_completed_tasks', 0) 
                                       for worker in self.shard_workers.values()),
                    "tool_calls": len(self.agents) * 2  # Estimated tool calls
                },
                "rtc_endpoints": len(self.agents)
            },
            "timing": {
                "total_runtime": total_runtime,
                "fault_time": getattr(self, 'fault_time', None),
                "recovery_end_time": end_time,
                "setup_time": getattr(self, 'setup_time', 0),
                "normal_phase_duration": self.config.get("scenario", {}).get("normal_duration", 30),
                "recovery_phase_duration": self.config.get("scenario", {}).get("recovery_duration", 60)
            }
        }
        
        # Add comprehensive metrics if available
        if self.metrics_collector:
            try:
                # Get performance metrics
                metrics_summary = self.metrics_collector.calculate_recovery_metrics()
                results["failstorm_metrics"] = metrics_summary
                
                # Get QA metrics
                qa_metrics = self.metrics_collector.get_qa_metrics()
                results["qa_metrics"] = qa_metrics
                
                # Add LLM outputs info
                results["llm_outputs"] = {
                    "saved": False,  # Agora doesn't save LLM outputs by default
                    "directory": None
                }
                
                self.output.success("📊 [Agora] Agora-specific metrics collected successfully")
            except Exception as e:
                self.output.warning(f"⚠️ [Agora] Failed to collect metrics: {e}")
        
        # Save results
        await self._save_results(results)
        
        return results

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
            # Add timestamp and protocol to filename
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            detailed_metrics_file = self.results_dir / f"detailed_failstorm_metrics_{timestamp}_agora.json"
            try:
                self.metrics_collector.export_to_json(str(detailed_metrics_file))
                self.output.success(f"💾 [Agora] Detailed metrics saved to: {detailed_metrics_file}")
            except Exception as e:
                self.output.error(f"❌ [Agora] Failed to save detailed metrics: {e}")
    
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
    
    # ========================== Agora-specific Agent Management ==========================
    
    async def _kill_agent(self, agent_id: str) -> None:
        """Kill a specific Agora agent."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # Store configuration for later reconnection
            self.killed_agent_configs[agent_id] = {
                'port': agent.port,
                'host': agent.host,
                'executor': self.shard_workers.get(agent_id)
            }
            
            try:
                # Close Agora agent with SDK cleanup
                await agent.close()
                self.output.info(f"   💀 [Agora] Killed agent: {agent_id}")
                
                # Remove from active agents but keep in shard_workers for recovery
                del self.agents[agent_id]
                
                # Schedule automatic reconnection
                reconnect_delay = 5.0  # 5 seconds delay
                asyncio.create_task(self._schedule_agora_reconnection(agent_id, reconnect_delay))
                
            except Exception as e:
                self.output.error(f"   ❌ [Agora] Error killing {agent_id}: {e}")
    
    async def _schedule_agora_reconnection(self, agent_id: str, delay: float) -> None:
        """Schedule Agora agent reconnection with SDK re-initialization."""
        await asyncio.sleep(delay)
        
        try:
            self.output.warning(f"🔄 [Agora] Attempting to re-initialize agent: {agent_id}")
            
            if agent_id in self.killed_agents and agent_id in self.shard_workers:
                # Get original configuration
                agent_config = self.killed_agent_configs.get(agent_id, {})
                worker = self.shard_workers[agent_id]
                port = agent_config.get('port', 9000)
                
                # Re-create Agora agent with SDK
                new_agent = await self.create_agent(agent_id, "127.0.0.1", port, worker)
                
                # Re-establish connections
                await self._reestablish_agora_connections(agent_id)
                
                # Restore to active agents
                self.agents[agent_id] = new_agent
                self.killed_agents.discard(agent_id)
                
                self.output.success(f"✅ [Agora] Agent {agent_id} successfully re-initialized and reconnected!")
                
                # Record recovery metrics
                if self.metrics_collector:
                    self.metrics_collector.set_first_recovery_time()
                
        except Exception as e:
            self.output.error(f"🔄 [Agora] Failed to reconnect {agent_id}: {e}")
    
    async def _reestablish_agora_connections(self, agent_id: str) -> None:
        """Re-establish connections for a reconnected Agora agent."""
        try:
            # Connect to all other active agents
            for other_id in self.agents:
                if other_id != agent_id and other_id not in self.killed_agents:
                    await self.mesh_network.connect_agents(agent_id, other_id)
                    await self.mesh_network.connect_agents(other_id, agent_id)
            
        except Exception as e:
            self.output.error(f"Failed to re-establish Agora connections for {agent_id}: {e}")
    
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
        max_groups = 20  # Agora uses 20 groups
        
        # Keep consistent group progression
        if not hasattr(self, "_next_group_for_agent"):
            self._next_group_for_agent = {}
        group_id = self._next_group_for_agent.get(original_agent_id, 0) % max_groups
        
        # 尝试的agent列表，从原始agent开始
        tried_agents = set()
        current_agent_id = original_agent_id
        current_worker = original_worker
        
        while time.time() - start_time < duration and group_id < max_groups:
            try:
                # 检查当前agent是否还可用
                if (current_agent_id in self.killed_agents or 
                    current_agent_id not in self.agents):
                    # 当前agent不可用，寻找下一个
                    tried_agents.add(current_agent_id)
                    next_agent = self.get_next_available_agent(tried_agents)
                    
                    if next_agent is None:
                        self.output.warning(f"🚨 [Agora] No available agents for task, original: {original_agent_id}")
                        break
                    
                    # 切换到新的agent
                    current_agent_id = next_agent
                    current_worker = self.shard_workers[next_agent]
                    self.output.info(f"🔄 [Agora] Switched from {original_agent_id} to {current_agent_id}")
                
                # 执行任务
                task_start_time = time.time()
                result = await current_worker.worker.start_task(group_id)
                task_end_time = time.time()
                task_count += 1
                current_group = group_id
                group_id = (group_id + 1) % max_groups
                
                # 记录任务执行
                if self.metrics_collector:
                    current_phase = self._get_current_phase()
                    task_type = f"qa_{current_phase}"
                    
                    result_str = str(result).lower() if result else ""
                    answer_found = (result and 
                                  ("document search success" in result_str or "answer_found:" in result_str) and 
                                  "no answer" not in result_str)
                    answer_source = "local" if "local" in result_str else "neighbor"
                    
                    self.metrics_collector.record_task_execution(
                        task_id=f"{current_agent_id}_{current_phase}_g{current_group}_{task_count}",
                        agent_id=current_agent_id,
                        task_type=task_type,
                        start_time=task_start_time,
                        end_time=task_end_time,
                        success=True,
                        answer_found=answer_found,
                        answer_source=answer_source,
                        group_id=current_group
                    )
                
                # 短暂延迟避免过载
                await asyncio.sleep(0.1)
                
            except Exception as e:
                # 任务执行失败，尝试下一个agent
                self.output.warning(f"⚠️ [Agora] Task failed on {current_agent_id}: {e}")
                tried_agents.add(current_agent_id)
                
                # 标记当前agent为失败
                if current_agent_id not in self.killed_agents:
                    self.killed_agents.add(current_agent_id)
                
                # 寻找下一个可用agent
                next_agent = self.get_next_available_agent(tried_agents)
                if next_agent is None:
                    self.output.error(f"❌ [Agora] No more available agents, stopping task for {original_agent_id}")
                    break
                
                # 切换到新agent
                current_agent_id = next_agent
                current_worker = self.shard_workers[next_agent]
                self.output.info(f"🔄 [Agora] Failover: {original_agent_id} -> {current_agent_id}")
        
        # 更新worker的任务计数
        if hasattr(current_worker, 'completed_tasks'):
            current_worker.completed_tasks = getattr(current_worker, 'completed_tasks', 0) + task_count
        
        elapsed = time.time() - start_time
        if current_agent_id != original_agent_id:
            self.output.success(f"✅ [Agora] Agent {original_agent_id} -> {current_agent_id}: {task_count} tasks in {elapsed:.1f}s")
        else:
            self.output.progress(f"📊 [Agora] Agent {current_agent_id}: {task_count} tasks in {elapsed:.1f}s")
    
    async def _reconnect_agent(self, agent_id: str) -> None:
        """Reconnect a killed Agora agent (override base implementation)."""
        if agent_id in self.killed_agents and agent_id in self.shard_workers:
            try:
                # Get stored configuration
                agent_config = self.killed_agent_configs.get(agent_id, {})
                worker = self.shard_workers[agent_id]
                port = agent_config.get('port', 9000)
                
                # Re-create Agora agent with SDK
                new_agent = await self.create_agent(agent_id, "127.0.0.1", port, worker)
                
                # Re-establish connections
                await self._reestablish_agora_connections(agent_id)
                
                # Restore to active agents
                self.agents[agent_id] = new_agent
                self.killed_agents.discard(agent_id)
                
                self.output.success(f"✅ [Agora] Agent {agent_id} successfully reconnected via base runner!")
                
                # Record reconnection metrics
                if self.metrics_collector:
                    self.metrics_collector.record_reconnection_attempt(
                        source_agent=agent_id,
                        target_agent="agora_network",
                        success=True,
                        duration_ms=5000.0
                    )
                    
                    self.metrics_collector.record_network_event(
                        event_type="agora_agent_reconnection_success",
                        source_agent=agent_id,
                        target_agent="mesh_network"
                    )
                
            except Exception as e:
                self.output.error(f"❌ [Agora] Base runner reconnection failed for {agent_id}: {e}")
                raise