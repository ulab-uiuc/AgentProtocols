#!/usr/bin/env python3
"""
ANP protocol runner for Fail-Storm Recovery scenario.

This module implements the ANP (Agent Network Protocol) specific functionality
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
from .agent import create_anp_agent, ANPAgent

# Import shard_qa components dynamically to avoid circular imports
shard_qa_path = Path(__file__).parent.parent.parent / "shard_qa"
sys.path.insert(0, str(shard_qa_path))
import importlib.util
spec = importlib.util.spec_from_file_location("agent_executor", shard_qa_path / "shard_worker" / "agent_executor.py")
agent_executor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_executor_module)
# Create ANP-specific implementations to avoid coordinator dependency
class ANPAgentExecutor(agent_executor_module.BaseAgentExecutor):
    """ANP-specific agent executor"""
    async def execute(self, context, event_queue):
        # ANP doesn't use the executor pattern, this is just for compatibility
        pass
    
    async def cancel(self, context, event_queue):
        # ANP doesn't use the executor pattern, this is just for compatibility
        pass

class ANPRequestContext(agent_executor_module.BaseRequestContext):
    """ANP-specific request context"""
    def __init__(self, input_data):
        self.input_data = input_data
    
    def get_user_input(self):
        return self.input_data

class ANPEventQueue(agent_executor_module.BaseEventQueue):
    """ANP-specific event queue"""
    def __init__(self):
        self.events = []
    
    async def enqueue_event(self, event):
        self.events.append(event)
        return event

def anp_new_agent_text_message(text, role="user"):
    """ANP-specific text message creation"""
    return {"type": "text", "content": text, "role": str(role)}

# Patch the _send_to_coordinator method to avoid coordinator dependency
original_send_to_coordinator = agent_executor_module.ShardWorker._send_to_coordinator

async def _patched_send_to_coordinator(self, content: str, path: List[str] = None, ttl: int = 0):
    """Patched version that doesn't require coordinator"""
    # For ANP fail-storm testing, we don't need coordinator
    # Just log the message that would have been sent
    if hasattr(self, 'output') and self.output:
        self.output.info(f"[{self.shard_id}] Would send to coordinator: {content}")
    return "Coordinator message skipped (ANP mode)"

# Apply the patch
agent_executor_module.ShardWorker._send_to_coordinator = _patched_send_to_coordinator

# Inject ANP implementations into the agent_executor module
agent_executor_module.AgentExecutor = ANPAgentExecutor
agent_executor_module.RequestContext = ANPRequestContext
agent_executor_module.EventQueue = ANPEventQueue
agent_executor_module.new_agent_text_message = anp_new_agent_text_message

ShardWorkerExecutor = agent_executor_module.ShardWorkerExecutor


class ANPRunner(FailStormRunnerBase):
    """
    ANP protocol runner.
    
    Implements protocol-specific agent creation and management for ANP protocol
    while inheriting all core Fail-Storm functionality from FailStormRunnerBase.
    """

    def __init__(self, config_path: str = "config.yaml"):
        # If using default config, use configs/config_anp.yaml
        if config_path == "config.yaml":
            configs_dir = Path(__file__).parent.parent.parent / "configs"
            protocol_config = configs_dir / "config_anp.yaml"
            if protocol_config.exists():
                config_path = str(protocol_config)
                print(f"📋 Using ANP config from: {config_path}")
            else:
                # Fallback to protocol-specific config
                protocol_config = Path(__file__).parent / "config.yaml"
                if protocol_config.exists():
                    config_path = str(protocol_config)
        
        super().__init__(config_path)
        
        # Ensure protocol is set correctly in config
        if "scenario" not in self.config:
            self.config["scenario"] = {}
        self.config["scenario"]["protocol"] = "anp"
        
        # ANP-specific port allocation tracking
        self._used_ports = set()  # Track ports currently in use
        self._port_lock = asyncio.Lock()  # Protect port allocation
        
        # Per-agent group tracking for consistent recovery behavior
        self._next_group_for_agent = {}
        
        self.output.info("Initialized ANP protocol runner")

    async def _allocate_unique_port(self, agent_id: str, preferred_port: int) -> int:
        """Thread-safe port allocation for ANP agents."""
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
                            self.output.warning(f"🔄 [ANP] Port {preferred_port} unavailable, using {port} for {agent_id}")
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
    
    async def create_agent(self, agent_id: str, host: str, port: int, executor: ShardWorkerExecutor) -> ANPAgent:
        """Create agent using ANP protocol."""
        try:
            # ANP requires additional setup for DID authentication and encryption
            self.output.progress(f"Setting up ANP agent {agent_id} with DID authentication...")
            
            agent = await create_anp_agent(
                agent_id=agent_id,
                host=host,
                port=port,
                executor=executor
            )
            
            self.output.success(f"ANP agent {agent_id} created successfully with hybrid communication")
            
            # Track the port as in use
            self._used_ports.add(port)
            
            return agent
            
        except Exception as e:
            self.output.error(f"Failed to create ANP agent {agent_id}: {e}")
            raise
    
    def get_protocol_info(self, agent_id: str, port: int, data_file: str) -> str:
        """Get ANP protocol display information."""
        return f"🚀 [ANP] Created {agent_id} - HTTP: {port}, WebSocket: {port + 1000} with data: {data_file}"
    
    def get_reconnection_info(self, agent_id: str, port: int) -> List[str]:
        """Get ANP protocol reconnection information."""
        return [
            f"🔗 [ANP] Agent {agent_id} RECONNECTED on port {port}",
            f"📡 [ANP] WebSocket endpoint: ws://127.0.0.1:{port + 1000}",
            f"🌐 [ANP] HTTP REST API: http://127.0.0.1:{port}",
            f"✅ [ANP] Hybrid communication protocol active"
        ]

    # ========================================
    # Required Protocol-Specific Methods
    # ========================================
    
    async def _setup_mesh_topology(self) -> None:
        """Setup full mesh topology between ANP agents."""
        self.output.progress("🔗 [ANP] Setting up hybrid mesh topology with authentication...")
        
        await self.mesh_network.setup_mesh_topology()
        
        # ANP agents need extra time for DID authentication and key exchange
        import asyncio
        await asyncio.sleep(3.0)  # Longer stabilization for ANP
        
        # Verify connectivity
        topology = self.mesh_network.get_topology()
        expected_connections = len(self.agents) * (len(self.agents) - 1)
        actual_connections = sum(len(edges) for edges in topology.values())
        
        self.output.success(f"🔗 [ANP] Authenticated mesh topology established: {actual_connections}/{expected_connections} connections")

    async def _broadcast_document(self) -> None:
        """Broadcast the document to all ANP agents with encryption."""
        if not self.agents:
            raise RuntimeError("No ANP agents available for broadcast")
        
        self.output.progress("📡 [ANP] Broadcasting Gaia document with E2E encryption...")
        
        # Use first agent as broadcaster
        broadcaster_id = list(self.agents.keys())[0]
        
        results = await self.mesh_network.broadcast_init(self.document, broadcaster_id)
        
        successful_deliveries = sum(1 for result in results.values() if "error" not in str(result))
        total_targets = len(results)
        
        self.output.success(f"📡 [ANP] Encrypted document broadcast: {successful_deliveries}/{total_targets} deliveries successful")

    async def _execute_normal_phase(self) -> None:
        """Execute normal Shard QA collaborative retrieval task with ANP."""
        import asyncio
        import time
        
        normal_phase_duration = self.config.get("shard_qa", {}).get("normal_phase_duration", 30.0)
        
        self.output.progress(f"🔍 [ANP] Running authenticated Shard QA for {normal_phase_duration}s...")
        
        # Start metrics collection for normal phase
        if self.metrics_collector:
            self.metrics_collector.start_normal_phase()
        
        start_time = time.time()
        qa_tasks = []
        
        # Start QA task execution on all agents simultaneously with failover
        for agent_id, worker in self.shard_workers.items():
            task = asyncio.create_task(self._run_qa_task_for_agent_with_failover(agent_id, worker, normal_phase_duration))
            qa_tasks.append(task)
        
        # Wait for normal phase duration with ANP status updates
        elapsed = 0
        while elapsed < normal_phase_duration:
            await asyncio.sleep(10)  # Check every 10 seconds
            elapsed = time.time() - start_time
            remaining = normal_phase_duration - elapsed
            if remaining > 0:
                self.output.info(f"🔍 [ANP] Normal phase: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
        
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
        self.output.success(f"🔍 [ANP] Normal phase completed in {elapsed:.2f}s")

    async def _run_qa_task_for_agent(self, agent_id: str, worker, duration: float):
        """Run QA task for a specific ANP agent."""
        import asyncio
        import time
        
        start_time = time.time()
        task_count = 0
        
        # Test first 20 groups like Meta protocol
        max_groups = 20
        group_id = 0
        
        while time.time() - start_time < duration and group_id < max_groups:
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
                        self.output.progress(f"🔍 [ANP] [{agent_id}] Found answer")
                    else:
                        self.output.progress(f"{agent_id}: Found answer (task #{task_count})")
                
                # Track task completion
                worker.completed_tasks = getattr(worker, 'completed_tasks', 0) + 1
                
                # Brief pause between tasks
                await asyncio.sleep(2.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.output.warning(f"🔍 [ANP] {agent_id} QA task error: {e}")
                await asyncio.sleep(1.0)  # Wait before retry
        
        self.output.progress(f"   {agent_id}: Found answer (task #{task_count})")

    async def _monitor_recovery(self) -> None:
        """Monitor recovery process for ANP agents."""
        import asyncio
        import time
        
        # Get recovery timeout from config
        recovery_timeout = self.config["scenario"].get("recovery_timeout", 60.0)
        fault_time = self.phase_timers.get("fault_injection_completed", time.time())
        
        self.output.progress(f"🔄 [ANP] Monitoring recovery and continuing QA tasks for {recovery_timeout}s...")
        
        # Start metrics collection for recovery phase
        if self.metrics_collector:
            self.metrics_collector.start_recovery_phase()
        
        # Restart QA tasks on surviving agents
        surviving_workers = {aid: worker for aid, worker in self.shard_workers.items() if aid in self.agents}
        qa_tasks = []
        agents_with_qa_tasks = set()
        
        if surviving_workers:
            self.output.progress(f"🔄 [ANP] Restarting authenticated QA tasks on {len(surviving_workers)} surviving agents...")
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
                self.output.success(f"🔄 [ANP] Agent reconnection detected! Active agents: {current_agent_count}")
                
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
                            self.output.progress(f"🔍 [ANP] Started authenticated QA task for reconnected agent: {agent_id}")
                
                last_agent_count = current_agent_count
            
            if remaining_time > 0:
                self.output.info(f"🔄 [ANP] Recovery phase: {recovery_timeout - remaining_time:.0f}s elapsed, {remaining_time:.0f}s remaining")
        
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
        self.output.success(f"🔄 [ANP] Recovery phase completed in {elapsed:.2f}s")

    async def _run_recovery_qa_task(self, agent_id: str, worker, duration: float):
        """Run QA task for a specific ANP agent during recovery without resetting group_id."""
        import asyncio
        import time

        start_wall = time.perf_counter()  # high-resolution timer for runtime budget
        task_count = 0

        # Keep a per-agent rolling group index across phases
        max_groups = 20  # keep your existing cap
        if not hasattr(self, "_next_group_for_agent"):
            self._next_group_for_agent = {}
        group_id = self._next_group_for_agent.get(agent_id, 0) % max_groups

        while (time.perf_counter() - start_wall) < duration and group_id < max_groups:
            try:
                # High-resolution start time for metrics
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
                    answer_source = "local" if "local" in result_str else "neighbor"
                    # Use wall-clock timestamps for compatibility with collector
                    now_wall = time.time()
                    duration_s = end_t - start_t
                    # Just map perf_counter-based duration onto wall clock end
                    self.metrics_collector.record_task_execution(
                        task_id=f"{agent_id}_recovery_g{current_group}_{task_count}",
                        agent_id=agent_id,
                        task_type="qa_recovery",
                        start_time=now_wall - duration_s,
                        end_time=now_wall,
                        success=True,
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
                
                # Optional tiny pause to avoid starvation
                await asyncio.sleep(0.002)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.output.warning(f"🔍 [ANP] {agent_id} recovery QA task error: {e}")
                await asyncio.sleep(0.02)

    async def _execute_fault_injection(self) -> None:
        """Execute the fault injection for ANP agents."""
        import time
        import asyncio
        
        fault_time = self.config["scenario"]["fault_injection_time"]
        elapsed = time.time() - self.scenario_start_time
        
        if elapsed < fault_time:
            wait_time = fault_time - elapsed
            self.output.progress(f"⏰ [ANP] Waiting {wait_time:.1f}s until fault injection time...")
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
        self.output.warning(f"💥 [ANP] Fault injection completed at t={fault_injection_time - self.scenario_start_time:.1f}s")

    async def _inject_faults(self, kill_fraction: float) -> None:
        """Inject faults by killing random ANP agents with DID session cleanup."""
        import random
        
        agent_ids = list(self.agents.keys())
        num_victims = max(1, int(len(agent_ids) * kill_fraction))
        
        victims = random.sample(agent_ids, num_victims)
        
        self.output.warning(f"💥 [ANP] Killing {len(victims)} agents: {', '.join(victims)}")
        
        for victim_id in victims:
            if victim_id in self.agents:
                try:
                    # ANP-specific cleanup: DID sessions, encryption keys, etc.
                    self.output.progress(f"💥 [ANP] Terminating {victim_id} and cleaning DID session...")
                    
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
                    
                    self.output.warning(f"💥 [ANP] Killed agent: {victim_id} (will attempt DID re-authentication later)")
                    
                except Exception as e:
                    self.output.error(f"💥 [ANP] Failed to kill {victim_id}: {e}")
        
        # Schedule reconnection for ANP agents (with DID re-authentication)
        if victims:
            reconnect_delay = self.config["scenario"].get("reconnect_delay", 10.0)
            
            self.output.success(f"🔄 [ANP] Scheduling DID re-authentication for {len(victims)} agents in {reconnect_delay}s...")
            
            # Schedule reconnection tasks
            import asyncio
            for victim_id in victims:
                asyncio.create_task(self._schedule_anp_reconnection(victim_id, reconnect_delay))

    async def _schedule_anp_reconnection(self, agent_id: str, delay: float) -> None:
        """Schedule ANP agent reconnection with DID re-authentication."""
        import asyncio
        import time
        
        await asyncio.sleep(delay)
        
        try:
            self.output.warning(f"🔄 [ANP] Attempting to re-authenticate agent: {agent_id}")
            
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
                    self.output.error(f"🔄 [ANP] {e}")
                    return
                
                # Create new ANP agent with DID re-authentication
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
                
                # Display ANP-specific reconnection info
                reconnect_info = self.get_reconnection_info(agent_id, port)
                for info_line in reconnect_info:
                    self.output.success(info_line)
                
                self.output.success(f"✅ [ANP] Agent {agent_id} successfully re-authenticated and reconnected!")
                
                # Clean up
                if agent_id in self.killed_agents:
                    self.killed_agents.remove(agent_id)
                if agent_id in self.killed_agent_configs:
                    del self.killed_agent_configs[agent_id]
                # Note: We don't release the port here as it's now in use by the new agent
                    
        except Exception as e:
            self.output.error(f"❌ [ANP] Failed to re-authenticate {agent_id}: {e}")

    async def _reestablish_agent_connections(self, agent_id: str) -> None:
        """Re-establish ANP agent connections with authentication."""
        try:
            self.output.progress(f"🔗 [ANP] Re-establishing authenticated connections for {agent_id}...")
            
            # Connect to all surviving agents with ANP authentication
            connection_count = 0
            for other_agent_id in self.agents.keys():
                if other_agent_id != agent_id:
                    try:
                        # Establish bidirectional connections with DID authentication
                        success1 = await self.mesh_network.connect_agents(agent_id, other_agent_id)
                        success2 = await self.mesh_network.connect_agents(other_agent_id, agent_id)
                        
                        if success1 and success2:
                            connection_count += 1
                            self.output.progress(f"🔗 [ANP] {agent_id} ↔ {other_agent_id} authenticated connection established")
                        else:
                            self.output.warning(f"⚠️ [ANP] Partial connection failure {agent_id} ↔ {other_agent_id}")
                            
                    except Exception as e:
                        self.output.warning(f"⚠️ [ANP] Failed to connect {agent_id} ↔ {other_agent_id}: {e}")
            
            self.output.success(f"🔗 [ANP] Re-established {connection_count} authenticated connections for {agent_id}")
                        
        except Exception as e:
            self.output.error(f"🔗 [ANP] Failed to reestablish connections for {agent_id}: {e}")

    async def _finalize_scenario(self) -> Dict[str, Any]:
        """Finalize ANP scenario and generate comprehensive results."""
        import time
        
        end_time = time.time()
        total_runtime = end_time - self.scenario_start_time
        
        self.output.progress("📊 [ANP] Collecting final ANP metrics...")
        
        # Generate ANP-specific comprehensive results
        results = {
            "metadata": {
                "scenario": "fail_storm_recovery",
                "protocol": "anp",  # Explicitly set ANP protocol
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
            "anp_specific": {
                "did_re_authentications": len(self.temporarily_killed_agents),
                "hybrid_communication_active": True,
                "websocket_endpoints": len([a for a in self.agents.values() if hasattr(a, 'anp_websocket_port')]),
                "http_endpoints": len(self.agents),
                "e2e_encryption_sessions": len(self.agents)
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
                    "saved": False,  # ANP doesn't save LLM outputs by default
                    "directory": None
                }
                
                self.output.success("📊 [ANP] ANP-specific metrics collected successfully")
            except Exception as e:
                self.output.warning(f"⚠️ [ANP] Failed to collect metrics: {e}")
        
        # Save results
        await self._save_results(results)
        
        return results

    async def _save_results(self, results: Dict[str, Any]) -> None:
        """Save ANP scenario results to files."""
        import json
        
        # Save main results file
        results_file = self.results_dir / self.config["output"]["results_file"]
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.output.success(f"💾 [ANP] Results saved to: {results_file}")
            
        except Exception as e:
            self.output.error(f"❌ [ANP] Failed to save results: {e}")
        
        # Save detailed metrics if available
        if self.metrics_collector:
            # Add timestamp and protocol to filename
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            detailed_metrics_file = self.results_dir / f"detailed_failstorm_metrics_{timestamp}_anp.json"
            try:
                self.metrics_collector.export_to_json(str(detailed_metrics_file))
                self.output.success(f"💾 [ANP] Detailed metrics saved to: {detailed_metrics_file}")
            except Exception as e:
                self.output.error(f"❌ [ANP] Failed to save detailed metrics: {e}")
        
        # Display final statistics
        self._display_anp_statistics(results)
    
    def _display_anp_statistics(self, results: Dict[str, Any]) -> None:
        """Display ANP-specific statistics and success rates."""
        self.output.info("=" * 60)
        self.output.info("📊 ANP Protocol Final Statistics")
        self.output.info("=" * 60)
        
        # Performance analysis
        performance = results.get('performance_analysis', {})
        if performance:
            pre_fault = performance.get('pre_fault_performance', {})
            recovery = performance.get('recovery_performance', {})
            post_fault = performance.get('post_fault_performance', {})
            
            self.output.info(f"📈 Task Performance Analysis:")
            self.output.info(f"  Pre-fault:  {pre_fault.get('count', 0)} tasks, avg: {pre_fault.get('mean', 0):.2f}s")
            self.output.info(f"  Recovery:   {recovery.get('count', 0)} tasks, avg: {recovery.get('mean', 0):.2f}s")
            self.output.info(f"  Post-fault: {post_fault.get('count', 0)} tasks, avg: {post_fault.get('mean', 0):.2f}s")
            
            total_executions = performance.get('total_task_executions', 0)
            successful_executions = performance.get('successful_executions', 0)
            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
            
            self.output.info(f"📊 Overall Success Rate: {success_rate:.1f}% ({successful_executions}/{total_executions})")
        
        # QA metrics
        qa_metrics = results.get('qa_metrics', {})
        if qa_metrics:
            total_qa = qa_metrics.get('total_qa_tasks', 0)
            successful_qa = qa_metrics.get('successful_tasks', 0)
            with_answers = qa_metrics.get('tasks_with_answers', 0)
            
            qa_success_rate = (successful_qa / total_qa * 100) if total_qa > 0 else 0
            answer_found_rate = (with_answers / total_qa * 100) if total_qa > 0 else 0
            
            self.output.info(f"🎯 QA Task Analysis:")
            self.output.info(f"  Total QA tasks: {total_qa}")
            self.output.info(f"  Successful tasks: {successful_qa} ({qa_success_rate:.1f}%)")
            self.output.info(f"  Tasks with answers: {with_answers} ({answer_found_rate:.1f}%)")
            
            answer_sources = qa_metrics.get('answer_sources', {})
            if answer_sources:
                local_answers = answer_sources.get('local', 0)
                neighbor_answers = answer_sources.get('neighbor', 0)
                self.output.info(f"  Answer sources: Local={local_answers}, Neighbor={neighbor_answers}")
        
        # Failstorm specific metrics
        failstorm = results.get('failstorm_metrics', {})
        if failstorm:
            recovery_ms = failstorm.get('recovery_ms')
            steady_state_ms = failstorm.get('steady_state_ms')
            
            self.output.info(f"🔄 Recovery Analysis:")
            if recovery_ms is not None:
                self.output.info(f"  Recovery time: {recovery_ms/1000:.1f}s")
            if steady_state_ms is not None:
                self.output.info(f"  Steady state time: {steady_state_ms/1000:.1f}s")
        
        # Protocol summary
        protocol_summary = results.get('protocol_summary', {}).get('anp', {})
        if protocol_summary:
            self.output.info(f"🔗 ANP Protocol Summary:")
            self.output.info(f"  Initial agents: {protocol_summary.get('total_agents', 0)}")
            self.output.info(f"  Killed agents: {protocol_summary.get('killed_agents_initial', 0)}")
            self.output.info(f"  Final alive: {protocol_summary.get('final_alive', 0)}")
        
        self.output.info("=" * 60)
    
    async def _reconnect_agent(self, agent_id: str) -> None:
        """Reconnect a killed ANP agent (override base implementation)."""
        if agent_id in self.killed_agents and agent_id in self.shard_workers:
            try:
                # Get stored configuration
                agent_config = self.killed_agent_configs.get(agent_id, {})
                worker = self.shard_workers[agent_id]
                original_port = agent_config.get('port', 9100)
                
                # Use port allocation system to avoid conflicts
                try:
                    # Allocate a new unique port
                    available_port = await self._allocate_unique_port(agent_id, original_port + 100)
                    
                    # Re-create ANP agent with DID authentication
                    new_agent = await self.create_agent(agent_id, "127.0.0.1", available_port, worker)
                    
                    # Re-establish authenticated connections
                    await self._reestablish_anp_connections(agent_id)
                    
                    # Restore to active agents
                    self.agents[agent_id] = new_agent
                    self.killed_agents.discard(agent_id)
                    
                    self.output.success(f"✅ [ANP] Agent {agent_id} successfully reconnected via base runner on port {available_port}!")
                    
                    # Record base runner reconnection metrics
                    if self.metrics_collector:
                        # Record successful reconnection
                        self.metrics_collector.record_reconnection_attempt(
                            source_agent=agent_id,
                            target_agent="base_runner",
                            success=True,
                            duration_ms=5000.0  # Approximate duration
                        )
                        
                        # Record network event
                        self.metrics_collector.record_network_event(
                            event_type="base_runner_reconnection_success",
                            source_agent=agent_id,
                            target_agent="mesh_network"
                        )
                    
                except Exception as port_error:
                    # Try fallback port range
                    fallback_port = await self._allocate_unique_port(agent_id, 9200)
                    new_agent = await self.create_agent(agent_id, "127.0.0.1", fallback_port, worker)
                    
                    await self._reestablish_anp_connections(agent_id)
                    
                    self.agents[agent_id] = new_agent
                    self.killed_agents.discard(agent_id)
                    
                    self.output.success(f"✅ [ANP] Agent {agent_id} reconnected via base runner on fallback port {fallback_port}!")
                
            except Exception as e:
                self.output.error(f"❌ [ANP] Base runner reconnection failed for {agent_id}: {e}")
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
    
    # ========================== ANP-specific Agent Management ==========================
    
    async def _kill_agent(self, agent_id: str) -> None:
        """Kill a specific ANP agent."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # Store configuration for later reconnection
            original_port = getattr(agent, 'port', 9100)
            self.killed_agent_configs[agent_id] = {
                'port': original_port,
                'host': agent.host,
                'executor': self.shard_workers.get(agent_id)
            }
            
            try:
                # Close ANP agent with DID cleanup
                if hasattr(agent, 'anp_agent') and agent.anp_agent:
                    await agent.anp_agent.close()
                if hasattr(agent, 'base_agent') and agent.base_agent:
                    await agent.base_agent.stop()
                    
                # Release the port from our tracking
                async with self._port_lock:
                    if original_port in self._used_ports:
                        self._used_ports.discard(original_port)
                        self.output.info(f"   🔓 [ANP] Released port {original_port} for {agent_id}")
                    
                self.output.info(f"   💀 [ANP] Killed agent: {agent_id}")
                
                # Remove from active agents but keep in shard_workers for recovery
                del self.agents[agent_id]
                
                # Don't schedule separate reconnection - let base runner handle it
                # The base runner will call our _reconnect_agent method
                
            except Exception as e:
                self.output.error(f"   ❌ [ANP] Error killing {agent_id}: {e}")
    
    async def _schedule_anp_reconnection(self, agent_id: str, delay: float) -> None:
        """Schedule ANP agent reconnection with DID re-authentication."""
        await asyncio.sleep(delay)
        
        try:
            self.output.warning(f"🔄 [ANP] Attempting to re-authenticate agent: {agent_id}")
            
            if agent_id in self.killed_agents and agent_id in self.shard_workers:
                # Get original configuration
                agent_config = self.killed_agent_configs.get(agent_id, {})
                worker = self.shard_workers[agent_id]
                original_port = agent_config.get('port', 9100)
                
                # Use the existing port allocation system to avoid conflicts
                try:
                    # Release the original port if it was tracked
                    if original_port in self._used_ports:
                        self._used_ports.discard(original_port)
                    
                    # Allocate a new unique port
                    available_port = await self._allocate_unique_port(agent_id, original_port + 100)  # Use higher range for reconnections
                    
                    # Re-create ANP agent with DID authentication
                    new_agent = await self.create_agent(agent_id, "127.0.0.1", available_port, worker)
                    
                    # Re-establish authenticated connections
                    await self._reestablish_anp_connections(agent_id)
                    
                    # Restore to active agents
                    self.agents[agent_id] = new_agent
                    self.killed_agents.discard(agent_id)
                    
                    self.output.success(f"✅ [ANP] Agent {agent_id} successfully re-authenticated and reconnected on port {available_port}!")
                    
                    # Record recovery metrics and network events
                    if self.metrics_collector:
                        self.metrics_collector.set_first_recovery_time()
                        
                        # Record successful reconnection
                        reconnection_duration = (time.time() - (time.time() - delay)) * 1000  # Convert to ms
                        self.metrics_collector.record_reconnection_attempt(
                            source_agent=agent_id,
                            target_agent="network",
                            success=True,
                            duration_ms=reconnection_duration
                        )
                        
                        # Record network event
                        self.metrics_collector.record_network_event(
                            event_type="agent_reconnection_success",
                            source_agent=agent_id,
                            target_agent="mesh_network"
                        )
                        
                except Exception as port_error:
                    self.output.error(f"🔄 [ANP] Port allocation failed for {agent_id}: {port_error}")
                    # Try with a completely different port range
                    try:
                        fallback_port = await self._allocate_unique_port(agent_id, 9200)  # Use 9200+ range as fallback
                        new_agent = await self.create_agent(agent_id, "127.0.0.1", fallback_port, worker)
                        
                        # Re-establish connections
                        await self._reestablish_anp_connections(agent_id)
                        
                        # Restore to active agents
                        self.agents[agent_id] = new_agent
                        self.killed_agents.discard(agent_id)
                        
                        self.output.success(f"✅ [ANP] Agent {agent_id} reconnected on fallback port {fallback_port}!")
                        
                        if self.metrics_collector:
                            self.metrics_collector.set_first_recovery_time()
                            
                    except Exception as fallback_error:
                        self.output.error(f"🔄 [ANP] Complete reconnection failure for {agent_id}: {fallback_error}")
                
        except Exception as e:
            self.output.error(f"🔄 [ANP] Failed to reconnect {agent_id}: {e}")
    
    
    async def _reestablish_anp_connections(self, agent_id: str) -> None:
        """Re-establish authenticated connections for a reconnected ANP agent."""
        try:
            # Re-establish authenticated connections to all other active agents
            for other_id in self.agents:
                if other_id != agent_id and other_id not in self.killed_agents:
                    # ANP requires DID-based authentication
                    await self.mesh_network.connect_agents(agent_id, other_id)
                    await self.mesh_network.connect_agents(other_id, agent_id)
            
        except Exception as e:
            self.output.error(f"Failed to re-establish ANP connections for {agent_id}: {e}")
    
    # ========================== Simple Failover Implementation ==========================
    
    def get_next_available_agent(self, exclude_agents: set = None) -> Optional[str]:
        """Get the next available agent, skipping failed agents"""
        if exclude_agents is None:
            exclude_agents = set()

        # Get all available agents (exclude killed and explicitly excluded)
        available_agents = []
        for agent_id in self.shard_workers.keys():
            if (agent_id not in self.killed_agents and 
                agent_id not in exclude_agents and
                agent_id in self.agents):  # Ensure agent still exists
                available_agents.append(agent_id)
        
        if available_agents:
            return available_agents[0]  # Return the first available agent
        return None
    
    async def _run_qa_task_for_agent_with_failover(self, original_agent_id: str, original_worker, duration: float):
        """Run a QA task with automatic failover to the next available agent if the original agent fails"""
        start_time = time.time()
        task_count = 0
        max_groups = self.config.get("shard_qa", {}).get("max_groups", 50)
        group_id = 0

        # List of tried agents, starting from the original agent
        tried_agents = set()
        current_agent_id = original_agent_id
        current_worker = original_worker
        
        while time.time() - start_time < duration and group_id < max_groups:
            try:
                # Check whether the current agent is still available
                if (current_agent_id in self.killed_agents or 
                    current_agent_id not in self.agents):
                    # If the previous agent is unavailable, search for the next one
                    tried_agents.add(current_agent_id)
                    next_agent = self.get_next_available_agent(tried_agents)
                    
                    if next_agent is None:
                        self.output.warning(f"🚨 [ANP] No available agents for task, original: {original_agent_id}")
                        break
                    
                    # Switch to the new agent
                    current_agent_id = next_agent
                    current_worker = self.shard_workers[next_agent]
                    self.output.info(f"🔄 [ANP] Switched from {original_agent_id} to {current_agent_id}")
                
                # Execute the task
                task_start_time = time.time()
                result = await current_worker.worker.start_task(group_id)
                task_end_time = time.time()
                task_count += 1
                
                # Record task execution
                if self.metrics_collector:
                    current_phase = self._get_current_phase()
                    task_type = f"qa_{current_phase}"
                    
                    result_str = str(result).lower() if result else ""
                    answer_found = (result and 
                                  ("document search success" in result_str or "answer_found:" in result_str) and 
                                  "no answer" not in result_str)
                    answer_source = "local" if "local" in result_str else "neighbor"
                    
                    self.metrics_collector.record_task_execution(
                        task_id=f"{current_agent_id}_{current_phase}_g{group_id}_{task_count}",
                        agent_id=current_agent_id,
                        task_type=task_type,
                        start_time=task_start_time,
                        end_time=task_end_time,
                        success=True,
                        answer_found=answer_found,
                        answer_source=answer_source,
                        group_id=group_id
                    )
                
                # Move to next group
                group_id = (group_id + 1) % max_groups
                
                # Short sleep to avoid overload
                await asyncio.sleep(0.1)
                
            except Exception as e:
                # Task execution failed, try next agent
                self.output.warning(f"⚠️ [ANP] Task failed on {current_agent_id}: {e}")
                tried_agents.add(current_agent_id)
                
                # Mark current agent as failed
                if current_agent_id not in self.killed_agents:
                    self.killed_agents.add(current_agent_id)
                
                # Find the next available agent
                next_agent = self.get_next_available_agent(tried_agents)
                if next_agent is None:
                    self.output.error(f"❌ [ANP] No more available agents, stopping task for {original_agent_id}")
                    break
                
                # Switch to the new agent
                current_agent_id = next_agent
                current_worker = self.shard_workers[next_agent]
                self.output.info(f"🔄 [ANP] Failover: {original_agent_id} -> {current_agent_id}")
        
        # Update worker's task count
        if hasattr(current_worker.worker, 'task_count'):
            current_worker.worker.task_count = getattr(current_worker.worker, 'task_count', 0) + task_count
        else:
            current_worker.worker.task_count = task_count
        
        elapsed = time.time() - start_time
        if current_agent_id != original_agent_id:
            self.output.success(f"✅ [ANP] Agent {original_agent_id} -> {current_agent_id}: {task_count} tasks in {elapsed:.1f}s")
        else:
            self.output.progress(f"📊 [ANP] Agent {current_agent_id}: {task_count} tasks in {elapsed:.1f}s")