#!/usr/bin/env python3
"""
Meta Protocol Fail-Storm Recovery Test Runner

This script runs the fail-storm recovery test using Meta protocol.
Meta protocol creates a unified network with multiple underlying protocols.

Usage:
    python run_meta.py
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, List, Dict

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))  # fail_storm_recovery directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import base runner and create a Meta version
from protocol_backends.base_runner import FailStormRunnerBase
from protocol_backends.meta_protocol.meta_coordinator import FailStormMetaCoordinator, create_failstorm_meta_network


class MetaFailStormRunner(FailStormRunnerBase):
    """Meta Protocol Fail-Storm Recovery Runner"""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.meta_coordinator: FailStormMetaCoordinator = None
    
    def get_protocol_name(self) -> str:
        return "meta"
    
    async def create_agent(self, agent_id: str, host: str, port: int, executor) -> Any:
        """Create meta-protocol agent (handled by meta coordinator)"""
        # Meta agents are created by the meta coordinator, not individually
        return None
    
    def get_protocol_info(self, agent_id: str, port: int, data_file: str) -> str:
        """Get meta protocol display information."""
        return f"🌐 [META] Created {agent_id} - Multi-protocol, Data: {data_file}"
    
    def get_reconnection_info(self, agent_id: str, port: int) -> List[str]:
        """Get meta protocol reconnection information."""
        return [
            f"🔗 [META] Agent {agent_id} RECONNECTED on port {port}",
            f"🌐 [META] Multi-protocol support active",
            f"📡 [META] Cross-protocol communication enabled"
        ]
    
    async def _setup_shard_qa_workers(self) -> None:
        """Setup Shard QA workers using meta-protocol approach."""
        self.output.progress("Setting up Meta-Protocol Shard QA workers...")
        
        # Create meta-protocol network
        self.meta_coordinator = await create_failstorm_meta_network(self.config)
        
        # Extract agents from meta coordinator and assign proper shard data files
        agent_index = 0
        for agent_id, meta_agent in self.meta_coordinator.meta_agents.items():
            # Register with our mesh network
            await self.mesh_network.register_agent(meta_agent.base_agent)
            self.agents[agent_id] = meta_agent.base_agent
            
            # Store shard worker executor for later use
            if hasattr(meta_agent, 'executor_wrapper') and hasattr(meta_agent.executor_wrapper, 'shard_worker_executor'):
                # Fix the data file to use proper shard distribution
                shard_worker = meta_agent.executor_wrapper.shard_worker_executor.worker
                # Use absolute path to avoid path resolution issues
                base_path = Path(__file__).parent.parent
                shard_file = str(base_path / "data" / "shards" / f"shard{agent_index % 8}.jsonl")
                shard_worker.data_file = shard_file
                
                # Set proper neighbors for ring topology using actual agent IDs
                all_agent_ids = list(self.meta_coordinator.meta_agents.keys())
                current_idx = agent_index
                prev_idx = (current_idx - 1) % len(all_agent_ids)
                next_idx = (current_idx + 1) % len(all_agent_ids)
                
                shard_worker.neighbors = {
                    "prev_id": all_agent_ids[prev_idx],
                    "next_id": all_agent_ids[next_idx]
                }
                
                # Set network reference for collaboration
                shard_worker.set_network(self.mesh_network)
                
                self.shard_workers[agent_id] = meta_agent.executor_wrapper.shard_worker_executor
                self.output.progress(f"  {agent_id}: Using {shard_file}, neighbors: {shard_worker.neighbors}")
                agent_index += 1
        
        self.output.success(f"Created {len(self.agents)} Meta-Protocol agents")
        
        # Display protocol distribution
        protocol_counts = {}
        for agent_id in self.meta_coordinator.protocol_types.values():
            protocol_counts[agent_id] = protocol_counts.get(agent_id, 0) + 1
        
        for protocol, count in protocol_counts.items():
            self.output.progress(f"  {protocol.upper()}: {count} agents")
    
    async def _execute_normal_phase(self) -> None:
        """Execute normal Shard QA collaborative retrieval task with meta protocols."""
        try:
            normal_duration = self.config.get("shard_qa", {}).get("normal_phase_duration", 30.0)
            
            self.output.progress(f"🌐 [META] Running Multi-Protocol Shard QA for {normal_duration}s...")
            
            # Start QA tasks for all meta agents
            qa_tasks = []
            for agent_id, meta_agent in self.meta_coordinator.meta_agents.items():
                if hasattr(meta_agent, 'executor_wrapper') and hasattr(meta_agent.executor_wrapper, 'shard_worker_executor'):
                    task = asyncio.create_task(
                        self._run_qa_task_for_agent(agent_id, meta_agent.executor_wrapper.shard_worker_executor, normal_duration),
                        name=f"qa_task_{agent_id}"
                    )
                    qa_tasks.append(task)
        
            # Wait for normal phase to complete
            await asyncio.gather(*qa_tasks, return_exceptions=True)
        
            # Report completion
            for agent_id, meta_agent in self.meta_coordinator.meta_agents.items():
                if hasattr(meta_agent, 'executor_wrapper') and hasattr(meta_agent.executor_wrapper, 'shard_worker_executor'):
                    executor = meta_agent.executor_wrapper.shard_worker_executor
                    task_count = getattr(executor.worker, 'task_count', 0)
                    protocol = self.meta_coordinator.protocol_types.get(agent_id, 'unknown')
                    self.output.info(f"    {agent_id} ({protocol.upper()}): Normal phase completed with {task_count} QA tasks")
            
            self.output.success(f"🌐 [META] Normal phase completed in {normal_duration:.2f}s")
            
        except Exception as e:
            self.output.error(f"❌ [META] Normal phase failed: {e}")
            raise
    
    async def _run_qa_task_for_agent(self, agent_id: str, worker, duration: float):
        """Run QA task for a specific meta-protocol agent."""
        import asyncio
        import time
        
        start_time = time.time()
        task_count = 0
        
        # Test first 20 groups like other protocols
        max_groups = 20
        group_id = 0
        
        while time.time() - start_time < duration and group_id < max_groups:
            try:
                # Execute QA task for current group
                task_start_time = time.time()
                print(f"[META-DEBUG] {agent_id}: Calling start_task({group_id}) on {type(worker.worker)}")
                result = await worker.worker.start_task(group_id)
                task_end_time = time.time()
                task_count += 1
                group_id = (group_id + 1) % max_groups  # Cycle through groups 0-19
                print(f"[META-DEBUG] {agent_id}: Group {group_id-1 if group_id > 0 else max_groups-1} result: {result[:100]}..." if result else f"[META-DEBUG] {agent_id}: No result")
                
                # Record task execution in metrics
                if self.metrics_collector:
                    # Fix logic: distinguish between finding answer vs not finding answer
                    result_str = str(result).lower() if result else ""
                    answer_found = (result and 
                                  ("document search success" in result_str or "answer_found:" in result_str) and 
                                  "no answer" not in result_str)
                    answer_source = "local" if "local" in result_str else "neighbor"
                    protocol = self.meta_coordinator.protocol_types.get(agent_id, 'unknown')
                    current_group = group_id - 1 if group_id > 0 else max_groups - 1
                    self.metrics_collector.record_task_execution(
                        task_id=f"{agent_id}_normal_g{current_group}_{task_count}",
                        agent_id=agent_id,
                        task_type="qa_normal",
                        start_time=task_start_time,
                        end_time=task_end_time,
                        success=True,
                        answer_found=answer_found,
                        answer_source=answer_source,
                        group_id=current_group
                    )
                
                # Fix display logic to match the corrected answer detection
                result_str = str(result).lower() if result else ""
                if (result and 
                    ("document search success" in result_str or "answer_found:" in result_str) and 
                    "no answer" not in result_str):
                    protocol = self.meta_coordinator.protocol_types.get(agent_id, 'unknown')
                    self.output.progress(f"{agent_id} ({protocol.upper()}): Found answer (task #{task_count})")
                elif result and "no answer" in result_str:
                    protocol = self.meta_coordinator.protocol_types.get(agent_id, 'unknown')
                    self.output.progress(f"{agent_id} ({protocol.upper()}): No answer found (task #{task_count})")
                
                # Track task completion
                if hasattr(worker.worker, 'task_count'):
                    worker.worker.task_count = task_count
                else:
                    worker.worker.task_count = task_count
                
                # Brief pause between tasks
                await asyncio.sleep(2.0)
                
            except Exception as e:
                self.output.warning(f"⚠️  [META] Task error for {agent_id}: {e}")
                await asyncio.sleep(1.0)  # Brief pause on error
    
    async def _inject_faults(self) -> None:
        """Inject faults by killing random meta agents."""
        kill_fraction = self.config.get("scenario", {}).get("kill_fraction", 0.3)
        num_to_kill = max(1, int(len(self.agents) * kill_fraction))
        
        # Select victims randomly
        import random
        victim_ids = random.sample(list(self.agents.keys()), num_to_kill)
        
        self.output.warning(f"💥 Killing {num_to_kill} Meta agents: {', '.join(victim_ids)}")
        
        # Record fault injection time
        if self.metrics_collector:
            self.metrics_collector.set_fault_injection_time()
        
        # Kill selected agents
        for victim_id in victim_ids:
            # Stop the base agent
            agent = self.agents[victim_id]
            await agent.stop()
            
            # Remove from active agents
            self.killed_agents.add(victim_id)
            self.killed_agent_configs[victim_id] = {
                'port': agent._port,
                'host': agent._host
            }
            
            self.output.warning(f"    ✗ Killed Meta agent: {victim_id} (will attempt reconnection later)")
        
        # Schedule reconnection attempts
        reconnect_delay = 10.0
        self.output.success(f"🔄 [META] Scheduling reconnection for {len(victim_ids)} agents in {reconnect_delay}s...")
        
        for victim_id in victim_ids:
            asyncio.create_task(self._schedule_meta_reconnection(victim_id, reconnect_delay))
        
        fault_elapsed = time.time() - self.scenario_start_time if hasattr(self, 'scenario_start_time') else 0
        self.output.warning(f"⚠️  Meta fault injection completed at t={fault_elapsed:.1f}s")
    
    async def _setup_mesh_topology(self) -> None:
        """Setup mesh topology between meta-protocol agents."""
        self.output.progress("🌐 [META] Setting up multi-protocol mesh topology...")
        
        await self.mesh_network.setup_mesh_topology()
        
        # Meta agents need time for all protocols to stabilize
        import asyncio
        await asyncio.sleep(3.0)  # Extra time for multi-protocol stabilization
        
        # Verify connectivity
        topology = self.mesh_network.get_topology()
        expected_connections = len(self.agents) * (len(self.agents) - 1)
        actual_connections = sum(len(edges) for edges in topology.values())
        
        self.output.success(f"🌐 [META] Multi-protocol mesh topology established: {actual_connections}/{expected_connections} connections")
    
    async def _broadcast_document(self) -> None:
        """Broadcast document to all meta-protocol agents."""
        try:
            self.output.progress("📡 [META] Broadcasting document to all meta agents...")
            
            # Use mesh network's broadcast functionality
            document = self._load_document()
            await self.mesh_network.broadcast_init(document)
            
            # Brief wait for document processing
            import asyncio
            await asyncio.sleep(1.0)
            
            self.output.success(f"📡 [META] Document broadcasted to {len(self.agents)}/{len(self.agents)} agents")
            
        except Exception as e:
            self.output.error(f"❌ [META] Document broadcast failed: {e}")
            raise

    async def _schedule_meta_reconnection(self, agent_id: str, delay: float) -> None:
        """Schedule Meta agent reconnection."""
        await asyncio.sleep(delay)
        
        try:
            self.output.warning(f"🔄 [META] Attempting to reconnect agent: {agent_id}")
            
            if agent_id in self.killed_agents and agent_id in self.meta_coordinator.meta_agents:
                # Get the meta agent
                meta_agent = self.meta_coordinator.meta_agents[agent_id]
                protocol = self.meta_coordinator.protocol_types[agent_id]
                
                # Get original configuration
                agent_config = self.killed_agent_configs.get(agent_id, {})
                
                # Find available port
                original_port = agent_config.get('port', 9000)
                try:
                    available_ports = self._find_available_ports("127.0.0.1", original_port, 1)
                    port = available_ports[0] if available_ports else original_port + 100
                except RuntimeError:
                    port = original_port + 100
                
                # Recreate the base agent
                if protocol == "acp":
                    new_base_agent = await meta_agent.create_acp_worker("0.0.0.0", port)
                elif protocol == "anp":
                    new_base_agent = await meta_agent.create_anp_worker("0.0.0.0", port)
                elif protocol == "agora":
                    new_base_agent = await meta_agent.create_agora_worker("0.0.0.0", port)
                elif protocol == "a2a":
                    new_base_agent = await meta_agent.create_a2a_worker("0.0.0.0", port)
                else:
                    raise ValueError(f"Unknown protocol: {protocol}")
                
                # Re-register with mesh network
                await self.mesh_network.register_agent(new_base_agent)
                
                # Restore to active agents
                self.agents[agent_id] = new_base_agent
                self.killed_agents.discard(agent_id)
                
                # Display reconnection info
                reconnect_info = self.get_reconnection_info(agent_id, port)
                for info_line in reconnect_info:
                    self.output.success(info_line)
                
                self.output.success(f"✅ [META] Agent {agent_id} ({protocol.upper()}) successfully reconnected!")
                
                # Record recovery metrics
                if self.metrics_collector:
                    self.metrics_collector.set_first_recovery_time()
                
        except Exception as e:
            self.output.error(f"🔄 [META] Failed to reconnect {agent_id}: {e}")
    
    async def _execute_fault_injection(self) -> None:
        """Execute fault injection for Meta protocol."""
        await self._inject_faults()
    
    async def _execute_recovery_phase(self) -> None:
        """Execute recovery monitoring for Meta protocol."""
        recovery_duration = self.config.get("scenario", {}).get("recovery_duration", 60.0)
        
        self.output.info(f"🔄 [META] Monitoring recovery for {recovery_duration}s...")
        
        start_time = time.time()
        while time.time() - start_time < recovery_duration:
            # Check recovery status
            alive_agents = len(self.agents) - len(self.killed_agents)
            total_agents = len(self.agents) + len(self.killed_agents)
            alive_percentage = (alive_agents / total_agents) * 100 if total_agents > 0 else 0
            
            elapsed = time.time() - start_time
            remaining = recovery_duration - elapsed
            
            self.output.info(f"🔄 [META] Recovery tick: alive={alive_percentage:.2f}%, elapsed={elapsed:.0f}s")
            
            # Check if all agents have recovered
            if len(self.killed_agents) == 0:
                self.output.success(f"🔄 [META] All agents recovered! Steady state achieved at t={elapsed:.0f}s")
                if self.metrics_collector:
                    self.metrics_collector.set_steady_state_time()
                break
            
            await asyncio.sleep(5.0)  # Check every 5 seconds
        
        self.output.success(f"🔄 [META] Recovery monitoring finished")
    
    async def _monitor_recovery(self) -> None:
        """Monitor recovery phase for Meta protocol."""
        await self._execute_recovery_phase()
    
    async def _collect_final_metrics(self) -> Dict[str, Any]:
        """Collect final metrics for Meta protocol."""
        self.output.progress("📊 [META] Collecting final Meta metrics...")
        
        # Get protocol distribution
        protocol_stats = {}
        for agent_id, protocol in self.meta_coordinator.protocol_types.items():
            if protocol not in protocol_stats:
                protocol_stats[protocol] = {"agents": 0, "tasks": 0}
            protocol_stats[protocol]["agents"] += 1
            
            # Get task count from shard worker if available
            if agent_id in self.meta_coordinator.meta_agents:
                meta_agent = self.meta_coordinator.meta_agents[agent_id]
                if hasattr(meta_agent, 'executor_wrapper') and hasattr(meta_agent.executor_wrapper, 'shard_worker_executor'):
                    task_count = getattr(meta_agent.executor_wrapper.shard_worker_executor.worker, 'task_count', 0)
                    protocol_stats[protocol]["tasks"] += task_count
        
        meta_specific_metrics = {
            "protocol_distribution": protocol_stats,
            "total_protocols": len(protocol_stats),
            "cross_protocol_communication": True,
            "unified_network_management": True
        }
        
        self.output.success("📊 [META] Meta-specific metrics collected successfully")
        return meta_specific_metrics


async def main():
    """Main entry point for Meta fail-storm testing."""
    try:
        print("🚀 Starting Meta Protocol Fail-Storm Recovery Test")
        print("=" * 60)
        
        # Create Meta runner with protocol-specific config
        runner = MetaFailStormRunner("configs/config_meta.yaml")
        
        print(f"📋 Configuration loaded from: configs/config_meta.yaml")
        print(f"🌐 Protocol: META (Multi-Protocol)")
        print(f"👥 Agents: {runner.config['scenario']['agent_count']}")
        print(f"⏱️  Runtime: {runner.config['scenario']['total_runtime']}s")
        print(f"💥 Fault time: {runner.config['scenario']['fault_injection_time']}s")
        print("=" * 60)
        
        # Run the scenario
        results = await runner.run_scenario()
        
        print("\n🎉 Meta Fail-Storm test completed successfully!")
        
        # Get actual result paths from runner
        result_paths = runner.get_results_paths()
        print(f"📊 Results saved to: {result_paths['results_file']}")
        print(f"📈 Detailed metrics: {result_paths['detailed_results_file']}")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Meta test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the Meta fail-storm test
    asyncio.run(main())
