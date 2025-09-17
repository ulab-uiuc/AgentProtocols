#!/usr/bin/env python3
"""
ACP protocol runner for Fail-Storm Recovery scenario.

This module implements the ACP (Agent Communication Protocol) specific functionality
using native acp-sdk while following A2A patterns and inheriting all core logic from the base runner.
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
from .agent import create_acp_agent, ACPAgent

# Import shard_qa components dynamically to avoid circular imports
shard_qa_path = Path(__file__).parent.parent.parent / "shard_qa"
sys.path.insert(0, str(shard_qa_path))
import importlib.util
spec = importlib.util.spec_from_file_location("agent_executor", shard_qa_path / "shard_worker" / "agent_executor.py")
agent_executor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_executor_module)

# Create ACP-specific implementations (following A2A pattern)
class ACPAgentExecutor(agent_executor_module.BaseAgentExecutor):
    """ACP-specific agent executor"""
    async def execute(self, context, event_queue):
        pass
    
    async def cancel(self, context, event_queue):
        pass

class ACPRequestContext(agent_executor_module.BaseRequestContext):
    """ACP-specific request context"""
    def __init__(self, input_data):
        self.input_data = input_data
    
    def get_user_input(self):
        return self.input_data

class ACPEventQueue(agent_executor_module.BaseEventQueue):
    """ACP-specific event queue"""
    def __init__(self):
        self.events = []
    
    async def enqueue_event(self, event):
        self.events.append(event)
        return event

def acp_new_agent_text_message(text, role="user"):
    """ACP-specific text message creation"""
    return {"type": "text", "content": text, "role": str(role)}

# Inject ACP implementations into the agent_executor module
agent_executor_module.AgentExecutor = ACPAgentExecutor
agent_executor_module.RequestContext = ACPRequestContext
agent_executor_module.EventQueue = ACPEventQueue
agent_executor_module.new_agent_text_message = acp_new_agent_text_message

ShardWorkerExecutor = agent_executor_module.ShardWorkerExecutor

class ACPRunner(FailStormRunnerBase):
    """ACP protocol runner following A2A patterns."""

    def __init__(self, config_path: str = "config.yaml"):
        # If using default config, use configs/config_acp.yaml
        if config_path == "config.yaml":
            configs_dir = Path(__file__).parent.parent.parent / "configs"
            protocol_config = configs_dir / "config_acp.yaml"
            if protocol_config.exists():
                config_path = str(protocol_config)
                print(f"📋 Using ACP config from: {config_path}")
            else:
                # Fallback to protocol-specific config
                protocol_config = Path(__file__).parent / "config.yaml"
                if protocol_config.exists():
                    config_path = str(protocol_config)
        
        super().__init__(config_path)
        
        # Ensure protocol is set correctly in config
        if "scenario" not in self.config:
            self.config["scenario"] = {}
        self.config["scenario"]["protocol"] = "acp"

        # ACP-specific attributes (following A2A pattern)
        self.acp_sessions: Dict[str, Any] = {}
        self.killed_agents: set = set()
        self.killed_agent_configs: Dict[str, Any] = {}
        
        # Per-agent group tracking for consistent recovery behavior
        self._next_group_for_agent = {}
        
        self.output.info("Initialized ACP protocol runner")

    async def create_agent(self, agent_id: str, host: str, port: int, executor: ShardWorkerExecutor) -> ACPAgent:
        """Create an ACP agent using native acp-sdk."""
        try:
            # Create ACP agent using the factory method
            agent = await create_acp_agent(
                agent_id=agent_id,
                host=host,
                port=port,
                executor=executor
            )
            
            # Store ACP-specific info
            self.acp_sessions[agent_id] = {
                "base_url": f"http://{host}:{port}",
                "session_id": f"session_{agent_id}_{int(time.time())}",
                "executor": executor
            }
            
            self.output.success(f"ACP agent {agent_id} created successfully")
            return agent
            
        except Exception as e:
            self.output.error(f"Failed to create ACP agent {agent_id}: {e}")
            raise

    def get_protocol_info(self, agent_id: str, port: int, data_file: str) -> str:
        """Get ACP protocol display information."""
        return f"🔗 [ACP] Created {agent_id} - HTTP: {port}, Data: {data_file}"

    def get_reconnection_info(self, agent_id: str, port: int) -> List[str]:
        """Get ACP protocol reconnection information."""
        return [
            f"🔗 [ACP] Agent {agent_id} RECONNECTED on port {port}",
            f"📡 [ACP] ACP protocol active",
            f"🌐 [ACP] HTTP REST API: http://127.0.0.1:{port}"
        ]

    async def _setup_mesh_topology(self) -> None:
        """Setup mesh topology between ACP agents (following A2A pattern)."""
        self.output.progress("🔗 [ACP] Setting up mesh topology...")
        
        await self.mesh_network.setup_mesh_topology()
        
        # Wait for all ACP agents to fully initialize
        self.output.progress("⏳ [ACP] Waiting for all agents to fully initialize...")
        await asyncio.sleep(3.0)  # Give ACP agents time to start their servers
        
        # ACP-specific: Register endpoints between all agents (like A2A)
        self.output.progress("🔗 [ACP] Registering agent endpoints...")
        endpoint_count = 0
        for agent_id, agent in self.agents.items():
            for other_id, other_agent in self.agents.items():
                if agent_id != other_id:
                    try:
                        # Register each agent's endpoint with every other agent
                        endpoint_url = f"http://127.0.0.1:{other_agent.port}"
                        await agent.register_endpoint(other_id, endpoint_url)
                        endpoint_count += 1
                    except Exception as e:
                        self.output.error(f"❌ [ACP] Failed to register endpoint {other_id} with {agent_id}: {e}")
        
        self.output.success(f"🔗 [ACP] Registered {endpoint_count} endpoint pairs")
        
        # ACP agents may need time for connection establishment
        await asyncio.sleep(2.0)  # Brief stabilization for ACP
        
        # Verify connectivity
        topology = self.mesh_network.get_topology()
        expected_connections = len(self.agents) * (len(self.agents) - 1)
        actual_connections = sum(len(edges) for edges in topology.values())
        
        self.output.success(f"🔗 [ACP] Mesh topology established: {actual_connections}/{expected_connections} connections")

    async def _broadcast_document(self) -> None:
        """Broadcast the document to all ACP agents."""
        if not self.agents:
            raise RuntimeError("No ACP agents available for broadcast")
        
        self.output.progress("📡 [ACP] Broadcasting document...")
        
        # Use first agent as broadcaster
        broadcaster_id = list(self.agents.keys())[0]
        
        results = await self.mesh_network.broadcast_init(self.document, broadcaster_id)
        
        successful_deliveries = sum(1 for result in results.values() if "error" not in str(result))
        total_targets = len(results)
        
        self.output.success(f"📡 [ACP] Document broadcast: {successful_deliveries}/{total_targets} deliveries successful")

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

    # ========================== ACP-specific Agent Management ==========================
    
    async def _kill_agent(self, agent_id: str) -> None:
        """Kill a specific ACP agent."""
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
                self.output.info(f"   💀 [ACP] Killed agent: {agent_id}")
                
                # Remove from active agents but keep in shard_workers for recovery
                del self.agents[agent_id]
                
            except Exception as e:
                self.output.error(f"   ❌ [ACP] Error killing {agent_id}: {e}")
    
    async def _reconnect_agent(self, agent_id: str) -> None:
        """Reconnect a killed ACP agent (override base implementation)."""
        if agent_id in self.killed_agents and agent_id in self.shard_workers:
            try:
                # Get stored configuration
                agent_config = self.killed_agent_configs.get(agent_id, {})
                worker = self.shard_workers[agent_id]
                port = agent_config.get('port', 9000)
                
                # Re-create ACP agent
                new_agent = await self.create_agent(agent_id, "127.0.0.1", port, worker)
                
                # Re-establish connections
                await self._reestablish_acp_connections(agent_id)
                
                # Restore to active agents
                self.agents[agent_id] = new_agent
                self.killed_agents.discard(agent_id)
                
                self.output.success(f"✅ [ACP] Agent {agent_id} successfully reconnected via base runner!")
                
                # Record reconnection metrics
                if self.metrics_collector:
                    self.metrics_collector.record_reconnection_attempt(
                        source_agent=agent_id,
                        target_agent="acp_network",
                        success=True,
                        duration_ms=5000.0
                    )
                    
                    self.metrics_collector.record_network_event(
                        event_type="acp_agent_reconnection_success",
                        source_agent=agent_id,
                        target_agent="mesh_network"
                    )
                
            except Exception as e:
                self.output.error(f"❌ [ACP] Base runner reconnection failed for {agent_id}: {e}")
                raise
    
    async def _reestablish_acp_connections(self, agent_id: str) -> None:
        """Re-establish connections for a reconnected ACP agent."""
        try:
            self.output.progress(f"🔗 [ACP] Re-establishing connections and endpoints for {agent_id}...")
            
            # Register endpoints with all other active agents
            reconnected_agent = self.agents[agent_id]
            for other_id, other_agent in self.agents.items():
                if other_id != agent_id and other_id not in self.killed_agents:
                    # Register reconnected agent's endpoint with other agents
                    await other_agent.register_endpoint(agent_id, f"http://127.0.0.1:{reconnected_agent.port}")
                    # Register other agents' endpoints with reconnected agent
                    await reconnected_agent.register_endpoint(other_id, f"http://127.0.0.1:{other_agent.port}")
                    
                    # Also establish mesh network connections
                    await self.mesh_network.connect_agents(agent_id, other_id)
                    await self.mesh_network.connect_agents(other_id, agent_id)
            
            self.output.success(f"🔗 [ACP] Re-established connections for {agent_id}")
            
        except Exception as e:
            self.output.error(f"Failed to re-establish ACP connections for {agent_id}: {e}")

    async def _run_qa_task_for_agent(self, agent_id: str, worker, duration: float):
        """Run QA task for a specific ACP agent during normal phase."""
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
                    
                    # Update group tracking
                    self._next_group_for_agent[agent_id] = group_id
                    
                    # Record task execution in metrics
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
                    await asyncio.sleep(0.002)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.output.error(f"[ACP] Error in QA task for {agent_id}: {e}")
                    await asyncio.sleep(1.0)
                    
        except asyncio.CancelledError:
            pass
        finally:
            # Store task count for reporting
            if hasattr(worker, 'worker'):
                worker.worker.task_count = task_count

    async def _finalize_scenario(self) -> Dict[str, Any]:
        """Finalize ACP scenario and generate comprehensive results."""
        end_time = time.time()
        total_runtime = end_time - self.scenario_start_time
        
        self.output.progress("📊 [ACP] Collecting final ACP metrics...")
        
        # Generate ACP-specific comprehensive results
        results = {
            "metadata": {
                "scenario": "fail_storm_recovery",
                "protocol": "acp",
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
            "acp_specific": {
                "sessions_created": len(self.acp_sessions),
                "native_sdk_active": True,
                "http_endpoints": len(self.agents),
                "protocol_version": "1.0.3"
            }
        }
        
        # Add metrics if available
        if self.metrics_collector:
            metrics_summary = self.metrics_collector.calculate_recovery_metrics()
            results["failstorm_metrics"] = metrics_summary
            
            # Add performance analysis
            performance = self.metrics_collector.get_performance_summary()
            results["performance_analysis"] = performance
            
            # Add QA metrics
            qa_metrics = self.metrics_collector.get_qa_metrics()
            results["qa_metrics"] = qa_metrics
        
        # Save results
        await self._save_results(results)
        
        # Display summary
        self._display_acp_statistics(results)
        
        self.output.success("✅ Fail-Storm scenario completed successfully!")
        return results
    
    def _display_acp_statistics(self, results: Dict[str, Any]) -> None:
        """Display ACP-specific statistics and success rates."""
        self.output.info("=" * 60)
        self.output.info("📊 ACP Protocol Final Statistics")
        self.output.info("=" * 60)
        
        # QA Task Analysis
        qa = results.get('qa_metrics', {})
        if qa:
            total_qa = qa.get('total_qa_tasks', 0)
            successful = qa.get('successful_tasks', 0)
            with_answers = qa.get('tasks_with_answers', 0)
            
            success_rate = (successful / total_qa * 100) if total_qa > 0 else 0
            answer_rate = (with_answers / total_qa * 100) if total_qa > 0 else 0
            
            self.output.info(f"🎯 QA Task Analysis:")
            self.output.info(f"  Total QA tasks: {total_qa}")
            self.output.info(f"  Successful tasks: {successful} ({success_rate:.1f}%)")
            self.output.info(f"  Tasks with answers: {with_answers} ({answer_rate:.1f}%)")
            
            answer_sources = qa.get('answer_sources', {})
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
        
        self.output.info("=" * 60)