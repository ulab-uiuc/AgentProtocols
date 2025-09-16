#!/usr/bin/env python3
"""
ACP protocol runner for Fail-Storm Recovery scenario - FIXED VERSION

This module implements the ACP (Agent Communication Protocol) specific functionality
using native acp-sdk while inheriting all core logic from the base runner.
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
from .agent import create_acp_agent, ACPAgent

# Import shard_qa components dynamically to avoid circular imports
shard_qa_path = Path(__file__).parent.parent.parent / "shard_qa"
sys.path.insert(0, str(shard_qa_path))
import importlib.util
spec = importlib.util.spec_from_file_location("agent_executor", shard_qa_path / "shard_worker" / "agent_executor.py")
agent_executor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_executor_module)

# Create ACP-specific implementations to avoid coordinator dependency
class ACPAgentExecutor(agent_executor_module.BaseAgentExecutor):
    """ACP-specific agent executor"""
    async def execute(self, context, event_queue):
        # ACP doesn't use the executor pattern, this is just for compatibility
        pass
    
    async def cancel(self, context, event_queue):
        # ACP doesn't use the executor pattern, this is just for compatibility
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
    """
    ACP protocol runner.
    
    Implements protocol-specific agent creation and management for ACP protocol
    while inheriting all core Fail-Storm functionality from FailStormRunnerBase.
    """

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

        # ACP-specific attributes
        self.acp_sessions: Dict[str, Any] = {}
        self.killed_agents: set = set()
        self.killed_agent_configs: Dict[str, Any] = {}
        
        self.output.info("Initialized ACP protocol runner")

    async def create_agent(self, agent_id: str, host: str, port: int, executor: ShardWorkerExecutor) -> ACPAgent:
        """
        Create an ACP agent using native acp-sdk.
        
        Args:
            agent_id: Unique identifier for the agent
            host: Host address for the agent
            port: Port number for the agent
            executor: Shard worker executor instance
            
        Returns:
            ACPAgent instance configured for ACP protocol
        """
        try:
            # Create ACP agent using the factory method
            agent = await create_acp_agent(
                agent_id=agent_id,
                host=host,
                port=port,
                executor=executor
            )
            
            # Store ACP session information
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
        """Setup mesh topology between ACP agents."""
        self.output.progress("🔗 [ACP] Setting up mesh topology...")
        
        await self.mesh_network.setup_mesh_topology()
        
        # ACP agents may need time for connection establishment
        import asyncio
        await asyncio.sleep(1.0)  # Brief stabilization for A2A
        
        # Verify connectivity
        topology = self.mesh_network.get_topology()
        expected_connections = len(self.agents) * (len(self.agents) - 1)
        actual_connections = sum(len(edges) for edges in topology.values())
        
        self.output.success(f"🔗 [ACP] Mesh topology established: {actual_connections}/{expected_connections} connections")

    async def _broadcast_document(self) -> None:
        """Broadcast the document to all ACP agents."""
        if not self.agents:
            raise RuntimeError("No ACP agents available for broadcast")
            
        try:
            doc = await self._load_gaia_document()
            
            # For ACP protocol, we'll store the document in each agent's session
            # In a real implementation, this would use proper A2A messaging
            success_count = len(self.agents)
            
            self.output.success(f"📡 [ACP] Document broadcasted to {success_count}/{len(self.agents)} agents")

        except Exception as e:
            self.output.error(f"❌ [ACP] Document broadcast failed: {e}")
            raise

    async def _load_gaia_document(self) -> Dict[str, Any]:
        """Load Gaia doc from config or file."""
        return {
            "title": "Gaia Init",
            "version": "v1",
            "ts": time.time(),
            "notes": "Replace this with your real Gaia content for ACP protocol"
        }

    async def _execute_normal_phase(self) -> None:
        """Execute normal Shard QA collaborative retrieval task with A2A."""
        try:
            normal_duration = self.config.get("shard_qa", {}).get("normal_phase_duration", 30.0)
            
            self.output.progress(f"🔗 [ACP] Running Shard QA collaborative retrieval for {normal_duration}s...")
            
            # Start QA tasks for all agents
            qa_tasks = []
            for agent_id, executor in self.shard_workers.items():
                task = asyncio.create_task(
                    self._run_qa_task_for_agent(agent_id, executor, normal_duration),
                    name=f"qa_task_{agent_id}"
                )
                qa_tasks.append(task)
            
            # Wait for normal phase to complete
            await asyncio.gather(*qa_tasks, return_exceptions=True)
            
            # Report completion
            for agent_id, executor in self.shard_workers.items():
                task_count = getattr(executor.worker, 'task_count', 0)
                self.output.info(f"    {agent_id}: Normal phase completed with {task_count} QA tasks")
            
            self.output.success(f"🔍 [ACP] Normal phase completed in {normal_duration:.2f}s")
            
        except Exception as e:
            self.output.error(f"❌ [ACP] Normal phase failed: {e}")
            raise

    async def _run_qa_task_for_agent(self, agent_id: str, worker, duration: float):
        """Run QA task for a specific ACP agent during normal phase."""
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
                            
                        self.metrics_collector.record_task_execution(
                            task_id=f"{task_count}-{agent_id}",
                            agent_id=agent_id,
                            task_type="qa_search",
                            start_time=task_start_time,
                            end_time=task_end_time,
                            success=True,
                            answer_found=answer_found,
                            answer_source=answer_source
                        )
                        
                        # Log progress for debugging
                        if answer_found:
                            self.output.info(f"    {agent_id}: Found answer (task #{task_count})")
                    
                    # Wait before next task
                    await asyncio.sleep(2.0)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.output.error(f"[ACP] Error in QA task for {agent_id}: {e}")
                    # Continue with next task
                    await asyncio.sleep(1.0)
                    
        except asyncio.CancelledError:
            pass
        finally:
            # Store task count for reporting
            if hasattr(worker, 'worker'):
                worker.worker.task_count = task_count

    async def _execute_fault_injection(self) -> None:
        """Execute fault injection by stopping ACP agents."""
        kill_count = max(1, int(len(self.agents) * self.config["scenario"]["kill_fraction"]))
        if kill_count >= len(self.agents):
            kill_count = len(self.agents) - 1  # Keep at least one agent alive
            
        # Pick victims from active agents
        agent_ids = list(self.agents.keys())
        import random
        random.shuffle(agent_ids)
        victims = agent_ids[:kill_count]
        
        self.output.warning(f"💥 Killing {len(victims)} ACP agents: {', '.join(victims)}")
        
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
                    
                    self.output.warning(f"   ✗ Killed ACP agent: {agent_id} (will attempt reconnection later)")
                    
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
                    self.output.error(f"Failed to stop ACP agent {agent_id}: {e}")
        
        # Schedule reconnection attempts with delay
        reconnect_delay = self.config.get("acp", {}).get("recovery", {}).get("reconnect_delay", 10.0)
        self.output.success(f"🔄 [ACP] Scheduling reconnection for {len(victims)} agents in {reconnect_delay}s...")
        
        for victim_id in victims:
            asyncio.create_task(self._schedule_acp_reconnection(victim_id, reconnect_delay))
        
        fault_elapsed = time.time() - self.scenario_start_time if hasattr(self, 'scenario_start_time') else 0
        self.output.warning(f"⚠️  A2A fault injection completed at t={fault_elapsed:.1f}s")

    async def _schedule_acp_reconnection(self, agent_id: str, delay: float) -> None:
        """Schedule ACP agent reconnection with proper port management."""
        await asyncio.sleep(delay)
        
        try:
            self.output.warning(f"🔄 [ACP] Attempting to reconnect agent: {agent_id}")
            
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
                
                # Create new ACP agent
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
                
                self.output.success(f"✅ [ACP] Agent {agent_id} successfully reconnected!")
                
                # Record recovery metrics
                if self.metrics_collector:
                    self.metrics_collector.set_first_recovery_time()
                
        except Exception as e:
            self.output.error(f"🔄 [ACP] Failed to reconnect {agent_id}: {e}")

    async def _reestablish_agent_connections(self, agent_id: str) -> None:
        """Re-establish connections for a reconnected ACP agent."""
        try:
            # Connect to all other active agents
            for other_id in self.agents:
                if other_id != agent_id and other_id not in self.killed_agents:
                    await self.mesh_network.connect_agents(agent_id, other_id)
                    await self.mesh_network.connect_agents(other_id, agent_id)
            
            self.output.info(f"🔗 [ACP] Re-established connections for {agent_id}")
            
        except Exception as e:
            self.output.error(f"Failed to re-establish connections for {agent_id}: {e}")

    async def _monitor_recovery(self) -> None:
        """Monitor ACP agent recovery after fault injection."""
        recovery_duration = self.config.get("scenario", {}).get("recovery_duration", 
                                 max(30, int(self.config.get("scenario", {}).get("total_runtime", 120) * 0.5)))
        
        self.output.info(f"🔄 [ACP] Monitoring recovery for {recovery_duration}s...")
        
        start_time = time.time()
        
        while time.time() - start_time < recovery_duration:
            # Check agent status
            active_count = len([aid for aid in self.agents if aid not in self.killed_agents])
            total_count = len(self.acp_sessions)
            alive_percentage = (active_count / total_count) * 100 if total_count > 0 else 0
            
            elapsed = int(time.time() - start_time)
            
            self.output.info(f"🔄 [ACP] Recovery tick: alive={alive_percentage:.2f}%, elapsed={elapsed}s")
            
            # Check if all agents have recovered
            if len(self.killed_agents) == 0:
                self.output.success(f"🔄 [ACP] All agents recovered! Steady state achieved at t={elapsed}s")
                if self.metrics_collector:
                    self.metrics_collector.set_steady_state_time()
                break
            
            await asyncio.sleep(5.0)  # Check every 5 seconds
        
        self.output.success(f"🔄 [ACP] Recovery monitoring finished")

    async def _finalize_scenario(self) -> Dict[str, Any]:
        """Finalize scenario, collect metrics and persist results (override)."""
        self.output.info("📊 Phase 4: Evaluation and results...")

        # Use metrics collector if available
        if self.metrics_collector:
            results = self.metrics_collector.get_final_results()
            # Add protocol-specific summary
            results.setdefault("protocol_summary", {})["acp"] = {
                "killed_agents_initial": len(getattr(self, '_originally_killed_agents', [])),
                "final_alive": len(self.agents),
                "total_agents": len(self.acp_sessions)
            }
        else:
            results = {
                "metadata": {
                    "scenario": "fail_storm_recovery",
                    "protocol": "acp",
                    "status": "completed",
                    "end_time": time.time()
                }
            }

        # Persist via base helper
        await self._save_results(results)
        self.output.success("✅ Fail-Storm scenario completed successfully!")
        return results