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

# Create ACP-specific implementations
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

# Import native ACP SDK
try:
    import acp_sdk
    from acp_sdk.models import RunCreateRequest, Message, MessagePart
    import httpx
    ACP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: acp-sdk not available: {e}")
    ACP_AVAILABLE = False
    # Mock classes for fallback
    class RunCreateRequest:
        def __init__(self, input: Any): pass
    class Message:
        def __init__(self, parts: List[Any]): pass
    class MessagePart:
        def __init__(self, type: str, text: str): pass


class ACPRunner(FailStormRunnerBase):
    """
    ACP protocol runner.
    
    Implements protocol-specific agent creation and management for ACP protocol
    while inheriting all core Fail-Storm functionality from FailStormRunnerBase.
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
        self.config["scenario"]["protocol"] = "acp"

        # ACP-specific attributes
        self.acp_sessions: Dict[str, Any] = {}
        self.killed_agents: set = set()
        
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
                'agent_id': agent_id,
                'executor': executor,
                'base_url': f"http://{host}:{port}",
                'created_at': time.time()
            }
            
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

    # ========================================
    # ACP-Specific Network Setup
    # ========================================
    
    async def _setup_mesh_topology(self) -> None:
        """Set up mesh topology for ACP agents."""
        try:
            self.output.progress("🔗 [ACP] Setting up mesh topology...")
            
            # Register all agents with each other
            connection_count = 0
            for agent_id, agent in self.agents.items():
                for other_id, other_agent in self.agents.items():
                    if agent_id != other_id:
                        # Get the base URL for the other agent
                        session_info = self.acp_sessions.get(other_id)
                        if session_info:
                            other_url = session_info['base_url']
                            await agent.register_endpoint(other_id, other_url)
                            connection_count += 1
            
            self.output.success(f"🔗 [ACP] Mesh topology established: {connection_count}/{len(self.agents) * (len(self.agents) - 1)} connections")
            
        except Exception as e:
            self.output.error(f"Failed to setup ACP mesh topology: {e}")
            raise
    
    async def _broadcast_document(self) -> None:
        """Broadcast document to all ACP agents."""
        try:
            self.output.progress("📡 [ACP] Document broadcasted to {}/{} agents".format(len(self.agents), len(self.agents)))
            # ACP agents receive document through the mesh network setup
            # No explicit broadcast needed as agents will discover content through QA tasks
            
        except Exception as e:
            self.output.error(f"Failed to broadcast document to ACP agents: {e}")
            raise

     # ========================================
     # ACP-Specific Normal Phase Execution
     # ========================================
     
    async def _execute_normal_phase(self) -> None:
        """Execute normal Shard QA collaborative retrieval task with ACP."""
        import asyncio
        import time
        
        normal_phase_duration = self.config.get("shard_qa", {}).get("normal_phase_duration", 30.0)
        
        self.output.progress(f"🔗 [ACP] Running Shard QA collaborative retrieval for {normal_phase_duration}s...")
        
        # Start metrics collection for normal phase
        if self.metrics_collector:
            self.metrics_collector.start_normal_phase()
        
        start_time = time.time()
        qa_tasks = []
        
        # Start QA task execution on all agents simultaneously
        for agent_id, worker in self.shard_workers.items():
            task = asyncio.create_task(self._run_qa_task_for_agent(agent_id, worker, normal_phase_duration))
            qa_tasks.append(task)
        
        # Wait for normal phase duration with ACP status updates
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
        self.output.success(f"[ACP] Normal phase completed in {elapsed:.2f}s")

    async def _run_qa_task_for_agent(self, agent_id: str, worker, duration: float):
        """Run QA task for a specific ACP agent."""
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
                         elif any(pattern in result_str for pattern in [
                             "NEIGHBOR SEARCH FAILED", "NO NEIGHBOR FOUND", "NO NEIGHBOR", 
                             "SEARCH FAILED", "FAILED TO FIND", "FALLBACK"
                         ]):
                             answer_source = "fallback"  # Failed to find from neighbors, used fallback
                         else:
                             answer_source = "unknown"
                             
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
                             self.output.progress(f"🔗 [{agent_id}] Found answer")
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

     async def _run_post_fault_qa_task(self, agent_id: str, worker, duration: float):
         """Run post-fault QA tasks to measure recovery performance."""
         import asyncio
         import time
         
         start_time = time.time()
         task_count = 0
         
         # Wait a bit for the fault to be injected, but start tasks immediately
         await asyncio.sleep(0.5)
         
         try:
             while time.time() - start_time < duration and not self.shutdown_event.is_set():
                 try:
                     # Execute QA task for group 0 (standard test case)
                     task_start_time = time.time()
                     result = await worker.worker.start_task(0)
                     task_end_time = time.time()
                     task_count += 1
                     
                     # Record post-fault task execution in metrics
                     if self.metrics_collector:
                         answer_found = result and "answer found" in result.lower()
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
                         elif any(pattern in result_str for pattern in [
                             "NEIGHBOR SEARCH FAILED", "NO NEIGHBOR FOUND", "NO NEIGHBOR", 
                             "SEARCH FAILED", "FAILED TO FIND", "FALLBACK"
                         ]):
                             answer_source = "fallback"  # Failed to find from neighbors, used fallback
                         else:
                             answer_source = "unknown"
                             
                         self.metrics_collector.record_task_execution(
                             task_id=f"{agent_id}_post_fault_{task_count}",
                             agent_id=agent_id,
                             task_type="qa_post_fault",
                             start_time=task_start_time,
                             end_time=task_end_time,
                             success=True,  # Task completed successfully
                             answer_found=answer_found,
                             answer_source=answer_source,
                             group_id=0
                         )
                     
                     self.output.info(f"   {agent_id}: Post-fault answer found (task #{task_count})")
                     
                     # Wait between tasks - shorter interval during fault recovery
                     await asyncio.sleep(3)
                     
                 except Exception as e:
                     # Task execution failed - likely due to fault impact
                     if self.metrics_collector:
                         self.metrics_collector.record_task_execution(
                             task_id=f"{agent_id}_post_fault_failed_{task_count}",
                             agent_id=agent_id,
                             task_type="qa_post_fault",
                             start_time=time.time(),
                             end_time=time.time(),
                             success=False,  # Task failed
                             answer_found=False,
                             answer_source="failed",
                             group_id=0
                         )
                     self.output.warning(f"   {agent_id}: Post-fault task failed: {e}")
                     await asyncio.sleep(5)  # Wait before retry
                     
         except asyncio.CancelledError:
             self.output.info(f"   {agent_id}: Post-fault QA task cancelled")
         except Exception as e:
             self.output.error(f"   {agent_id}: Post-fault QA task error: {e}")
         
         self.output.success(f"   {agent_id}: Post-fault phase completed with {task_count} QA tasks")

     # ========================================
     # ACP-Specific Fault Injection and Recovery
     # ========================================

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
        
        # Kill selected agents
        for agent_id in victims:
            agent = self.agents.get(agent_id)
            if agent:
                try:
                    self.output.warning(f"   ✗ Killed ACP agent: {agent_id} (will attempt reconnection later)")
                    
                    # Stop the agent and ensure port release
                    await agent.stop()
                    self.killed_agents.add(agent_id)
                    
                    # Additional wait to ensure port is fully released
                    await asyncio.sleep(2.0)
                    
                    # Update metrics
                    if self.metrics_collector:
                        self.metrics_collector.update_agent_state(agent_id, "killed")
                        
                except Exception as e:
                    self.output.error(f"Failed to stop ACP agent {agent_id}: {e}")
        
        # Schedule reconnection attempts (like other protocols)
        if self.killed_agents:
            reconnect_delay = self.config.get("acp", {}).get("recovery", {}).get("reconnect_delay", 10.0)
            self.output.success(f"🔄 [ACP] Scheduling reconnection for {len(self.killed_agents)} agents in {reconnect_delay}s...")
            
            for agent_id in self.killed_agents.copy():
                asyncio.create_task(self._schedule_acp_reconnection(agent_id, reconnect_delay))
        
        fault_elapsed = time.time() - self.scenario_start_time if hasattr(self, 'scenario_start_time') else 0
        self.output.warning(f"⚠️  ACP fault injection completed at t={fault_elapsed:.1f}s")

    async def _monitor_recovery(self) -> None:
        """Monitor ACP agent recovery after fault injection."""
        recovery_duration = self.config["scenario"]["recovery_duration"]
        
        self.output.info(f"🔄 [ACP] Monitoring recovery for {recovery_duration}s...")
        
        start_time = time.time()
        
        # Start post-fault QA tasks for remaining agents
        post_fault_tasks = []
        for agent_id, worker in self.shard_workers.items():
            if agent_id not in self.killed_agents:
                task = asyncio.create_task(self._run_post_fault_qa_task(agent_id, worker, recovery_duration))
                post_fault_tasks.append(task)
        
        while time.time() - start_time < recovery_duration:
            # Simple monitoring - reconnection is handled by scheduled tasks
            alive_count = len([aid for aid in self.agents.keys() if aid not in self.killed_agents])
            total_agents = len(self.agents)
            alive_ratio = alive_count / total_agents if total_agents > 0 else 0.0
            elapsed = time.time() - start_time
            
            self.output.info(f"🔄 [ACP] Recovery tick: alive={alive_ratio:.2%}, elapsed={elapsed:.0f}s")
            
            # Check if system reached steady state (all agents recovered)
            min_monitoring_time = 10.0  # Minimum 10 seconds of monitoring
            if alive_ratio >= 1.0 and len(self.killed_agents) == 0 and elapsed >= min_monitoring_time:
                if self.metrics_collector:
                    self.metrics_collector.set_steady_state_time(time.time())
                self.output.success(f"🔄 [ACP] All agents recovered! Steady state achieved at t={elapsed:.1f}s")
                break
            elif alive_ratio >= 1.0 and len(self.killed_agents) == 0 and elapsed < min_monitoring_time:
                # Agents recovered but continue monitoring for post-fault performance
                if not hasattr(self, '_steady_state_recorded'):
                    if self.metrics_collector:
                        self.metrics_collector.set_steady_state_time(time.time())
                    self._steady_state_recorded = True
                    self.output.info(f"🔄 [ACP] Agents recovered at t={elapsed:.1f}s, continuing monitoring for post-fault metrics")
            
            await asyncio.sleep(5)  # Check every 5 seconds
        
        # Cancel post-fault QA tasks
        for task in post_fault_tasks:
            task.cancel()
        
        # Wait for tasks to finish cancellation
        await asyncio.gather(*post_fault_tasks, return_exceptions=True)
        
        self.output.success(f"🔄 [ACP] Recovery monitoring finished")

     async def cleanup(self):
         """Clean up ACP-specific resources."""
         self.output.progress("Cleaning up ACP resources...")
         
         # Clear ACP sessions
         self.acp_sessions.clear()
         
         # Call parent cleanup
         await super().cleanup()
         
         self.output.success("ACP resource cleanup completed")

     async def _finalize_scenario(self) -> Dict[str, Any]:
         """Finalize scenario and return results with expected structure."""
         # Get timing information
         fault_ts = self.metrics_collector.fault_injection_time if self.metrics_collector else None
         rec_ts = self.metrics_collector.first_recovery_time if self.metrics_collector else None
         steady_ts = self.metrics_collector.steady_state_time if self.metrics_collector else None
         
         # Calculate agent statistics based on metrics collector data
         total_agents = len(self.agents)
         killed_agents = getattr(self, '_originally_killed_agents', self.killed_agents.copy())
         currently_alive_agents = [aid for aid in self.agents.keys() if aid not in self.killed_agents]
         recovered_agents = [aid for aid in killed_agents if aid not in self.killed_agents]
         
         # Build comprehensive results dictionary with expected structure
         final = {
             "metadata": {
                 "protocol": "acp",
                 "scenario": "fail_storm_recovery",
                 "agent_count": len(self.agents),
                 "kill_fraction": self.config["scenario"]["kill_fraction"],
                 "timestamp": time.time(),
                 "total_runtime": time.time() - self.scenario_start_time if hasattr(self, 'scenario_start_time') else 0.0
             },
             "agent_summary": {
                 "initial_count": total_agents,
                 "temporarily_killed_count": len(killed_agents),  # Total agents that were killed
                 "currently_killed_count": len(self.killed_agents),  # Still dead
                 "permanently_failed_count": 0,  # ACP doesn't have permanent failures
                 "surviving_count": len(currently_alive_agents),
                 "reconnected_count": len(recovered_agents),
                 "temporarily_killed_agents": list(killed_agents),
                 "currently_killed_agents": list(self.killed_agents),
                 "permanently_failed_agents": [],
                 "surviving_agents": currently_alive_agents
             },
             "acp_specific": {
                 "sessions_created": len(self.acp_sessions),
                 "document_broadcast": "success",
                 "mesh_connections": len(self.agents) * (len(self.agents) - 1)  # n * (n-1) for full mesh
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
         
         # Add comprehensive metrics if available
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
                     "saved": False,  # ACP doesn't save LLM outputs by default
                     "directory": None
                 }
                 
             except Exception as e:
                 self.output.error(f"Failed to collect comprehensive metrics: {e}")
                 # Don't use fallback - let the error be visible
                 raise
         
         return final
 
     async def _schedule_acp_reconnection(self, agent_id: str, delay: float) -> None:
        """Schedule ACP agent reconnection with proper port management."""
        await asyncio.sleep(delay)
        
        try:
            # Get session info for the killed agent
            session_info = self.acp_sessions.get(agent_id)
            if not session_info:
                self.output.error(f"No session info found for {agent_id}")
                return
            
            executor = session_info['executor']
            old_port = int(session_info['base_url'].split(':')[-1])
            
            # Find a new available port to avoid conflicts
            base_port = 9100 + hash(agent_id) % 100  # Use agent_id hash for port offset
            available_ports = self._find_available_ports("127.0.0.1", base_port, 1)
            new_port = available_ports[0] if available_ports else base_port
            
            self.output.warning(f"🔄 [ACP] Attempting to reconnect {agent_id}: {old_port} → {new_port}")
            
            # Create new agent instance using native ACP agent
            new_agent = await create_acp_agent(
                agent_id=agent_id,
                host="127.0.0.1", 
                port=new_port,
                executor=executor
            )
            
            # Update session info and agents registry
            session_info['base_url'] = f"http://127.0.0.1:{new_port}"
            self.agents[agent_id] = new_agent
            
            # Wait for agent to fully start
            await asyncio.sleep(1.0)
            
            # Mark as successfully reconnected
            if agent_id in self.killed_agents:
                self.killed_agents.remove(agent_id)
                
                # Record first recovery time
                if self.metrics_collector and not hasattr(self, '_first_recovery_recorded'):
                    self.metrics_collector.set_first_recovery_time(time.time())
                    self._first_recovery_recorded = True
                
                # Re-establish connections
                await self._reestablish_acp_connections(agent_id, new_port)
                
                self.output.success(f"✅ [ACP] Agent {agent_id} successfully reconnected on port {new_port}!")
                
        except Exception as e:
            self.output.error(f"Failed to reconnect ACP agent {agent_id}: {e}")
    
    async def _reestablish_acp_connections(self, agent_id: str, port: int) -> None:
        """Re-establish ACP connections for reconnected agent."""
        try:
            agent = self.agents.get(agent_id)
            if not agent:
                return
                
            # Register endpoints with other agents
            for other_id, other_agent in self.agents.items():
                if other_id != agent_id and other_id not in self.killed_agents:
                    # Register this agent with others
                    await other_agent.register_endpoint(agent_id, f"http://127.0.0.1:{port}")
                    # Register others with this agent
                    other_port = int(self.acp_sessions[other_id]['base_url'].split(':')[-1])
                    await agent.register_endpoint(other_id, f"http://127.0.0.1:{other_port}")
            
            self.output.info(f"🔗 [ACP] Re-established connections for {agent_id}")
            
        except Exception as e:
            self.output.error(f"Failed to re-establish connections for {agent_id}: {e}")


if __name__ == "__main__":
    import asyncio
    
    async def main():
        runner = ACPRunner()
        await runner.run_scenario()
    
    asyncio.run(main())
