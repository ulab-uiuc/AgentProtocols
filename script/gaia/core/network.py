"""Enhanced MeshNetwork with dynamic agent management and intelligent routing."""
import asyncio
import json
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import inspect
from abc import ABC, abstractmethod
import sys
import os
import logging

from pytest import Config
from .agent import MeshAgent
from .schema import AgentState, Message, Role, ToolCall, Memory, NetworkMemoryPool, ExecutionStatus, Colors

# GAIA 根目录 (script/gaia)
GAIA_ROOT = Path(__file__).resolve().parent.parent

# Setup logger for core network
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AgentTaskTimeout(Exception):
    """Exception raised when agent task times out."""
    pass


class AgentTaskFailed(Exception):
    """Exception raised when agent task fails."""
    pass

class MeshNetwork(ABC):
    """Enhanced MeshNetwork with dynamic agent management and intelligent routing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.agents: List[MeshAgent] = []
        self.config: Dict[str, Any] = config
        self.task_id = self.config.get("task_id", "unknown") if isinstance(self.config, dict) else "unknown"
        self.network_memory: NetworkMemoryPool = NetworkMemoryPool()  # Step-based memory pool
        self.running = False
        
        # Store original query for task reflection
        self.original_query: str = ""
        
    # Metrics
        self.bytes_tx = 0
        self.bytes_rx = 0
        self.pkt_cnt = 0
        self.header_overhead = 0
        self.token_sum = 0
        self.start_ts = None
        self.done_ts = None
        self.done_payload = None
        self.connections = []
        
        # Get timeout from config
        network_config = self.config.get("network", {}) if isinstance(self.config, dict) else {}
        self.timeout = network_config.get("timeout_seconds", 30)
    # ==================== Abstract Communication Methods ====================
    
    @abstractmethod
    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """
        Deliver message to specific agent.
        
        This is an abstract method that must be implemented by concrete network classes
        to provide the specific message delivery mechanism for their protocol.
        
        Args:
            dst: Destination agent ID
            msg: Message payload
        """
        pass

    # ==================== Network Management ====================

    async def start(self):
        """Start the network and message processing."""
        print("🌐 Starting multi-agent network...")
        
        # Set start timestamp for metrics
        self.start_ts = time.time() * 1000  # Convert to milliseconds
        
        # Start all agents concurrently in separate tasks
        agent_tasks = []
        for agent in self.agents:
            async def start_agent(agent):
                try:
                    await agent.start()
                    print(f"✅ Started agent {agent.id} ({agent.name}) on port {agent.port}")
                except Exception as e:
                    print(f"❌ Failed to start agent {agent.id}: {e}")
            
            # Create a task for each agent to start concurrently
            task = asyncio.create_task(start_agent(agent))
            agent_tasks.append(task)
        
        # Store agent tasks for later cleanup but don't wait for them
        # (agent.start() is an infinite loop, so we shouldn't wait for completion)
        self._agent_tasks = agent_tasks
        
        # Give agents a moment to initialize their connections
        await asyncio.sleep(0.1)
        print(f"🤖 All {len(self.agents)} agent tasks have been started")
        
        # Start message processing loop
        # asyncio.create_task(self._message_processing_loop())
        
        self.running = True
        print("🚀 Network started successfully")
    
    async def stop(self):
        """Stop the network and all agents."""
        print("🛑 Stopping network...")
        self.running = False
        
        # Cancel all agent tasks first
        if hasattr(self, '_agent_tasks'):
            for task in self._agent_tasks:
                if not task.done():
                    task.cancel()
            # Wait a moment for cancellation to complete
            await asyncio.sleep(0.1)
        
        # Stop all agents concurrently
        stop_tasks = []
        for agent in self.agents:
            async def stop_agent(agent):
                try:
                    await agent.stop()
                    print(f"✅ Stopped agent {agent.id}")
                except Exception as e:
                    print(f"❌ Error stopping agent {agent.id}: {e}")
            
            # Create a task for each agent to stop concurrently
            task = asyncio.create_task(stop_agent(agent))
            stop_tasks.append(task)
        
        # Wait for all agents to stop (with timeout)
        if stop_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*stop_tasks, return_exceptions=True),
                    timeout=5.0  # 5 seconds timeout for all agents to stop
                )
            except asyncio.TimeoutError:
                print("⚠️ Some agents took longer than expected to stop")
        
        print("✅ Network stopped")

    # ==================== Health Check (Protocol-agnostic) ====================
    async def monitor_agent_health(self) -> bool:
        """Check health of network and all agents."""
        try:
            # Check individual agent health (via agent.health_check)
            healthy_agents = 0
            for agent in self.agents:
                try:
                    # agent maintains its endpoint; mirror provided sample logic
                    if hasattr(agent, 'health_check') and callable(getattr(agent, 'health_check')):
                        healthy = await agent.health_check()
                    else:
                        # Fallback to backend check if agent lacks method
                        raise NotImplementedError("Agent lacks health_check method")
                    if healthy:
                        healthy_agents += 1
                except Exception as e:
                    print(f"❌ Error checking agent {agent.id} health: {e}")
            
            agent_health_ratio = healthy_agents / len(self.agents) if self.agents else 0
            
            if agent_health_ratio == 1.0:
                # All agents healthy - green
                logger.info(f"{Colors.GREEN}✅ Network: {healthy_agents}/{len(self.agents)} agents are healthy{Colors.RESET}")
                return True
            elif agent_health_ratio >= 0.5:
                # Most agents healthy - yellow  
                logger.warning(f"{Colors.YELLOW}⚠️  Network: {healthy_agents}/{len(self.agents)} agents are healthy{Colors.RESET}")
                return True
            else:
                # Too few healthy - red
                logger.info(f"{Colors.RED}❌ Network: Only {healthy_agents}/{len(self.agents)} agents are healthy{Colors.RESET}")
                return False
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Network health check failed: {e}{Colors.RESET}")
            return False

    def register_agent(self, agent: MeshAgent) -> None:
        """Register a new agent in the system."""
        self.agents.append(agent)
        print(f"📝 Registered agent {agent.id} ({agent.name}) with tool {agent.tool_name}")
    
    def unregister_agent(self, agent_id: int) -> None:
        """Remove agent from the system."""
        self.agents = [agent for agent in self.agents if agent.id != agent_id]
        print(f"🗑️ Unregistered agent {agent_id}")
    
    def get_agent_by_id(self, agent_id: int) -> Optional[MeshAgent]:
        """Get agent by ID from the agents list."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    # ==================== Workflow Execution ==================== 
    async def execute_workflow(self, config: Dict[str, Any], initial_task: str = None) -> str:
        """
        Execute workflow based on configuration with proper memory management.
        
        Args:
            config: Configuration containing workflow, agents, and agent_prompts
            initial_task: Initial task to start the workflow with
            
        Returns:
            Final result from the workflow execution
        """
        try:
            workflow = config.get('workflow', {})
            
            start_agent_id = workflow.get('start_agent', 0)
            message_flow = workflow.get('message_flow', [])
            execution_pattern = workflow.get('execution_pattern', 'sequential')
            
            # Store original query for task reflection throughout the workflow
            self.original_query = initial_task or "Begin task execution"
            
            print(f"🚀 Starting workflow execution with pattern: {execution_pattern}")
            print(f"📋 Starting agent: {start_agent_id}")
            print(f"🎯 Original Query: {self.original_query[:100]}...")
            
            # Initialize workflow state
            workflow_results = {}
            final_result = None
            current_input = initial_task or "Begin task execution"
            
            # Process workflow according to message flow
            for step_idx, flow_step in enumerate(message_flow):
                from_agent_id = flow_step.get('from')
                to_agents = flow_step.get('to')
                message_type = flow_step.get('message_type', 'task_result')
                
                print(f"🔄 Processing step {step_idx + 1}: Agent {from_agent_id} -> {to_agents} ({message_type})")
                
                try:
                    # Get previous agent's assistant memory if exists with original query
                    context_message = self._build_workflow_context(current_input, step_idx, self.original_query)
                    
                    # Execute agent with step information
                    agent_result = await self._execute_agent_step(from_agent_id, context_message, step_idx)
                    workflow_results[f'step_{step_idx}'] = agent_result
                    
                    # Check if this is the final step
                    if to_agents == 'final' or to_agents == ['final']:
                        final_result = agent_result
                        print(f"🎯 Final result obtained: {final_result}")
                        break
                    
                    # Use result as input for next step
                    current_input = agent_result
                    
                except (AgentTaskTimeout, AgentTaskFailed) as e:
                    print(f"❌ Agent {from_agent_id} failed: {e}")
                    print(f"⏭️ Moving to next agent in workflow")
                    
                    # Use error message as input for next step
                    current_input = f"Previous agent failed: {e}"
                    workflow_results[f'step_{step_idx}'] = f"FAILED: {e}"
                    continue
                
                # Add delay between steps
                await asyncio.sleep(0.5)
            

            # Return final result, and use LLM to summarize all memory
            if final_result:
                print(f"🎉 Workflow execution completed successfully")
                # Use LLM to summarize network memory
                try:
                    summary = await self.network_memory.summarize(initial_task=initial_task)
                    print(f"📝 Final summary: {summary}")
                    await self._log_message_pool_to_workspace()
                    return summary
                except Exception as e:
                    print(f"⚠️ Memory summary failed: {e}")
                    await self._log_message_pool_to_workspace()
                    return final_result
            else:
                # Return the last available result
                last_result = list(workflow_results.values())[-1] if workflow_results else "No results generated"
                print(f"⚠️ Workflow completed but no final result marked, returning last result")
                await self._log_message_pool_to_workspace()
                return last_result
                
        except Exception as e:
            print(f"❌ Error executing workflow: {e}")
            import traceback
            traceback.print_exc()
            return f"Workflow execution failed: {e}"
    
    async def _execute_agent_step(self, agent_id: int, context_message: str, step_idx: int) -> str:
        """
        Execute agent for a single workflow step using direct message passing.
        
        Args:
            agent_id: Target agent ID
            context_message: Context message built from previous agents
            step_idx: Current step index
            
        Returns:
            Agent execution result
            
        Raises:
            AgentTaskTimeout: If agent doesn't respond within timeout
            AgentTaskFailed: If agent execution fails
        """
        agent = self.get_agent_by_id(agent_id)
        if not agent:
            raise AgentTaskFailed(f"Agent {agent_id} not found in network")
        
        print(f"🔄 Executing agent {agent_id} for step {step_idx + 1}")
        
        # Add step execution to network memory
        self.network_memory.add_step_execution(
            step=step_idx, 
            agent_id=str(agent_id), 
            agent_name=agent.name,
            task_id=self.task_id,
            user_message=context_message
        )
        
        # Create a result container to capture agent response
        result_container = {"result": None, "received": False}
        
        # Set up callback to capture agent result directly
        async def capture_result(message_data):
            result_container["result"] = message_data["assistant_response"]
            result_container["received"] = True
            print(f"📥 Captured result from agent {agent_id}: {message_data['assistant_response']}")
        
        try:
            # Update step status to processing
            self.network_memory.update_step_status(step_idx, ExecutionStatus.PROCESSING)
            
            # Set callback for this agent to capture result
            agent.set_result_callback(capture_result)
            
            # Send context to agent 
            await self.deliver(agent_id, {
                "type": "task_execution",
                "sender_id": 0,  # Network sender
                "content": context_message,
                "step": step_idx,
                "timestamp": time.time()
            })
            
            # Wait for agent to process and callback to be triggered
            timeout = self.timeout
            elapsed = 0.0
            poll_interval = 0.1
            
            while not result_container["received"] and elapsed < timeout:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
            
            # Clear callback after receiving result
            agent.set_result_callback(None)
            
            # Check if we got a result
            if not result_container["received"]:
                raise AgentTaskTimeout(f"Agent {agent_id} did not respond within {timeout} seconds")
            
            result = result_container["result"]
            if not result or not result.strip():
                try:
                    print(f"[MeshNetwork] Empty result from agent id={agent_id}, name={agent.name}, step={step_idx}")
                except Exception:
                    pass
                self.network_memory.update_step_status(step_idx, ExecutionStatus.ERROR, error_message="Empty result from agent")
                return "No meaningful result generated by agent"
            else:
                # Get memory messages for logging purposes only
                messages = agent.get_memory_messages()
                self.network_memory.update_step_status(step_idx, ExecutionStatus.SUCCESS, 
                                                     messages=messages)
            
            print(f"✅ Agent {agent_id} completed step {step_idx + 1}")
            print(f"� Result: {result}")
            
            return result
            
        except Exception as e:
            # Clear callback on error
            agent.set_result_callback(None)
            self.network_memory.update_step_status(step_idx, ExecutionStatus.ERROR, 
                                                 error_message=str(e))
            raise AgentTaskFailed(f"Agent {agent_id} execution failed: {e}")

    def _build_workflow_context(self, current_input: str, step_idx: int, original_query: str = "") -> str:
        """
        Build context message from previous steps using step-based memory.
        """
        context_parts = []
        
        # Add original query at the top for constant reminder
        if original_query:
            context_parts.append("🎯 ORIGINAL TASK REQUIREMENT:")
            context_parts.append(original_query)
            context_parts.append("=" * 80)
        
        context_parts.append(f"Step {step_idx + 1} - Current Task Input:")
        context_parts.append(current_input)
        
        # Add context from previous steps using NetworkMemoryPool
        if step_idx > 0:
            previous_context = self.network_memory.get_step_chain_context(step_idx)
            
            if previous_context:
                context_parts.append("\nPrevious Steps Context:")
                for ctx in previous_context:
                    step_num = ctx["step"] + 1  # Human-readable step number
                    agent_name = ctx["agent_name"]
                    result = ctx["result"]
                    status = ctx["status"]
                    
                    context_parts.append(f"Step {step_num} ({agent_name} - {status}): {result}")
        
        # Remind about original task again at the end
        if original_query:
            context_parts.append("\n⚠️  REMEMBER: All your work must contribute to answering the ORIGINAL TASK above!")
        
        return "\n".join(context_parts)

    async def _log_message_pool_to_workspace(self):
        """Log network memory pool with step-based structure to workspace (under GAIA root)."""
        try:
            task_id = self.config.get("task_id", "unknown") if isinstance(self.config, dict) else "unknown"
            protocol_name = (self.config.get("protocol") if isinstance(self.config, dict) else None) or "general"
            logs_dir = GAIA_ROOT / "workspaces" / protocol_name / task_id
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Save network memory with step-based structure
            # Helper to safely serialize messages and truncate very large contents
            def _safe_msg_dict(msg):
                d = msg.to_dict()
                content = d.get('content')
                try:
                    if content is None:
                        pass
                    else:
                        # If content looks binary-ish or is extremely large, replace with placeholder
                        if isinstance(content, (bytes, bytearray)):
                            d['content'] = "<BINARY_PAYLOAD - omitted; saved to workspace file if needed>"
                        else:
                            s = str(content)
                            if len(s) > 2000:
                                # Truncate long contents but keep a preview
                                d['content'] = s[:2000] + "...<truncated>"
                            else:
                                d['content'] = s
                except Exception:
                    d['content'] = "<UNSERIALIZABLE_CONTENT>"
                return d

            network_log = {
                "network_memory": [(_safe_msg_dict(msg)) for step_exec in self.network_memory.step_executions.values() if step_exec.messages for msg in step_exec.messages],
                "workflow_progress": self.network_memory.get_workflow_progress(),
                "step_executions": {},
                "timestamp": time.time()
            }
            
            # Save step-based execution records
            for step, step_exec in self.network_memory.step_executions.items():
                network_log["step_executions"][f"step_{step}"] = {
                    "step": step_exec.step,
                    "agent_id": step_exec.agent_id,
                    "agent_name": step_exec.agent_name,
                    "status": step_exec.execution_status.value,
                    "start_time": step_exec.start_time,
                    "end_time": step_exec.end_time,
                    "duration": step_exec.duration(),
                    "messages": [_safe_msg_dict(msg) for msg in step_exec.messages],
                    "error_message": step_exec.error_message
                }
            
            # Write to file
            log_file = logs_dir / "network_execution_log.json"
            with open(log_file, "w", encoding='utf-8') as f:
                json.dump(network_log, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Step-based message pool logged to {log_file}")
            
        except Exception as e:
            print(f"❌ Failed to log message pool: {e}")
        
    # ==================== Archive and Evaluation ====================
    
    async def _archive_artifacts(self):
        """Archive workspace artifacts (under GAIA root)."""
        task_id = self.config.get("task_id", "unknown") if isinstance(self.config, dict) else "unknown"
        protocol_name = (self.config.get("protocol") if isinstance(self.config, dict) else None) or "default"
        try:
            workspaces_path = GAIA_ROOT / "workspaces" / protocol_name / task_id
            if workspaces_path.exists():
                with tarfile.open("run_artifacts.tar.gz", "w:gz") as tar:
                    for workspace in workspaces_path.iterdir():
                        if workspace.is_dir():
                            tar.add(workspace, arcname=workspace.name)
                print("📦 Artifacts archived to run_artifacts.tar.gz")
            else:
                print("📁 No workspaces found to archive")
        except Exception as e:
            print(f"Error archiving artifacts: {e}")
            
    async def evaluate(self):
        """Run final evaluation and archive artifacts."""
        print("📊 Running evaluation...")
        
        # # Run quality evaluation
        # try:
        #     quality = await eval_runner(self.done_payload or "", "ground_truth.json")
        # except Exception as e:
        #     print(f"Evaluation error: {e}")
        #     quality = {"quality_score": 0.0, "exact_match": 0, "error": str(e)}
        
        # Compile metrics report
        elapsed_ms = 0
        if self.done_ts and self.start_ts:
            elapsed_ms = self.done_ts - self.start_ts
        elif self.start_ts:
            # If no done_ts, calculate from current time
            elapsed_ms = time.time() * 1000 - self.start_ts
        
        report = {
            "performance_metrics": {
                "bytes_tx": self.bytes_tx,
                "bytes_rx": self.bytes_rx,
                "pkt_cnt": self.pkt_cnt,
                "header_overhead": self.header_overhead,
                "token_sum": self.token_sum,
                "elapsed_ms": elapsed_ms
            },
            # "quality_metrics": quality,
            "configuration": {
                "num_agents": len(self.agents),
                "task_id": self.config.get("task_id", "unknown"),
                "complexity": self.config.get("task_analysis", {}).get("complexity", "unknown")
            },
            "final_answer": self.done_payload
        }
        
        # Save metrics
        with open("metrics.json", "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("💾 Metrics saved to metrics.json")
        
        # Archive workspaces
        await self._archive_artifacts()
        
        print("✅ Evaluation complete!")
        # print(f"📈 Quality Score: {quality.get('quality_score', 0):.2f}")
        print(f"⚡ Total Time: {report['performance_metrics']['elapsed_ms']:.0f}ms")
        print(f"🔢 Total Tokens: {report['performance_metrics']['token_sum']}")