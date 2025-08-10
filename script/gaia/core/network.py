"""Enhanced MeshNetwork with dynamic agent management and intelligent routing."""
import asyncio
import json
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import sys
import os

# Add tenacity for retry mechanism
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:
    print("Warning: tenacity not found, please install with: pip install tenacity")
    # Mock retry decorator for fallback
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    stop_after_attempt = wait_exponential = retry_if_exception_type = lambda x: x

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .agent_base import MeshAgent


class AgentTaskTimeout(Exception):
    """Exception raised when agent task times out."""
    pass


class AgentTaskFailed(Exception):
    """Exception raised when agent task fails."""
    pass


async def eval_runner(pred: str, truth_path: str) -> Dict[str, Any]:
    """Calculate quality score for final answer."""
    try:
        with open(truth_path, encoding='utf-8') as f:
            truth = json.load(f)
        
        # Simple exact match and basic similarity
        em = int(pred.strip().lower() == truth.get("answer", "").strip().lower())
        
        # Basic similarity score (can be enhanced with ROUGE-L)
        pred_words = set(pred.lower().split())
        truth_words = set(truth.get("answer", "").lower().split())
        
        if pred_words and truth_words:
            similarity = len(pred_words & truth_words) / len(pred_words | truth_words)
        else:
            similarity = 0.0
        
        quality_score = (em + similarity) / 2
        
        return {
            "quality_score": quality_score,
            "exact_match": em,
            "similarity": similarity
        }
    except Exception as e:
        return {
            "quality_score": 0.0,
            "exact_match": 0,
            "similarity": 0.0,
            "error": str(e)
        }


class MeshNetwork:
    """Enhanced MeshNetwork with dynamic agent creation and intelligent routing."""
    
    def __init__(self):
        self.agents: List[MeshAgent] = []
        self.config: Dict[str, Any] = {}
        
        # Control flags
        self.running = False
        
        # Message processing
        self._message_buffer: List[Tuple[int, Dict[str, Any]]] = []
        
        # Message Pool for context management
        self._message_pool = {
            "agent_inputs": {},      # 存储每个agent接收到的输入消息
            "agent_outputs": {},     # 存储每个agent的输出/完成记录
            "conversation_history": [],  # 完整的对话历史
            "context_metadata": {    # 上下文元数据
                "session_start": time.time(),
                "message_count": 0,
                "active_agents": set(),
                "workflow_steps": []
            }
        }
        
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
        
        # Message dispatch table
        self._message_dispatch = {
            "doc_init": self._handle_doc_init,
            "task_result": self._handle_task_result,
            "search_results": self._handle_search_results,
            "file_result": self._handle_file_result,
            "code_result": self._handle_code_result,
            "data_event": self._handle_data_event,
            "agent_shutdown": self._handle_agent_shutdown,
            "error": self._handle_error_message,
            "workflow_task": self._handle_workflow_task,
            "workflow_result": self._handle_workflow_result,
        }
    
    # ==================== Communication Methods ====================
    
    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """
        Deliver message to specific agent by putting it directly into their receive queue.
        
        Args:
            dst: Destination agent ID
            msg: Message payload
        """
        # Find the destination agent
        target_agent = self.get_agent_by_id(dst)
        if target_agent:
            try:
                # For DummyAgent, directly put message into agent's mailbox
                if hasattr(target_agent, '_client') and hasattr(target_agent._client, '_mailbox'):
                    # Add network metadata
                    enhanced_msg = {
                        **msg,
                        "_network_meta": {
                            "delivered_by": "network",
                            "target_agent": dst,
                            "delivery_timestamp": time.time(),
                            "message_id": f"net_{int(time.time() * 1000000)}"
                        }
                    }
                    
                    # Record message in message pool
                    await self._record_input_message(dst, enhanced_msg)
                    
                    # Directly deliver to agent's mailbox
                    await target_agent._client._mailbox.put(enhanced_msg)
                    
                    # Update metrics
                    self.pkt_cnt += 1
                    msg_size = len(json.dumps(msg).encode('utf-8'))
                    self.bytes_tx += msg_size
                    
                    print(f"📤 Delivered message to agent {dst}: {msg.get('type', 'unknown')}")
                else:
                    print(f"❌ Agent {dst} does not support direct message delivery")
            except Exception as e:
                print(f"❌ Failed to deliver message to agent {dst}: {e}")
        else:
            print(f"❌ Agent {dst} not found in network")
    
    async def poll(self) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Poll for incoming messages from agents using agent's recv_msg method.
        
        Returns:
            List of (sender_id, message) tuples
        """
        messages = []
        
        # Poll each agent for incoming messages
        for agent in self.agents:
            try:
                # Use non-blocking receive (timeout=0.0)
                msg = await agent.recv_msg(timeout=0.0)
                if msg:
                    # Update metrics
                    self.pkt_cnt += 1
                    msg_size = len(json.dumps(msg).encode('utf-8'))
                    self.bytes_rx += msg_size
                    
                    messages.append((agent.id, msg))
                    print(f"📥 Polled message from agent {agent.id}: {msg.get('type', 'unknown')}")
            except Exception as e:
                # Don't log errors for non-blocking poll operations
                pass
        
        return messages
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
        
        # Wait for all agents to start (with timeout to avoid hanging)
        if agent_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*agent_tasks, return_exceptions=True), 
                    timeout=10.0  # 10 seconds timeout for all agents to start
                )
                print(f"🤖 All {len(self.agents)} agents startup tasks initiated")
            except asyncio.TimeoutError:
                print("⚠️ Some agents took longer than expected to start, continuing...")
        
        # Start message processing loop
        asyncio.create_task(self._message_processing_loop())
        
        self.running = True
        print("🚀 Network started successfully")
    
    async def stop(self):
        """Stop the network and all agents."""
        print("🛑 Stopping network...")
        self.running = False
        
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

    # ==================== Message Processing ====================
    
    def register_message_handler(self, message_type: str, handler) -> None:
        """Register a new message handler in the dispatch table."""
        self._message_dispatch[message_type] = handler
        print(f"📝 Registered handler for message type: {message_type}")
    
    def unregister_message_handler(self, message_type: str) -> None:
        """Unregister a message handler from the dispatch table."""
        if message_type in self._message_dispatch:
            del self._message_dispatch[message_type]
            print(f"🗑️ Unregistered handler for message type: {message_type}")
    
    def get_registered_message_types(self) -> List[str]:
        """Get list of all registered message types."""
        return list(self._message_dispatch.keys())
    
    async def process_messages(self) -> None:
        """Process all incoming messages from agents."""
        # Poll for new messages using the protocol-specific implementation
        try:
            messages = await self.poll()
            self._message_buffer.extend(messages)
        except Exception as e:
            print(f"Error polling messages: {e}")
            return
        
        # Process buffered messages
        for sender_id, msg in self._message_buffer:
            await self._handle_message(sender_id, msg)
        
        self._message_buffer.clear()
    
    async def _handle_message(self, sender_id: int, msg: Dict[str, Any]) -> None:
        """Handle different types of messages from agents using dispatch table."""
        msg_type = msg.get("type")
        
        # Use dispatch table for message handling
        handler = self._message_dispatch.get(msg_type)
        if handler:
            try:
                await handler(sender_id, msg)
            except Exception as e:
                print(f"❌ Error handling {msg_type} from agent {sender_id}: {e}")
        else:
            print(f"❓ Unknown message type: {msg_type} from agent {sender_id}")
    
    async def _handle_doc_init(self, sender_id: int, msg: Dict[str, Any]) -> None:
        """Handle document initialization messages."""
        print(f"📄 Agent {sender_id} processed document initialization")
        # This is typically sent by the network to agents, not the other way around
    
    async def _handle_task_result(self, sender_id: int, msg: Dict[str, Any]) -> None:
        """Handle task result messages between agents."""
        result = msg.get("result", "")
        source = msg.get("source", f"agent_{sender_id}")
        
        print(f"📤 Task result from {source} (Agent {sender_id}): {result[:100]}...")
    
    async def _handle_search_results(self, sender_id: int, msg: Dict[str, Any]) -> None:
        """Handle search results from search agents."""
        result = msg.get("result", "")
        print(f"🔍 Search results from Agent {sender_id}: {result[:100]}...")
    
    async def _handle_file_result(self, sender_id: int, msg: Dict[str, Any]) -> None:
        """Handle file operation results."""
        result = msg.get("result", "")
        print(f"📁 File operation result from Agent {sender_id}: {result[:100]}...")
    
    async def _handle_code_result(self, sender_id: int, msg: Dict[str, Any]) -> None:
        """Handle code execution results."""
        result = msg.get("result", "")
        print(f"💻 Code execution result from Agent {sender_id}: {result[:100]}...")
    
    async def _handle_data_event(self, sender_id: int, msg: Dict[str, Any]) -> None:
        """Handle data events like final answers."""
        tag = msg.get("tag", "")
        payload = msg.get("payload", "")
        
        if tag == "final_answer":
            if not self.done_ts:
                self.done_ts = time.time() * 1000
                self.done_payload = payload
                print(f"🎯 Final answer received from Agent {sender_id}: {payload[:100]}...")
        else:
            print(f"📊 Data event '{tag}' from Agent {sender_id}")
    
    async def _handle_agent_shutdown(self, sender_id: int, msg: Dict[str, Any]) -> None:
        """Handle agent shutdown notifications."""
        agent_name = msg.get("agent_name", f"Agent_{sender_id}")
        tokens_used = msg.get("total_tokens_used", 0)
        
        print(f"🔻 Agent {agent_name} (ID: {sender_id}) shutdown. Tokens used: {tokens_used}")
    
    async def _handle_error_message(self, sender_id: int, msg: Dict[str, Any]) -> None:
        """Handle error messages from agents."""
        error = msg.get("error", "Unknown error")
        agent_name = msg.get("agent_name", f"Agent_{sender_id}")
        
        print(f"❌ Error from {agent_name}: {error}")
    
    async def _handle_workflow_task(self, sender_id: int, msg: Dict[str, Any]) -> None:
        """Handle workflow task messages sent to agents."""
        task_input = msg.get("task_input", "")
        step = msg.get("step", 0)
        from_agent = msg.get("from", "system")
        
        print(f"📥 Workflow task received for agent {sender_id} from {from_agent} (step {step})")
        
        # Find the target agent and execute its tool with the task input
        agent = self.get_agent_by_id(sender_id)
        if agent and hasattr(agent, '_execute_tool'):
            try:
                # Execute the agent's tool with the task input
                result = await agent._execute_tool(task_input)
                
                # Send result back via deliver
                result_message = {
                    'type': 'workflow_result',
                    'result': result,
                    'step': step,
                    'timestamp': int(time.time())
                }
                
                # The result will be picked up by poll() in execute_workflow
                await self.deliver(sender_id, result_message)
                
            except Exception as e:
                print(f"❌ Error executing workflow task for agent {sender_id}: {e}")
                
                # Send error result
                error_message = {
                    'type': 'workflow_result',
                    'result': f"Error: {e}",
                    'step': step,
                    'timestamp': int(time.time())
                }
                await self.deliver(sender_id, error_message)
    
    async def _handle_workflow_result(self, sender_id: int, msg: Dict[str, Any]) -> None:
        """Handle workflow result messages from agents."""
        result = msg.get("result", "")
        step = msg.get("step", 0)
        
        print(f"📤 Workflow result received from agent {sender_id} (step {step}): {result[:100]}...")
        
        # This message will be processed by the execute_workflow polling loop
        # No additional processing needed here, just log it
    
    async def _message_processing_loop(self) -> None:
        """Main message processing loop."""
        print("🔄 Starting message processing loop...")
        
        while self.running:
            try:
                await self.process_messages()
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.05)
            except Exception as e:
                print(f"Error in message processing loop: {e}")
                await asyncio.sleep(0.1)
        
        print("🔄 Message processing loop stopped")

    # ==================== Workflow Execution with Tenacity ====================
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((AgentTaskTimeout, AgentTaskFailed))
    )
    async def _execute_agent_with_retry(self, agent_id: int, task_input: str, step: int) -> str:
        """
        Execute a single agent with tenacity retry mechanism.
        
        Args:
            agent_id: ID of the agent to execute
            task_input: Task input for the agent
            step: Current workflow step
            
        Returns:
            Result from the agent
            
        Raises:
            AgentTaskTimeout: If agent doesn't respond within timeout
            AgentTaskFailed: If agent execution fails
        """
        print(f"🔄 Executing agent {agent_id} (attempt, step {step})")
        
        # Send task to agent using deliver
        initial_message = {
            'type': 'workflow_task',
            'task_input': task_input,
            'from': 'system',
            'step': step,
            'timestamp': int(time.time())
        }
        
        try:
            await self.deliver(agent_id, initial_message)
        except Exception as e:
            raise AgentTaskFailed(f"Failed to deliver task to agent {agent_id}: {e}")
        
        # Wait for result with timeout
        timeout = 30.0  # 30 seconds timeout per attempt
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Poll for messages
                messages = await self.poll()
                
                # Look for result from the expected agent
                for sender_id, msg in messages:
                    if sender_id == agent_id and msg.get('type') == 'workflow_result':
                        result = msg.get('result', '')
                        print(f"✅ Received result from agent {agent_id}: {result[:100]}...")
                        return result
                
                # Small delay to avoid busy waiting
                await asyncio.sleep(0.2)
                
            except Exception as e:
                raise AgentTaskFailed(f"Error polling for agent {agent_id} response: {e}")
        
        # If we reach here, timeout occurred
        raise AgentTaskTimeout(f"Agent {agent_id} did not respond within {timeout} seconds")

    async def execute_workflow(self, config: Dict[str, Any], initial_task: str = None) -> str:
        """
        Execute workflow based on configuration using tenacity retry mechanism.
        
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
            
            print(f"🚀 Starting workflow execution with pattern: {execution_pattern}")
            print(f"📋 Starting agent: {start_agent_id}")
            
            # Initialize workflow state
            workflow_results = {}
            final_result = None
            current_input = initial_task or "Begin task execution"
            
            # Process workflow according to message flow with retry
            for step_idx, flow_step in enumerate(message_flow):
                from_agent_id = flow_step.get('from')
                to_agents = flow_step.get('to')
                message_type = flow_step.get('message_type', 'task_result')
                
                print(f"🔄 Processing step {step_idx + 1}: Agent {from_agent_id} -> {to_agents} ({message_type})")
                
                try:
                    # Execute agent with retry mechanism
                    agent_result = await self._execute_agent_with_retry(from_agent_id, current_input, step_idx)
                    workflow_results[f'step_{step_idx}'] = agent_result
                    
                    # Check if this is the final step
                    if to_agents == 'final' or to_agents == ['final']:
                        final_result = agent_result
                        print(f"🎯 Final result obtained: {final_result[:100]}...")
                        break
                    
                    # Use result as input for next step
                    current_input = agent_result
                    
                except (AgentTaskTimeout, AgentTaskFailed) as e:
                    print(f"❌ Agent {from_agent_id} failed after 3 retries: {e}")
                    print(f"⏭️ Moving to next agent in workflow")
                    
                    # Use error message as input for next step
                    current_input = f"Previous agent failed: {e}"
                    workflow_results[f'step_{step_idx}'] = f"FAILED: {e}"
                    continue
                
                # Add delay between steps
                await asyncio.sleep(0.5)
            
            # Return final result
            if final_result:
                print(f"🎉 Workflow execution completed successfully")
                
                # Log message pool to workspace if available
                await self._log_message_pool_to_workspace()
                
                return final_result
            else:
                # Return the last available result
                last_result = list(workflow_results.values())[-1] if workflow_results else "No results generated"
                print(f"⚠️ Workflow completed but no final result marked, returning last result")
                
                # Log message pool to workspace if available
                await self._log_message_pool_to_workspace()
                
                return last_result
                
        except Exception as e:
            print(f"❌ Error executing workflow: {e}")
            import traceback
            traceback.print_exc()
            return f"Workflow execution failed: {e}"
    
    async def _log_message_pool_to_workspace(self) -> None:
        """
        Log message pool information to workspace directory if available.
        This method checks if the network has message pool functionality and logs it.
        """
        try:
            # Get task_id from config
            task_id = self.config.get("task_id", "unknown") if isinstance(self.config, dict) else "unknown"
            
            # Create workspace directory
            workspace_dir = Path(f"workspaces/{task_id}")
            workspace_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if this network has message pool functionality
            if hasattr(self, '_message_pool') and hasattr(self, 'get_full_conversation_context'):
                print("📝 Logging message pool to workspace...")
                
                # Get full conversation context
                context_data = self.get_full_conversation_context()
                
                # Create comprehensive log in markdown format for better readability
                log_content = self._format_message_pool_log(context_data, task_id)
                
                # Save as markdown file
                log_file = workspace_dir / f"message_pool_log_{task_id}.md"
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(log_content)
                
                # Also save raw JSON data for programmatic access
                json_file = workspace_dir / f"message_pool_data_{task_id}.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(context_data, f, indent=2, ensure_ascii=False, default=str)
                
                print(f"📊 Message pool logged to:")
                print(f"  📄 Readable log: {log_file}")
                print(f"  📊 Raw data: {json_file}")
                
            else:
                # Fallback: log basic network metrics
                print("📝 Logging basic network metrics to workspace...")
                
                basic_log = self._format_basic_network_log(task_id)
                log_file = workspace_dir / f"network_log_{task_id}.md"
                
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(basic_log)
                
                print(f"📊 Basic network log saved to: {log_file}")
                
        except Exception as e:
            print(f"❌ Error logging message pool to workspace: {e}")
    
    def _format_message_pool_log(self, context_data: Dict[str, Any], task_id: str) -> str:
        """
        Format message pool data into a readable markdown log.
        
        Args:
            context_data: Full conversation context data
            task_id: Task identifier
            
        Returns:
            Formatted markdown content
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        content = f"""# Message Pool Log - Task {task_id}

**Generated:** {timestamp}
**Network Type:** {self.__class__.__name__}

## Workflow Summary

"""
        
        # Add summary information
        summary = context_data.get('summary', {})
        content += f"""
### Key Metrics
- **Session Duration:** {summary.get('session_duration', 0):.2f} seconds
- **Total Agents:** {summary.get('total_agents', 0)}
- **Input Messages:** {summary.get('total_input_messages', 0)}
- **Output Completions:** {summary.get('total_output_completions', 0)}
- **Conversation Turns:** {summary.get('conversation_turns', 0)}

### Active Agents
{', '.join(map(str, summary.get('active_agents', [])))}

"""
        
        # Add conversation history
        history = context_data.get('message_pool', {}).get('conversation_history', [])
        if history:
            content += "## Conversation Timeline\n\n"
            for i, entry in enumerate(history, 1):
                timestamp_str = time.strftime("%H:%M:%S", time.localtime(entry.get('timestamp', 0)))
                direction = "📤" if entry.get('direction') == 'output' else "📥"
                agent_id = entry.get('agent_id', 'unknown')
                
                if entry.get('direction') == 'input':
                    msg_type = entry.get('message_type', 'unknown')
                    content_summary = entry.get('content_summary', '')[:200]
                    content += f"{i}. **{timestamp_str}** {direction} Agent {agent_id} received `{msg_type}`\n"
                    content += f"   - {content_summary}\n\n"
                else:
                    result_type = entry.get('result_type', 'unknown')
                    result_summary = entry.get('result_summary', '')[:200]
                    content += f"{i}. **{timestamp_str}** {direction} Agent {agent_id} completed `{result_type}`\n"
                    content += f"   - {result_summary}\n\n"
        
        # Add agent-specific contexts
        agent_contexts = context_data.get('agent_contexts', {})
        if agent_contexts:
            content += "## Agent Details\n\n"
            for agent_id, agent_context in agent_contexts.items():
                context_summary = agent_context.get('context_summary', {})
                content += f"### Agent {agent_id}\n"
                content += f"- **Input Messages:** {context_summary.get('input_messages_count', 0)}\n"
                content += f"- **Output Completions:** {context_summary.get('output_completions_count', 0)}\n"
                content += f"- **Unprocessed Inputs:** {context_summary.get('unprocessed_inputs', 0)}\n"
                
                # Add recent inputs and outputs
                inputs = agent_context.get('inputs', [])
                outputs = agent_context.get('outputs', [])
                
                if inputs:
                    content += "\n**Recent Inputs:**\n"
                    for inp in inputs[-3:]:  # Last 3 inputs
                        ts = time.strftime("%H:%M:%S", time.localtime(inp.get('timestamp', 0)))
                        msg_type = inp.get('message_type', 'unknown')
                        processed = "✅" if inp.get('processed') else "⏳"
                        content += f"- `{ts}` {processed} {msg_type}\n"
                
                if outputs:
                    content += "\n**Recent Outputs:**\n"
                    for out in outputs[-3:]:  # Last 3 outputs
                        ts = time.strftime("%H:%M:%S", time.localtime(out.get('timestamp', 0)))
                        result_type = out.get('result_type', 'unknown')
                        success = "✅" if out.get('success') else "❌"
                        content += f"- `{ts}` {success} {result_type}\n"
                
                content += "\n"
        
        # Add network performance metrics
        content += f"""## Network Performance

### Communication Metrics
- **Packets Sent:** {self.pkt_cnt}
- **Bytes Transmitted:** {self.bytes_tx}
- **Bytes Received:** {self.bytes_rx}
- **Token Usage:** {self.token_sum}

### Timing
- **Workflow Start:** {time.strftime("%H:%M:%S", time.localtime(self.start_ts / 1000)) if self.start_ts else "Unknown"}
- **Workflow End:** {time.strftime("%H:%M:%S", time.localtime(self.done_ts / 1000)) if self.done_ts else "Unknown"}

---
*Log generated by {self.__class__.__name__} at {timestamp}*
"""
        
        return content
    
    def _format_basic_network_log(self, task_id: str) -> str:
        """
        Format basic network metrics into a readable log for networks without message pool.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Formatted markdown content
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        content = f"""# Network Log - Task {task_id}

**Generated:** {timestamp}
**Network Type:** {self.__class__.__name__}

## Workflow Summary

### Network Configuration
- **Total Agents:** {len(self.agents)}
- **Agent Details:**
"""
        
        for agent in self.agents:
            content += f"  - Agent {agent.id}: {agent.name} ({agent.tool_name})\n"
        
        content += f"""
### Communication Metrics
- **Packets Processed:** {self.pkt_cnt}
- **Bytes Transmitted:** {self.bytes_tx}
- **Bytes Received:** {self.bytes_rx}
- **Total Tokens:** {self.token_sum}
- **Header Overhead:** {self.header_overhead}

### Timing Information
- **Workflow Start:** {time.strftime("%H:%M:%S", time.localtime(self.start_ts / 1000)) if self.start_ts else "Unknown"}
- **Workflow End:** {time.strftime("%H:%M:%S", time.localtime(self.done_ts / 1000)) if self.done_ts else "Unknown"}
- **Duration:** {((self.done_ts - self.start_ts) / 1000):.2f if self.done_ts and self.start_ts else 0} seconds

### Final Result
{self.done_payload if self.done_payload else "No final result recorded"}

---
*Log generated by {self.__class__.__name__} at {timestamp}*
"""
        
        return content

    # ==================== Configuration and Agent Management ====================
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        print(f"📋 Loading configuration from {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.json'):
                self.config = json.load(f)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                import yaml
                self.config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
        
        print(f"✅ Configuration loaded successfully")

    # ==================== Archive and Evaluation ====================
    
    async def _archive_artifacts(self):
        """Archive workspace artifacts."""
        task_id = self.config.get("task_id", "unknown") if isinstance(self.config, dict) else "unknown"
        try:
            workspaces_path = Path(f"workspaces/{task_id}")
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
        
        # Run quality evaluation
        try:
            quality = await eval_runner(self.done_payload or "", "ground_truth.json")
        except Exception as e:
            print(f"Evaluation error: {e}")
            quality = {"quality_score": 0.0, "exact_match": 0, "error": str(e)}
        
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
            "quality_metrics": quality,
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
        print(f"📈 Quality Score: {quality.get('quality_score', 0):.2f}")
        print(f"⚡ Total Time: {report['performance_metrics']['elapsed_ms']:.0f}ms")
        print(f"🔢 Total Tokens: {report['performance_metrics']['token_sum']}")
    
    # ==================== Message Pool Management Methods ====================
    
    async def _record_input_message(self, agent_id: int, msg: Dict[str, Any]) -> None:
        """
        Record input message for an agent in the message pool.
        
        Args:
            agent_id: ID of the receiving agent
            msg: Input message
        """
        if agent_id not in self._message_pool["agent_inputs"]:
            self._message_pool["agent_inputs"][agent_id] = []
        
        # Create input record
        input_record = {
            "timestamp": time.time(),
            "message_id": msg.get("_network_meta", {}).get("message_id", f"msg_{int(time.time() * 1000000)}"),
            "message_type": msg.get("type", "unknown"),
            "content": msg,
            "processed": False,
            "response_generated": False
        }
        
        self._message_pool["agent_inputs"][agent_id].append(input_record)
        
        # Update conversation history
        self._message_pool["conversation_history"].append({
            "direction": "input",
            "agent_id": agent_id,
            "timestamp": input_record["timestamp"],
            "message_type": input_record["message_type"],
            "content_summary": str(msg)[:200] + "..." if len(str(msg)) > 200 else str(msg)
        })
        
        # Update metadata
        self._message_pool["context_metadata"]["message_count"] += 1
        self._message_pool["context_metadata"]["active_agents"].add(agent_id)
        
        print(f"📝 Recorded input message for agent {agent_id}: {input_record['message_type']}")
    
    async def _record_output_completion(self, agent_id: int, result: Any) -> None:
        """
        Record output/completion for an agent in the message pool.
        
        Args:
            agent_id: ID of the agent
            result: Task completion result
        """
        if agent_id not in self._message_pool["agent_outputs"]:
            self._message_pool["agent_outputs"][agent_id] = []
        
        # Create output record
        output_record = {
            "timestamp": time.time(),
            "completion_id": f"comp_{agent_id}_{int(time.time() * 1000000)}",
            "result": result,
            "result_type": type(result).__name__,
            "result_summary": str(result)[:300] + "..." if len(str(result)) > 300 else str(result),
            "success": True,
            "context_used": self._get_agent_context_summary(agent_id)
        }
        
        self._message_pool["agent_outputs"][agent_id].append(output_record)
        
        # Update conversation history
        self._message_pool["conversation_history"].append({
            "direction": "output",
            "agent_id": agent_id,
            "timestamp": output_record["timestamp"],
            "result_type": output_record["result_type"],
            "result_summary": output_record["result_summary"]
        })
        
        # Mark related input messages as processed
        await self._mark_inputs_as_processed(agent_id)
        
        print(f"📝 Recorded output completion for agent {agent_id}: {output_record['result_type']}")
    
    def _get_agent_context_summary(self, agent_id: int) -> Dict[str, Any]:
        """
        Get context summary for an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Context summary dictionary
        """
        inputs = self._message_pool["agent_inputs"].get(agent_id, [])
        outputs = self._message_pool["agent_outputs"].get(agent_id, [])
        
        return {
            "input_messages_count": len(inputs),
            "output_completions_count": len(outputs),
            "last_input_time": inputs[-1]["timestamp"] if inputs else None,
            "last_output_time": outputs[-1]["timestamp"] if outputs else None,
            "unprocessed_inputs": len([msg for msg in inputs if not msg["processed"]])
        }
    
    async def _mark_inputs_as_processed(self, agent_id: int) -> None:
        """
        Mark recent input messages as processed for an agent.
        
        Args:
            agent_id: Agent ID
        """
        if agent_id in self._message_pool["agent_inputs"]:
            # Mark unprocessed inputs as processed
            for msg_record in self._message_pool["agent_inputs"][agent_id]:
                if not msg_record["processed"]:
                    msg_record["processed"] = True
                    msg_record["response_generated"] = True
    
    def get_message_pool_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the message pool state.
        
        Returns:
            Message pool summary
        """
        total_inputs = sum(len(inputs) for inputs in self._message_pool["agent_inputs"].values())
        total_outputs = sum(len(outputs) for outputs in self._message_pool["agent_outputs"].values())
        
        return {
            "session_duration": time.time() - self._message_pool["context_metadata"]["session_start"],
            "total_agents": len(self._message_pool["context_metadata"]["active_agents"]),
            "total_input_messages": total_inputs,
            "total_output_completions": total_outputs,
            "conversation_turns": len(self._message_pool["conversation_history"]),
            "active_agents": list(self._message_pool["context_metadata"]["active_agents"]),
            "latest_activity": self._message_pool["conversation_history"][-5:] if self._message_pool["conversation_history"] else []
        }
    
    def get_agent_context(self, agent_id: int) -> Dict[str, Any]:
        """
        Get full context for a specific agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent's full context including inputs and outputs
        """
        return {
            "agent_id": agent_id,
            "inputs": self._message_pool["agent_inputs"].get(agent_id, []),
            "outputs": self._message_pool["agent_outputs"].get(agent_id, []),
            "context_summary": self._get_agent_context_summary(agent_id),
            "conversation_participation": [
                entry for entry in self._message_pool["conversation_history"] 
                if entry["agent_id"] == agent_id
            ]
        }
    
    def get_full_conversation_context(self) -> Dict[str, Any]:
        """
        Get the complete conversation context across all agents.
        
        Returns:
            Full conversation context
        """
        return {
            "message_pool": self._message_pool,
            "summary": self.get_message_pool_summary(),
            "agent_contexts": {
                agent_id: self.get_agent_context(agent_id) 
                for agent_id in self._message_pool["context_metadata"]["active_agents"]
            }
        }
    
    async def export_context_to_file(self, filepath: str) -> None:
        """
        Export the complete message pool context to a JSON file.
        
        Args:
            filepath: Path to export file
        """
        context_data = self.get_full_conversation_context()
        
        # Add export metadata
        context_data["export_metadata"] = {
            "export_timestamp": time.time(),
            "export_format_version": "1.0",
            "network_type": self.__class__.__name__,
            "total_bytes_tx": self.bytes_tx,
            "total_bytes_rx": self.bytes_rx,
            "total_packets": self.pkt_cnt
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(context_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📊 Exported message pool context to: {filepath}")
    
    def clear_message_pool(self) -> None:
        """
        Clear the message pool while preserving metadata structure.
        """
        print("🧹 Clearing message pool...")
        
        self._message_pool = {
            "agent_inputs": {},
            "agent_outputs": {},
            "conversation_history": [],
            "context_metadata": {
                "session_start": time.time(),
                "message_count": 0,
                "active_agents": set(),
                "workflow_steps": []
            }
        }
        
        print("✅ Message pool cleared")
