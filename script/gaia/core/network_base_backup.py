"""Enhanced MeshNetwork with dynamic agent management and intelligent routing."""
import asyncio
import json
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import sys
import os
import abc

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .agent_base import MeshAgent
from protocols.base_adapter import ProtocolAdapter


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
    
    def __init__(self, adapter: ProtocolAdapter):
        self.adapter = adapter
        self.agents: List[MeshAgent] = []
        self.config: Dict[str, Any] = {}
        
        # Control flags
        self.running = False
        
        # Message processing
        self._message_buffer: List[Tuple[int, Dict[str, Any]]] = []
        
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
    
    # ==================== Abstract Protocol Methods ====================
    
    @abc.abstractmethod
    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """
        Deliver message to specific agent.
        
        Args:
            dst: Destination agent ID
            msg: Message payload
        """
        pass
    
    @abc.abstractmethod
    async def poll(self) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Poll for incoming messages from agents.
        
        Returns:
            List of (sender_id, message) tuples
        """
        pass

    # ==================== Startup and Shutdown ====================
    async def start(self):
        """Start the network and message processing."""
        print("� Starting multi-agent network...")
        
        # Start all agents
        for agent in self.agents:
            try:
                await agent.start()
                print(f"✅ Started agent {agent.id} ({agent.name}) on port {agent.port}")
            except Exception as e:
                print(f"❌ Failed to start agent {agent.id}: {e}")
        
        # Start message processing loop
        asyncio.create_task(self._message_processing_loop())
        
        self.running = True
        print("🚀 Network started successfully")
    
    async def stop(self):
        """Stop the network and all agents."""
        print("🛑 Stopping network...")
        self.running = False
        
        # Stop all agents
        for agent in self.agents:
            try:
                await agent.stop()
                print(f"✅ Stopped agent {agent.id}")
            except Exception as e:
                print(f"❌ Error stopping agent {agent.id}: {e}")
        
        print("✅ Network stopped")

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
    
    async def _relay(self, reader: asyncio.StreamReader, src_port: int):
        """Relay messages according to configuration routing rules."""
        try:
            while self.running:
                # Read packet size
                size_data = await reader.readexactly(4)
                size = int.from_bytes(size_data, "big")
                
                # Read packet data
                data = await reader.readexactly(size)
                
                # Update metrics
                self.bytes_rx += 4 + size
                packet = self.adapter.decode(data)
                self.pkt_cnt += 1
                self.header_overhead += self.adapter.header_size(packet)
                self.token_sum += packet.get("token_used", 0)
                
                # Check for final answer
                if packet.get("tag") == "final_answer" and not self.done_ts:
                    self.done_ts = time.time() * 1000
                    self.done_payload = packet.get("payload", "")
                    print(f"🎯 Final answer received: {self.done_payload[:100]}...")
                
                # Route message according to configuration
                await self._route_message(packet, src_port)
                
        except asyncio.IncompleteReadError:
            # Connection closed
            pass
        except Exception as e:
            print(f"Error in relay for port {src_port}: {e}")
    
    async def _route_message(self, packet: Dict[str, Any], src_port: int):
        """Route message according to configuration rules."""
        message_type = packet.get("type", "unknown")
        
        # Check for broadcast types
        broadcast_types = self.config.get("communication_rules", {}).get("broadcast_types", [])
        if message_type in broadcast_types:
            # Broadcast to all agents except sender using deliver
            for agent in self.agents:
                if agent.port != src_port:  # Don't send back to sender
                    try:
                        await self.deliver(agent.id, packet)
                    except Exception as e:
                        print(f"Error delivering broadcast message to agent {agent.id}: {e}")
        else:
            # Direct routing based on configuration using deliver
            await self._direct_route(packet, src_port)
    
    async def _direct_route(self, packet: Dict[str, Any], src_port: int):
        """Handle direct routing based on workflow configuration using deliver."""
        # Find source agent
        src_agent_id = None
        for agent in self.agents:
            if agent.port == src_port:
                src_agent_id = agent.id
                break
        
        if src_agent_id is None:
            return
        
        # Find target agents based on workflow
        workflow = self.config.get("workflow", {})
        message_flow = workflow.get("message_flow", [])
        
        target_agents = []
        for flow in message_flow:
            if flow["from"] == src_agent_id:
                if flow["to"] == "final":
                    # This is a final answer, no routing needed
                    return
                target_agents.extend(flow["to"])
        
        # Route to target agents using deliver
        for target_id in target_agents:
            try:
                await self.deliver(target_id, packet)
            except Exception as e:
                print(f"Error delivering message to agent {target_id}: {e}")
    
    def _get_writer_by_agent_id(self, agent_id: int) -> Optional[asyncio.StreamWriter]:
        """Get writer for specific agent ID."""
        for agent in self.agents:
            if agent.id == agent_id:
                # Find corresponding writer
                for i, a in enumerate(self.agents):
                    if a.id == agent_id and i < len(self.connections):
                        return self.connections[i]
        return None
    
    async def _monitor_done(self):
        """Monitor for completion and trigger evaluation."""
        print("👀 Monitoring for final answer...")
        
        # Wait for final answer or timeout
        timeout = self.config.get("performance_targets", {}).get("max_execution_time", 300000)
        timeout_seconds = timeout / 1000
        
        start_time = time.time()
        while self.done_ts is None and self.running:
            await asyncio.sleep(1)
            
            # Check for timeout
            if time.time() - start_time > timeout_seconds:
                print("⏰ Execution timeout reached")
                self.done_ts = time.time() * 1000
                self.done_payload = "TIMEOUT: No final answer received within time limit"
                break
        
        if self.done_payload:
            print("🔄 Starting evaluation...")
            await self._evaluate()
    
    async def _evaluate(self):
        """Run final evaluation and archive artifacts."""
        print("📊 Running evaluation...")
        
        # Run quality evaluation
        try:
            quality = await eval_runner(self.done_payload or "", "ground_truth.json")
        except Exception as e:
            print(f"Evaluation error: {e}")
            quality = {"quality_score": 0.0, "exact_match": 0, "error": str(e)}
        
        # Compile metrics report
        report = {
            "performance_metrics": {
                "bytes_tx": self.bytes_tx,
                "bytes_rx": self.bytes_rx,
                "pkt_cnt": self.pkt_cnt,
                "header_overhead": self.header_overhead,
                "token_sum": self.token_sum,
                "elapsed_ms": self.done_ts - self.start_ts if self.done_ts else 0
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

    # ==================== Workflow Execution ====================
    async def execute_workflow(self, config: Dict[str, Any], initial_task: str = None) -> str:
        """
        Execute workflow based on configuration using deliver/poll for agent communication.
        
        Args:
            config: Configuration containing workflow, agents, and agent_prompts
            initial_task: Initial task to start the workflow with
            
        Returns:
            Final result from the workflow execution
        """
        try:
            workflow = config.get('workflow', {})
            agent_prompts = config.get('agent_prompts', {})
            
            start_agent_id = workflow.get('start_agent', 0)
            message_flow = workflow.get('message_flow', [])
            execution_pattern = workflow.get('execution_pattern', 'sequential')
            
            print(f"🚀 Starting workflow execution with pattern: {execution_pattern}")
            print(f"📋 Starting agent: {start_agent_id}")
            
            # Initialize workflow state
            workflow_results = {}
            final_result = None
            
            # Send initial task to start agent using deliver
            initial_message = {
                'type': 'workflow_task',
                'task_input': initial_task or "Begin task execution",
                'from': 'system',
                'step': 0,
                'timestamp': int(time.time())
            }
            
            print(f"📤 Sending initial task to agent {start_agent_id}")
            await self.deliver(start_agent_id, initial_message)
            
            # Process workflow according to message flow
            for step_idx, flow_step in enumerate(message_flow):
                from_agent_id = flow_step.get('from')
                to_agents = flow_step.get('to')
                message_type = flow_step.get('message_type', 'task_result')
                
                print(f"� Processing step {step_idx + 1}: Agent {from_agent_id} -> {to_agents} ({message_type})")
                
                # Wait for result from current agent
                timeout = 30.0  # 30 seconds timeout
                start_time = time.time()
                agent_result = None
                
                while time.time() - start_time < timeout:
                    # Poll for messages
                    messages = await self.poll()
                    
                    # Look for result from the expected agent
                    for sender_id, msg in messages:
                        if sender_id == from_agent_id and msg.get('type') == 'workflow_result':
                            agent_result = msg.get('result', '')
                            workflow_results[f'step_{step_idx}'] = agent_result
                            print(f"✅ Received result from agent {from_agent_id}: {agent_result[:100]}...")
                            break
                    
                    if agent_result:
                        break
                    
                    # Small delay to avoid busy waiting
                    await asyncio.sleep(0.1)
                
                if not agent_result:
                    print(f"⏰ Timeout waiting for result from agent {from_agent_id}")
                    continue
                
                # Check if this is the final step
                if to_agents == 'final' or to_agents == ['final']:
                    final_result = agent_result
                    print(f"🎯 Final result obtained: {final_result[:100]}...")
                    break
                
                # Send result to next agents
                if isinstance(to_agents, list):
                    for target_id in to_agents:
                        if target_id != 'final':
                            next_message = {
                                'type': 'workflow_task',
                                'task_input': agent_result,
                                'from': from_agent_id,
                                'step': step_idx + 1,
                                'timestamp': int(time.time())
                            }
                            
                            print(f"📤 Sending result to agent {target_id}")
                            await self.deliver(target_id, next_message)
                
                # Add delay between steps
                await asyncio.sleep(0.5)
            
            # Return final result
            if final_result:
                print(f"🎉 Workflow execution completed successfully")
                return final_result
            else:
                # Return the last available result
                last_result = list(workflow_results.values())[-1] if workflow_results else "No results generated"
                print(f"⚠️ Workflow completed but no final result marked, returning last result")
                return last_result
                
        except Exception as e:
            print(f"❌ Error executing workflow: {e}")
            import traceback
            traceback.print_exc()
            return f"Workflow execution failed: {e}"

    def get_agent_by_id(self, agent_id: int):
        """Get agent by ID from the agents list."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None
