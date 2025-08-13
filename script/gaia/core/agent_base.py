"""Enhanced MeshAgent with dynamic configuration and personalization."""
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import os
import abc

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.registry import ToolRegistry
from tools.tool_collection import ToolCollection


class MeshAgent(abc.ABC):
    """
    Enhanced multi-agent network node with configuration-driven dynamic creation and personalization.
    
    Key features:
    1. Configuration-driven initialization: Dynamic agent parameter setup via JSON config
    2. Agent naming: Meaningful agent names for debugging and monitoring
    3. Workspace naming: Uses "id_name" format workspaces for better readability
    4. Specialized processing: Adjusts message processing logic based on specialization
    5. Token limit management: Configurable token usage limits with warnings
    6. Priority support: Agent priority settings for task scheduling
    7. Enhanced logging: Detailed execution logs with agent metadata
    8. Tool configuration: Dynamic tool configuration and parameter passing

    Abstract methods:
    - send_msg: Send message to other agents (protocol-specific)
    - recv_msg: Receive message from other agents (protocol-specific)
    """
    
    def __init__(self, node_id: int, name: str, tool: str, port: int, 
                 config: Dict[str, Any], task_id: Optional[str] = None):
        """
        Initialize enhanced agent.
        
        Args:
            node_id: Unique agent identifier
            name: Human-readable agent name
            tool: Tool name
            port: Listening port
            config: Configuration dictionary with personalization parameters
            task_id: Optional task identifier
        """
        # Basic attributes
        self.id = node_id
        self.name = name
        self.tool_name = tool
        self.port = port
        self.config = config
        self.task_id = task_id or "default"
        
        # Configuration-based personalization
        self.max_tokens = config.get("max_tokens", 500)
        self.priority = config.get("priority", 1)
        self.specialization = config.get("specialization", "general")
        
        # Enhanced workspace setup: use "id_name" format
        self.ws = f"workspaces/{self.task_id}/{self.id}_{self.name}"
        Path(self.ws).mkdir(parents=True, exist_ok=True)
        
        # Initialize tool system
        self.tool_registry = ToolRegistry()
        self.tool_collection = self._setup_tools()
        self.token_used = 0
        
        # Initialize state
        self.running = False
        self.server = None

    # ==================== Abstract Communication Methods ====================
    
    @abc.abstractmethod
    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """
        Send message to another agent.
        
        Args:
            dst: Destination agent ID
            payload: Message content
        """
        pass
    
    @abc.abstractmethod
    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Receive message with optional timeout.
        
        Args:
            timeout: Maximum wait time in seconds (0 = non-blocking)
            
        Returns:
            Received message or None if timeout
        """
        pass
    
    # ==================== Core Agent Methods ====================
    
    def _setup_tools(self) -> Optional[ToolCollection]:
        """Setup tools for this agent based on configuration."""
        try:
            # Create tool instance based on tool name using registry
            tool_instance = self.tool_registry.create_tool(self.tool_name)
            if tool_instance:
                return ToolCollection(tool_instance)
            else:
                self._log(f"Warning: Tool {self.tool_name} not found, agent will have limited functionality")
                return None
        except Exception as e:
            self._log(f"Error setting up tools for agent {self.name}: {e}")
            return None
    
    def _count_token(self, _):
        """Token counter callback."""
        self.token_used += 1

    async def start(self):
        """Start agent and main execution loop."""
        self._log(f"Starting agent {self.name} (ID: {self.id}) on port {self.port}")
        self.running = True
        
        try:
            # Main execution loop
            while self.running:
                # Process incoming messages
                await self.process_messages()
                
                # Monitor token usage and performance
                await self._monitor_agent_health()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.05)
                
        except Exception as e:
            self._log(f"Error in main loop: {e}")
        finally:
            await self.stop()
            
            # Notify completion when shutting down
            completion_msg = {
                "type": "agent_shutdown",
                "agent_id": self.id,
                "agent_name": self.name,
                "final_status": "completed",
                "total_tokens_used": self.token_used
            }
            
            # Send completion notification
            try:
                await self.send_msg(dst=0, payload=completion_msg)  # Broadcast
                self._log(f"Agent {self.name} completed execution successfully")
            except Exception as e:
                self._log(f"Error sending completion notification: {e}")
    
    async def stop(self):
        """Stop agent."""
        self.running = False
        self._log(f"Agent {self.name} stopped")

    async def process_messages(self) -> None:
        """Process incoming messages and update coordination state."""
        msg = await self.recv_msg(timeout=0.0)  # Non-blocking, single message
        if msg:
            await self._handle_message(msg)
    
    async def _handle_message(self, msg: Dict[str, Any]) -> None:
        """Handle different types of messages."""
        msg_type = msg.get("type")
        
        if msg_type == "doc_init":
            await self._handle_doc_init(msg)
        elif msg_type == "task_result":
            await self._handle_task_result(msg)
        elif msg_type in ["search_results", "file_result", "code_result"]:
            await self._handle_intermediate_result(msg)
        elif msg_type == "workflow_task":
            await self._handle_workflow_task(msg)
        else:
            self._log(f"Unknown message type: {msg_type}")
    
    async def _handle_doc_init(self, msg: Dict[str, Any]):
        """Handle initial document broadcast."""
        chunks = msg.get("chunks", [])
        full_doc = "".join(chunks)
        
        self._log(f"Received initial document ({len(full_doc)} chars)")
        
        # Save document to workspace
        doc_path = Path(self.ws) / "initial_document.txt"
        doc_path.write_text(full_doc, encoding='utf-8')
        
        # If this is the first agent in the workflow, start processing
        if self.priority == 1:
            result = await self._execute_tool(full_doc)
            await self._send_result("task_result", {"result": result, "source": "doc_init"})
    
    async def _handle_task_result(self, msg: Dict[str, Any]):
        """Handle task result from previous agent."""
        result = msg.get("result", "")
        source = msg.get("source", "unknown")
        
        self._log(f"Received task result from {source}")
        
        # Execute tool with the received result
        tool_result = await self._execute_tool(result)
        
        # Check if this is the final agent
        if self.specialization == "reasoning_synthesis":
            await self._send_result("data_event", {
                "tag": "final_answer",
                "payload": tool_result,
                "log_uri": f"{self.ws}/execution.log",
                "agent_id": self.id,
                "agent_name": self.name,
                "specialization": self.specialization
            })
            self._log(f"Generated final answer: {tool_result[:100]}...")
        else:
            await self._send_result("task_result", {
                "result": tool_result,
                "source": self.name
            })
    
    async def _handle_intermediate_result(self, msg: Dict[str, Any]):
        """Handle intermediate results from other agents."""
        result = msg.get("result", "")
        self._log(f"Processing intermediate result: {result[:100]}...")
        
        tool_result = await self._execute_tool(result)
        await self._send_result("task_result", {
            "result": tool_result,
            "source": self.name
        })
    
    async def _handle_workflow_task(self, msg: Dict[str, Any]):
        """Handle workflow task messages."""
        task_input = msg.get("task_input", "")
        step = msg.get("step", 0)
        from_agent = msg.get("from", "system")
        
        self._log(f"Received workflow task from {from_agent} (step {step})")
        
        try:
            # Execute the agent's tool with the task input
            result = await self._execute_tool(task_input)
            
            # Send result back
            result_message = {
                'type': 'workflow_result',
                'result': result,
                'step': step,
                'timestamp': int(time.time())
            }
            
            await self.send_msg(0, result_message)
            
        except Exception as e:
            self._log(f"Error executing workflow task: {e}")
            
            # Send error result
            error_message = {
                'type': 'workflow_result',
                'result': f"Error: {e}",
                'step': step,
                'timestamp': int(time.time())
            }
            await self.send_msg(0, error_message)
    
    async def _execute_tool(self, input_data: str) -> str:
        """Execute the agent's tool with input data."""
        if not self.tool_collection:
            return f"No tool available for agent {self.name}"
        
        try:
            # Get the first (and typically only) tool
            tool = list(self.tool_collection.tools)[0]
            
            # Execute tool based on its type and parameters
            if self.tool_name == "web_search":
                result = await tool.execute(query=input_data)
            elif self.tool_name == "file_operators":
                result = await tool.execute(operation="read", content=input_data)
            elif self.tool_name == "python_execute":
                result = await tool.execute(code=input_data)
            elif self.tool_name == "create_chat_completion":
                result = await tool.execute(input=input_data)
            else:
                # Generic execution - try common parameter names
                if hasattr(tool, 'execute'):
                    # Try different parameter names
                    for param_name in ['input', 'query', 'task', 'content']:
                        try:
                            result = await tool.execute(**{param_name: input_data})
                            break
                        except TypeError:
                            continue
                    else:
                        # If no common parameter worked, try without parameters
                        result = await tool.execute()
                else:
                    result = f"Tool {self.tool_name} does not have execute method"
            
            # Extract result output
            if hasattr(result, 'output') and result.output:
                return str(result.output)
            elif hasattr(result, 'error') and result.error:
                return f"Tool error: {result.error}"
            else:
                return str(result)
        
        except Exception as e:
            error_msg = f"Tool execution error: {e}"
            self._log(error_msg)
            return error_msg
    
    async def _send_result(self, msg_type: str, extra: Dict[str, Any]):
        """Send result message using the abstract send_msg method."""
        # Add agent metadata to packet
        message = {
            "type": msg_type,
            "token_used": self.token_used,
            "agent_id": self.id,
            "agent_name": self.name,
            "priority": self.priority,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **extra
        }
        
        # Use the abstract send_msg method
        await self.send_msg(dst=0, payload=message)  # dst=0 for broadcast or coordinator
        
        # Enhanced logging
        self._log_message(message)
    
    def _log(self, message: str):
        """Enhanced logging function with agent metadata."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Agent-{self.id}({self.name}): {message}"
        
        log_path = Path(self.ws) / "agent.log"
        with open(log_path, "a", encoding='utf-8') as f:
            f.write(log_entry + "\n")
        
        # Also print to console for debugging
        print(log_entry)
    
    def _log_message(self, message: Dict[str, Any]):
        """Log message in structured format."""
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agent_id": self.id,
            "agent_name": self.name,
            "specialization": self.specialization,
            "message": message
        }
        
        log_path = Path(self.ws) / "message.log"
        with open(log_path, "a", encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    async def _monitor_agent_health(self) -> None:
        """Monitor agent health and performance metrics."""
        # Check token usage limits
        if self.token_used > self.max_tokens * 0.8:  # 80% warning threshold
            self._log(f"WARNING: High token usage ({self.token_used}/{self.max_tokens})")
        
        # Check if workspace is accessible
        if not Path(self.ws).exists():
            self._log("WARNING: Workspace directory not accessible")
            Path(self.ws).mkdir(parents=True, exist_ok=True)
