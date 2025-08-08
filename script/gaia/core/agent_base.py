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

from protocols.base_adapter import ProtocolAdapter
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
    - send_message: Send message to other agents (protocol-specific)
    - receive_message: Receive message from other agents (protocol-specific)
    """
    
    def __init__(self, node_id: int, name: str, tool: str, adapter: ProtocolAdapter, 
                 port: int, config: Dict[str, Any], task_id: Optional[List[str]] = None):
        """
        Initialize enhanced agent.
        
        Args:
            node_id: Unique agent identifier
            name: Human-readable agent name
            tool: Tool name
            adapter: Protocol adapter
            port: Listening port
            config: Configuration dictionary with personalization parameters
        """
        # Basic attributes
        self.id = node_id
        self.name = name
        self.tool_name = tool
        self.adapter = adapter
        self.port = port
        self.config = config
        self.task_id = task_id
        
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
    
    # ==================== Enhanced Agent Methods ====================
    def _setup_tools(self) -> Optional[ToolCollection]:
        """Setup tools for this agent based on configuration."""
        try:
            # Create tool instance based on tool name using adapter
            tool_instance = self.tool_registry.create_tool(self.tool_name)
            if tool_instance:
                return ToolCollection(tool_instance)
            else:
                print(f"Warning: Tool {self.tool_name} not found, agent will have limited functionality")
                return None
        except Exception as e:
            print(f"Error setting up tools for agent {self.name}: {e}")
            return None
    
    def _count_token(self, _):
        """Token counter callback."""
        self.token_used += 1

    async def start(self):
        """Start agent server and main execution loop."""
        print(f"🤖 Starting agent {self.name} (ID: {self.id}) on port {self.port}")
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", self.port)
        self.running = True
        
        self._log(f"Starting main execution loop for agent {self.name}")
        
        # Start both server and main execution loop concurrently
        async with self.server:
            # Create server task
            server_task = asyncio.create_task(self.server.serve_forever())
            
            try:
                # Main execution loop runs alongside the server
                while self.running:
                    # Process incoming messages
                    await self.process_messages()
                    
                    # Monitor token usage and performance
                    await self._monitor_agent_health()
                    
                    # Small delay to prevent busy waiting
                    await asyncio.sleep(0.05)
                    print('running')  # Debugging output
                    
            except Exception as e:
                self._log(f"Error in main loop: {e}")
            finally:
                # Cancel server task and ensure clean shutdown
                server_task.cancel()
                await self.stop()
                
                # Notify completion when shutting down
                completion_msg = {
                    "type": "agent_shutdown",
                    "agent_id": self.id,
                    "agent_name": self.name,
                    "final_status": "completed",
                    "total_tokens_used": self.token_used
                }
                
                # Broadcast completion to other agents
                try:
                    await self.send_msg(dst=0, payload=completion_msg)  # Broadcast
                    self._log(f"Agent {self.name} completed execution successfully")
                except Exception as e:
                    self._log(f"Error sending completion notification: {e}")
    
    async def stop(self):
        """Stop agent server."""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def process_messages(self) -> None:
        """Process incoming messages and update coordination state."""
        msg = await self.recv_msg(timeout=0.0)  # Non-blocking, single message
        if msg:
            await self._handle_message(msg)
            self.running = False # Stop after processing one message
    
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle network connection."""
        try:
            while self.running:
                # Read packet size
                size_data = await reader.readexactly(4)
                size = int.from_bytes(size_data, "big")
                
                # Read packet data
                packet_data = await reader.readexactly(size)
                packet = self.adapter.decode(packet_data)
                
                # Process packet asynchronously
                asyncio.create_task(self._handle_message(packet))
                
        except asyncio.IncompleteReadError:
            # Connection closed
            pass
        except Exception as e:
            self._log(f"Error handling connection: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def _handle_message(self, msg: Dict[str, Any]) -> None:
        """Handle different types of coordination messages."""
        msg_type = msg.get("type")
        
        if msg_type == "doc_init":
            await self._handle_doc_init(msg)
        elif msg_type == "task_result":
            await self._handle_task_result(msg)
        elif msg_type in ["search_results", "file_result", "code_result"]:
            await self._handle_intermediate_result(msg)
        else:
            self._log(f"Unknown message type: {msg_type}")
    
    async def _handle_doc_init(self, packet: Dict[str, Any]):
        """Handle initial document broadcast."""
        chunks = packet.get("chunks", [])
        full_doc = "".join(chunks)
        
        self._log(f"Received initial document ({len(full_doc)} chars)")
        
        # Save document to workspace
        doc_path = Path(self.ws) / "initial_document.txt"
        doc_path.write_text(full_doc, encoding='utf-8')
        
        # If this is the first agent in the workflow, start processing
        if self.priority == 1:
            result = await self._execute_tool(full_doc)
            await self._send_result("task_result", {"result": result, "source": "doc_init"})
    
    async def _handle_task_result(self, packet: Dict[str, Any]):
        """Handle task result from previous agent."""
        result = packet.get("result", "")
        source = packet.get("source", "unknown")
        
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
    
    async def _handle_intermediate_result(self, packet: Dict[str, Any]):
        """Handle intermediate results from other agents."""
        result = packet.get("result", "")
        self._log(f"Processing intermediate result: {result[:100]}...")
        
        tool_result = await self._execute_tool(result)
        await self._send_result("task_result", {
            "result": tool_result,
            "source": self.name
        })
    
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
    
    async def _send_result(self, pkt_type: str, extra: Dict[str, Any]):
        """Send result message using the abstract send_msg method."""
        # Add agent metadata to packet
        packet = {
            "type": pkt_type,
            "token_used": self.token_used,
            "agent_id": self.id,
            "agent_name": self.name,
            "priority": self.priority,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **extra
        }
        
        # Use the abstract send_msg method
        # Note: This assumes the destination is handled by the protocol adapter
        # You may need to adjust this based on your specific routing logic
        await self.send_msg(dst=0, payload=packet)  # dst=0 for broadcast or coordinator
        
        # Enhanced logging
        self._log_packet(packet)
    
    async def _emit(self, writer: asyncio.StreamWriter, pkt_type: str, extra: Dict[str, Any]):
        """
        Send message packet with enhanced agent metadata.
        
        Features:
        1. Add agent metadata to packet
        2. Enhanced logging format
        """
        # Add agent metadata to packet
        packet = {
            "type": pkt_type,
            "token_used": self.token_used,
            "agent_id": self.id,
            "agent_name": self.name,
            "priority": self.priority,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **extra
        }
        
        blob = self.adapter.encode(packet)
        writer.write(len(blob).to_bytes(4, "big") + blob)
        await writer.drain()
        
        # Enhanced logging
        self._log_packet(packet)
    
    async def _emit_warning(self, writer: asyncio.StreamWriter, warning_type: str):
        """Send warning message."""
        await self._emit(writer, "warning", {
            "warning_type": warning_type,
            "message": f"Agent {self.name} encountered {warning_type}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    async def _emit_error(self, writer: asyncio.StreamWriter, error_message: str):
        """Send error message."""
        await self._emit(writer, "error", {
            "error": error_message,
            "agent_id": self.id,
            "agent_name": self.name
        })
    
    def _log(self, message: str):
        """Enhanced logging function with agent metadata."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Agent-{self.id}({self.name}): {message}"
        
        log_path = Path(self.ws) / "agent.log"
        with open(log_path, "a", encoding='utf-8') as f:
            f.write(log_entry + "\n")
        
        # Also print to console for debugging
        print(log_entry)
    
    def _log_packet(self, packet: Dict[str, Any]):
        """Log packet in structured format."""
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agent_id": self.id,
            "agent_name": self.name,
            "specialization": self.specialization,
            "packet": packet
        }
        
        log_path = Path(self.ws) / "packet.log"
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
