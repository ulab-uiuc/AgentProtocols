import asyncio
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .llm import LLM
from tools.utils.config import config  
from tools.utils.logger import logger
from tools.tool_collection import ToolCollection
from tools.registry import ToolRegistry
from tools.exceptions import TokenLimitExceeded
from core.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice, Memory

# GAIA 根目录 (script/gaia)
GAIA_ROOT = Path(__file__).resolve().parent.parent


TOOL_CALL_REQUIRED = "Tool calls required but none provided"
SYSTEM_PROMPT = "You are an agent that can execute tool calls"

NEXT_STEP_PROMPT = (
    "Continue with your assigned task using the available tools."
)

class ReActAgent(BaseModel, ABC):
    """Simplified ReAct Agent base class without memory management"""
    name: str
    description: Optional[str] = None
    
    system_prompt: Optional[str] = None
    next_step_prompt: Optional[str] = None
    
    state: AgentState = AgentState.IDLE
    max_steps: int = 10
    current_step: int = 0

    @abstractmethod
    async def think(self) -> bool:
        """Process current state and decide next action"""

    @abstractmethod
    async def act(self) -> str:
        """Execute decided actions"""

    async def step(self) -> str:
        """Execute a single step: think and act."""
        should_act = await self.think()
        if not should_act:
            return "Thinking complete - no action needed"
        return await self.act()
    
class ToolCallAgent(ReActAgent):
    """Tool calling agent without memory management"""
    
    model_config = {"arbitrary_types_allowed": True}

    name: str = "toolcall"
    description: str = "an agent that can execute tool calls."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT
    max_steps: int = 30
    max_observe: int = 10000
    
    # LLM and tool management
    llm: Optional[LLM] = Field(default_factory=LLM)
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolRegistry().available_tools
    )

    special_tool_names: list[str] = Field(default_factory=lambda: ["CreateChatCompletion"])
    
    # Track connected MCP servers
    connected_servers: Dict[str, str] = Field(default_factory=dict)
    _initialized: bool = False
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore

    tool_calls: List[ToolCall] = Field(default_factory=list)
    _current_base64_image: Optional[str] = None
    
    # Message handling without memory
    messages: List[Message] = Field(default_factory=list)

    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages.append(user_msg)

        try:
            # Get response with tool options
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            # Check if this is a RetryError containing TokenLimitExceeded
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                self.messages.append(
                    Message.assistant_message(
                        f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}"
                    )
                )
                self.state = AgentState.FINISHED
                return False
            raise

        self.tool_calls = tool_calls = (
            response.get("tool_calls", []) if response else []
        )
        content = response.get("content", "") if response else ""

        # Log response info with blue color
        logger.info(f"\033[94m✨ {self.name}'s thoughts: {content}\033[0m")
        logger.info(
            f"\033[94m🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use\033[0m"
        )
        if tool_calls:
            # Handle both dict and object formats for tool calls
            tool_names = []
            for call in tool_calls:
                if hasattr(call, 'function'):
                    tool_names.append(call.function.name)
                elif isinstance(call, dict) and 'function' in call:
                    tool_names.append(call['function']['name'])
                else:
                    tool_names.append("unknown")
            
            logger.info(f"\033[94m🧰 Tools being prepared: {tool_names}\033[0m")
            
            # Log first tool arguments
            if tool_calls and len(tool_calls) > 0:
                first_call = tool_calls[0]
                if hasattr(first_call, 'function'):
                    logger.info(f"\033[94m🔧 Tool arguments: {first_call.function.arguments}\033[0m")
                elif isinstance(first_call, dict) and 'function' in first_call:
                    logger.info(f"\033[94m🔧 Tool arguments: {first_call['function'].get('arguments', '')}\033[0m")
                else:
                    logger.info(f"\033[94m🔧 Tool arguments: {str(first_call)}\033[0m")

        try:
            if response is None:
                raise RuntimeError("No response received from the LLM")

            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.messages.append(Message.assistant_message(content))
                    return True
                return False

            # Create and add assistant message
            if self.tool_calls:
                # Convert dict tool_calls to ToolCall objects if needed
                converted_tool_calls = []
                for call in self.tool_calls:
                    if isinstance(call, dict):
                        # Convert dict to ToolCall format for Message.from_tool_calls
                        from core.schema import ToolCall, Function
                        function_data = call.get('function', {})
                        tool_call = ToolCall(
                            id=call.get('id', 'unknown'),
                            function=Function(
                                name=function_data.get('name', 'unknown'),
                                arguments=function_data.get('arguments', '{}')
                            ),
                            type=call.get('type', 'function')
                        )
                        converted_tool_calls.append(tool_call)
                    else:
                        # Already a ToolCall object
                        converted_tool_calls.append(call)
                
                assistant_msg = Message.from_tool_calls(content=content, tool_calls=converted_tool_calls)
            else:
                assistant_msg = Message.assistant_message(content)
            self.messages.append(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # Will be handled in act()

            # For 'auto' mode, continue with content if no commands but content exists
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.messages.append(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        """Execute tool calls and handle their results"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # Return last message content if no tool calls
            return self.messages[-1].content or "No content or commands to execute"

        results = []
        for command in self.tool_calls:
            # Reset base64_image for each tool call
            self._current_base64_image = None

            result = await self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]

            # Get tool name and ID in compatible way
            if hasattr(command, 'function'):
                tool_name = command.function.name
                tool_id = command.id
            elif isinstance(command, dict) and 'function' in command:
                tool_name = command['function'].get('name', 'unknown')
                tool_id = command.get('id', 'unknown')
            else:
                tool_name = 'unknown'
                tool_id = 'unknown'

            logger.info(
                f"🎯 Tool '{tool_name}' completed its mission! Result: {result}"
            )

            # Add tool response to messages
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=tool_id,
                name=tool_name,
                base64_image=self._current_base64_image,
            )
            self.messages.append(tool_msg)
            results.append(result)

        return "\n\n".join(results)

    async def execute_tool(self, command) -> str:
        """Execute a single tool call with robust error handling and detailed logging"""
        import time
        start_time = time.time()
        # Ensure per-agent accumulators exist
        if not hasattr(self, '_toolcall_total'):
            self._toolcall_total = 0.0
        if not hasattr(self, '_toolcall_count'):
            self._toolcall_count = 0
        
        # Handle both ToolCall object and dict formats
        if hasattr(command, 'function'):
            # ToolCall object format
            if not command or not command.function or not command.function.name:
                return "Error: Invalid command format"
            name = command.function.name
            arguments = command.function.arguments or "{}"
            tool_call_id = command.id
        elif isinstance(command, dict) and 'function' in command:
            # Dict format from LLM response
            function_data = command.get('function', {})
            name = function_data.get('name')
            if not name:
                return "Error: Invalid command format - missing function name"
            arguments = function_data.get('arguments', '{}')
            tool_call_id = command.get('id', 'unknown')
        else:
            return "Error: Invalid command format"

        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        try:
            # Parse arguments
            args = json.loads(arguments)
            
            # Enhanced logging: Tool call initiation
            print(f"🔧 TOOL CALL START - Agent: {self.name}, Tool: {name}, ID: {tool_call_id}")
            logger.info(f"🔧 TOOL CALL START - Agent: {self.name}, Tool: {name}, ID: {tool_call_id}")
            print(f"📋 Tool Arguments: {json.dumps(args, indent=2)}")
            logger.info(f"📋 Tool Arguments: {json.dumps(args, indent=2)}")

            # Inject per-agent environment for tools (workspace/task/protocol)
            try:
                os.environ["GAIA_AGENT_WORKSPACE_DIR"] = self.ws
                os.environ["GAIA_TASK_ID"] = self.task_id
                protocol_name = (self.config.get("protocol") if isinstance(self.config, dict) else None) or "default"
                os.environ["GAIA_PROTOCOL_NAME"] = protocol_name
                logger.info(f"🌍 Environment: workspace={self.ws}, task_id={self.task_id}, protocol={protocol_name}")
            except Exception as env_error:
                logger.warning(f"⚠️  Environment setup warning: {env_error}")

            # Execute the tool
            logger.info(f"⚡ Executing tool: '{name}'...")
            # Measure toolcall duration separately
            tool_start = time.time()
            result = await self.available_tools.execute(name=name, tool_input=args)
            tool_duration = time.time() - tool_start

            # Accumulate toolcall metrics
            try:
                self._toolcall_total += float(tool_duration)
                self._toolcall_count += 1
            except Exception:
                # Defensive: if attributes are not numeric for any reason
                pass

            # Enhanced logging: Tool execution result
            execution_time = time.time() - start_time
            print(f"✅ TOOL CALL SUCCESS - Tool: {name}, Duration: {execution_time:.2f}s (toolcall: {tool_duration:.2f}s)")
            logger.info(f"✅ TOOL CALL SUCCESS - Tool: {name}, Duration: {execution_time:.2f}s (toolcall: {tool_duration:.2f}s)")
            
            # Log result summary (truncated for readability)
            result_str = str(result)
            result_preview = result_str[:2000] + "..." if len(result_str) > 2000 else result_str
            print(f"📤 Tool Result Preview: {result_preview}")
            logger.info(f"📤 Tool Result Preview: {result_preview}")
            
            # Log full result for detailed analysis
            logger.debug(f"📋 Full Tool Result: {result_str}")

            # Handle special tools
            await self._handle_special_tool(name=name, result=result)

            # Check if result is a ToolResult with base64_image
            if hasattr(result, "base64_image") and result.base64_image:
                # Store the base64_image for later use in tool_message
                self._current_base64_image = result.base64_image
                logger.info(f"🖼️  Base64 image captured from tool result")

            # Format result for display (standard case)
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            return observation
        except json.JSONDecodeError as json_error:
            tool_duration = time.time() - start_time
            try:
                self._toolcall_total += float(tool_duration)
                self._toolcall_count += 1
            except Exception:
                pass
            execution_time = time.time() - start_time
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(f"❌ TOOL CALL FAILED - Tool: {name}, Duration: {execution_time:.2f}s")
            logger.error(f"� JSON Parse Error: {json_error}")
            logger.error(f"📝 Invalid arguments: {arguments}")
            return f"Error: {error_msg}"
        except Exception as e:
            tool_duration = time.time() - start_time
            try:
                self._toolcall_total += float(tool_duration)
                self._toolcall_count += 1
            except Exception:
                pass
            execution_time = time.time() - start_time
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.error(f"❌ TOOL CALL FAILED - Tool: {name}, Duration: {execution_time:.2f}s")
            logger.error(f"💥 Exception Details: {str(e)}")
            logger.exception(f"🔍 Full Exception Traceback for tool '{name}':")
            return f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return False  # Don't auto-finish for normal tools

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        """Clean up resources used by the agent's tools."""
        logger.info(f"🧹 Cleaning up resources for agent '{self.name}'...")
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 Cleaning up tool: {tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(
                        f"🚨 Error cleaning up tool '{tool_name}': {e}", exc_info=True
                    )
        logger.info(f"✨ Cleanup complete for agent '{self.name}'.")

    async def run(self, request: Optional[str] = None) -> str:
        """Run the agent with cleanup when done."""
        try:
            if request:
                # Add user request to messages
                self.messages.append(Message.user_message(request))
            
            # Main execution loop
            self.current_step = 0
            while self.current_step < self.max_steps and self.state != AgentState.FINISHED:
                result = await self.step()
                self.current_step += 1
                
                # If agent finished, break
                if self.state == AgentState.FINISHED:
                    break
            
            # Return final result or status
            return f"Agent completed after {self.current_step} steps"
        finally:
            await self.cleanup()
    
class MeshAgent(ToolCallAgent, ABC):
    """Abstract Mesh Agent that can deliver and receive messages"""
    model_config = {"arbitrary_types_allowed": True}
    
    # Define Pydantic fields for MeshAgent
    id: int = Field(description="Unique agent identifier")
    tool_name: str = Field(description="Tool name")
    port: int = Field(description="Listening port")
    config: Dict[str, Any] = Field(description="Configuration dictionary")
    task_id: str = Field(default="default", description="Task identifier")
    max_tokens: int = Field(default=500, description="Maximum tokens")
    priority: int = Field(default=1, description="Agent priority")
    ws: str = Field(description="Workspace path")
    tool_registry: Any = Field(description="Tool registry instance")
    tool_collection: Any = Field(description="Tool collection instance")
    token_used: int = Field(default=0, description="Tokens used")
    running: bool = Field(default=False, description="Running state")
    server: Optional[Any] = Field(default=None, description="Server instance")
    memory: Memory = Field(default_factory=Memory, description="Message memory")
    
    # 回调函数用于返回处理结果，不中断循环
    result_callback: Optional[callable] = Field(default=None, description="Callback for returning results")
    
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
        # Setup workspace path directly under task directory (no agent-specific subdirs)
        task_id = task_id or "default"
        protocol_name = (config.get("protocol") if isinstance(config, dict) else None) or "default"
        ws = GAIA_ROOT / "workspaces" / protocol_name / task_id
        
        # Initialize tool system
        tool_registry = ToolRegistry()
        tool_collection = tool_registry.available_tools
        
        # Initialize parent ToolCallAgent with all required fields
        super().__init__(
            name=name,
            description=f"Mesh agent {name} with {tool} capabilities",
            max_steps=config.get("max_steps", 30),
            system_prompt=config.get("system_prompt", SYSTEM_PROMPT),
            available_tools=tool_collection,  # Pass tool collection to parent
            # MeshAgent specific fields
            id=node_id,
            tool_name=tool,
            port=port,
            config=config,
            task_id=task_id,
            max_tokens=config.get("max_tokens", 500),
            priority=config.get("priority", 1),
            ws=str(ws),
            tool_registry=tool_registry,
            tool_collection=tool_collection,
            token_used=0,
            running=False,
            server=None
        )
        
        # Create workspace directory (shared by all agents in the task)
        Path(self.ws).mkdir(parents=True, exist_ok=True)
    
    # ==================== Abstract Communication Methods ====================
    
    @abstractmethod
    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """
        Send message to another agent.
        
        Args:
            dst: Destination agent ID
            payload: Message content
        """
        pass
    
    @abstractmethod
    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Receive message with optional timeout.
        
        Args:
            timeout: Maximum wait time in seconds (0 = non-blocking)
            
        Returns:
            Received message or None if timeout
        """
        pass

    @abstractmethod
    async def health_check(self):
        """Monitor agent health and resource usage (abstract).
        
        协议实现应在此处进行自身需要的健康检查，例如：
        - token 使用量/速率限制
        - 连接/服务端口/线程存活检查
        - 关键依赖可用性（如 SDK、API Key）
        """
        pass
    
    # ==================== Core Agent Methods ====================
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
                await self.health_check()
                
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
                # await self.send_msg(dst=0, payload=completion_msg)  # Broadcast
                self._log(f"Agent {self.name} completed execution successfully, completion messages are as follows:\n{completion_msg}")
            except Exception as e:
                self._log(f"Error sending completion notification: {e}")

    async def stop(self):
        """Stop agent."""
        self.running = False
        self._log(f"Agent {self.name} stopped")

    async def process_messages(self):
        """Process incoming messages and generate responses using ToolCallAgent logic."""
        try:
            # Receive message (non-blocking)
            msg = await self.recv_msg(timeout=0.0)
            if not msg:
                return  # No message to process
            
            self._log(f"Received message: {msg}")
            
            # Extract message content and sender info
            sender_id = msg.get("sender_id")
            message_type = msg.get("type", "user_message")
            content = msg.get("content", "")
            
            if not content or sender_id is None:
                self._log("Invalid message format - missing content or sender_id")
                return
            
            # Clear previous messages to start fresh for this task
            # self.messages.clear()
            
            # Add user message to both internal messages and memory
            user_msg = Message.user_message(content)
            self.messages.append(user_msg)
            self.memory.add_message(user_msg)
            
            self._log(f"Processing user request: {content}")
            
            # Use parent class ToolCallAgent logic to process the message
            try:
                # Reset for new task
                self.current_step = 0
                self.state = AgentState.IDLE
                
                # Execute thinking and acting steps until completion or max steps
                while self.current_step < self.max_steps and self.state != AgentState.FINISHED:
                    step_result = await self.step()  # Use parent's step method
                    self.current_step += 1
                    
                    # Check if we have tool calls or meaningful content
                    if self.tool_calls or (self.messages and self.messages[-1].content):
                        break  # We got some result, can proceed
                    
                    # If agent finished, break
                    if self.state == AgentState.FINISHED:
                        break
                
                # Extract the final result from agent's internal messages
                final_result = self._extract_final_result()
                
                # Create assistant response message and add to memory
                assistant_msg = Message.assistant_message(content=final_result)
                self.memory.add_message(assistant_msg)
                
                # 通过回调函数返回完整消息，不中断循环
                if self.result_callback:
                    complete_message = {
                        "agent_id": self.id,
                        "agent_name": self.name,
                        "sender_id": sender_id,
                        "message_type": message_type,
                        "original_content": content,
                        "assistant_response": final_result,
                        "assistant_msg": assistant_msg,
                        "processing_steps": self.current_step,
                        "status": "completed"
                    }
                    try:
                        # 异步调用回调函数，不等待返回值
                        if asyncio.iscoroutinefunction(self.result_callback):
                            asyncio.create_task(self.result_callback(complete_message))
                        else:
                            self.result_callback(complete_message)
                    except Exception as e:
                        self._log(f"Error in result callback: {e}")
                
                self._log(f"Task completed. Result stored in memory: {final_result[:200]}...")
                logger.info(f"\033[91m Agent {self.name} memory updated with result\033[0m")
                
            except Exception as e:
                # Create error message and add to memory
                error_content = f"Error processing message: {str(e)}"
                error_msg = Message.assistant_message(content=error_content)
                self.memory.add_message(error_msg)
                self._log(f"Added error to memory: {e}")
                
        except Exception as e:
            self._log(f"Error in process_messages: {e}")
    
    def _extract_final_result(self) -> str:
        """Extract the final result from agent's processing."""
        # Look for the last assistant message with actual content
        for msg in reversed(self.messages):
            if msg.role == "assistant" and msg.content and msg.content.strip():
                return msg.content.strip()
        
        # If no assistant message found, check tool results
        for msg in reversed(self.messages):
            if msg.role == "tool" and msg.content and msg.content.strip():
                return f"Tool result: {msg.content.strip()}"
        
        return "No result generated by agent"
    
    def set_result_callback(self, callback: callable):
        """设置结果回调函数，用于在不中断循环的情况下返回处理结果"""
        self.result_callback = callback
        self._log(f"Result callback function set for agent {self.name}")
    
    def get_memory_messages(self) -> List[Message]:
        """获取内存中的所有消息"""
        return self.memory.messages
    
    def get_recent_assistant_responses(self, n: int = 5) -> List[str]:
        """获取最近的n个assistant回复"""
        assistant_messages = [
            msg.content for msg in self.memory.messages 
            if msg.role == "assistant" and msg.content
        ]
        return assistant_messages[-n:] if assistant_messages else []
    
    def get_last_processing_result(self) -> Optional[Dict[str, Any]]:
        """获取最后一次处理的完整结果（基于Memory）"""
        recent_messages = self.memory.get_recent_messages(10)
        
        # 查找最近的user和assistant消息对
        user_msg = None
        assistant_msg = None
        
        for msg in reversed(recent_messages):
            if msg.role == "assistant" and assistant_msg is None:
                assistant_msg = msg
            elif msg.role == "user" and user_msg is None and assistant_msg is not None:
                user_msg = msg
                break
        
        if user_msg and assistant_msg:
            return {
                "agent_id": self.id,
                "agent_name": self.name,
                "user_message": user_msg.content,
                "assistant_response": assistant_msg.content,
                "processing_steps": self.current_step,
                "status": "completed"
            }
        
        return None
    
    def get_latest_result(self) -> Optional[str]:
        """Get the latest processing result from memory."""
        # Get the most recent assistant message from memory
        for msg in reversed(self.memory.messages):
            if msg.role == "assistant" and msg.content and msg.content.strip():
                return msg.content.strip()
        return None
    
    def is_processing_complete(self) -> bool:
        """Check if the agent has completed processing the current task."""
        # Agent is considered complete if it has an assistant message in memory
        # or if it's in FINISHED state
        return (self.state == AgentState.FINISHED or 
                any(msg.role == "assistant" for msg in self.memory.messages[-3:]))

    def _setup_tools(self) -> ToolCollection:
        """Setup tools for this agent."""
        # Use registry's default tools
        return self.tool_registry.available_tools

    def _log(self, message: str):
        """Log message with agent context."""
        logger.info(f"[Agent-{self.id}:{self.name}] {message}")
    
    def get_memory_messages(self) -> List[Message]:
        """Get all messages from memory."""
        return self.memory.messages
    
    def get_recent_messages(self, n: int = 10) -> List[Message]:
        """Get n most recent messages from memory."""
        return self.memory.get_recent_messages(n)
    
    def clear_memory(self):
        """Clear all messages from memory."""
        self.memory.clear()
        self._log("Memory cleared")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the memory state."""
        messages = self.memory.messages
        message_types = {}
        for msg in messages:
            msg_type = msg.role
            message_types[msg_type] = message_types.get(msg_type, 0) + 1
        
        return {
            "total_messages": len(messages),
            "message_types": message_types,
            "memory_limit": self.memory.max_messages
        }

    # async def _save_code_before_execution(self, tool_name: str, args: dict):
    #     """Save code to workspace before execution for supported tools."""
    #     import os
    #     import datetime
        
    #     # Only save code for SandboxPythonExecute tool
    #     if tool_name != "SandboxPythonExecute":
    #         return
        
    #     # Extract code from arguments
    #     code = args.get("code", "")
    #     if not code or not code.strip():
    #         return
            
    #     try:
    #         # Get workspace directory from environment
    #         workspace_dir = os.environ.get("GAIA_AGENT_WORKSPACE_DIR", "")
    #         if not workspace_dir:
    #             logger.warning("No workspace directory set - skipping code save")
    #             return
                
    #         # Ensure workspace directory exists
    #         os.makedirs(workspace_dir, exist_ok=True)
            
    #         # Generate filename with timestamp
    #         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #         filename = f"executed_code_{timestamp}.py"
    #         filepath = os.path.join(workspace_dir, filename)
            
    #         # Save code to file
    #         with open(filepath, 'w', encoding='utf-8') as f:
    #             f.write(f"# Code executed at {datetime.datetime.now().isoformat()}\n")
    #             f.write(f"# Tool: {tool_name}\n")
    #             f.write(f"# Agent: {self.name} (ID: {self.id})\n")
    #             f.write(f"# Task: {self.task_id}\n\n")
    #             f.write(code)
                
    #         logger.info(f"💾 Code saved to: {filepath}")
            
    #     except Exception as e:
    #         logger.warning(f"Failed to save code before execution: {e}")