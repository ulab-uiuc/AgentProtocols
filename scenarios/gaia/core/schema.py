import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

from .llm import LLM

# Shared terminal colors for consistent logging output across modules
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"

class Role(str, Enum):
    """Message role options"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore


class ToolChoice(str, Enum):
    """Tool choice options"""

    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES]  # type: ignore


class AgentState(str, Enum):
    """Agent execution states"""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class Function(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    """Represents a tool/function call in a message"""

    id: str
    type: str = "function"
    function: Function


class Message(BaseModel):
    """Represents a chat message in the conversation"""

    role: ROLE_TYPE = Field(...)  # type: ignore
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(default=None)

    def __add__(self, other) -> List["Message"]:
        """Support Message + list or Message + Message operations"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["Message"]:
        """Support list + Message operations"""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> dict:
        """Convert message to dictionary format"""
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [tool_call.dict() for tool_call in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.base64_image is not None:
            message["base64_image"] = self.base64_image
        return message

    @classmethod
    def user_message(
        cls, content: str, base64_image: Optional[str] = None
    ) -> "Message":
        """Create a user message"""
        return cls(role=Role.USER, content=content, base64_image=base64_image)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        """Create a system message"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(
        cls, content: Optional[str] = None, base64_image: Optional[str] = None
    ) -> "Message":
        """Create an assistant message"""
        return cls(role=Role.ASSISTANT, content=content, base64_image=base64_image)

    @classmethod
    def tool_message(
        cls, content: str, name, tool_call_id: str, base64_image: Optional[str] = None
    ) -> "Message":
        """Create a tool message"""
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            base64_image=base64_image,
        )

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: List[Any],
        content: Union[str, List[str]] = "",
        base64_image: Optional[str] = None,
        **kwargs,
    ):
        """Create ToolCallsMessage from raw tool calls.

        Args:
            tool_calls: Raw tool calls from LLM
            content: Optional message content
            base64_image: Optional base64 encoded image
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=formatted_calls,
            base64_image=base64_image,
            **kwargs,
        )


class ExecutionStatus(str, Enum):
    """Agent execution status for tasks"""
    PENDING = "pending"
    PROCESSING = "processing" 
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class StepExecution(BaseModel):
    """Single step execution record"""
    step: int
    agent_id: str
    agent_name: str
    execution_status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    messages: List[Message] = Field(default_factory=list)
    error_message: Optional[str] = None
    
    def is_completed(self) -> bool:
        """Check if step execution is completed"""
        return self.execution_status in [ExecutionStatus.SUCCESS, ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT]
    
    def duration(self) -> Optional[float]:
        """Get step execution duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class AgentExecution(BaseModel):
    """Agent execution record in network memory pool"""
    agent_id: str
    agent_name: str
    task_id: str
    user_message: str
    execution_status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    memory_messages: List[Message] = Field(default_factory=list)
    error_message: Optional[str] = None
    
    def is_completed(self) -> bool:
        """Check if execution is completed (success or error)"""
        return self.execution_status in [ExecutionStatus.SUCCESS, ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT]
    
    def duration(self) -> Optional[float]:
        """Get execution duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class NetworkMemoryPool(BaseModel):
    """Network-level memory pool to track all agent executions with step-based structure"""
    model_config = {"arbitrary_types_allowed": True}
    executions: List[AgentExecution] = Field(default_factory=list)
    step_executions: Dict[int, StepExecution] = Field(default_factory=dict)  # step -> execution
    max_executions: int = Field(default=1000)
    llm: Optional[LLM] = Field(default_factory=LLM)

    def add_step_execution(self, step: int, agent_id: str, agent_name: str, task_id: str, user_message: str) -> str:
        """Add new step execution record"""
        step_execution = StepExecution(
            step=step,
            agent_id=agent_id,
            agent_name=agent_name
        )
        self.step_executions[step] = step_execution
        
        # Also add to legacy executions for compatibility
        execution_id = f"{agent_id}_{task_id}_{step}"
        execution = AgentExecution(
            agent_id=agent_id,
            agent_name=agent_name, 
            task_id=task_id,
            user_message=user_message
        )
        self.executions.append(execution)
        
        # Limit executions
        if len(self.executions) > self.max_executions:
            self.executions = self.executions[-self.max_executions:]
            
        return execution_id
    
    def update_step_status(self, step: int, status: ExecutionStatus, 
                          messages: Optional[List[Message]] = None,
                          error_message: Optional[str] = None) -> bool:
        """Update step execution status and messages"""
        import time
        
        if step not in self.step_executions:
            return False
            
        step_exec = self.step_executions[step]
        step_exec.execution_status = status
        
        if status == ExecutionStatus.PROCESSING and step_exec.start_time is None:
            step_exec.start_time = time.time()
        
        if status in [ExecutionStatus.SUCCESS, ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT]:
            step_exec.end_time = time.time()
        
        if messages:
            step_exec.messages = messages
            
        if error_message:
            step_exec.error_message = error_message
            
        return True
    
    def get_step_status(self, step: int) -> Optional[ExecutionStatus]:
        """Get current step execution status"""
        if step in self.step_executions:
            return self.step_executions[step].execution_status
        return None
    
    def get_step_messages(self, step: int) -> Optional[List[Message]]:
        """Get messages for specific step"""
        if step in self.step_executions:
            return self.step_executions[step].messages
        return None
    
    def get_completed_steps(self) -> List[int]:
        """Get all completed step numbers"""
        return [step for step, exec in self.step_executions.items() if exec.is_completed()]
    
    def is_step_completed(self, step: int) -> bool:
        """Check if step is completed"""
        if step in self.step_executions:
            return self.step_executions[step].is_completed()
        return False
    
    def get_workflow_progress(self) -> Dict[str, Any]:
        """Get overall workflow progress based on steps"""
        if not self.step_executions:
            return {"status": "no_steps", "total": 0}
        
        total_steps = len(self.step_executions)
        completed_steps = len(self.get_completed_steps())
        
        status_counts = {}
        for status in ExecutionStatus:
            status_counts[status.value] = sum(1 for exec in self.step_executions.values() 
                                            if exec.execution_status == status)
        
        return {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "completion_rate": completed_steps / total_steps if total_steps > 0 else 0,
            "status_breakdown": status_counts,
            "step_details": {step: exec.execution_status.value for step, exec in self.step_executions.items()}
        }
    
    def get_step_chain_context(self, current_step: int) -> List[Dict[str, Any]]:
        """Get context from previous steps for current step"""
        context = []
        for step in range(current_step):
            if step in self.step_executions:
                step_exec = self.step_executions[step]
                if step_exec.is_completed() and step_exec.messages:
                    # Extract assistant messages as results
                    assistant_messages = [msg for msg in step_exec.messages if msg.role == "assistant"]
                    if assistant_messages:
                        context.append({
                            "step": step,
                            "agent_id": step_exec.agent_id,
                            "agent_name": step_exec.agent_name,
                            "result": assistant_messages[-1].content,
                            "status": step_exec.execution_status.value
                        })
        return context
    
    # Legacy methods for backward compatibility
    def add_execution(self, agent_id: str, agent_name: str, task_id: str, user_message: str) -> str:
        """Legacy method: Add new agent execution record"""
        execution_id = f"{agent_id}_{task_id}_{len(self.executions)}"
        execution = AgentExecution(
            agent_id=agent_id,
            agent_name=agent_name, 
            task_id=task_id,
            user_message=user_message
        )
        self.executions.append(execution)
        
        if len(self.executions) > self.max_executions:
            self.executions = self.executions[-self.max_executions:]
            
        return execution_id

    async def summarize(self, initial_task: str = None, max_tokens: int = 256) -> str:
        if not self.llm:
            return "LLM not initialized for summarization."

        all_messages: List[Message] = []
        for step_exec in self.step_executions.values():
            if step_exec.messages:
                all_messages.extend(step_exec.messages)

        if not all_messages:
            return "No messages to summarize."
        
        context_msgs = all_messages[-20:]

        # 1. Create a list specifically for system messages
        system_prompts = [
            {"role": "system", "content": "You are a helpful AI assistant skilled at summarizing conversations to provide a final, direct answer."}
        ]

        # 2. Create a list containing only conversation history (user/assistant)
        history_messages = []
        for msg in context_msgs:
            content_str = ""
            if isinstance(msg.content, (dict, list)):
                content_str = json.dumps(msg.content, ensure_ascii=False)
            elif msg.content is not None:
                content_str = str(msg.content)
            
            # Ensure we don't add 'system' role to this list
            if msg.role != "system":
                 history_messages.append({"role": str(msg.role), "content": content_str})

        # 3. Add the final summary instruction to the conversation history
        summarization_prompt = "Based on our conversation history, please provide a concise and clear final summary of what happened."
        if initial_task:
            summarization_prompt += f"\nRemember, the original goal was: '{initial_task[:200]}...'"
        
        history_messages.append({"role": "user", "content": summarization_prompt})
        
        # 4. Use the correct, separated parameters to call 'ask'
        response = await self.llm.ask(
            messages=history_messages,      # user/assistant conversation history
            system_msgs=system_prompts,     # standalone system instructions
            temperature=0.2
        )

        return response.strip()


class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)
    
    def add_message(self, message: Message) -> None:
        """Add a message to memory"""
        self.messages.append(message)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: List[Message]) -> None:
        """Add multiple messages to memory"""
        self.messages.extend(messages)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        """Get n most recent messages"""
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """Convert messages to list of dicts"""
        return [msg.to_dict() for msg in self.messages]