"""
A2A Meta Agent for GAIA Framework.
Integrates A2A protocol with meta protocol capabilities using A2A SDK.
"""

print(f"[A2A-META] Module path: {__file__}")

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')

# Setup paths for imports - follow streaming_queue pattern
current_file = Path(__file__).resolve()
gaia_root = current_file.parents[2]  # Go up to gaia root
agent_network_root = gaia_root.parent.parent  # Go up to agent_network root
src_path = agent_network_root / "src"

sys.path.insert(0, str(agent_network_root))
sys.path.insert(0, str(gaia_root))
sys.path.insert(0, str(src_path))

from src.core.base_agent import BaseAgent

# A2A SDK imports
try:
    from a2a.server.agent_execution import AgentExecutor, RequestContext
except ImportError as e:
    print(f"[A2A-META] A2A SDK not available: {e}")
    raise ImportError(f"A2A protocol requires A2A SDK: {e}")

logger = logging.getLogger(__name__)


class A2AExecutorWrapper:
    """Adapter for A2A protocol integration with GAIA framework."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.capabilities = ["text_processing", "fast_communication", "high_throughput", "a2a_native"]
        
    async def execute(self, context, event_queue: Optional[Any] = None) -> None:
        """Execute A2A task within GAIA framework using actual GAIA tools, and reply via SDK context."""
        import inspect
        import json

        def _extract_text_from_context(ctx) -> str:
            """
            Be liberal in what you accept:
            - A2A SDK RequestContext: try ctx.params.message.{text|parts[*].text}
            - Generic objects: try ctx.message / ctx.data / ctx.payload / ctx.body
            - Dict-like: look for ['message']['text'|'content'|parts], or top-level 'text'/'content'
            - Fallback: JSON-dump shallow attrs or repr(ctx)
            """
            import json

            # 1) A2A SDK style
            try:
                params = getattr(ctx, "params", None)
                if params is not None:
                    msg = getattr(params, "message", None)
                    if msg is not None:
                        # direct text
                        t = getattr(msg, "text", None)
                        if isinstance(t, str) and t.strip():
                            return t
                        # parts list
                        parts = getattr(msg, "parts", None)
                        if isinstance(parts, (list, tuple)):
                            for p in parts:
                                pt = getattr(p, "text", None)
                                if isinstance(pt, str) and pt.strip():
                                    return pt
                        # dict-like message
                        if isinstance(msg, dict):
                            t = msg.get("text") or msg.get("content")
                            if isinstance(t, str) and t.strip():
                                return t
                            if "parts" in msg and isinstance(msg["parts"], list):
                                for p in msg["parts"]:
                                    if isinstance(p, dict) and isinstance(p.get("text"), str):
                                        return p["text"]
            except Exception:
                pass

            # 2) Attribute fallbacks on ctx.*
            for attr in ("message", "data", "payload", "body"):
                v = getattr(ctx, attr, None)
                if isinstance(v, str) and v.strip():
                    return v
                if isinstance(v, dict):
                    t = v.get("text") or v.get("content")
                    if isinstance(t, str) and t.strip():
                        return t
                    parts = v.get("parts")
                    if isinstance(parts, list):
                        for p in parts:
                            if isinstance(p, dict) and isinstance(p.get("text"), str):
                                return p["text"]

            # 3) Dict-like ctx itself
            if hasattr(ctx, "get"):
                try:
                    msg = ctx.get("message")
                    if isinstance(msg, dict):
                        t = msg.get("text") or msg.get("content")
                        if isinstance(t, str) and t.strip():
                            return t
                        parts = msg.get("parts")
                        if isinstance(parts, list):
                            for p in parts:
                                if isinstance(p, dict) and isinstance(p.get("text"), str):
                                    return p["text"]
                    for k in ("content", "text", "body"):
                        v = ctx.get(k)
                        if isinstance(v, str) and v.strip():
                            return v
                except Exception:
                    pass

            # 4) Shallow JSON view of known attrs
            for attr in ("params", "message", "data", "payload", "body"):
                v = getattr(ctx, attr, None)
                if v is not None:
                    try:
                        return json.dumps(v, default=str, ensure_ascii=False)
                    except Exception:
                        pass

            # 5) Final fallback
            return str(ctx)
        
        async def _execute_with_gaia_tools(text_content: str) -> str:
            """Execute using GAIA tool system."""
            try:
                # Import GAIA core components
                import sys
                import os
                from pathlib import Path
                
                # Add GAIA path
                gaia_root = Path(__file__).resolve().parents[2]  # Go up to gaia root
                if str(gaia_root) not in sys.path:
                    sys.path.insert(0, str(gaia_root))
                
                from core.agent import ToolCallAgent
                from tools.registry import ToolRegistry
                from core.llm import LLM
                
                # Create tool registry and get tools based on config
                registry = ToolRegistry()
                agent_tool = self.config.get('tool', 'create_chat_completion')  # Default tool
                
                # Create tool collection for this agent
                if agent_tool == 'browser_use':
                    tools = registry.create_collection(['browser_use'], 'a2a_browser')
                elif agent_tool == 'str_replace_editor':
                    tools = registry.create_collection(['str_replace_editor'], 'a2a_editor')
                elif agent_tool == 'python_execute':
                    tools = registry.create_collection(['python_execute'], 'a2a_python')
                else:  # Default to create_chat_completion
                    tools = registry.create_collection(['create_chat_completion'], 'a2a_chat')
                
                # Compose input with agent prompt and tool usage guidelines
                agent_prompt = self.config.get('agent_prompt', '')
                tool_guidelines = ""
                
                if agent_tool == 'str_replace_editor':
                    tool_guidelines = """
STR_REPLACE_EDITOR GUIDELINES:
- First explore current directory with {"command":"view","path":"."}
- Do NOT guess file paths - explore systematically
- Avoid view_range parameter for directories
- Work within the current workspace directory
- Stop after 2-3 failed path attempts
"""
                elif agent_tool == 'browser_use':
                    tool_guidelines = """
BROWSER_USE GUIDELINES:
- Use targeted searches with specific terms and dates
- Avoid repeated searches with similar queries
- If search fails, try different keywords instead of retrying
"""
                
                if agent_prompt or tool_guidelines:
                    composed_input = f"{agent_prompt}\n{tool_guidelines}\n\nTASK:\n{text_content}"
                else:
                    composed_input = text_content
                
                # Create temporary GAIA agent with strict execution limits
                temp_agent = ToolCallAgent(
                    name=f"A2AAgent_{self.config.get('name', 'unknown')}",
                    available_tools=tools,
                    llm=LLM(),
                    task_id=self.config.get('task_id', 'meta_task'),
                    ws=os.environ.get('GAIA_AGENT_WORKSPACE_DIR', '/tmp'),
                    config=self.config,
                    max_steps=2  # Strict limit to prevent file system exploration loops
                )
                
                # Execute the task using GAIA agent WITHOUT auto-cleanup
                # Use manual execution loop to avoid ToolCallAgent.run()'s auto-cleanup
                if composed_input:
                    from core.schema import Message, AgentState
                    temp_agent.messages.append(Message.user_message(composed_input))
                
                # Execute steps manually without cleanup
                temp_agent.current_step = 0
                final_result = "No result generated"
                
                while temp_agent.current_step < temp_agent.max_steps and temp_agent.state != AgentState.FINISHED:
                    step_result = await temp_agent.step()
                    temp_agent.current_step += 1
                    
                    # Extract result if available
                    if temp_agent.messages and temp_agent.messages[-1].content:
                        final_result = temp_agent.messages[-1].content
                        break
                    
                    # If agent finished, break
                    if temp_agent.state == AgentState.FINISHED:
                        break
                
                result = final_result or f"Agent completed after {temp_agent.current_step} steps"
                
                # DO NOT cleanup - lifecycle managed by MetaProtocol network
                
                return str(result)
                
            except Exception as e:
                logger.error(f"[A2A-META] GAIA tool execution failed: {e}")
                return f"[A2A-GAIA] Error executing tools: {e}"
        
        async def _send_event(eq: Any, payload: Dict[str, Any]) -> None:
            """Send event to queue safely."""
            try:
                if hasattr(eq, 'enqueue_event'):
                    res = eq.enqueue_event(payload)
                    if inspect.isawaitable(res):
                        await res
                elif hasattr(eq, 'put'):
                    await eq.put(payload)
                elif hasattr(eq, 'put_nowait'):
                    eq.put_nowait(payload)
            except Exception as e:
                logger.error(f"Failed to send event: {e}")
        
        async def _safe_send_event(eq: Any, payload: Dict[str, Any]) -> None:
            """Send event to queue safely (fallback only)."""
            try:
                if hasattr(eq, 'enqueue_event'):
                    res = eq.enqueue_event(payload)
                    if inspect.isawaitable(res):
                        await res
                elif hasattr(eq, 'put'):
                    await eq.put(payload)
                elif hasattr(eq, 'put_nowait'):
                    eq.put_nowait(payload)
            except Exception as e:
                logger.error(f"Failed to send event: {e}")

        try:
            text_content = _extract_text_from_context(context)
            logger.info(f"[A2A-META] Processing: {text_content[:80]}...")

            result_text = await _execute_with_gaia_tools(text_content)

            # A2A SDK uses EventQueue for communication, not RequestContext reply methods
            if event_queue:
                try:
                    # Create A2A SDK compatible event with proper messageId
                    from a2a.utils import new_agent_text_message
                    from a2a.types import Message, TextPart, Role
                    import uuid
                    import time
                    
                    # Create message manually to ensure all required fields
                    message_id = str(uuid.uuid4())
                    text_part = TextPart(text=result_text)
                    
                    result_message = Message(
                        messageId=message_id,
                        role=Role.agent,
                        parts=[text_part],
                        taskId=self.config.get('task_id', str(uuid.uuid4())),
                        contextId=str(uuid.uuid4()),
                        kind='message'
                    )
                    
                    await event_queue.enqueue_event(result_message)
                    logger.info(f"[A2A-META] Sent result via EventQueue: {result_text[:50]}...")
                    
                except Exception as e:
                    logger.error(f"[A2A-META] Failed to send A2A event: {e}")
                    # Fallback to basic event
                    await _safe_send_event(event_queue, {
                        "type": "agent_text_message",
                        "data": result_text,
                        "protocol": "a2a"
                    })
            else:
                logger.warning(f"[A2A-META] No event_queue provided, cannot send result")

        except Exception as e:
            error_msg = f"[A2A-META] Execution failed: {e}"
            logger.error(error_msg)
            # Try to notify sender about the failure using A2A EventQueue
            if event_queue:
                try:
                    from a2a.utils import new_agent_text_message
                    error_message = new_agent_text_message(error_msg)
                    await event_queue.enqueue_event(error_message)
                except Exception:
                    await _safe_send_event(event_queue, {
                        "type": "agent_text_message", 
                        "data": error_msg,
                        "protocol": "a2a",
                        "error": True
                    })


class A2AMetaAgent:
    """A2A Meta Protocol Agent for GAIA."""
    
    def __init__(self, agent_id: str, config: Dict[str, Any], install_loopback: bool = False):
        self.agent_id = agent_id
        self.config = config
        self.install_loopback = install_loopback
        self.base_agent: Optional[BaseAgent] = None
        self.a2a_executor: Optional[A2AExecutorWrapper] = None
        
        logger.info(f"[A2A-META] Initialized A2A meta agent: {agent_id}")
        
    async def create_a2a_worker(self, host: str = "0.0.0.0", port: Optional[int] = None) -> BaseAgent:
        """Create A2A worker with BaseAgent integration."""
        try:
            # Create A2A executor wrapper
            self.a2a_executor = A2AExecutorWrapper(self.config)
            
            # Create BaseAgent with A2A executor
            self.base_agent = await BaseAgent.create_a2a(
                agent_id=self.agent_id,
                executor=self.a2a_executor,
                host=host,
                port=port or 0
            )
            
            logger.info(f"[A2A-META] Created A2A meta worker: {self.agent_id}")
            return self.base_agent
            
        except Exception as e:
            logger.error(f"[A2A-META] Failed to create worker {self.agent_id}: {e}")
            raise
    
    async def close(self):
        """Close A2A meta agent."""
        if self.base_agent:
            await self.base_agent.stop()
            logger.info(f"[A2A-META] Closed agent: {self.agent_id}")


async def create_a2a_meta_worker(
    agent_id: str, 
    config: Dict[str, Any], 
    host: str = "0.0.0.0", 
    port: Optional[int] = None,
    install_loopback: bool = False
) -> A2AMetaAgent:
    """Factory function to create A2A meta protocol worker."""
    agent = A2AMetaAgent(agent_id, config, install_loopback)
    await agent.create_a2a_worker(host=host, port=port)
    return agent
