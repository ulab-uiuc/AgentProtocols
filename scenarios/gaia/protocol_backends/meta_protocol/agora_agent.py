"""
Agora Meta Agent for GAIA Framework.
Integrates Agora protocol with meta protocol capabilities using Agora SDK.
"""

import asyncio
import uuid
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Setup paths for imports - follow streaming_queue pattern
current_file = Path(__file__).resolve()
gaia_root = current_file.parents[2]  # Go up to gaia root
agent_network_root = gaia_root.parent.parent  # Go up to agent_network root
src_path = agent_network_root / "src"

sys.path.insert(0, str(agent_network_root))
sys.path.insert(0, str(gaia_root))
sys.path.insert(0, str(src_path))

from src.core.base_agent import BaseAgent

# Agora SDK imports
try:
    import agora
except ImportError as e:
    print(f"[AGORA-META] Agora SDK not available: {e}")
    raise ImportError(f"Agora protocol requires Agora SDK: {e}")

logger = logging.getLogger(__name__)


class AgoraExecutorWrapper:
    """Adapter for Agora protocol integration with GAIA framework."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.capabilities = ["text_processing", "protocol_optimization", "efficiency_enhancement", "agora_native"]
        # Add agora_qa_worker for AgoraServerAdapter compatibility
        self.agora_qa_worker = self
    
    async def answer(self, message: str) -> str:
        """Answer method for AgoraServerAdapter compatibility."""
        logger.info(f"[AGORA-META] answer() called with message: {message[:100]}...")
        result = await self._async_execute(message, "")
        logger.info(f"[AGORA-META] answer() returning result: {result[:100]}...")
        return result
        
    def __call__(self, message: str, context: str = "") -> str:
        """Synchronous tool interface for Agora ReceiverServer."""
        try:
            # Run async execution in the current event loop
            loop = asyncio.get_running_loop()
            coro = self._async_execute(message, context)
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            result = future.result(timeout=60)  # Wait for result
            return result
        except Exception as e:
            logger.error(f"[AGORA-META] Sync execution failed: {e}")
            return f"[AGORA-META] Error: {e}"
    
    async def _async_execute(self, message: str, context: str = "") -> str:
        """Async execution implementation."""
        import inspect
        import json
        
        # For Agora, message is already a string, so we can use it directly
        text_content = message if isinstance(message, str) else str(message)
        
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
                from core.schema import Message, AgentState
                
                # Create tool registry and get tools based on config
                registry = ToolRegistry()
                agent_tool = self.config.get('tool', 'create_chat_completion')  # Default tool
                
                # Create tool collection for this agent with rate limiting
                if agent_tool == 'browser_use':
                    tools = registry.create_collection(['browser_use'], 'agora_browser')
                    # Add search rate limiting to avoid CAPTCHA
                    import os
                    os.environ['BROWSER_USE_SEARCH_DELAY'] = '3'  # 3 seconds between searches
                    os.environ['BROWSER_USE_MAX_RETRIES'] = '1'   # Reduce retries to avoid CAPTCHA
                elif agent_tool == 'str_replace_editor':
                    tools = registry.create_collection(['str_replace_editor'], 'agora_editor')
                elif agent_tool == 'python_execute':
                    tools = registry.create_collection(['python_execute'], 'agora_python')
                else:  # Default to create_chat_completion
                    tools = registry.create_collection(['create_chat_completion'], 'agora_chat')
                
                # Compose input with agent prompt and tool usage guidelines
                agent_prompt = self.config.get('agent_prompt', '')
                tool_guidelines = ""
                
                if agent_tool == 'browser_use':
                    tool_guidelines = """
BROWSER_USE GUIDELINES (ANTI-CAPTCHA):
- ONLY perform ONE search attempt per task
- If search fails, immediately conclude with available information
- DO NOT retry failed searches
- Use simple, direct search terms without complex site: filters
- If no results found, return "Search failed - unable to locate document"
- NEVER perform more than 1 web_search action
"""
                elif agent_tool == 'str_replace_editor':
                    tool_guidelines = """
STR_REPLACE_EDITOR GUIDELINES:
- First explore current directory with {"command":"view","path":"."}
- Do NOT guess file paths - explore systematically
- Avoid view_range parameter for directories
- Work within the current workspace directory
"""
                
                if agent_prompt or tool_guidelines:
                    composed_input = f"{agent_prompt}\n{tool_guidelines}\n\nTASK:\n{text_content}"
                else:
                    composed_input = text_content
                
                # Create temporary GAIA agent with strict execution limits
                temp_agent = ToolCallAgent(
                    name=f"AgoraAgent_{self.config.get('name', 'unknown')}",
                    available_tools=tools,
                    llm=LLM(),
                    task_id=self.config.get('task_id', 'meta_task'),
                    ws=os.environ.get('GAIA_AGENT_WORKSPACE_DIR', '/tmp'),
                    config=self.config,
                    max_steps=2  # Strict limit to prevent cascading failures
                )
                
                # Execute the task using GAIA agent WITHOUT auto-cleanup
                # Use manual execution loop to avoid ToolCallAgent.run()'s auto-cleanup
                if composed_input:
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
                logger.error(f"[AGORA-META] GAIA tool execution failed: {e}")
                return f"[AGORA-GAIA] Error executing tools: {e}"
        
        try:
            logger.info(f"[AGORA-META] Processing: {text_content[:50]}...")
            
            # Execute using GAIA tools and return result synchronously
            result = await _execute_with_gaia_tools(text_content)
            
            logger.info(f"[AGORA-META] Completed execution: {result[:50]}...")
            return result
            
        except Exception as e:
            error_msg = f"[AGORA-META] Execution failed: {e}"
            logger.error(error_msg)
            return error_msg
    
    async def execute(self, context: Dict[str, Any], event_queue: Optional[Any] = None) -> None:
        """Legacy async execute method for compatibility."""
        try:
            # Extract text from context dict
            if isinstance(context, dict):
                text_content = context.get("message", {}).get("content", str(context))
            else:
                text_content = str(context)
            
            result = await self._async_execute(text_content, "")
            
            # Send result event if queue is provided
            if event_queue:
                event = {
                    "type": "agent_text_message",
                    "data": result,
                    "protocol": "agora",
                    "timestamp": asyncio.get_event_loop().time()
                }
                try:
                    if hasattr(event_queue, 'enqueue_event'):
                        res = event_queue.enqueue_event(event)
                        if inspect.isawaitable(res):
                            await res
                    elif hasattr(event_queue, 'put'):
                        await event_queue.put(event)
                    elif hasattr(event_queue, 'put_nowait'):
                        event_queue.put_nowait(event)
                except Exception as e:
                    logger.error(f"Failed to send event: {e}")
            
        except Exception as e:
            logger.error(f"[AGORA-META] Legacy execute failed: {e}")


class AgoraMetaAgent:
    """Agora Meta Protocol Agent for GAIA."""
    
    def __init__(self, agent_id: str, config: Dict[str, Any], install_loopback: bool = False):
        self.agent_id = agent_id
        self.config = config
        self.install_loopback = install_loopback
        self.base_agent: Optional[BaseAgent] = None
        self.agora_executor: Optional[AgoraExecutorWrapper] = None
        
        logger.info(f"[AGORA-META] Initialized Agora meta agent: {agent_id}")
        
    async def create_agora_worker(self, host: str = "0.0.0.0", port: Optional[int] = None) -> BaseAgent:
        """Create Agora worker with BaseAgent integration."""
        try:
            # Create Agora executor wrapper
            self.agora_executor = AgoraExecutorWrapper(self.config)
            
            # Create BaseAgent with Agora executor
            self.base_agent = await BaseAgent.create_agora(
                agent_id=self.agent_id,
                executor=self.agora_executor,
                host=host,
                port=port or 0
            )
            
            logger.info(f"[AGORA-META] Created Agora meta worker: {self.agent_id}")
            return self.base_agent
            
        except Exception as e:
            logger.error(f"[AGORA-META] Failed to create worker {self.agent_id}: {e}")
            raise
    
    async def close(self):
        """Close Agora meta agent."""
        if self.base_agent:
            await self.base_agent.stop()
            logger.info(f"[AGORA-META] Closed agent: {self.agent_id}")


async def create_agora_meta_worker(
    agent_id: str, 
    config: Dict[str, Any], 
    host: str = "0.0.0.0", 
    port: Optional[int] = None,
    install_loopback: bool = False
) -> AgoraMetaAgent:
    """Factory function to create Agora meta protocol worker."""
    agent = AgoraMetaAgent(agent_id, config, install_loopback)
    await agent.create_agora_worker(host=host, port=port)
    return agent