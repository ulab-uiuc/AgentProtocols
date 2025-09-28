"""
ACP Meta Agent for GAIA Framework.
Integrates ACP protocol with meta protocol capabilities using ACP SDK.
"""

import asyncio
import uuid
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, AsyncGenerator

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

# ACP SDK imports
try:
    from acp_sdk.models import Message, MessagePart
    from acp_sdk.server import Context, RunYield
except ImportError as e:
    print(f"[ACP-META] ACP SDK not available: {e}")
    raise ImportError(f"ACP protocol requires ACP SDK: {e}")

logger = logging.getLogger(__name__)


class ACPExecutorWrapper:
    """Adapter for ACP protocol integration with GAIA framework."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.capabilities = ["text_processing", "async_generation", "acp_sdk_1.0.3"]
        
    async def __call__(self, messages: list[Message], context: Context) -> AsyncGenerator[RunYield, None]:
        """ACP server async generator interface."""
        logger.debug(f"[ACP-META] Processing {len(messages)} messages")
        
        try:
            # Process each message
            for i, message in enumerate(messages):
                run_id = str(uuid.uuid4())
                logger.debug(f"[ACP-META] Processing message {i+1}/{len(messages)} with run_id: {run_id}")
                
                # Extract text from message
                text_content = ""
                if hasattr(message, 'parts') and message.parts:
                    for part in message.parts:
                        if hasattr(part, 'type') and part.type == "text":
                            text_content += getattr(part, 'text', getattr(part, 'content', ""))
                else:
                    text_content = str(message)
                
                # Execute using GAIA tools with strict limits
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
                    
                    # Create tool collection for this agent
                    if agent_tool == 'browser_use':
                        tools = registry.create_collection(['browser_use'], 'acp_browser')
                    elif agent_tool == 'file_operators':
                        tools = registry.create_collection(['file_operators'], 'acp_file_ops')
                    elif agent_tool == 'python_execute':
                        tools = registry.create_collection(['python_execute'], 'acp_python')
                    else:  # Default to create_chat_completion
                        tools = registry.create_collection(['create_chat_completion'], 'acp_chat')
                    
                    # Compose input with agent prompt and tool usage guidelines
                    agent_prompt = self.config.get('agent_prompt', '')
                    tool_guidelines = ""
                    
                    if agent_tool == 'create_chat_completion':
                        tool_guidelines = """
CREATE_CHAT_COMPLETION GUIDELINES:
- You are the FINAL agent - provide a definitive answer
- Use the tool ONLY ONCE to give the final response
- Do NOT ask for more information or clarification
- Synthesize available information into a clear conclusion
- If information is insufficient, state what can be concluded
"""
                    
                    if agent_prompt or tool_guidelines:
                        composed_input = f"{agent_prompt}\n{tool_guidelines}\n\nTASK:\n{text_content}"
                    else:
                        composed_input = text_content
                    
                    # Create temporary GAIA agent with strict execution limits
                    temp_agent = ToolCallAgent(
                        name=f"ACPAgent_{self.config.get('name', 'unknown')}",
                        available_tools=tools,
                        llm=LLM(),
                        task_id=self.config.get('task_id', 'meta_task'),
                        ws=os.environ.get('GAIA_AGENT_WORKSPACE_DIR', '/tmp'),
                        config=self.config,
                        max_steps=1  # Strict limit for final synthesis agents
                    )
                    
                    # Execute the task using GAIA agent WITHOUT auto-cleanup
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
                    final_result = str(result)
                    
                except Exception as e:
                    logger.error(f"[ACP-META] GAIA tool execution failed: {e}")
                    final_result = f"[ACP-GAIA] Error executing tools: {e}"
                
                # Yield result
                yield final_result
                
        except Exception as e:
            logger.error(f"[ACP-META] Execution failed: {e}")
            yield f"[ACP-META] Error: {e}"


class ACPMetaAgent:
    """ACP Meta Protocol Agent for GAIA."""
    
    def __init__(self, agent_id: str, config: Dict[str, Any], install_loopback: bool = False):
        self.agent_id = agent_id
        self.config = config
        self.install_loopback = install_loopback
        self.base_agent: Optional[BaseAgent] = None
        self.acp_executor: Optional[ACPExecutorWrapper] = None
        
        logger.info(f"[ACP-META] Initialized ACP meta agent: {agent_id}")
        
    async def create_acp_worker(self, host: str = "0.0.0.0", port: Optional[int] = None) -> BaseAgent:
        """Create ACP worker with BaseAgent integration."""
        try:
            # Create ACP executor wrapper
            self.acp_executor = ACPExecutorWrapper(self.config)
            
            # Create BaseAgent with ACP executor
            self.base_agent = await BaseAgent.create_acp(
                agent_id=self.agent_id,
                executor=self.acp_executor,
                host=host,
                port=port or 0
            )
            
            logger.info(f"[ACP-META] Created ACP meta worker: {self.agent_id}")
            return self.base_agent
            
        except Exception as e:
            logger.error(f"[ACP-META] Failed to create worker {self.agent_id}: {e}")
            raise
    
    async def close(self):
        """Close ACP meta agent."""
        if self.base_agent:
            await self.base_agent.stop()
            logger.info(f"[ACP-META] Closed agent: {self.agent_id}")


async def create_acp_meta_worker(
    agent_id: str, 
    config: Dict[str, Any], 
    host: str = "0.0.0.0", 
    port: Optional[int] = None,
    install_loopback: bool = False
) -> ACPMetaAgent:
    """Factory function to create ACP meta protocol worker."""
    agent = ACPMetaAgent(agent_id, config, install_loopback)
    await agent.create_acp_worker(host=host, port=port)
    return agent
