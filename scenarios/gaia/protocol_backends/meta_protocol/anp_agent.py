"""
ANP Meta Agent for GAIA Framework.
Integrates ANP protocol with meta protocol capabilities using AgentConnect SDK.
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

# AgentConnect SDK imports
try:
    agentconnect_path = agent_network_root / "agentconnect_src"
    sys.path.insert(0, str(agentconnect_path))

    from agent_connect.simple_node import SimpleNode, SimpleNodeSession
    from agent_connect.authentication import create_did_wba_document
    from agent_connect.utils.did_generate import did_generate
    from agent_connect.utils.crypto_tool import get_pem_from_private_key
except ImportError as e:
    print(f"[ANP-META] AgentConnect SDK not available: {e}")
    raise ImportError(f"ANP protocol requires AgentConnect SDK: {e}")

logger = logging.getLogger(__name__)


class ANPExecutorWrapper:
    """Adapter for ANP protocol integration with GAIA framework."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.capabilities = ["text_processing", "did_authentication", "e2e_encryption", "anp_native"]
        
    async def execute(self, context: Dict[str, Any], event_queue: Optional[Any] = None) -> None:
        """Execute ANP task within GAIA framework."""
        import json
        
        def _extract_text_from_context(ctx: Dict[str, Any]) -> str:
            # Extract text from various context formats
            msg = ctx.get("message")
            if isinstance(msg, dict):
                if "content" in msg and isinstance(msg["content"], str):
                    return msg["content"]
                if "text" in msg and isinstance(msg["text"], str):
                    return msg["text"]
                # parts-based
                parts = msg.get("parts")
                if isinstance(parts, list) and parts:
                    first = parts[0]
                    if isinstance(first, dict) and "text" in first:
                        return first["text"]
            # flat keys
            for k in ("content", "text", "body"):
                v = ctx.get(k)
                if isinstance(v, str):
                    return v
            try:
                return json.dumps(ctx, ensure_ascii=False)
            except Exception:
                return str(ctx)
        
        async def _send_event(eq: Any, payload: Dict[str, Any]) -> None:
            """Send event to queue safely."""
            import inspect
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
                
                # Create tool collection for this agent
                if agent_tool == 'browser_use':
                    tools = registry.create_collection(['browser_use'], 'anp_browser')
                elif agent_tool == 'str_replace_editor':
                    tools = registry.create_collection(['str_replace_editor'], 'anp_editor')
                elif agent_tool == 'python_execute':
                    tools = registry.create_collection(['python_execute'], 'anp_python')
                else:  # Default to create_chat_completion
                    tools = registry.create_collection(['create_chat_completion'], 'anp_chat')
                
                # Compose input with agent prompt
                agent_prompt = self.config.get('agent_prompt', '')
                if agent_prompt:
                    composed_input = f"{agent_prompt}\n\nTASK:\n{text_content}"
                else:
                    composed_input = text_content
                
                # Create temporary GAIA agent with execution limits
                temp_agent = ToolCallAgent(
                    name=f"ANPAgent_{self.config.get('name', 'unknown')}",
                    available_tools=tools,
                    llm=LLM(),
                    task_id=self.config.get('task_id', 'meta_task'),
                    ws=os.environ.get('GAIA_AGENT_WORKSPACE_DIR', '/tmp'),
                    config=self.config,
                    max_steps=2  # Limit steps for ANP agents
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
                return str(result)
                
            except Exception as e:
                logger.error(f"[ANP-META] GAIA tool execution failed: {e}")
                return f"[ANP-GAIA] Error executing tools: {e}"
        
        try:
            text_content = _extract_text_from_context(context)
            logger.info(f"[ANP-META] Processing: {text_content[:50]}...")
            
            # Execute using GAIA tools instead of just returning a string
            result = await _execute_with_gaia_tools(text_content)
            
            # Send result event
            if event_queue:
                event = {
                    "type": "agent_text_message",
                    "data": result,
                    "protocol": "anp",
                    "timestamp": asyncio.get_event_loop().time()
                }
                await _send_event(event_queue, event)
            
        except Exception as e:
            error_msg = f"[ANP-META] Execution failed: {e}"
            logger.error(error_msg)
            if event_queue:
                error_event = {
                    "type": "agent_text_message", 
                    "data": error_msg,
                    "protocol": "anp",
                    "error": True
                }
                await _send_event(event_queue, error_event)


class ANPMetaAgent:
    """ANP Meta Protocol Agent for GAIA."""
    
    def __init__(self, agent_id: str, config: Dict[str, Any], install_loopback: bool = False):
        self.agent_id = agent_id
        self.config = config
        self.install_loopback = install_loopback
        self.base_agent: Optional[BaseAgent] = None
        self.anp_executor: Optional[ANPExecutorWrapper] = None
        
        logger.info(f"[ANP-META] Initialized ANP meta agent: {agent_id}")
        
    async def create_anp_worker(self, host: str = "0.0.0.0", port: Optional[int] = None) -> BaseAgent:
        """Create ANP worker with BaseAgent integration."""
        try:
            # Create ANP executor wrapper
            self.anp_executor = ANPExecutorWrapper(self.config)
            
            # Create BaseAgent with ANP executor
            self.base_agent = await BaseAgent.create_anp(
                agent_id=self.agent_id,
                executor=self.anp_executor,
                host=host,
                port=port or 0
            )
            
            logger.info(f"[ANP-META] Created ANP meta worker: {self.agent_id}")
            return self.base_agent
            
        except Exception as e:
            logger.error(f"[ANP-META] Failed to create worker {self.agent_id}: {e}")
            raise
    
    async def close(self):
        """Close ANP meta agent."""
        if self.base_agent:
            await self.base_agent.stop()
            logger.info(f"[ANP-META] Closed agent: {self.agent_id}")


async def create_anp_meta_worker(
    agent_id: str, 
    config: Dict[str, Any], 
    host: str = "0.0.0.0", 
    port: Optional[int] = None,
    install_loopback: bool = False
) -> ANPMetaAgent:
    """Factory function to create ANP meta protocol worker."""
    agent = ANPMetaAgent(agent_id, config, install_loopback)
    await agent.create_anp_worker(host=host, port=port)
    return agent