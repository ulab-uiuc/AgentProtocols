"""
A2A Agent Meta Protocol Integration

Integrates A2A agent with src/core/base_agent.py using Meta-Protocol (UTE) system.
Based on fail_storm_recovery A2A agent implementation.
"""

import asyncio
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Production imports - add proper paths
current_file = Path(__file__).resolve()
fail_storm_path = current_file.parents[2]  # Go up to fail_storm_recovery
project_root = fail_storm_path.parent.parent  # Go up to agent_network
src_path = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(fail_storm_path))
sys.path.insert(0, str(src_path))

from src.core.base_agent import BaseAgent
from protocol_backends.a2a.agent import A2AAgent, create_a2a_agent

logger = logging.getLogger(__name__)


class A2AMetaAgent:
    """
    A2A Agent integrated with Meta-Protocol system
    
    Uses BaseAgent.create_a2a() to wrap A2A native agent into a Meta-Protocol node
    with UTE encoding/decoding, health endpoints, and agent card.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], install_loopback: bool = False):
        self.agent_id = agent_id
        self.config = config
        self.install_loopback = install_loopback
        self.base_agent: Optional[BaseAgent] = None
        self.a2a_agent: Optional[A2AAgent] = None
        self.native_executor = None
        
        logger.info(f"Initialized A2A meta protocol agent: {agent_id}")
    
    def _convert_config_for_executor(self) -> Dict[str, Any]:
        """Convert config for A2A executor based on fail_storm pattern"""
        # Support both 'llm' (new) and 'core' (legacy) config keys
        llm_config = self.config.get("llm") or self.config.get("core", {})
        if llm_config.get("type") in ["openai", "local"]:
            return {
                "model": {
                    "type": llm_config.get("type", "openai"),
                    "name": llm_config.get("model") or llm_config.get("name", "gpt-4o"),
                    "openai_api_key": llm_config.get("openai_api_key"),
                    "openai_base_url": llm_config.get("openai_base_url", "https://api.openai.com/v1"),
                    "temperature": llm_config.get("temperature", 0.0),
                    "max_tokens": llm_config.get("max_tokens", 4096)
                }
            }
        return {"model": llm_config}
    
    async def create_a2a_worker(self, host: str = "0.0.0.0", port: Optional[int] = None) -> BaseAgent:
        """
        Start an A2A server via BaseAgent.create_a2a().
        Uses fail_storm_recovery A2A native implementation.
        """
        try:
            # 1) Create native A2A executor from shard_qa pattern
            shard_qa_path = fail_storm_path / "shard_qa"
            sys.path.insert(0, str(shard_qa_path))
            
            from shard_worker.agent_executor import ShardWorkerExecutor
            
            executor_config = self._convert_config_for_executor()
            # ShardWorkerExecutor needs: config, global_config, shard_id, data_file, neighbors, output, force_llm
            global_config = {
                'shard_qa': {
                    'history': {'max_len': 20},
                    'search': {'max_results': 10}
                }
            }
            self.native_executor = ShardWorkerExecutor(
                config=executor_config,
                global_config=global_config,
                shard_id=self.agent_id,
                data_file=None,
                neighbors=[],
                output=None,
                force_llm=True
            )
            
            # 2) Create executor wrapper to adapt A2A SDK RequestContext to ShardWorkerExecutor expectations
            class A2AExecutorWrapper:
                """Wrapper that adapts A2A SDK RequestContext for ShardWorkerExecutor"""
                def __init__(self, shard_executor):
                    self.shard_executor = shard_executor
                
                async def execute(self, context, event_queue):
                    """Adapt A2A SDK context to ShardWorkerExecutor expectations"""
                    # Import A2A SDK utilities
                    from a2a.utils import new_agent_text_message as a2a_new_text_message
                    from a2a.server.events import EventQueue as A2AEventQueue
                    
                    # Wrap event_queue to convert dict events to A2A SDK Events
                    class EventQueueWrapper:
                        def __init__(self, a2a_queue):
                            object.__setattr__(self, '_a2a_queue', a2a_queue)
                        
                        def __getattr__(self, name):
                            """Delegate all attribute access to original queue"""
                            a2a_queue = object.__getattribute__(self, '_a2a_queue')
                            return getattr(a2a_queue, name)
                        
                        async def enqueue_event(self, event):
                            """Convert dict events to A2A SDK Events before enqueueing"""
                            a2a_queue = object.__getattribute__(self, '_a2a_queue')
                            
                            # If event is a simple dict (from agent_executor.py's create_text_message)
                            if isinstance(event, dict):
                                # Convert to A2A SDK Event using new_agent_text_message
                                text_content = event.get('content') or event.get('text', str(event))
                                a2a_event = a2a_new_text_message(text_content)
                                result = a2a_queue.enqueue_event(a2a_event)
                            else:
                                # Already an A2A Event object
                                result = a2a_queue.enqueue_event(event)
                            
                            # Handle async/sync result
                            import inspect
                            if inspect.isawaitable(result):
                                return await result
                            return result
                    
                    # Wrap the A2A SDK RequestContext to add get_user_input() method
                    class ContextWrapper:
                        def __init__(self, a2a_context):
                            # Don't set any attributes - use __getattr__ for everything except get_user_input
                            object.__setattr__(self, '_a2a_context', a2a_context)
                        
                        def __getattr__(self, name):
                            """Delegate all attribute access to the original context"""
                            a2a_context = object.__getattribute__(self, '_a2a_context')
                            # Map 'params' to '_params' for ShardWorkerExecutor compatibility
                            if name == 'params':
                                return getattr(a2a_context, '_params', None)
                            return getattr(a2a_context, name)
                        
                        def __setattr__(self, name, value):
                            """Delegate attribute setting to original context"""
                            if name == '_a2a_context':
                                object.__setattr__(self, name, value)
                            else:
                                setattr(object.__getattribute__(self, '_a2a_context'), name, value)
                        
                        def get_user_input(self):
                            """Extract text from A2A message parts"""
                            result = ""
                            try:
                                a2a_context = object.__getattribute__(self, '_a2a_context')
                                params = a2a_context._params if hasattr(a2a_context, '_params') else None
                                if params and hasattr(params, 'message'):
                                    message = params.message
                                    
                                    # Extract from parts array (standard A2A format)
                                    if hasattr(message, 'parts') and isinstance(message.parts, list):
                                        for part in message.parts:
                                            # A2A SDK Part has a 'root' attribute containing TextPart/JsonPart etc
                                            if hasattr(part, 'root'):
                                                root = part.root
                                                if hasattr(root, 'text'):
                                                    text_content = root.text
                                                    
                                                    # The text might be a JSON string that needs parsing
                                                    if isinstance(text_content, str):
                                                        # Try to parse as JSON
                                                        try:
                                                            import json
                                                            import ast
                                                            # Try json.loads first
                                                            try:
                                                                parsed = json.loads(text_content)
                                                            except:
                                                                # Fallback to ast.literal_eval for Python dict strings
                                                                parsed = ast.literal_eval(text_content)
                                                            
                                                            # Now extract from the parsed structure
                                                            if isinstance(parsed, dict):
                                                                # Check if it's a UTE-wrapped message
                                                                if 'params' in parsed and 'message' in parsed['params']:
                                                                    inner_parts = parsed['params']['message'].get('parts', [])
                                                                    if inner_parts and len(inner_parts) > 0:
                                                                        inner_part = inner_parts[0]
                                                                        if isinstance(inner_part, dict):
                                                                            inner_text = inner_part.get('text', {})
                                                                            if isinstance(inner_text, dict):
                                                                                # Extract actual question/text
                                                                                actual_text = inner_text.get('text') or inner_text.get('question')
                                                                                if actual_text:
                                                                                    result = str(actual_text)
                                                                                    return result
                                                                
                                                                # Fallback: try direct text/question keys
                                                                actual_text = parsed.get('text') or parsed.get('question') or parsed.get('content')
                                                                if actual_text:
                                                                    result = str(actual_text)
                                                                    return result
                                                        except Exception as e:
                                                            # If parsing fails, use the raw string
                                                            result = text_content
                                                            return result
                                                    
                                                    result = str(text_content)
                                                    return result
                                
                                return ""
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                return result if result else ""
                    
                    # Call wrapped executor with adapted context and wrapped event_queue
                    wrapped_context = ContextWrapper(context)
                    wrapped_queue = EventQueueWrapper(event_queue)
                    try:
                        await self.shard_executor.execute(wrapped_context, wrapped_queue)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise
            
            executor_wrapper = A2AExecutorWrapper(self.native_executor)
            
            # 3) Create BaseAgent with A2A server using wrapped executor
            self.base_agent = await BaseAgent.create_a2a(
                agent_id=self.agent_id,
                host=host,
                port=None,  # Use dynamic port allocation
                executor=executor_wrapper  # Use wrapped executor
            )
            
            # 4) Optional loopback adapter (for local testing only)
            if self.install_loopback:
                try:
                    from src.agent_adapters.a2a_adapter import A2AAdapter
                    listen_url = self.base_agent.get_listening_address()
                    
                    # Fix URL for Windows - replace 0.0.0.0 with 127.0.0.1
                    if "0.0.0.0" in listen_url:
                        listen_url = listen_url.replace("0.0.0.0", "127.0.0.1")
                    
                    adapter = A2AAdapter(
                        httpx_client=self.base_agent._httpx_client,
                        base_url=listen_url
                    )
                    await adapter.initialize()
                    self.base_agent.add_outbound_adapter(self.agent_id, adapter)
                    logger.info(f"Installed loopback A2A adapter at {listen_url}")
                except ImportError as e:
                    logger.warning(f"Cannot install loopback adapter: A2A adapter not available ({e})")
                except Exception as e:
                    logger.error(f"Failed to install loopback adapter: {e}")
                    # Don't raise - loopback is optional

            # 5) Diagnostics
            logger.info(f"Agent URL: {self.base_agent.get_listening_address()}")
            logger.debug(f"Agent Card: {self.base_agent.get_card()}")
            return self.base_agent
            
        except Exception as e:
            logger.error(f"Failed to create A2A meta worker: {e}")
            raise
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Return health info using BaseAgent public API."""
        if not self.base_agent:
            return {"status": "not_initialized"}
        try:
            is_healthy = await self.base_agent.health_check()
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "agent_id": self.agent_id,
                "protocol": "a2a",
                "url": self.base_agent.get_listening_address(),
                "meta_protocol": "ute",
                "native_agent_status": self.a2a_agent.get_status() if self.a2a_agent else None
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def close(self):
        """Close the A2A meta agent"""
        if self.base_agent:
            await self.base_agent.stop()
        logger.info(f"Closed A2A BaseAgent: {self.agent_id}")


async def create_a2a_meta_worker(
    agent_id: str, 
    config: Dict[str, Any], 
    host: str = "0.0.0.0", 
    port: Optional[int] = None,
    install_loopback: bool = False
) -> A2AMetaAgent:
    """
    Factory function to create A2A meta protocol worker for network integration
    
    Args:
        agent_id: Unique agent identifier
        config: Configuration dict with core.openai_api_key etc.
        host: Server host
        port: Server port (auto-assigned if None)
        install_loopback: Whether to install loopback adapter (default False for production)
    
    Returns:
        Initialized A2AMetaAgent instance ready for network registration
    """
    agent = A2AMetaAgent(agent_id, config, install_loopback)
    await agent.create_a2a_worker(host=host, port=port)
    
    logger.info(f"Created A2A meta worker: {agent_id}")
    return agent


async def integrate_a2a_into_network(network, agent_id: str, config: Dict[str, Any], port: Optional[int] = None) -> str:
    """
    Integrate A2A worker into existing NetworkBase
    
    Args:
        network: NetworkBase instance for registration
        agent_id: Agent identifier
        config: Configuration dict
        port: Server port
    
    Returns:
        Agent listening URL for network registration
    """
    # Create A2A meta agent (no loopback for network deployment)
    a2a_agent = await create_a2a_meta_worker(agent_id, config, port=port, install_loopback=False)
    
    # Register with network
    agent_url = a2a_agent.base_agent.get_listening_address()
    await network.register_agent(agent_id, agent_url)
    
    logger.info(f"Integrated A2A agent {agent_id} into network at {agent_url}")
    return agent_url
