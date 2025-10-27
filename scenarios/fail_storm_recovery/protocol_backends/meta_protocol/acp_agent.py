"""
ACP Meta Agent - Production Version

Production-ready ACP agent integration with src/core/base_agent.py Meta-Protocol system.
Based on fail_storm_recovery ACP agent implementation with ACP SDK native features.
"""

import os
import asyncio
import uuid
import logging
from typing import Dict, Any, Optional, AsyncGenerator

# Configure logging with consistent format
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Production imports - add proper paths
import sys
from pathlib import Path

# Add paths
current_file = Path(__file__).resolve()
fail_storm_path = current_file.parents[2]  # Go up to fail_storm_recovery
project_root = fail_storm_path.parent.parent  # Go up to agent_network
src_path = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(fail_storm_path))
sys.path.insert(0, str(src_path))

try:
    from src.core.base_agent import BaseAgent
except ImportError as e:
    raise ImportError(
        f"BaseAgent is required but not available: {e}. "
        "Please ensure src/core/base_agent.py is available."
    )

try:
    from protocol_backends.acp.agent import ACPAgent, create_acp_agent
except ImportError as e:
    raise ImportError(
        f"ACP native agent is required but not available: {e}. "
        "Please ensure protocol_backends/acp/agent.py is available."
    )

# ACP SDK imports
try:
    from acp_sdk.models import Message, MessagePart
    from acp_sdk.server import Context, RunYield
except ImportError as e:
    raise ImportError(
        f"ACP SDK is required but not available: {e}. "
        "Please install with: pip install acp-sdk"
    )

logger = logging.getLogger(__name__)


class ACPExecutorWrapper:
    """
    Wrapper to adapt fail_storm ACPAgent to ACP server async generator interface
    
    Converts: ACPAgent -> ACP SDK async generator interface
    To: executor(messages, context) -> AsyncGenerator[RunYield, None]
    """
    
    def __init__(self, shard_worker_executor):
        self.shard_worker_executor = shard_worker_executor
        self.capabilities = ["text_processing", "async_generation", "acp_sdk_native", "fail_storm_integration"]
    
    async def __call__(self, messages: list[Message], context: Context) -> AsyncGenerator[RunYield, None]:
        """
        ACP server async generator interface - called directly by ACP server adapter
        RunYield is Union type, so we yield Message/MessagePart/str directly
        """
        logger.debug(f"[ACP] ACPExecutorWrapper called with {len(messages)} messages")
        
        try:
            # Process each message (typically just one)
            for i, message in enumerate(messages):
                # Generate run_id for this execution
                run_id = str(uuid.uuid4())
                
                logger.debug(f"[ACP] Processing message {i+1}/{len(messages)} with run_id: {run_id}")
                
                # Extract text from message for processing
                text_content = ""
                
                if hasattr(message, 'parts') and message.parts:
                    for part in message.parts:
                        if hasattr(part, 'type') and part.type == "text":
                            text_content += getattr(part, 'text', getattr(part, 'content', ""))
                else:
                    text_content = str(message)
                
                # Use ShardWorkerExecutor directly for QA processing
                if hasattr(self.shard_worker_executor, 'worker') and hasattr(self.shard_worker_executor.worker, 'answer'):
                    result = await self.shard_worker_executor.worker.answer(text_content)
                    yield result  # Yield LLM result
                else:
                    yield f"ACP processing: {text_content}"
                
                logger.debug(f"[ACP] Executor yielded result for message {i+1}")
                
        except Exception as e:
            # Yield error as string
            error_msg = f"ACP processing error: {e}"
            yield error_msg
            logger.error(f"[ACP] Executor error: {e}", exc_info=True)


class ACPMetaAgent:
    """
    ACP Agent integrated with Meta-Protocol system
    
    Uses BaseAgent.create_acp() to wrap fail_storm ACPAgent into a Meta-Protocol node
    with UTE encoding/decoding and ACP SDK native features.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], install_loopback: bool = False):
        self.agent_id = agent_id
        self.config = config
        self.install_loopback = install_loopback
        self.base_agent: Optional[BaseAgent] = None
        self.acp_agent: Optional[ACPAgent] = None
        self.executor_wrapper: Optional[ACPExecutorWrapper] = None
        
        logger.info(f"[ACP-META] Initialized agent: {agent_id}")
    
    def _convert_config_for_executor(self) -> Dict[str, Any]:
        """Convert config for fail_storm ACP executor"""
        # Support both 'llm' (new) and 'core' (legacy) config keys
        llm_config = self.config.get("llm") or self.config.get("core", {})
        if llm_config.get("type") in ["openai", "local"]:
            # Prioritize environment variables
            api_key = os.getenv("OPENAI_API_KEY") or llm_config.get("openai_api_key")
            base_url = os.getenv("OPENAI_BASE_URL") or llm_config.get("openai_base_url", "https://api.openai.com/v1")
            return {
                "model": {
                    "type": llm_config.get("type", "openai"),
                    "name": llm_config.get("model") or llm_config.get("name", "gpt-4o"),
                    "openai_api_key": api_key,
                    "openai_base_url": base_url,
                    "temperature": llm_config.get("temperature", 0.0),
                    "max_tokens": llm_config.get("max_tokens", 4096)
                }
            }
        return {"model": llm_config}
    
    async def create_acp_worker(self, host: str = "0.0.0.0", port: Optional[int] = None):
        """
        Start an ACP server via BaseAgent.create_acp().
        Uses fail_storm ACP native implementation with async generator wrapper.
        """
        try:
            # 1) Create native ACP executor from shard_qa pattern
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
            native_executor = ShardWorkerExecutor(
                config=executor_config,
                global_config=global_config,
                shard_id=self.agent_id,
                data_file=None,
                neighbors=[],
                output=None,
                force_llm=True
            )
            
            # 2) Wrap in async generator interface for ACP server
            self.executor_wrapper = ACPExecutorWrapper(native_executor)
            
            if not callable(self.executor_wrapper):
                raise RuntimeError("ACPExecutorWrapper must be callable as async generator")

            # 4) Create BaseAgent with ACP server adapter (using factory method)
            logger.info(f"[{self.agent_id}] Creating BaseAgent.create_acp on {host}:{port or 8082}")
            self.base_agent = await BaseAgent.create_acp(
                agent_id=self.agent_id,
                host=host,
                port=None,  # Use dynamic port allocation
                executor=self.executor_wrapper  # callable async generator
                # server_adapter defaults to ACPServerAdapter
            )
            logger.info(f"[{self.agent_id}] BaseAgent ACP server created successfully at {self.base_agent.get_listening_address()}")

            # 5) Optional loopback adapter (for local testing only - default False for production)
            if self.install_loopback:
                try:
                    from src.agent_adapters.acp_adapter import ACPAdapter
                    listen_url = self.base_agent.get_listening_address()
                    
                    # Fix URL for Windows - replace 0.0.0.0 with 127.0.0.1
                    if "0.0.0.0" in listen_url:
                        listen_url = listen_url.replace("0.0.0.0", "127.0.0.1")
                    
                    adapter = ACPAdapter(
                        httpx_client=self.base_agent._httpx_client,
                        base_url=listen_url,
                        agent_id=self.agent_id
                    )
                    await adapter.initialize()
                    self.base_agent.add_outbound_adapter(self.agent_id, adapter)
                    logger.info(f"[{self.agent_id}] Installed loopback ACP adapter at {listen_url}")
                except ImportError as e:
                    logger.warning(f"[{self.agent_id}] Cannot install loopback adapter: ACP adapter not available ({e})")
                except Exception as e:
                    logger.error(f"[{self.agent_id}] Failed to install loopback adapter: {e}")
                    # Don't raise - loopback is optional

            # 6) Diagnostics
            logger.info(f"Agent URL: {self.base_agent.get_listening_address()}")
            logger.debug(f"Agent Card: {self.base_agent.get_card()}")
            return self.base_agent
            
        except Exception as e:
            logger.error(f"[ACP-META] Failed to create ACP worker: {e}")
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
                "protocol": "acp",
                "url": self.base_agent.get_listening_address(),
                "meta_protocol": "ute",
                "sdk_version": "native",
                "native_agent_status": self.acp_agent.get_status() if self.acp_agent else None
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def close(self):
        """Close the ACP meta agent"""
        if self.base_agent:
            await self.base_agent.stop()
        logger.info(f"Closed ACP BaseAgent: {self.agent_id}")


async def create_acp_meta_worker(
    agent_id: str, 
    config: Dict[str, Any], 
    host: str = "0.0.0.0", 
    port: Optional[int] = None,
    install_loopback: bool = False
) -> ACPMetaAgent:
    """
    Factory function to create ACP meta protocol worker for network integration
    
    Args:
        agent_id: Unique agent identifier
        config: Configuration dict with core.openai_api_key etc.
        host: Server host
        port: Server port (auto-assigned if None)
        install_loopback: Whether to install loopback adapter (default False for production)
    
    Returns:
        Initialized ACPMetaAgent instance ready for network registration
    """
    agent = ACPMetaAgent(agent_id, config, install_loopback)
    try:
        await agent.create_acp_worker(host=host, port=port)
        logger.info(f"[ACP-META] Created meta worker: {agent_id}")
        return agent
    except Exception as e:
        logger.error(f"[ACP-META] Failed to create worker for {agent_id}: {e}")
        raise


async def integrate_acp_into_network(network, agent_id: str, config: Dict[str, Any], port: Optional[int] = None) -> str:
    """
    Integrate ACP worker into existing NetworkBase
    
    Args:
        network: NetworkBase instance for registration
        agent_id: Agent identifier
        config: Configuration dict
        port: Server port
    
    Returns:
        Agent listening URL for network registration
    """
    # Create ACP meta agent (no loopback for network deployment)
    acp_agent = await create_acp_meta_worker(agent_id, config, port=port, install_loopback=False)
    
    # Register with network
    agent_url = acp_agent.base_agent.get_listening_address()
    await network.register_agent(agent_id, agent_url)
    
    logger.info(f"[ACP-META] Integrated agent {agent_id} into network at {agent_url}")
    return agent_url
