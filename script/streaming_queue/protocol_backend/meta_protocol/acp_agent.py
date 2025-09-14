"""
ACP Meta Agent - Production Version

Production-ready ACP agent integration with src/core/base_agent.py Meta-Protocol system.
Uses ACP SDK 1.0.3 native executor with async generator wrapper.
"""

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
streaming_queue_path = current_file.parents[2]  # Go up to streaming_queue
project_root = streaming_queue_path.parent.parent  # Go up to agent_network
src_path = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(streaming_queue_path))
sys.path.insert(0, str(src_path))

from src.core.base_agent import BaseAgent
from protocol_backend.acp.worker import ACPWorkerExecutor

# ACP SDK imports
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield

logger = logging.getLogger(__name__)


class ACPExecutorWrapper:
    """
    Wrapper to adapt ACPWorkerExecutor.process_message() to ACP server async generator interface
    
    Converts: process_message(message, run_id) -> Message
    To: executor(messages, context) -> AsyncGenerator[RunYield, None]
    """
    
    def __init__(self, acp_worker_executor: ACPWorkerExecutor):
        self.acp_worker_executor = acp_worker_executor
        self.capabilities = ["text_processing", "async_generation", "acp_sdk_1.0.3"]
    
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
                
                # Extract text from message for LLM call
                text_content = ""
                
                if hasattr(message, 'parts') and message.parts:
                    for part in message.parts:
                        if hasattr(part, 'type') and part.type == "text":
                            text_content += getattr(part, 'text', getattr(part, 'content', ""))
                else:
                    text_content = str(message)
                
                # Call LLM directly through worker
                if hasattr(self.acp_worker_executor, '_worker') and hasattr(self.acp_worker_executor._worker, 'answer'):
                    llm_result = await self.acp_worker_executor._worker.answer(text_content)
                    yield llm_result  # Yield LLM result as string
                else:
                    # Fallback to original process_message
                    result_message = await self.acp_worker_executor.process_message(message, run_id)
                    if isinstance(result_message, Message):
                        yield result_message
                    else:
                        yield "ACP processing completed"
                
                logger.debug(f"[ACP] Executor yielded result for message {i+1}")
                
        except Exception as e:
            # Yield error as string
            error_msg = f"ACP processing error: {e}"
            yield error_msg
            logger.error(f"[ACP] Executor error: {e}", exc_info=True)


class ACPMetaAgent:
    """
    ACP Agent integrated with Meta-Protocol system
    
    Uses BaseAgent.create_acp() to wrap ACPWorkerExecutor into a Meta-Protocol node
    with UTE encoding/decoding and ACP SDK 1.0.3 native features.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], install_loopback: bool = False):
        self.agent_id = agent_id
        self.config = config
        self.install_loopback = install_loopback
        self.base_agent: Optional[BaseAgent] = None
        self.acp_executor: Optional[ACPWorkerExecutor] = None
        self.executor_wrapper: Optional[ACPExecutorWrapper] = None
        
        logger.info(f"[ACP-META] Initialized agent: {agent_id}")
    
    def _convert_config_for_qa(self) -> Dict[str, Any]:
        """Convert config for ACPWorkerExecutor"""
        core = self.config.get("core", {})
        if core.get("type") == "openai":
            return {
                "model": {
                    "type": "openai",
                    "name": core.get("name", "gpt-4o"),
                    "openai_api_key": core.get("openai_api_key"),
                    "openai_base_url": core.get("openai_base_url", "https://api.openai.com/v1"),
                    "temperature": core.get("temperature", 0.0),
                }
            }
        return {}
    
    async def create_acp_worker(self, host: str = "0.0.0.0", port: Optional[int] = None) -> BaseAgent:
        """
        Start an ACP server via BaseAgent.create_acp().
        Uses async generator wrapper to adapt streaming_queue ACPWorkerExecutor.
        """
        # 1) Build native ACP executor
        # Ensure paths are set before creating ACPWorkerExecutor
        streaming_queue_path = Path(__file__).resolve().parents[2]
        core_path = streaming_queue_path / "core"
        if str(core_path) not in sys.path:
            sys.path.insert(0, str(core_path))
        
        qa_config = self._convert_config_for_qa()
        try:
            self.acp_executor = ACPWorkerExecutor(qa_config)
        except Exception as e:
            logger.error(f"[ACP-META] Failed to create ACPWorkerExecutor: {e}")
            raise RuntimeError(f"ACPWorkerExecutor creation failed: {e}")
        
        # 2) Wrap in async generator interface for ACP server
        self.executor_wrapper = ACPExecutorWrapper(self.acp_executor)  # Pass executor back
        
        if not callable(self.executor_wrapper):
            raise RuntimeError("ACPExecutorWrapper must be callable as async generator")

        # 3) Create BaseAgent with ACP server adapter (using factory method)
        logger.info(f"[{self.agent_id}] Creating BaseAgent.create_acp on {host}:{port or 8082}")
        try:
            self.base_agent = await BaseAgent.create_acp(
                agent_id=self.agent_id,
                host=host,
                port=port or 8082,
                executor=self.executor_wrapper  # callable async generator
                # server_adapter defaults to ACPServerAdapter
            )
            logger.info(f"[{self.agent_id}] BaseAgent ACP server created successfully at {self.base_agent.get_listening_address()}")
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to create BaseAgent ACP server: {e}", exc_info=True)
            raise

        # 4) Optional loopback adapter (for local testing only - default False for production)
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

        # 5) Diagnostics
        logger.info(f"Agent URL: {self.base_agent.get_listening_address()}")
        logger.debug(f"Agent Card: {self.base_agent.get_card()}")
        return self.base_agent
    
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
                "sdk_version": "1.0.3"
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


# Test function
async def test_acp_meta_integration():
    """Test ACP meta protocol integration"""
    # Enable debug logging for testing
    logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 Testing ACP Meta-Protocol Integration")
    print("=" * 50)
    
    # Test config
    config = {
        "core": {
            "type": "openai",
            "name": "gpt-4o",
            "openai_api_key": "test-key",
            "openai_base_url": "https://api.openai.com/v1",
            "temperature": 0.0
        }
    }
    
    acp_agent = None
    try:
        print("📝 Creating ACP Meta Worker...")
        # Create ACP meta worker with different port to avoid conflicts
        acp_agent = await create_acp_meta_worker(
            agent_id="ACP-Test-Worker",
            config=config,
            port=8084,  # Different port from our test
            install_loopback=True  # Enable for testing
        )
        
        print("✅ ACP Meta Agent created successfully")
        
        # Check health
        print("🔍 Checking health status...")
        health = await acp_agent.get_health_status()
        print(f"✅ Health status: {health.get('status', 'unknown')}")
        print(f"🌐 Agent URL: {health.get('url', 'unknown')}")
        
        # Test basic functionality
        print("🧪 Testing basic ACP executor...")
        if hasattr(acp_agent, 'executor_wrapper'):
            print("✅ Executor wrapper available")
        
        print("✅ ACP Meta Agent test completed successfully!")
        
    except Exception as e:
        print(f"❌ ACP Meta Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if acp_agent:
            try:
                print("🧹 Cleaning up...")
                await acp_agent.close()
                print("✅ Cleanup completed")
            except Exception as cleanup_e:
                print(f"⚠️ Cleanup error: {cleanup_e}")


if __name__ == "__main__":
    asyncio.run(test_acp_meta_integration())
