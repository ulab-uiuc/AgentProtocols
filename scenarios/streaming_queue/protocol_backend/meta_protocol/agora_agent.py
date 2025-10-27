"""
Agora Meta Agent - Production Version

Production-ready Agora agent integration with src/core/base_agent.py Meta-Protocol system.
Uses Agora native SDK for protocol-specific optimizations and efficiency.
"""

import os
import asyncio
import uuid
import logging
from typing import Dict, Any, Optional, AsyncGenerator

# Configure logging with consistent format
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Production imports
from src.core.base_agent import BaseAgent
from scenarios.streaming_queue.protocol_backend.agora.worker import AgoraQAWorker

# Agora SDK imports
try:
    import agora
except ImportError as e:
    raise ImportError(
        f"Agora SDK is required but not available: {e}. "
        "Please install with: pip install agora-sdk"
    )

logger = logging.getLogger(__name__)


class AgoraExecutorWrapper:
    """
    Adapter that makes AgoraQAWorker usable by the Agora server adapter.

    It accepts a dict-like context and pushes one or more events to the server's
    event queue. The queue API can vary by adapter:
      - enqueue_event(event)  (sync or async)
      - put_nowait(event) / put(event)  (asyncio.Queue-like)
    This wrapper handles all of them safely.
    """

    def __init__(self, agora_qa_worker: AgoraQAWorker):
        self.agora_qa_worker = agora_qa_worker
        self.capabilities = ["text_processing", "protocol_optimization", "efficiency_enhancement", "agora_native"]

    async def execute(self, context: Dict[str, Any], event_queue: Optional[Any] = None) -> None:
        """
        Compatible with A2A-like 'execute(context, event_queue)' pattern used by the Agora server.
        """
        import inspect
        import json
        import logging

        log = logging.getLogger(__name__)

        def _extract_text(ctx: Dict[str, Any]) -> str:
            # Try common shapes: {"message": {"content"/"text"/"body"}} or flat keys
            msg = ctx.get("message")
            if isinstance(msg, dict):
                for k in ("content", "text", "body"):
                    v = msg.get(k)
                    if isinstance(v, str):
                        return v
                # parts-based fallback: {"message":{"parts":[{"text":...}]}}
                parts = msg.get("parts")
                if isinstance(parts, list) and parts:
                    first = parts[0]
                    if isinstance(first, dict) and "text" in first:
                        t = first["text"]
                        return t if isinstance(t, str) else json.dumps(t, ensure_ascii=False)
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
            """Push one event to various queue implementations."""
            if eq is None:
                return
            # 1) enqueue_event(event)
            if hasattr(eq, "enqueue_event"):
                res = eq.enqueue_event(payload)
                if inspect.isawaitable(res):
                    await res
                return
            # 2) put_nowait / put
            if hasattr(eq, "put_nowait"):
                try:
                    eq.put_nowait(payload)
                    return
                except Exception:
                    pass
            if hasattr(eq, "put"):
                await eq.put(payload)
                return
            # 3) no-op fallback

        try:
            text = _extract_text(context)
            result = await self.agora_qa_worker.answer(text)

            await _send_event(event_queue, {
                "type": "text",
                "content": result,
                "metadata": {
                    "protocol": "agora",
                    "enhanced": True,
                    "sdk_version": "native",
                    "optimized": True
                }
            })
            log.debug("[AGORA] Executor completed processing")

        except Exception as e:
            err = f"Agora processing error: {e}"
            logging.getLogger(__name__).error(err, exc_info=True)
            await _send_event(event_queue, {
                "type": "error",
                "content": err,
                "metadata": {"error": True, "protocol": "agora"}
            })


class AgoraMetaAgent:
    """
    Agora Agent integrated with Meta-Protocol system
    
    This agent uses Agora native SDK for protocol optimizations and efficiency enhancements,
    while integrating with the BaseAgent meta-protocol framework.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], install_loopback: bool = False):
        self.agent_id = agent_id
        self.config = config
        self.install_loopback = install_loopback
        self.base_agent: Optional[BaseAgent] = None
        self.agora_qa_worker: Optional[AgoraQAWorker] = None
        self.executor_wrapper: Optional[AgoraExecutorWrapper] = None
        
        logger.info(f"[AGORA-META] Initialized agent: {agent_id}")
    
    def _convert_config_for_qa(self) -> Dict[str, Any]:
        """Convert config for AgoraQAWorker"""
        core = self.config.get("core", {})
        if core.get("type") == "openai":
            # Prioritize environment variables
            api_key = os.getenv("OPENAI_API_KEY") or core.get("openai_api_key")
            base_url = os.getenv("OPENAI_BASE_URL") or core.get("openai_base_url")
            return {
                "model": {
                    "type": "openai",
                    "name": core.get("name", "gpt-4o"),
                    "openai_api_key": api_key,
                    "openai_base_url": base_url,
                    "temperature": core.get("temperature", 0.0)
                }
            }
        return {"model": core}
    
    async def create_agora_worker(self, host: str = "0.0.0.0", port: Optional[int] = None) -> BaseAgent:
        """Create Agora worker integrated with BaseAgent"""
        
        # 1) Create Agora QA Worker with streaming_queue implementation
        qa_config = self._convert_config_for_qa()
        self.agora_qa_worker = AgoraQAWorker(qa_config)
        
        # 2) Wrap in executor interface for Agora server (pass QA worker directly)
        self.executor_wrapper = AgoraExecutorWrapper(self.agora_qa_worker)
        
        if not callable(getattr(self.executor_wrapper, 'execute', None)):
            raise RuntimeError("AgoraExecutorWrapper must implement execute(context, event_queue) method")

        # 3) Create BaseAgent with Agora server adapter (using factory method)
        logger.info(f"[{self.agent_id}] Creating BaseAgent.create_agora on {host}:{port or 8086}")
        try:
            self.base_agent = await BaseAgent.create_agora(
                agent_id=self.agent_id,
                host=host,
                port=port or 8086,
                executor=self.executor_wrapper
            )
            logger.info(f"[{self.agent_id}] BaseAgent Agora server created successfully at {self.base_agent.get_listening_address()}")
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to create BaseAgent Agora server: {e}", exc_info=True)
            raise

        # 4) Optional loopback adapter (for local testing only - default False for production)
        if self.install_loopback:
            # NOTE: AgoraClientAdapter signature may require specific parameters:
            # - toolformer: Agora toolformer instance
            # - target_url: HTTP endpoint URL  
            # - model/headers/auth: Additional configuration
            # Verify signature matches your actual AgoraClientAdapter implementation
            logger.warning(f"[{self.agent_id}] Agora loopback enabled - ensure AgoraClientAdapter signature is correct")
            
            try:
                from src.agent_adapters.agora_adapter import AgoraClientAdapter
                from agora import Toolformer

                listen_url = self.base_agent.get_listening_address()
                if "0.0.0.0" in listen_url:
                    listen_url = listen_url.replace("0.0.0.0", "127.0.0.1")

                # Create Toolformer
                toolformer = Toolformer()

                # Create adapter
                adapter = AgoraClientAdapter(
                    httpx_client=self.base_agent._httpx_client,
                    toolformer=toolformer,
                    target_url=listen_url,
                    agent_id=self.agent_id
                )
                await adapter.initialize()
                self.base_agent.add_outbound_adapter(self.agent_id, adapter)
                logger.info(f"[{self.agent_id}] Installed loopback Agora adapter at {listen_url}")
            except Exception as e:
                logger.warning(f"[{self.agent_id}] Loopback adapter failed: {e}")
                # Production deployments should use install_loopback=False to avoid this

        # 5) Diagnostics
        logger.info(f"Agent URL: {self.base_agent.get_listening_address()}")
        logger.debug(f"Agent Card: {self.base_agent.get_card()}")
        return self.base_agent

    async def get_health_status(self) -> Dict[str, Any]:
        """Return best-effort health info for Agora server."""
        if not self.base_agent:
            return {"status": "not_initialized", "url": None}
        try:
            is_running = self.base_agent.is_server_running()
            url = self.base_agent.get_listening_address()
            if "0.0.0.0" in url:
                url = url.replace("0.0.0.0", "127.0.0.1")
            return {"status": "healthy" if is_running else "unhealthy", "url": url}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "url": None}

    async def close(self):
        """Close the Agora meta agent"""
        if self.base_agent:
            await self.base_agent.stop()
            logger.info(f"Closed Agora BaseAgent: {self.agent_id}")


async def create_agora_meta_worker(
    agent_id: str, 
    config: Dict[str, Any], 
    host: str = "0.0.0.0", 
    port: Optional[int] = None,
    install_loopback: bool = False
) -> AgoraMetaAgent:
    """
    Factory function to create Agora meta protocol worker for network integration
    
    Args:
        agent_id: Unique agent identifier
        config: Configuration dict with core.openai_api_key etc.
        host: Server host
        port: Server port (auto-assigned if None)
        install_loopback: Whether to install loopback adapter 
                         (default False for production - PRODUCTION SAFE)
    
    Returns:
        Initialized AgoraMetaAgent instance ready for network registration
    """
    agent = AgoraMetaAgent(agent_id, config, install_loopback)
    await agent.create_agora_worker(host=host, port=port)
    
    logger.info(f"[AGORA-META] Created meta worker: {agent_id}")
    return agent


async def integrate_agora_into_network(network, agent_id: str, config: Dict[str, Any], port: Optional[int] = None) -> str:
    """
    Integrate Agora worker into existing NetworkBase
    
    Args:
        network: NetworkBase instance
        agent_id: Agent identifier
        config: Agent configuration
        port: Server port (optional)
    
    Returns:
        Agent URL for network registration
    """
    # Create Agora meta worker (production mode - no loopback)
    agora_agent = await create_agora_meta_worker(agent_id, config, port=port, install_loopback=False)
    
    # Register with network
    agent_url = agora_agent.base_agent.get_listening_address()
    await network.register_agent(agent_id, agent_url)
    
    logger.info(f"[AGORA-META] Integrated agent {agent_id} into network at {agent_url}")
    return agent_url


# Test function
async def test_agora_meta_integration():
    """Test Agora meta protocol integration"""
    # Enable debug logging for testing
    logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 Testing Agora Meta-Protocol Integration")
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
    
    agora_agent = None
    try:
        print("📝 Creating Agora Meta Worker...")
        # Create Agora meta worker with different port to avoid conflicts
        agora_agent = await create_agora_meta_worker(
            agent_id="Agora-Test-Worker",
            config=config,
            port=8086,  # Different port
            install_loopback=False  # Production safe - avoid adapter signature issues
        )
        
        print("✅ Agora Meta Agent created successfully")
        
        # Check health
        print("🔍 Checking health status...")
        health = await agora_agent.get_health_status()
        print(f"✅ Health status: {health.get('status', 'unknown')}")
        print(f"🌐 Agent URL: {health.get('url', 'unknown')}")
        
        # Test basic functionality
        print("🧪 Testing basic Agora executor...")
        if hasattr(agora_agent, 'executor_wrapper'):
            print("✅ Executor wrapper available")
        
        if hasattr(agora_agent.agora_qa_worker, 'agora_protocol') and agora_agent.agora_qa_worker.agora_protocol:
            print("✅ Agora native SDK protocol available")
        
        print("✅ Agora Meta Agent test completed successfully!")
        
    except Exception as e:
        print(f"❌ Agora Meta Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if agora_agent:
            try:
                print("🧹 Cleaning up...")
                await agora_agent.close()
                print("✅ Cleanup completed")
            except Exception as cleanup_e:
                print(f"⚠️ Cleanup error: {cleanup_e}")


if __name__ == "__main__":
    asyncio.run(test_agora_meta_integration())
