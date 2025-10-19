"""
Agora Meta Agent - Production Version

Production-ready Agora agent integration with src/core/base_agent.py Meta-Protocol system.
Based on fail_storm_recovery Agora agent implementation with native SDK optimizations.
"""

import asyncio
import uuid
import logging
from typing import Dict, Any, Optional, AsyncGenerator

# Configure logging with consistent format
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Production imports
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

from src.core.base_agent import BaseAgent
from protocol_backends.agora.agent import AgoraAgent, create_agora_agent

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
    Adapter that makes fail_storm AgoraAgent usable by the Agora server adapter.

    It accepts a dict-like context and pushes one or more events to the server's
    event queue. The queue API can vary by adapter:
      - enqueue_event(event)  (sync or async)
      - put_nowait(event) / put(event)  (asyncio.Queue-like)
    This wrapper handles all of them safely.
    """

    def __init__(self, shard_worker_executor):
        self.shard_worker_executor = shard_worker_executor
        self.capabilities = ["text_processing", "protocol_optimization", "efficiency_enhancement", "agora_native", "fail_storm_integration"]

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
            
            # Use ShardWorkerExecutor directly for QA processing
            if hasattr(self.shard_worker_executor, 'worker') and hasattr(self.shard_worker_executor.worker, 'answer'):
                result_text = await self.shard_worker_executor.worker.answer(text)
            else:
                result_text = f"Agora processing: {text}"

            await _send_event(event_queue, {
                "type": "text",
                "content": result_text,
                "metadata": {
                    "protocol": "agora",
                    "enhanced": True,
                    "sdk_version": "native",
                    "optimized": True,
                    "fail_storm_integration": True
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
    
    This agent uses fail_storm Agora native implementation for protocol optimizations and efficiency enhancements,
    while integrating with the BaseAgent meta-protocol framework.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], install_loopback: bool = False):
        self.agent_id = agent_id
        self.config = config
        self.install_loopback = install_loopback
        self.base_agent: Optional[BaseAgent] = None
        self.agora_agent: Optional[AgoraAgent] = None
        self.executor_wrapper: Optional[AgoraExecutorWrapper] = None
        
        logger.info(f"[AGORA-META] Initialized agent: {agent_id}")
    
    def _convert_config_for_executor(self) -> Dict[str, Any]:
        """Convert config for fail_storm Agora executor"""
        # Support both 'llm' (new) and 'core' (legacy) config keys
        llm_config = self.config.get("llm") or self.config.get("core", {})
        if llm_config.get("type") in ["openai", "local"]:
            return {
                "model": {
                    "type": llm_config.get("type", "openai"),
                    "name": llm_config.get("model") or llm_config.get("name", "gpt-4o"),
                    "openai_api_key": llm_config.get("openai_api_key"),
                    "openai_base_url": llm_config.get("openai_base_url"),
                    "temperature": llm_config.get("temperature", 0.0),
                    "max_tokens": llm_config.get("max_tokens", 4096)
                }
            }
        return {"model": llm_config}
    
    async def create_agora_worker(self, host: str = "0.0.0.0", port: Optional[int] = None) -> BaseAgent:
        """Create Agora worker integrated with BaseAgent"""
        
        try:
            # 1) Create native Agora executor from shard_qa pattern
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
            
            # 2) Wrap in executor interface for Agora server
            self.executor_wrapper = AgoraExecutorWrapper(native_executor)
            
            if not callable(getattr(self.executor_wrapper, 'execute', None)):
                raise RuntimeError("AgoraExecutorWrapper must implement execute(context, event_queue) method")

            # 4) Create BaseAgent with Agora server adapter (using factory method)
            logger.info(f"[{self.agent_id}] Creating BaseAgent.create_agora on {host}:{port or 8086}")
            self.base_agent = await BaseAgent.create_agora(
                agent_id=self.agent_id,
                host=host,
                port=None,  # Use dynamic port allocation
                executor=self.executor_wrapper
            )
            logger.info(f"[{self.agent_id}] BaseAgent Agora server created successfully at {self.base_agent.get_listening_address()}")

            # 5) Optional loopback adapter (for local testing only - default False for production)
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

            # 6) Diagnostics
            logger.info(f"Agent URL: {self.base_agent.get_listening_address()}")
            logger.debug(f"Agent Card: {self.base_agent.get_card()}")
            return self.base_agent
            
        except Exception as e:
            logger.error(f"[AGORA-META] Failed to create Agora worker: {e}")
            raise

    async def get_health_status(self) -> Dict[str, Any]:
        """Return best-effort health info for Agora server."""
        if not self.base_agent:
            return {"status": "not_initialized", "url": None}
        try:
            is_running = self.base_agent.is_server_running()
            url = self.base_agent.get_listening_address()
            if "0.0.0.0" in url:
                url = url.replace("0.0.0.0", "127.0.0.1")
            return {
                "status": "healthy" if is_running else "unhealthy", 
                "url": url,
                "agent_id": self.agent_id,
                "protocol": "agora",
                "meta_protocol": "ute",
                "native_agent_status": self.agora_agent.get_status() if self.agora_agent else None
            }
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
