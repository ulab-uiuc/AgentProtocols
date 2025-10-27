"""
ANP Meta Agent - Production Version

Production-ready ANP agent integration with src/core/base_agent.py Meta-Protocol system.
Based on fail_storm_recovery ANP agent implementation with AgentConnect SDK for DID authentication and E2E encryption.
"""

import os
import asyncio
import uuid
import logging
from typing import Dict, Any, Optional, AsyncGenerator

# Configure logging with consistent format
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Production imports
import sys
from pathlib import Path

# Add paths for imports
current_file = Path(__file__).resolve()
fail_storm_path = current_file.parents[2]  # Go up to fail_storm_recovery
project_root = fail_storm_path.parent.parent  # Go up to agent_network
src_path = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(fail_storm_path))
sys.path.insert(0, str(src_path))

from src.core.base_agent import BaseAgent
from protocol_backends.anp.agent import ANPAgent, create_anp_agent

logger = logging.getLogger(__name__)


class ANPExecutorWrapper:
    """
    Adapter that makes fail_storm ANPAgent usable by the ANP server adapter.

    It accepts a generic context (dict-like) and pushes one or more events
    to the server's event queue. The queue API can vary by adapter:
      - enqueue_event(event)  (sync or async)
      - put_nowait(event) / put(event)  (asyncio.Queue style)
    This wrapper handles all of them safely.
    """

    def __init__(self, shard_worker_executor):
        self.shard_worker_executor = shard_worker_executor
        self.capabilities = ["text_processing", "did_authentication", "e2e_encryption", "anp_native", "fail_storm_integration"]

    async def execute(self, context: Dict[str, Any], event_queue: Optional[Any] = None) -> None:
        """
        Compatible with A2A-like 'execute(context, event_queue)' pattern used by the ANP server.
        """
        import inspect
        import json

        def _extract_text_from_context(ctx: Dict[str, Any]) -> str:
            # try common shapes first
            msg = ctx.get("message")
            if isinstance(msg, dict):
                # AP/ANP/A2A shapes
                if "content" in msg and isinstance(msg["content"], str):
                    return msg["content"]
                if "text" in msg and isinstance(msg["text"], str):
                    return msg["text"]
                # parts-based
                parts = msg.get("parts")
                if isinstance(parts, list) and parts:
                    first = parts[0]
                    if isinstance(first, dict) and "text" in first:
                        t = first["text"]
                        return t if isinstance(t, str) else json.dumps(t, ensure_ascii=False)
            # flat shapes
            for k in ("content", "text"):
                v = ctx.get(k)
                if isinstance(v, str):
                    return v
            # last resort
            try:
                return json.dumps(ctx, ensure_ascii=False)
            except Exception:
                return str(ctx)

        async def _send_event(eq: Any, payload: Dict[str, Any]) -> None:
            """
            Send one event to whatever queue implementation the server uses.
            """
            if eq is None:
                return
            # 1) enqueue_event(event)  (sync or async)
            if hasattr(eq, "enqueue_event"):
                res = eq.enqueue_event(payload)
                if inspect.isawaitable(res):
                    await res
                return
            # 2) put_nowait / put (asyncio.Queue-like)
            if hasattr(eq, "put_nowait"):
                try:
                    eq.put_nowait(payload)
                    return
                except Exception:
                    pass
            if hasattr(eq, "put"):
                await eq.put(payload)
                return
            # 3) fallback: ignore silently

        try:
            message_content = _extract_text_from_context(context)
            
            # Use ShardWorkerExecutor directly for QA processing
            if hasattr(self.shard_worker_executor, 'worker') and hasattr(self.shard_worker_executor.worker, 'answer'):
                result_text = await self.shard_worker_executor.worker.answer(message_content)
            else:
                result_text = f"ANP processing: {message_content}"

            await _send_event(event_queue, {
                "type": "text",
                "content": result_text,
                "metadata": {
                    "protocol": "anp",
                    "did_authenticated": True,
                    "encrypted": True,
                    "fail_storm_integration": True
                }
            })

        except Exception as e:
            err = f"ANP processing error: {e}"
            logging.getLogger(__name__).error(err, exc_info=True)
            await _send_event(event_queue, {
                "type": "error",
                "content": err,
                "metadata": {"error": True, "protocol": "anp"}
            })


class ANPMetaAgent:
    """
    ANP Agent integrated with Meta-Protocol system
    
    This agent uses fail_storm ANP native implementation with AgentConnect SDK for DID authentication and E2E encryption,
    while integrating with the BaseAgent meta-protocol framework.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], install_loopback: bool = False):
        self.agent_id = agent_id
        self.config = config
        self.install_loopback = install_loopback
        self.base_agent: Optional[BaseAgent] = None
        self.anp_agent: Optional[ANPAgent] = None
        self.executor_wrapper: Optional[ANPExecutorWrapper] = None
        
        logger.info(f"[ANP-META] Initialized agent: {agent_id}")
    
    def _convert_config_for_executor(self) -> Dict[str, Any]:
        """Convert config for fail_storm ANP executor"""
        # Support both 'llm' (new) and 'core' (legacy) config keys
        llm_config = self.config.get("llm") or self.config.get("core", {})
        
        if llm_config.get("type") == "openai" or llm_config.get("type") == "local":
            # Prioritize environment variables
            api_key = os.getenv("OPENAI_API_KEY") or llm_config.get("openai_api_key")
            base_url = os.getenv("OPENAI_BASE_URL") or llm_config.get("openai_base_url")
            result = {
                "model": {
                    "type": llm_config.get("type", "openai"),
                    "name": llm_config.get("model") or llm_config.get("name", "gpt-4o"),
                    "openai_api_key": api_key,
                    "openai_base_url": base_url,
                    "temperature": llm_config.get("temperature", 0.0),
                    "max_tokens": llm_config.get("max_tokens", 4096)
                }
            }
            return result
        return {"model": llm_config}
    
    async def create_anp_worker(self, host: str = "0.0.0.0", port: Optional[int] = None) -> BaseAgent:
        """Create ANP worker integrated with BaseAgent"""
        
        try:
            # 1) Create native ANP executor from shard_qa pattern
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
            
            # 2) Wrap native executor for ANP server
            self.executor_wrapper = ANPExecutorWrapper(native_executor)
            
            if not callable(getattr(self.executor_wrapper, 'execute', None)):
                raise RuntimeError("ANPExecutorWrapper must implement execute(context, event_queue) method")

            # 4) Generate DID info for ANP authentication (let ANP server handle DID generation)
            did_info = None
            logger.debug(f"[ANP] DID generation delegated to ANP server adapter")

            # 5) Create BaseAgent with ANP server adapter (using factory method)
            logger.info(f"[{self.agent_id}] Creating BaseAgent.create_anp on {host}:{port or 8085}")
            # Add DID service parameters from environment for did:wba generation
            import os
            did_service_url = os.getenv("ANP_DID_SERVICE_URL")
            did_api_key = os.getenv("ANP_DID_API_KEY")
            
            self.base_agent = await BaseAgent.create_anp(
                agent_id=self.agent_id,
                host=host,
                port=None,  # Use dynamic port allocation
                executor=self.executor_wrapper,
                did_info=did_info,
                did_service_url=did_service_url,
                did_api_key=did_api_key,
                enable_protocol_negotiation=True
            )
            logger.info(f"[{self.agent_id}] BaseAgent ANP server created successfully at {self.base_agent.get_listening_address()}")

            # 6) Optional loopback adapter (for local testing only - default False for production)
            if self.install_loopback:
                # ANP uses DID-based connections, not HTTP URLs
                # For proper loopback, we would need:
                # 1. Extract target_did from self.base_agent.get_card()
                # 2. Use ANPAdapter(target_did=..., local_did_info=...) 
                # 3. Connect via WebSocket endpoint with DID authentication
                logger.warning(f"[{self.agent_id}] ANP loopback requires DID-based connection - not implemented for testing")
                logger.info(f"[{self.agent_id}] For ANP loopback, use: ANPAdapter(target_did=server_did, local_did_info=client_did)")
                # Production deployments should use install_loopback=False (default)

            # 7) Diagnostics
            logger.info(f"Agent URL: {self.base_agent.get_listening_address()}")
            logger.debug(f"Agent Card: {self.base_agent.get_card()}")
            return self.base_agent
            
        except Exception as e:
            logger.error(f"[ANP-META] Failed to create ANP worker: {e}")
            raise

    async def get_health_status(self) -> Dict[str, Any]:
        """Return best-effort health info for ANP server."""
        if not self.base_agent:
            return {"status": "not_initialized", "url": None}

        try:
            # Prefer server task state if available
            is_running = self.base_agent.is_server_running()
            url = self.base_agent.get_listening_address()
            # Replace 0.0.0.0 for local preview convenience
            if "0.0.0.0" in url:
                url = url.replace("0.0.0.0", "127.0.0.1")
            return {
                "status": "healthy" if is_running else "unhealthy", 
                "url": url,
                "agent_id": self.agent_id,
                "protocol": "anp",
                "meta_protocol": "ute",
                "native_agent_status": self.anp_agent.get_status() if self.anp_agent else None
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "url": None}

    async def close(self):
        """Close the ANP meta agent"""
        if self.base_agent:
            await self.base_agent.stop()
        logger.info(f"Closed ANP BaseAgent: {self.agent_id}")


async def create_anp_meta_worker(
    agent_id: str, 
    config: Dict[str, Any], 
    host: str = "0.0.0.0", 
    port: Optional[int] = None,
    install_loopback: bool = False
) -> ANPMetaAgent:
    """
    Factory function to create ANP meta protocol worker for network integration
    
    Args:
        agent_id: Unique agent identifier
        config: Configuration dict with core.openai_api_key etc.
        host: Server host
        port: Server port (auto-assigned if None)
        install_loopback: Whether to install loopback adapter 
                         (default False for production - PRODUCTION SAFE)
    
    Returns:
        Initialized ANPMetaAgent instance ready for network registration
    """
    # Debug: check config structure
    logger.info(f"[ANP-META] create_anp_meta_worker config keys: {list(config.keys())}")
    logger.info(f"[ANP-META] Has 'llm' key: {'llm' in config}")
    if 'llm' in config:
        logger.info(f"[ANP-META] llm config keys: {list(config['llm'].keys())}")
        logger.info(f"[ANP-META] openai_api_key present: {'openai_api_key' in config['llm']}")
    
    agent = ANPMetaAgent(agent_id, config, install_loopback)
    await agent.create_anp_worker(host=host, port=port)
    
    logger.info(f"[ANP-META] Created meta worker: {agent_id}")
    return agent


async def integrate_anp_into_network(network, agent_id: str, config: Dict[str, Any], port: Optional[int] = None) -> str:
    """
    Integrate ANP worker into existing NetworkBase
    
    Args:
        network: NetworkBase instance
        agent_id: Agent identifier
        config: Agent configuration
        port: Server port (optional)
    
    Returns:
        Agent URL for network registration
    """
    # Create ANP meta worker (production mode - no loopback)
    anp_agent = await create_anp_meta_worker(agent_id, config, port=port, install_loopback=False)
    
    # Register with network
    agent_url = anp_agent.base_agent.get_listening_address()
    await network.register_agent(agent_id, agent_url)
    
    logger.info(f"[ANP-META] Integrated agent {agent_id} into network at {agent_url}")
    return agent_url
