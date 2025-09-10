"""
Agora Meta Agent for Fail-Storm Recovery

Integrates Agora agents with BaseAgent meta-protocol system for unified network management.
Based on streaming_queue's meta implementation but adapted for fail_storm_recovery architecture.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add src path for BaseAgent access
project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

# Import BaseAgent from src/core
from src.core.base_agent import BaseAgent

# Import fail_storm Agora agent components
from ..agora.agent import AgoraAgent

logger = logging.getLogger(__name__)


class AgoraExecutorWrapper:
    """
    Wrapper to convert ShardWorkerExecutor to Agora-compatible interface.
    
    Agora expects a specific message handling interface that we adapt here.
    """
    
    def __init__(self, shard_worker_executor):
        self.shard_worker_executor = shard_worker_executor
        self.logger = logging.getLogger("AgoraExecutorWrapper")
    
    async def execute(self, context: Dict[str, Any], event_queue: Optional[Any] = None) -> None:
        """
        Handle Agora execution and convert to/from shard worker format.
        
        Args:
            context: Agora execution context
            event_queue: Optional event queue for async communication
        """
        try:
            # Extract question from Agora context
            question = context.get("text") or context.get("content") or context.get("question")
            if not question:
                # Try to extract from nested structures
                if "request" in context:
                    request = context["request"]
                    question = request.get("text") or request.get("content") or str(request)
                else:
                    question = str(context)
            
            # Create shard worker context
            shard_context = {
                "question": question,
                "agora_context": context,
                "event_queue": event_queue
            }
            
            # Execute shard worker
            await self.shard_worker_executor.execute(shard_context)
            
            # Store result back in context
            answer = shard_context.get("answer", "No response from shard worker")
            context["response"] = answer
            context["status"] = "completed"
            
            # If event queue is provided, send result
            if event_queue and hasattr(event_queue, 'put'):
                await event_queue.put({
                    "type": "response",
                    "text": answer,
                    "agent_id": getattr(self.shard_worker_executor, 'agent_id', 'agora_worker')
                })
            
        except Exception as e:
            self.logger.error(f"[AGORA] Executor error: {e}")
            error_msg = f"Error processing request: {str(e)}"
            context["response"] = error_msg
            context["status"] = "error"
            context["error"] = str(e)
            
            if event_queue and hasattr(event_queue, 'put'):
                await event_queue.put({
                    "type": "error",
                    "text": error_msg,
                    "error": str(e)
                })


class AgoraMetaAgent:
    """
    Agora Agent integrated with Meta-Protocol system for fail-storm recovery.
    
    Wraps the fail-storm Agora agent with BaseAgent interface to enable
    unified network management and cross-protocol communication.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], install_loopback: bool = False):
        self.agent_id = agent_id
        self.config = config
        self.install_loopback = install_loopback
        self.base_agent: Optional[BaseAgent] = None
        self.agora_agent: Optional[AgoraAgent] = None
        self.executor_wrapper: Optional[AgoraExecutorWrapper] = None
        
        logger.info(f"[AGORA-META] Initialized agent: {agent_id}")
    
    def _convert_config_for_shard_worker(self) -> Dict[str, Any]:
        """Convert config for ShardWorkerExecutor"""
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
        return {"model": core}
    
    async def create_agora_worker(self, host: str = "0.0.0.0", port: Optional[int] = None) -> "BaseAgent":
        """
        Create Agora worker using BaseAgent.create_agora() factory method.
        
        Args:
            host: Host address to bind to
            port: Port number (auto-assigned if None)
            
        Returns:
            BaseAgent instance with Agora protocol support
        """
        try:
            # Import ShardWorkerExecutor for fail-storm tasks
            from ...shard_qa.shard_worker.agent_executor import ShardWorkerExecutor
            
            # Create ShardWorkerExecutor with converted config
            shard_config = self._convert_config_for_shard_worker()
            shard_executor = ShardWorkerExecutor(shard_config)
            
            # Create Agora executor wrapper
            self.executor_wrapper = AgoraExecutorWrapper(shard_executor)
            
            # Create BaseAgent with Agora protocol
            self.base_agent = await BaseAgent.create_agora(
                agent_id=self.agent_id,
                executor=self.executor_wrapper,
                host=host,
                port=port,
                install_loopback=self.install_loopback
            )
            
            logger.info(f"[AGORA-META] Created BaseAgent for {self.agent_id} @ {self.base_agent.get_listening_address()}")
            return self.base_agent
            
        except Exception as e:
            logger.error(f"[AGORA-META] Failed to create Agora worker {self.agent_id}: {e}")
            raise
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the Agora agent"""
        if not self.base_agent:
            return {"status": "not_started", "url": None, "error": "BaseAgent not created"}
        
        try:
            url = self.base_agent.get_listening_address()
            # Use BaseAgent's health check if available
            if hasattr(self.base_agent, 'health_check'):
                is_healthy = await self.base_agent.health_check()
                status = "healthy" if is_healthy else "unhealthy"
            else:
                status = "healthy"  # Assume healthy if no health check method
            
            return {"status": status, "url": url, "error": None}
        except Exception as e:
            return {"status": "error", "url": None, "error": str(e)}
    
    async def close(self):
        """Close the Agora meta agent"""
        try:
            if self.base_agent:
                await self.base_agent.stop()
                logger.info(f"[AGORA-META] Closed BaseAgent for {self.agent_id}")
            
            if self.agora_agent:
                await self.agora_agent.stop()
                logger.info(f"[AGORA-META] Closed Agora agent for {self.agent_id}")
                
        except Exception as e:
            logger.error(f"[AGORA-META] Error closing {self.agent_id}: {e}")


async def create_agora_meta_worker(agent_id: str, config: Dict[str, Any], 
                                 host: str = "0.0.0.0", port: Optional[int] = None,
                                 install_loopback: bool = False) -> AgoraMetaAgent:
    """
    Factory function to create Agora meta worker.
    
    Args:
        agent_id: Unique agent identifier
        config: Agent configuration
        host: Host address to bind to
        port: Port number (auto-assigned if None)
        install_loopback: Whether to install loopback adapter
        
    Returns:
        AgoraMetaAgent instance with running BaseAgent
    """
    meta_agent = AgoraMetaAgent(agent_id, config, install_loopback)
    await meta_agent.create_agora_worker(host, port)
    return meta_agent
