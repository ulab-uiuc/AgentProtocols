"""
ANP Meta Agent for Fail-Storm Recovery

Integrates ANP agents with BaseAgent meta-protocol system for unified network management.
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

# Import fail_storm ANP agent components
from ..anp.agent import ANPAgent

logger = logging.getLogger(__name__)


class ANPExecutorWrapper:
    """
    Wrapper to convert ShardWorkerExecutor to ANP-compatible interface.
    
    ANP SimpleNode expects a specific message handling interface that we adapt here.
    """
    
    def __init__(self, shard_worker_executor):
        self.shard_worker_executor = shard_worker_executor
        self.logger = logging.getLogger("ANPExecutorWrapper")
    
    async def handle_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle ANP message and convert to/from shard worker format.
        
        Args:
            message_data: ANP message data
            
        Returns:
            Response data for ANP
        """
        try:
            # Extract question from ANP message
            question = message_data.get("text") or message_data.get("content") or str(message_data)
            
            # Create shard worker context
            context = {
                "question": question,
                "message_data": message_data
            }
            
            # Execute shard worker
            await self.shard_worker_executor.execute(context)
            
            # Return ANP-compatible response
            return {
                "text": context.get("answer", "No response from shard worker"),
                "status": "success",
                "agent_id": getattr(self.shard_worker_executor, 'agent_id', 'anp_worker')
            }
            
        except Exception as e:
            self.logger.error(f"[ANP] Executor error: {e}")
            return {
                "text": f"Error processing request: {str(e)}",
                "status": "error",
                "error": str(e)
            }


class ANPMetaAgent:
    """
    ANP Agent integrated with Meta-Protocol system for fail-storm recovery.
    
    Wraps the fail-storm ANP agent with BaseAgent interface to enable
    unified network management and cross-protocol communication.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], install_loopback: bool = False):
        self.agent_id = agent_id
        self.config = config
        self.install_loopback = install_loopback
        self.base_agent: Optional[BaseAgent] = None
        self.anp_agent: Optional[ANPAgent] = None
        self.executor_wrapper: Optional[ANPExecutorWrapper] = None
        
        logger.info(f"[ANP-META] Initialized agent: {agent_id}")
    
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
    
    async def create_anp_worker(self, host: str = "0.0.0.0", port: Optional[int] = None) -> "BaseAgent":
        """
        Create ANP worker using BaseAgent.create_anp() factory method.
        
        Args:
            host: Host address to bind to
            port: Port number (auto-assigned if None)
            
        Returns:
            BaseAgent instance with ANP protocol support
        """
        try:
            # Import ShardWorkerExecutor for fail-storm tasks
            from ...shard_qa.shard_worker.agent_executor import ShardWorkerExecutor
            
            # Create ShardWorkerExecutor with converted config
            shard_config = self._convert_config_for_shard_worker()
            shard_executor = ShardWorkerExecutor(shard_config)
            
            # Create ANP executor wrapper
            self.executor_wrapper = ANPExecutorWrapper(shard_executor)
            
            # Create BaseAgent with ANP protocol
            self.base_agent = await BaseAgent.create_anp(
                agent_id=self.agent_id,
                executor=self.executor_wrapper,
                host=host,
                port=port,
                install_loopback=self.install_loopback
            )
            
            logger.info(f"[ANP-META] Created BaseAgent for {self.agent_id} @ {self.base_agent.get_listening_address()}")
            return self.base_agent
            
        except Exception as e:
            logger.error(f"[ANP-META] Failed to create ANP worker {self.agent_id}: {e}")
            raise
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the ANP agent"""
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
        """Close the ANP meta agent"""
        try:
            if self.base_agent:
                await self.base_agent.stop()
                logger.info(f"[ANP-META] Closed BaseAgent for {self.agent_id}")
            
            if self.anp_agent:
                await self.anp_agent.stop()
                logger.info(f"[ANP-META] Closed ANP agent for {self.agent_id}")
                
        except Exception as e:
            logger.error(f"[ANP-META] Error closing {self.agent_id}: {e}")


async def create_anp_meta_worker(agent_id: str, config: Dict[str, Any], 
                               host: str = "0.0.0.0", port: Optional[int] = None,
                               install_loopback: bool = False) -> ANPMetaAgent:
    """
    Factory function to create ANP meta worker.
    
    Args:
        agent_id: Unique agent identifier
        config: Agent configuration
        host: Host address to bind to
        port: Port number (auto-assigned if None)
        install_loopback: Whether to install loopback adapter
        
    Returns:
        ANPMetaAgent instance with running BaseAgent
    """
    meta_agent = ANPMetaAgent(agent_id, config, install_loopback)
    await meta_agent.create_anp_worker(host, port)
    return meta_agent
