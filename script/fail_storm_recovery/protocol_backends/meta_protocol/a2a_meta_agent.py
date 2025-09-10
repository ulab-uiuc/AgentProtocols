"""
A2A Meta Agent for Fail-Storm Recovery

Integrates A2A agents with BaseAgent meta-protocol system for unified network management.
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

# Import fail_storm A2A agent
try:
    from ..a2a.agent import A2AAgent, A2AExecutorWrapper
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    current_path = Path(__file__).parent.parent
    sys.path.insert(0, str(current_path))
    from a2a.agent import A2AAgent, A2AExecutorWrapper

logger = logging.getLogger(__name__)


class A2AMetaAgent:
    """
    A2A Agent integrated with Meta-Protocol system for fail-storm recovery.
    
    Wraps the fail-storm A2A agent with BaseAgent interface to enable
    unified network management and cross-protocol communication.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], install_loopback: bool = False):
        self.agent_id = agent_id
        self.config = config
        self.install_loopback = install_loopback
        self.base_agent: Optional[BaseAgent] = None
        self.a2a_agent: Optional[A2AAgent] = None
        self.executor_wrapper: Optional[A2AExecutorWrapper] = None
        
        logger.info(f"[A2A-META] Initialized agent: {agent_id}")
    
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
    
    async def create_a2a_worker(self, host: str = "0.0.0.0", port: Optional[int] = None) -> "BaseAgent":
        """
        Create A2A worker using BaseAgent.create_a2a() factory method.
        
        Args:
            host: Host address to bind to
            port: Port number (auto-assigned if None)
            
        Returns:
            BaseAgent instance with A2A protocol support
        """
        try:
            # Import ShardWorkerExecutor for fail-storm tasks
            from ...shard_qa.shard_worker.agent_executor import ShardWorkerExecutor
            
            # Create ShardWorkerExecutor with converted config
            shard_config = self._convert_config_for_shard_worker()
            shard_executor = ShardWorkerExecutor(shard_config)
            
            # Create A2A executor wrapper
            self.executor_wrapper = A2AExecutorWrapper(shard_executor)
            
            # Create BaseAgent with A2A protocol
            self.base_agent = await BaseAgent.create_a2a(
                agent_id=self.agent_id,
                executor=self.executor_wrapper,
                host=host,
                port=port,
                install_loopback=self.install_loopback
            )
            
            logger.info(f"[A2A-META] Created BaseAgent for {self.agent_id} @ {self.base_agent.get_listening_address()}")
            return self.base_agent
            
        except Exception as e:
            logger.error(f"[A2A-META] Failed to create A2A worker {self.agent_id}: {e}")
            raise
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the A2A agent"""
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
        """Close the A2A meta agent"""
        try:
            if self.base_agent:
                await self.base_agent.stop()
                logger.info(f"[A2A-META] Closed BaseAgent for {self.agent_id}")
            
            if self.a2a_agent:
                await self.a2a_agent.stop()
                logger.info(f"[A2A-META] Closed A2A agent for {self.agent_id}")
                
        except Exception as e:
            logger.error(f"[A2A-META] Error closing {self.agent_id}: {e}")


async def create_a2a_meta_worker(agent_id: str, config: Dict[str, Any], 
                               host: str = "0.0.0.0", port: Optional[int] = None,
                               install_loopback: bool = False) -> A2AMetaAgent:
    """
    Factory function to create A2A meta worker.
    
    Args:
        agent_id: Unique agent identifier
        config: Agent configuration
        host: Host address to bind to
        port: Port number (auto-assigned if None)
        install_loopback: Whether to install loopback adapter
        
    Returns:
        A2AMetaAgent instance with running BaseAgent
    """
    meta_agent = A2AMetaAgent(agent_id, config, install_loopback)
    await meta_agent.create_a2a_worker(host, port)
    return meta_agent
