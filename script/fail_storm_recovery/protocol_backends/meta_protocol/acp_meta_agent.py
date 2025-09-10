"""
ACP Meta Agent for Fail-Storm Recovery

Integrates ACP agents with BaseAgent meta-protocol system for unified network management.
Based on streaming_queue's meta implementation but adapted for fail_storm_recovery architecture.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
from pathlib import Path
import sys

# Add src path for BaseAgent access
project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

# Import BaseAgent from src/core
from src.core.base_agent import BaseAgent

# Import ACP SDK components
from acp_sdk.models import Message, RunYield
from acp_sdk.server import Context

# Import fail_storm ACP agent components
from ..acp.agent import ACPAgent

logger = logging.getLogger(__name__)


class ACPExecutorWrapper:
    """
    Wrapper to convert ShardWorkerExecutor to ACP SDK interface.
    
    ACP SDK expects: async def executor(messages: List[Message], context: Context) -> AsyncGenerator[RunYield, None]
    """
    
    def __init__(self, shard_worker_executor):
        self.shard_worker_executor = shard_worker_executor
        self.logger = logging.getLogger("ACPExecutorWrapper")
    
    async def __call__(self, messages: List[Message], context: Context) -> AsyncGenerator[RunYield, None]:
        """
        ACP SDK executor interface implementation for fail-storm shard workers.
        
        Args:
            messages: List of ACP SDK Message objects
            context: ACP SDK Context object
            
        Yields:
            RunYield objects as required by ACP SDK
        """
        try:
            # Extract text from ACP messages
            question_parts = []
            for msg in messages:
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        if hasattr(part, 'text') and part.text:
                            question_parts.append(part.text)
                elif hasattr(msg, 'content'):
                    question_parts.append(str(msg.content))
            
            question = " ".join(question_parts) if question_parts else "No question provided"
            
            # Create shard worker context
            shard_context = {
                "question": question,
                "messages": messages,
                "acp_context": context
            }
            
            # Execute shard worker
            await self.shard_worker_executor.execute(shard_context)
            
            # Get response from context
            answer = shard_context.get("answer", "No response from shard worker")
            
            # Create ACP SDK response
            from acp_sdk.models import MessagePart
            
            # Yield ACP SDK RunYield with text response
            yield RunYield(
                type="message",
                content=Message(
                    parts=[MessagePart(type="text", text=answer)]
                )
            )
            
        except Exception as e:
            self.logger.error(f"[ACP] Executor error: {e}")
            
            # Yield error response in ACP format
            from acp_sdk.models import MessagePart
            error_msg = RunYield(
                type="message", 
                content=Message(
                    parts=[MessagePart(type="text", text=f"Error: {str(e)}")]
                )
            )
            yield error_msg


class ACPMetaAgent:
    """
    ACP Agent integrated with Meta-Protocol system for fail-storm recovery.
    
    Wraps the fail-storm ACP agent with BaseAgent interface to enable
    unified network management and cross-protocol communication.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], install_loopback: bool = False):
        self.agent_id = agent_id
        self.config = config
        self.install_loopback = install_loopback
        self.base_agent: Optional[BaseAgent] = None
        self.acp_agent: Optional[ACPAgent] = None
        self.executor_wrapper: Optional[ACPExecutorWrapper] = None
        
        logger.info(f"[ACP-META] Initialized agent: {agent_id}")
    
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
    
    async def create_acp_worker(self, host: str = "0.0.0.0", port: Optional[int] = None) -> "BaseAgent":
        """
        Create ACP worker using BaseAgent.create_acp() factory method.
        
        Args:
            host: Host address to bind to
            port: Port number (auto-assigned if None)
            
        Returns:
            BaseAgent instance with ACP protocol support
        """
        try:
            # Import ShardWorkerExecutor for fail-storm tasks
            from ...shard_qa.shard_worker.agent_executor import ShardWorkerExecutor
            
            # Create ShardWorkerExecutor with converted config
            shard_config = self._convert_config_for_shard_worker()
            shard_executor = ShardWorkerExecutor(shard_config)
            
            # Create ACP executor wrapper
            self.executor_wrapper = ACPExecutorWrapper(shard_executor)
            
            # Create BaseAgent with ACP protocol
            self.base_agent = await BaseAgent.create_acp(
                agent_id=self.agent_id,
                executor=self.executor_wrapper,
                host=host,
                port=port,
                install_loopback=self.install_loopback
            )
            
            logger.info(f"[ACP-META] Created BaseAgent for {self.agent_id} @ {self.base_agent.get_listening_address()}")
            return self.base_agent
            
        except Exception as e:
            logger.error(f"[ACP-META] Failed to create ACP worker {self.agent_id}: {e}")
            raise
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the ACP agent"""
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
        """Close the ACP meta agent"""
        try:
            if self.base_agent:
                await self.base_agent.stop()
                logger.info(f"[ACP-META] Closed BaseAgent for {self.agent_id}")
            
            if self.acp_agent:
                await self.acp_agent.stop()
                logger.info(f"[ACP-META] Closed ACP agent for {self.agent_id}")
                
        except Exception as e:
            logger.error(f"[ACP-META] Error closing {self.agent_id}: {e}")


async def create_acp_meta_worker(agent_id: str, config: Dict[str, Any], 
                               host: str = "0.0.0.0", port: Optional[int] = None,
                               install_loopback: bool = False) -> ACPMetaAgent:
    """
    Factory function to create ACP meta worker.
    
    Args:
        agent_id: Unique agent identifier
        config: Agent configuration
        host: Host address to bind to
        port: Port number (auto-assigned if None)
        install_loopback: Whether to install loopback adapter
        
    Returns:
        ACPMetaAgent instance with running BaseAgent
    """
    meta_agent = ACPMetaAgent(agent_id, config, install_loopback)
    await meta_agent.create_acp_worker(host, port)
    return meta_agent
