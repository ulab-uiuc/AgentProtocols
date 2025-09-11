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
try:
    from acp_sdk.models import Message, RunYield
    from acp_sdk.server import Context
except ImportError:
    # Fallback for older ACP SDK versions
    from acp_sdk.models import Message
    from acp_sdk.server import Context
    RunYield = None

# Import fail_storm ACP agent components
import importlib.util
acp_agent_path = Path(__file__).parent.parent / "acp" / "agent.py"
spec = importlib.util.spec_from_file_location("acp_agent", acp_agent_path)
acp_agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acp_agent_module)
ACPAgent = acp_agent_module.ACPAgent

logger = logging.getLogger(__name__)


class ACPExecutorWrapper:
    """
    Wrapper to convert ShardWorkerExecutor to ACP SDK interface.
    
    ACP SDK expects: async def executor(messages: List[Message], context: Context) -> AsyncGenerator[RunYield, None]
    """
    
    def __init__(self, shard_worker_executor):
        self.shard_worker_executor = shard_worker_executor
        self.logger = logging.getLogger("ACPExecutorWrapper")
    
    async def __call__(self, messages: List[Message], context: Context) -> AsyncGenerator[Any, None]:
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
            if RunYield:
                yield RunYield(
                    type="message",
                    content=Message(
                        parts=[MessagePart(type="text", text=answer)]
                    )
                )
            else:
                # Fallback for older ACP SDK versions
                yield {"type": "message", "content": answer}
            
        except Exception as e:
            self.logger.error(f"[ACP] Executor error: {e}")
            
            # Yield error response in ACP format
            from acp_sdk.models import MessagePart
            if RunYield:
                error_msg = RunYield(
                    type="message", 
                    content=Message(
                        parts=[MessagePart(type="text", text=f"Error: {str(e)}")]
                    )
                )
                yield error_msg
            else:
                # Fallback for older ACP SDK versions
                yield {"type": "error", "content": f"Error: {str(e)}"}


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
            print(f"[ACP-META] DEBUG: Starting creation for {self.agent_id}")
            print(f"[ACP-META] DEBUG: Config: {self.config}")
            print(f"[ACP-META] DEBUG: Host: {host}, Port: {port}")
            
            # Import ShardWorkerExecutor for fail-storm tasks
            shard_qa_path = Path(__file__).parent.parent.parent / "shard_qa" / "shard_worker"
            sys.path.insert(0, str(shard_qa_path))
            from agent_executor import ShardWorkerExecutor
            
            # Create ShardWorkerExecutor with converted config
            shard_config = self._convert_config_for_shard_worker()
            print(f"[ACP-META] DEBUG: Shard config: {shard_config}")
            
            shard_executor = ShardWorkerExecutor(
                config=shard_config,
                global_config=self.config,
                shard_id=self.agent_id,
                data_file="data/shards/shard0.jsonl",  # Default data file
                neighbors={"prev_id": "prev", "next_id": "next"},  # Default neighbors
                output=None,
                force_llm=True
            )
            print(f"[ACP-META] DEBUG: Shard executor created successfully")
            
            # Create ACP executor wrapper
            self.executor_wrapper = ACPExecutorWrapper(shard_executor)
            print(f"[ACP-META] DEBUG: Executor wrapper created successfully")
            
            # Create BaseAgent with ACP protocol
            print(f"[ACP-META] DEBUG: Creating BaseAgent with ACP protocol...")
            self.base_agent = await BaseAgent.create_acp(
                agent_id=self.agent_id,
                executor=self.executor_wrapper,
                host=host,
                port=port
            )
            print(f"[ACP-META] DEBUG: BaseAgent created: {type(self.base_agent)}")
            
            # Debug: check if base_agent is properly created
            if self.base_agent:
                listening_addr = self.base_agent.get_listening_address()
                logger.info(f"[ACP-META] Created BaseAgent for {self.agent_id} @ {listening_addr}")
            else:
                logger.error(f"[ACP-META] BaseAgent creation failed for {self.agent_id} - base_agent is None")
            return self.base_agent
            
        except Exception as e:
            print(f"[ACP-META] ERROR: Failed to create ACP worker {self.agent_id}: {e}")
            import traceback
            print(f"[ACP-META] ERROR: Traceback:")
            traceback.print_exc()
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
