"""
A2A Agent Meta Protocol Integration

Integrates A2A QAAgentExecutor with src/core/base_agent.py using Meta-Protocol (UTE) system.
The A2A executor already has the correct BaseAgent interface, so minimal adaptation is needed.
"""

import asyncio
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Production imports - no path hacks
import logging

# Production imports - require real dependencies
# Add proper paths
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parents[2]
project_root = streaming_queue_path.parent.parent
src_path = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(streaming_queue_path))
sys.path.insert(0, str(src_path))

from src.core.base_agent import BaseAgent
from protocol_backend.a2a.worker import QAAgentExecutor

logger = logging.getLogger(__name__)


class A2AMetaAgent:
    """
    A2A Agent integrated with Meta-Protocol system
    
    Uses BaseAgent.create_a2a() to wrap QAAgentExecutor into a Meta-Protocol node
    with UTE encoding/decoding, health endpoints, and agent card.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], install_loopback: bool = False):
        self.agent_id = agent_id
        self.config = config
        self.install_loopback = install_loopback
        self.base_agent: Optional[BaseAgent] = None
        self.qa_executor: Optional[QAAgentExecutor] = None
        
        logger.info(f"Initialized A2A meta protocol agent: {agent_id}")
    

    
    def _convert_config_for_qa(self) -> Dict[str, Any]:
        """Convert config for QAAgentExecutor"""
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
    
    async def create_a2a_worker(self, host: str = "0.0.0.0", port: Optional[int] = None) -> "BaseAgent":
        """
        Start an A2A server via BaseAgent.create_a2a().
        Optionally install a loopback outbound adapter for self-call tests.
        """
        # 1) Build native A2A executor
        qa_config = self._convert_config_for_qa()
        self.qa_executor = QAAgentExecutor(qa_config)
        if not hasattr(self.qa_executor, "execute"):
            raise RuntimeError("QAAgentExecutor must implement execute(context, event_queue)")

        # 2) Start BaseAgent server (A2A)
        self.base_agent = await BaseAgent.create_a2a(
            agent_id=self.agent_id,
            host=host,
            port=port or 8081,
            executor=self.qa_executor
        )

        # 3) Optional loopback adapter (for local testing only)
        if self.install_loopback:
            from src.agent_adapters.a2a_adapter import A2AAdapter  # must exist in runtime
            listen_url = self.base_agent.get_listening_address()
            adapter = A2AAdapter(
                httpx_client=self.base_agent._httpx_client,  # reuse pooled client
                base_url=listen_url
            )
            await adapter.initialize()
            self.base_agent.add_outbound_adapter(self.agent_id, adapter)
            logger.info(f"Installed loopback adapter for {self.agent_id}")

        # 4) Diagnostics
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
                "protocol": "a2a",
                "url": self.base_agent.get_listening_address(),  # <- use public method
                "meta_protocol": "ute",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def close(self):
        """Close the A2A meta agent"""
        if self.base_agent:
            await self.base_agent.stop()
            logger.info(f"Closed A2A BaseAgent: {self.agent_id}")


async def create_a2a_meta_worker(
    agent_id: str, 
    config: Dict[str, Any], 
    host: str = "0.0.0.0", 
    port: Optional[int] = None,
    install_loopback: bool = False
) -> A2AMetaAgent:
    """
    Factory function to create A2A meta protocol worker for network integration
    
    Args:
        agent_id: Unique agent identifier
        config: Configuration dict with core.openai_api_key etc.
        host: Server host
        port: Server port (auto-assigned if None)
        install_loopback: Whether to install loopback adapter (default False for production)
    
    Returns:
        Initialized A2AMetaAgent instance ready for network registration
    """
    agent = A2AMetaAgent(agent_id, config, install_loopback)
    await agent.create_a2a_worker(host=host, port=port)
    
    logger.info(f"Created A2A meta worker: {agent_id}")
    return agent


async def integrate_a2a_into_network(network, agent_id: str, config: Dict[str, Any], port: Optional[int] = None) -> str:
    """
    Integrate A2A worker into existing NetworkBase
    
    Args:
        network: NetworkBase instance for registration
        agent_id: Agent identifier
        config: Configuration dict
        port: Server port
    
    Returns:
        Agent listening URL for network registration
    """
    # Create A2A meta agent (no loopback for network deployment)
    a2a_agent = await create_a2a_meta_worker(agent_id, config, port=port, install_loopback=False)
    
    # Register with network
    agent_url = a2a_agent.base_agent.get_listening_address()
    await network.register_agent(agent_id, agent_url)
    
    logger.info(f"Integrated A2A agent {agent_id} into network at {agent_url}")
    return agent_url
