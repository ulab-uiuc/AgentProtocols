"""
A2A Agent Meta Protocol Integration

Integrates A2A agent with src/core/base_agent.py using Meta-Protocol (UTE) system.
Based on fail_storm_recovery A2A agent implementation.
"""

import asyncio
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Production imports - add proper paths
current_file = Path(__file__).resolve()
fail_storm_path = current_file.parents[2]  # Go up to fail_storm_recovery
project_root = fail_storm_path.parent.parent  # Go up to agent_network
src_path = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(fail_storm_path))
sys.path.insert(0, str(src_path))

from src.core.base_agent import BaseAgent
from protocol_backends.a2a.agent import A2AAgent, create_a2a_agent

logger = logging.getLogger(__name__)


class A2AMetaAgent:
    """
    A2A Agent integrated with Meta-Protocol system
    
    Uses BaseAgent.create_a2a() to wrap A2A native agent into a Meta-Protocol node
    with UTE encoding/decoding, health endpoints, and agent card.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], install_loopback: bool = False):
        self.agent_id = agent_id
        self.config = config
        self.install_loopback = install_loopback
        self.base_agent: Optional[BaseAgent] = None
        self.a2a_agent: Optional[A2AAgent] = None
        self.native_executor = None
        
        logger.info(f"Initialized A2A meta protocol agent: {agent_id}")
    
    def _convert_config_for_executor(self) -> Dict[str, Any]:
        """Convert config for A2A executor based on fail_storm pattern"""
        # Use llm config from fail_storm_recovery pattern
        llm_config = self.config.get("llm", {})
        if llm_config.get("type") == "openai":
            return {
                "model": {
                    "type": "openai",
                    "name": llm_config.get("model", "gpt-4o"),
                    "openai_api_key": llm_config.get("openai_api_key"),
                    "openai_base_url": llm_config.get("openai_base_url", "https://api.openai.com/v1"),
                    "temperature": llm_config.get("temperature", 0.0),
                }
            }
        return {"model": llm_config}
    
    async def create_a2a_worker(self, host: str = "0.0.0.0", port: Optional[int] = None) -> BaseAgent:
        """
        Start an A2A server via BaseAgent.create_a2a().
        Uses fail_storm_recovery A2A native implementation.
        """
        try:
            # 1) Create native A2A executor from shard_qa pattern
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
            self.native_executor = ShardWorkerExecutor(
                config=executor_config,
                global_config=global_config,
                shard_id=self.agent_id,
                data_file=None,
                neighbors=[],
                output=None,
                force_llm=True
            )
            
            # 2) Create BaseAgent with A2A server using native executor directly
            self.base_agent = await BaseAgent.create_a2a(
                agent_id=self.agent_id,
                host=host,
                port=None,  # Use dynamic port allocation
                executor=self.native_executor  # Use ShardWorkerExecutor directly
            )
            
            # 4) Optional loopback adapter (for local testing only)
            if self.install_loopback:
                try:
                    from src.agent_adapters.a2a_adapter import A2AAdapter
                    listen_url = self.base_agent.get_listening_address()
                    
                    # Fix URL for Windows - replace 0.0.0.0 with 127.0.0.1
                    if "0.0.0.0" in listen_url:
                        listen_url = listen_url.replace("0.0.0.0", "127.0.0.1")
                    
                    adapter = A2AAdapter(
                        httpx_client=self.base_agent._httpx_client,
                        base_url=listen_url
                    )
                    await adapter.initialize()
                    self.base_agent.add_outbound_adapter(self.agent_id, adapter)
                    logger.info(f"Installed loopback A2A adapter at {listen_url}")
                except ImportError as e:
                    logger.warning(f"Cannot install loopback adapter: A2A adapter not available ({e})")
                except Exception as e:
                    logger.error(f"Failed to install loopback adapter: {e}")
                    # Don't raise - loopback is optional

            # 5) Diagnostics
            logger.info(f"Agent URL: {self.base_agent.get_listening_address()}")
            logger.debug(f"Agent Card: {self.base_agent.get_card()}")
            return self.base_agent
            
        except Exception as e:
            logger.error(f"Failed to create A2A meta worker: {e}")
            raise
    
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
                "url": self.base_agent.get_listening_address(),
                "meta_protocol": "ute",
                "native_agent_status": self.a2a_agent.get_status() if self.a2a_agent else None
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
