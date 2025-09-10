"""
Meta-Protocol Coordinator for Fail-Storm Recovery

Unified coordinator that manages all four meta-protocol agents (ACP, ANP, Agora, A2A)
for fail-storm recovery scenarios. Based on streaming_queue's meta coordinator but
adapted for fail-storm architecture and shard QA tasks.
"""

import asyncio
import uuid
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Add src path for BaseAgent access
project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

# Import BaseAgent and network components
from src.core.base_agent import BaseAgent
from src.core.network import AgentNetwork

# Meta-protocol agent imports
from .acp_meta_agent import create_acp_meta_worker, ACPMetaAgent
from .anp_meta_agent import create_anp_meta_worker, ANPMetaAgent
from .agora_meta_agent import create_agora_meta_worker, AgoraMetaAgent
from .a2a_meta_agent import create_a2a_meta_worker, A2AMetaAgent

logger = logging.getLogger(__name__)


class FailStormMetaCoordinator:
    """
    Unified coordinator for all meta-protocol agents in fail-storm recovery.
    
    This coordinator manages ACP, ANP, Agora, and A2A agents through their 
    BaseAgent meta-protocol interfaces for fail-storm recovery scenarios. It provides:
    - Dynamic load balancing across protocols
    - Unified message routing and response handling
    - Protocol-agnostic shard QA task dispatching
    - Cross-protocol performance comparison
    - Fault tolerance and recovery metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, output=None):
        self.config = config or {}
        self.output = output
        
        # Meta-protocol agent instances
        self.meta_agents: Dict[str, Any] = {}  # agent_id -> MetaAgent instance
        self.protocol_types: Dict[str, str] = {}  # agent_id -> protocol type
        
        # Network management
        self.agent_network = AgentNetwork()
        self.worker_ids: List[str] = []
        
        # Performance tracking
        self.protocol_stats: Dict[str, Dict[str, Any]] = {}
        
        # Fail-storm specific metrics
        self.failure_stats: Dict[str, Dict[str, Any]] = {}
        self.recovery_stats: Dict[str, Dict[str, Any]] = {}
        
        logger.info("[FAILSTORM-META-COORDINATOR] Initialized meta-protocol coordinator")
    
    def _initialize_worker_stats(self, worker_ids: List[str]):
        """Initialize protocol stats for all workers"""
        for worker_id in worker_ids:
            if worker_id not in self.protocol_stats:
                self.protocol_stats[worker_id] = {
                    "questions_processed": 0,
                    "total_response_time": 0.0,
                    "avg_response_time": 0.0,
                    "errors": 0,
                    "protocol": self.protocol_types.get(worker_id, "unknown"),
                    "failures": 0,
                    "recoveries": 0
                }
                
                # Initialize fail-storm specific stats
                self.failure_stats[worker_id] = {
                    "total_failures": 0,
                    "network_failures": 0,
                    "timeout_failures": 0,
                    "protocol_failures": 0,
                    "last_failure_time": None
                }
                
                self.recovery_stats[worker_id] = {
                    "total_recoveries": 0,
                    "recovery_time_avg": 0.0,
                    "successful_recoveries": 0,
                    "failed_recoveries": 0,
                    "last_recovery_time": None
                }
    
    async def add_protocol_agent(self, protocol: str, agent_id: str, config: Dict[str, Any], 
                                port: Optional[int] = None) -> str:
        """
        Add a meta-protocol agent to the coordinator.
        
        Args:
            protocol: Protocol type ("acp", "anp", "agora", "a2a")
            agent_id: Unique agent identifier
            config: Agent configuration
            port: Server port (optional)
        
        Returns:
            Agent URL for network registration
        """
        protocol = protocol.lower()
        
        try:
            if protocol == "acp":
                agent = await create_acp_meta_worker(agent_id, config, port=port, install_loopback=False)
            elif protocol == "anp":
                agent = await create_anp_meta_worker(agent_id, config, port=port, install_loopback=False)
            elif protocol == "agora":
                agent = await create_agora_meta_worker(agent_id, config, port=port, install_loopback=False)
            elif protocol == "a2a":
                agent = await create_a2a_meta_worker(agent_id, config, port=port, install_loopback=False)
            else:
                raise ValueError(f"Unsupported protocol: {protocol}")
            
            self.meta_agents[agent_id] = agent
            self.protocol_types[agent_id] = protocol
            self.worker_ids.append(agent_id)
            
            # Register with agent network
            await self.agent_network.register_agent(agent.base_agent)
            
            # Initialize stats
            self._initialize_worker_stats([agent_id])
            
            agent_url = agent.base_agent.get_listening_address()
            logger.info(f"[FAILSTORM-META-COORDINATOR] Added {protocol.upper()} agent {agent_id} at {agent_url}")
            return agent_url
            
        except Exception as e:
            logger.error(f"[FAILSTORM-META-COORDINATOR] Failed to add {protocol} agent {agent_id}: {e}")
            raise
    
    async def install_outbound_adapters(self) -> None:
        """
        Install correct outbound adapters on a chosen router agent so that
        all sends go over the network (BaseAgent.send -> HTTP/WS).
        """
        # Choose router (prefer A2A, else any non-ANP)
        router_id: Optional[str] = None
        for aid, proto in self.protocol_types.items():
            if proto == "a2a":
                router_id = aid
                break
        if router_id is None:
            for aid, proto in self.protocol_types.items():
                if proto != "anp":
                    router_id = aid
                    break
        if router_id is None:
            logger.warning("[FAILSTORM-META-COORDINATOR] No suitable router found (only ANP agents?).")
            return

        router_meta = self.meta_agents[router_id]
        router_ba = router_meta.base_agent

        # Build directory: agent_id -> (protocol, reachable_url)
        directory: Dict[str, Any] = {}
        for aid, meta in self.meta_agents.items():
            url = meta.base_agent.get_listening_address()
            if "0.0.0.0" in url:
                url = url.replace("0.0.0.0", "127.0.0.1")
            directory[aid] = (self.protocol_types[aid], url)

        # Lazy import adapters (avoid hard failure if missing)
        try:
            from src.agent_adapters.a2a_adapter import A2AAdapter
        except Exception:
            A2AAdapter = None
        try:
            from src.agent_adapters.acp_adapter import ACPAdapter
        except Exception:
            ACPAdapter = None
        try:
            from src.agent_adapters.agora_adapter import AgoraClientAdapter
        except Exception:
            AgoraClientAdapter = None

        # Install from router -> every destination (including self for loopback)
        for dst_id, (proto, url) in directory.items():
            if proto == "anp":
                # Install DID-based ANP outbound adapter
                try:
                    await self._install_anp_outbound_adapter(router_ba, dst_id)
                    logger.info("[FAILSTORM-META-COORDINATOR] Installed ANP DID outbound adapter: %s", dst_id)
                except Exception as e:
                    logger.warning("[FAILSTORM-META-COORDINATOR] ANP DID adapter failed: %s", e)
                continue
            try:
                if proto == "a2a" and A2AAdapter:
                    adp = A2AAdapter(httpx_client=router_ba._httpx_client, base_url=url)
                    await adp.initialize()
                    router_ba.add_outbound_adapter(dst_id, adp)
                elif proto == "acp" and ACPAdapter:
                    adp = ACPAdapter(httpx_client=router_ba._httpx_client, base_url=url, agent_id=dst_id)
                    await adp.initialize()
                    router_ba.add_outbound_adapter(dst_id, adp)
                elif proto == "agora" and AgoraClientAdapter:
                    # Force no toolformer to avoid JSON schema building errors
                    toolformer = None
                    adp = AgoraClientAdapter(
                        httpx_client=router_ba._httpx_client,
                        toolformer=toolformer,
                        target_url=url,
                        agent_id=dst_id
                    )
                    await adp.initialize()
                    router_ba.add_outbound_adapter(dst_id, adp)
                else:
                    logger.warning("[FAILSTORM-META-COORDINATOR] No adapter available for %s -> %s", router_id, dst_id)
            except Exception as e:
                logger.error("[FAILSTORM-META-COORDINATOR] Install adapter %s -> %s failed: %s", router_id, dst_id, e)

        # Remember router id for later sends
        self._router_id = router_id
        logger.info("[FAILSTORM-META-COORDINATOR] Router is %s", router_id)

    async def _install_anp_outbound_adapter(self, router_ba, dst_id: str) -> None:
        """Install a DID-based ANP outbound adapter on the router BaseAgent."""
        import socket
        
        def _find_free_port() -> int:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                return s.getsockname()[1]
        
        # Read server DID from card
        anp_meta = self.meta_agents[dst_id]
        server_card = anp_meta.base_agent.get_card()
        
        server_did = server_card.get("id") \
                     or (server_card.get("authentication") or {}).get("did") \
                     or (server_card.get("endpoints") or {}).get("did_document")
        if not server_did:
            raise RuntimeError("Server card does not contain DID")

        # Create a local DID for the client side (ANPAdapter)
        from agent_connect.python.utils.did_generate import did_generate
        from agent_connect.python.utils.crypto_tool import get_pem_from_private_key
        
        local_ws_port = _find_free_port()
        local_ws_endpoint = f"ws://127.0.0.1:{local_ws_port}/ws"
        private_key, _, local_did, did_document_json = did_generate(local_ws_endpoint)
        local_did_info = {
            "private_key_pem": get_pem_from_private_key(private_key),
            "did": local_did,
            "did_document_json": did_document_json
        }

        # Build ANPAdapter with target DID + local DID info + DID service support
        from src.agent_adapters.anp_adapter import ANPAdapter
        import os
        
        anp_adp = ANPAdapter(
            httpx_client=router_ba._httpx_client,
            target_did=server_did,
            local_did_info=local_did_info,
            host_domain="127.0.0.1",
            host_port=str(local_ws_port),
            host_ws_path="/ws",
            did_service_url=os.getenv("ANP_DID_SERVICE_URL"),
            did_api_key=os.getenv("ANP_DID_API_KEY"),
            enable_protocol_negotiation=False,
            enable_e2e_encryption=True
        )
        await anp_adp.initialize()
        router_ba.add_outbound_adapter(dst_id, anp_adp)

    async def send_shard_task(self, worker_id: str, shard_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send shard QA task to worker via network using BaseAgent.send().
        
        Args:
            worker_id: Target worker ID
            shard_data: Shard task data including questions and context
            
        Returns:
            Task result with answers and metrics
        """
        import time
        start_time = time.time()

        if worker_id not in self.meta_agents:
            return {
                "answers": [],
                "raw": {"error": f"Worker {worker_id} not found"}, 
                "protocol": "unknown", 
                "response_time": 0.0, 
                "success": False
            }

        # Ensure router chosen
        router_id = getattr(self, "_router_id", None)
        if router_id is None:
            await self.install_outbound_adapters()
            router_id = getattr(self, "_router_id", None)
            if router_id is None:
                return {
                    "answers": [],
                    "raw": {"error": "No router available"}, 
                    "protocol": "unknown", 
                    "response_time": 0.0, 
                    "success": False
                }

        dst_protocol = self.protocol_types[worker_id]
        router_meta = self.meta_agents[router_id]
        router_ba = router_meta.base_agent

        try:
            # Send shard task over the network
            content = await router_ba.send(worker_id, shard_data)
            dt = time.time() - start_time

            # Update stats
            stats = self.protocol_stats[worker_id]
            stats["questions_processed"] += len(shard_data.get("questions", []))
            stats["total_response_time"] += dt
            stats["avg_response_time"] = stats["total_response_time"] / max(1, stats["questions_processed"])

            # Extract answers from response
            answers = []
            if isinstance(content, dict):
                answers = content.get("answers", [])
                if not answers and "response" in content:
                    # Handle single response
                    answers = [content["response"]]
                elif not answers and "text" in content:
                    answers = [content["text"]]
            elif isinstance(content, list):
                answers = content
            elif isinstance(content, str):
                answers = [content]

            return {
                "answers": answers,
                "raw": {"response": content, "protocol": dst_protocol}, 
                "protocol": dst_protocol, 
                "response_time": dt, 
                "success": True
            }

        except Exception as e:
            dt = time.time() - start_time
            self.protocol_stats[worker_id]["errors"] += 1
            self._record_failure(worker_id, "network_error", str(e))
            
            logger.error(f"[FAILSTORM-META-COORDINATOR] Network send {router_id} -> {worker_id} failed: {e}")
            return {
                "answers": [],
                "raw": {"error": str(e), "protocol": dst_protocol}, 
                "protocol": dst_protocol, 
                "response_time": dt, 
                "success": False
            }
    
    def _record_failure(self, worker_id: str, failure_type: str, error_msg: str):
        """Record failure statistics for fail-storm analysis"""
        import time
        
        if worker_id in self.failure_stats:
            stats = self.failure_stats[worker_id]
            stats["total_failures"] += 1
            stats["last_failure_time"] = time.time()
            
            if failure_type == "network_error":
                stats["network_failures"] += 1
            elif failure_type == "timeout":
                stats["timeout_failures"] += 1
            else:
                stats["protocol_failures"] += 1
    
    def _record_recovery(self, worker_id: str, recovery_time: float):
        """Record recovery statistics for fail-storm analysis"""
        import time
        
        if worker_id in self.recovery_stats:
            stats = self.recovery_stats[worker_id]
            stats["total_recoveries"] += 1
            stats["last_recovery_time"] = time.time()
            stats["successful_recoveries"] += 1
            
            # Update average recovery time
            if stats["total_recoveries"] > 1:
                stats["recovery_time_avg"] = (
                    (stats["recovery_time_avg"] * (stats["total_recoveries"] - 1) + recovery_time) / 
                    stats["total_recoveries"]
                )
            else:
                stats["recovery_time_avg"] = recovery_time
    
    async def get_failstorm_metrics(self) -> Dict[str, Any]:
        """Get comprehensive fail-storm recovery metrics."""
        return {
            "total_agents": len(self.meta_agents),
            "protocols": list(set(self.protocol_types.values())),
            "agent_stats": dict(self.protocol_stats),
            "failure_stats": dict(self.failure_stats),
            "recovery_stats": dict(self.recovery_stats),
            "network_health": await self.health_check_all(),
            "summary": {
                protocol: {
                    "agents": len([aid for aid, ptype in self.protocol_types.items() if ptype == protocol]),
                    "total_questions": sum(stats["questions_processed"] 
                                         for aid, stats in self.protocol_stats.items() 
                                         if self.protocol_types[aid] == protocol),
                    "avg_response_time": sum(stats["avg_response_time"] 
                                           for aid, stats in self.protocol_stats.items() 
                                           if self.protocol_types[aid] == protocol and stats["questions_processed"] > 0) / 
                                         max(1, len([aid for aid, ptype in self.protocol_types.items() if ptype == protocol])),
                    "total_errors": sum(stats["errors"] 
                                      for aid, stats in self.protocol_stats.items() 
                                      if self.protocol_types[aid] == protocol),
                    "total_failures": sum(self.failure_stats[aid]["total_failures"] 
                                        for aid in self.protocol_types if self.protocol_types[aid] == protocol),
                    "total_recoveries": sum(self.recovery_stats[aid]["total_recoveries"] 
                                          for aid in self.protocol_types if self.protocol_types[aid] == protocol)
                }
                for protocol in set(self.protocol_types.values())
            }
        }
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Check health status of all meta-protocol agents."""
        health_results = {}
        
        for agent_id, agent in self.meta_agents.items():
            protocol = self.protocol_types[agent_id]
            try:
                health = await agent.get_health_status()
                health_results[agent_id] = {
                    "protocol": protocol,
                    "status": health.get("status", "unknown"),
                    "url": health.get("url"),
                    "error": health.get("error")
                }
            except Exception as e:
                health_results[agent_id] = {
                    "protocol": protocol,
                    "status": "error",
                    "url": None,
                    "error": str(e)
                }
        
        return health_results
    
    async def close_all(self):
        """Close all meta-protocol agents."""
        logger.info("[FAILSTORM-META-COORDINATOR] Closing all meta-protocol agents")
        
        for agent_id, agent in self.meta_agents.items():
            try:
                await agent.close()
                logger.info(f"[FAILSTORM-META-COORDINATOR] Closed {self.protocol_types[agent_id]} agent {agent_id}")
            except Exception as e:
                logger.error(f"[FAILSTORM-META-COORDINATOR] Error closing {agent_id}: {e}")
        
        self.meta_agents.clear()
        self.protocol_types.clear()
        self.worker_ids.clear()


async def create_failstorm_meta_network(config: Dict[str, Any]) -> FailStormMetaCoordinator:
    """
    Factory function to create a complete fail-storm meta-protocol network.
    
    Args:
        config: Configuration dict with protocol-specific settings
    
    Returns:
        Configured FailStormMetaCoordinator with all agents
    """
    coordinator = FailStormMetaCoordinator(config)
    
    # Extract configurations
    protocols_config = config.get("protocols", {})
    base_port = config.get("network", {}).get("base_port", 9000)
    core_config = config.get("core", {})
    
    # Add agents for each configured protocol
    port_offset = 0
    for protocol in ["acp", "anp", "agora", "a2a"]:
        protocol_settings = protocols_config.get(protocol, {})
        if protocol_settings.get("enabled", True):
            agent_count = protocol_settings.get("agent_count", 2)
            
            # Create specified number of agents per protocol
            for i in range(agent_count):
                agent_id = f"{protocol.upper()}-FailStorm-{i+1}-{uuid.uuid4().hex[:8]}"
                port = base_port + port_offset
                
                # Use core config for agent configuration
                agent_config = {"core": core_config}
                
                try:
                    await coordinator.add_protocol_agent(protocol, agent_id, agent_config, port)
                    port_offset += 1
                except Exception as e:
                    logger.warning(f"[FAILSTORM-META-NETWORK] Failed to add {protocol} agent {agent_id}: {e}")
    
    logger.info(f"[FAILSTORM-META-NETWORK] Created fail-storm meta-protocol network with {len(coordinator.meta_agents)} agents")
    return coordinator
