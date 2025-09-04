"""
Meta-Protocol Coordinator - Production Version

Unified coordinator that manages all four meta-protocol agents (ACP, ANP, Agora, A2A).
Based on QACoordinatorBase from core and integrates with BaseAgent meta-protocol system.
"""

import asyncio
import uuid
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Core imports
from script.streaming_queue.core.qa_coordinator_base import QACoordinatorBase

# Meta-protocol agent imports
from .acp_agent import create_acp_meta_worker, ACPMetaAgent
from .anp_agent import create_anp_meta_worker, ANPMetaAgent
from .agora_agent import create_agora_meta_worker, AgoraMetaAgent
from .a2a_agent import create_a2a_meta_worker, A2AMetaAgent

logger = logging.getLogger(__name__)


class MetaProtocolCoordinator(QACoordinatorBase):
    """
    Unified coordinator for all meta-protocol agents.
    
    This coordinator extends QACoordinatorBase to manage ACP, ANP, Agora, and A2A
    agents through their BaseAgent meta-protocol interfaces. It provides:
    - Dynamic load balancing across protocols
    - Unified message routing and response handling
    - Protocol-agnostic question dispatching
    - Cross-protocol performance comparison
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, output=None):
        super().__init__(config, output)
        
        # Meta-protocol agent instances
        self.meta_agents: Dict[str, Any] = {}  # agent_id -> MetaAgent instance
        self.protocol_types: Dict[str, str] = {}  # agent_id -> protocol type
        
        # Performance tracking
        self.protocol_stats: Dict[str, Dict[str, Any]] = {}
        
        logger.info("[META-COORDINATOR] Initialized meta-protocol coordinator")
    
    def _initialize_worker_stats(self, worker_ids: List[str]):
        """Initialize protocol stats for all workers"""
        for worker_id in worker_ids:
            if worker_id not in self.protocol_stats:
                self.protocol_stats[worker_id] = {
                    "questions_processed": 0,
                    "total_response_time": 0.0,
                    "avg_response_time": 0.0,
                    "errors": 0,
                    "protocol": self.protocol_types.get(worker_id, "unknown")
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
            
            # Initialize stats
            self.protocol_stats[agent_id] = {
                "protocol": protocol,
                "questions_processed": 0,
                "total_response_time": 0.0,
                "errors": 0,
                "avg_response_time": 0.0
            }
            
            agent_url = agent.base_agent.get_listening_address()
            logger.info(f"[META-COORDINATOR] Added {protocol.upper()} agent {agent_id} at {agent_url}")
            return agent_url
            
        except Exception as e:
            logger.error(f"[META-COORDINATOR] Failed to add {protocol} agent {agent_id}: {e}")
            raise
    
    async def install_outbound_adapters(self) -> None:
        """
        Install correct outbound adapters on a chosen router agent so that
        all sends go over the network (BaseAgent.send -> HTTP/WS).
        We choose one non-ANP agent as the 'router' (prefer A2A), then
        install outbound adapters from it to all other agents.
        """
        # 1) choose router (prefer A2A, else any non-ANP)
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
            logger.warning("[META-COORDINATOR] No suitable router found (only ANP agents?).")
            return

        router_meta = self.meta_agents[router_id]
        router_ba = router_meta.base_agent

        # 2) build directory: agent_id -> (protocol, reachable_url)
        directory: Dict[str, Any] = {}
        for aid, meta in self.meta_agents.items():
            url = meta.base_agent.get_listening_address()
            if "0.0.0.0" in url:
                url = url.replace("0.0.0.0", "127.0.0.1")
            directory[aid] = (self.protocol_types[aid], url)

        # 3) lazy import adapters (avoid hard failure if missing)
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

        # 4) install from router -> every destination (including self for loopback)
        for dst_id, (proto, url) in directory.items():
            if proto == "anp":
                # Install DID-based ANP outbound adapter
                try:
                    await self._install_anp_outbound_adapter(router_ba, dst_id)
                    logger.info("[META-COORDINATOR] Installed ANP DID outbound adapter: %s", dst_id)
                except Exception as e:
                    logger.warning("[META-COORDINATOR] ANP DID adapter failed: %s", e)
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
                    logger.warning("[META-COORDINATOR] No adapter available for %s -> %s", router_id, dst_id)
            except Exception as e:
                logger.error("[META-COORDINATOR] Install adapter %s -> %s failed: %s", router_id, dst_id, e)

        # 5) remember router id for later sends
        self._router_id = router_id  # create an attribute
        logger.info("[META-COORDINATOR] Router is %s", router_id)

    async def _install_anp_outbound_adapter(self, router_ba, dst_id: str) -> None:
        """
        Install a DID-based ANP outbound adapter on the router BaseAgent.
        """
        import socket
        
        def _find_free_port() -> int:
            """Return a free TCP port for the local SimpleNode."""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                return s.getsockname()[1]
        
        # 1) read server DID from card
        anp_meta = self.meta_agents[dst_id]
        server_card = anp_meta.base_agent.get_card()
        
        # prefer "id", else try authentication.did or endpoints.did_document
        server_did = server_card.get("id") \
                     or (server_card.get("authentication") or {}).get("did") \
                     or (server_card.get("endpoints") or {}).get("did_document")
        if not server_did:
            raise RuntimeError("Server card does not contain DID")

        # 2) create a local DID for the client side (ANPAdapter)
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

        # 3) build ANPAdapter with target DID + local DID info + DID service support
        from src.agent_adapters.anp_adapter import ANPAdapter
        import os
        
        anp_adp = ANPAdapter(
            httpx_client=router_ba._httpx_client,
            target_did=server_did,
            local_did_info=local_did_info,
            host_domain="127.0.0.1",                # local SimpleNode domain
            host_port=str(local_ws_port),           # local SimpleNode port
            host_ws_path="/ws",                     # must match DID doc path
            did_service_url=os.getenv("ANP_DID_SERVICE_URL"),   # for did:wba support
            did_api_key=os.getenv("ANP_DID_API_KEY"),           # for did:wba support
            enable_protocol_negotiation=False,      # can enable later
            enable_e2e_encryption=True
        )
        await anp_adp.initialize()
        router_ba.add_outbound_adapter(dst_id, anp_adp)

    async def send_to_worker(self, worker_id: str, question: str) -> Dict[str, Any]:
        """
        Send via network using BaseAgent.send(router -> worker).
        Requires install_outbound_adapters() has been called to
        install proper outbound adapters on the router agent.
        """
        import time
        start_time = time.time()

        if worker_id not in self.meta_agents:
            return {"answer": None, "raw": {"error": f"Worker {worker_id} not found"}, "protocol": "unknown", "response_time": 0.0, "success": False}

        # Ensure router chosen
        router_id = getattr(self, "_router_id", None)
        if router_id is None:
            await self.install_outbound_adapters()
            router_id = getattr(self, "_router_id", None)
            if router_id is None:
                return {"answer": None, "raw": {"error": "No router available"}, "protocol": "unknown", "response_time": 0.0, "success": False}

        dst_protocol = self.protocol_types[worker_id]
        router_meta = self.meta_agents[router_id]
        router_ba = router_meta.base_agent

        try:
            payload = {"text": question, "question": question}
            content = await router_ba.send(worker_id, payload)  # <-- over the network
            dt = time.time() - start_time

            # update stats
            stats = self.protocol_stats[worker_id]
            stats["questions_processed"] += 1
            stats["total_response_time"] += dt
            stats["avg_response_time"] = stats["total_response_time"] / max(1, stats["questions_processed"])

            # Extract text content from response
            answer_text = None
            if isinstance(content, dict):
                # Try different content keys
                answer_text = content.get("text") or content.get("content") or content.get("answer")
                if not answer_text and "content" in content and isinstance(content["content"], dict):
                    answer_text = content["content"].get("text")
                
                # Handle A2A events structure
                if not answer_text and "events" in content and isinstance(content["events"], list):
                    events = content["events"]
                    for event in events:
                        if isinstance(event, dict) and "parts" in event:
                            parts = event["parts"]
                            for part in parts:
                                if isinstance(part, dict) and part.get("kind") == "text":
                                    answer_text = part.get("text", "")
                                    break
                            if answer_text:
                                break
                    pass  # Successfully extracted from A2A events
            elif isinstance(content, str):
                answer_text = content
            
            if not answer_text:
                logger.warning(f"[META-COORDINATOR] No text found in response from {worker_id}: {content}")
                answer_text = str(content)

            return {"answer": answer_text, "raw": {"response": content, "protocol": dst_protocol}, "protocol": dst_protocol, "response_time": dt, "success": True}

        except Exception as e:
            dt = time.time() - start_time
            self.protocol_stats[worker_id]["errors"] += 1
            logger.error(f"[META-COORDINATOR] Network send {router_id} -> {worker_id} failed: {e}")
            import traceback
            logger.error(f"[META-COORDINATOR] Traceback: {traceback.format_exc()}")
            return {"answer": None, "raw": {"error": str(e), "protocol": dst_protocol}, "protocol": dst_protocol, "response_time": dt, "success": False}
    
    async def get_protocol_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all protocols."""
        return {
            "total_agents": len(self.meta_agents),
            "protocols": list(set(self.protocol_types.values())),
            "agent_stats": dict(self.protocol_stats),
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
                                      if self.protocol_types[aid] == protocol)
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
        logger.info("[META-COORDINATOR] Closing all meta-protocol agents")
        
        for agent_id, agent in self.meta_agents.items():
            try:
                await agent.close()
                logger.info(f"[META-COORDINATOR] Closed {self.protocol_types[agent_id]} agent {agent_id}")
            except Exception as e:
                logger.error(f"[META-COORDINATOR] Error closing {agent_id}: {e}")
        
        self.meta_agents.clear()
        self.protocol_types.clear()
        self.worker_ids.clear()


async def create_meta_protocol_network(config: Dict[str, Any]) -> MetaProtocolCoordinator:
    """
    Factory function to create a complete meta-protocol network.
    
    Args:
        config: Configuration dict with protocol-specific settings
    
    Returns:
        Configured MetaProtocolCoordinator with all agents
    """
    coordinator = MetaProtocolCoordinator(config)
    
    # Extract protocol configurations
    protocols_config = config.get("protocols", {})
    base_port = config.get("base_port", 8080)
    
    # Add agents for each configured protocol
    port_offset = 0
    for protocol in ["acp", "anp", "agora", "a2a"]:
        if protocol in protocols_config:
            agent_config = protocols_config[protocol]
            agent_id = f"{protocol.upper()}-Meta-Worker-{uuid.uuid4().hex[:8]}"
            port = base_port + port_offset
            
            try:
                await coordinator.add_protocol_agent(protocol, agent_id, agent_config, port)
                port_offset += 1
            except Exception as e:
                logger.warning(f"[META-NETWORK] Failed to add {protocol} agent: {e}")
    
    logger.info(f"[META-NETWORK] Created meta-protocol network with {len(coordinator.meta_agents)} agents")
    return coordinator


# Test function
async def test_meta_protocol_coordination():
    """Test meta-protocol coordination with all four protocols."""
    print("🚀 Testing Meta-Protocol Coordination")
    print("=" * 50)
    
    # Test config for all protocols
    config = {
        "protocols": {
            "acp": {
                "core": {
                    "type": "openai",
                    "name": "gpt-4o",
                    "openai_api_key": "test-key",
                    "openai_base_url": "https://api.openai.com/v1",
                    "temperature": 0.0
                }
            },
            "anp": {
                "core": {
                    "type": "openai", 
                    "name": "gpt-4o",
                    "openai_api_key": "test-key",
                    "openai_base_url": "https://api.openai.com/v1",
                    "temperature": 0.0
                }
            },
            "agora": {
                "core": {
                    "type": "openai",
                    "name": "gpt-4o", 
                    "openai_api_key": "test-key",
                    "openai_base_url": "https://api.openai.com/v1",
                    "temperature": 0.0
                }
            },
            "a2a": {
                "core": {
                    "type": "openai",
                    "name": "gpt-4o",
                    "openai_api_key": "test-key", 
                    "openai_base_url": "https://api.openai.com/v1",
                    "temperature": 0.0
                }
            }
        },
        "base_port": 8090,
        "qa": {
            "coordinator": {
                "batch_size": 10,
                "first_50": True,
                "data_file": "data/top1000_simplified.jsonl",
                "result_file": "data/qa_results_meta.json"
            }
        }
    }
    
    coordinator = None
    try:
        print("📝 Creating meta-protocol network...")
        coordinator = await create_meta_protocol_network(config)
        
        print(f"✅ Created network with {len(coordinator.meta_agents)} agents")
        
        # Install outbound adapters for network routing
        print("\n🔗 Installing outbound adapters for network routing...")
        await coordinator.install_outbound_adapters()
        print("✅ Outbound adapters installed")
        
        # Health check
        print("\n🔍 Checking health of all agents...")
        health = await coordinator.health_check_all()
        for agent_id, status in health.items():
            print(f"  {agent_id} ({status['protocol']}): {status['status']}")
        
        # Test question dispatch
        print("\n🧪 Testing question dispatch...")
        test_question = "What is the capital of France?"
        
        for agent_id in coordinator.worker_ids:
            protocol = coordinator.protocol_types[agent_id]
            print(f"  Testing {protocol.upper()} agent...")
            result = await coordinator.send_to_worker(agent_id, test_question)
            if result["success"]:
                answer_text = str(result['answer']) if result['answer'] else "No response"
                print(f"    ✅ Response: {answer_text[:50]}... ({result['response_time']:.2f}s)")
            else:
                print(f"    ❌ Error: {result['raw'].get('error', 'Unknown error')}")
        
        # Protocol stats
        print("\n📊 Protocol Statistics:")
        stats = await coordinator.get_protocol_stats()
        for protocol, summary in stats["summary"].items():
            print(f"  {protocol.upper()}: {summary['agents']} agents, "
                  f"{summary['total_questions']} questions, "
                  f"{summary['avg_response_time']:.2f}s avg")
        
        print("\n✅ Meta-protocol coordination test completed successfully!")
        
    except Exception as e:
        print(f"❌ Meta-protocol coordination test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if coordinator:
            try:
                print("\n🧹 Cleaning up...")
                await coordinator.close_all()
                print("✅ Cleanup completed")
            except Exception as cleanup_e:
                print(f"⚠️ Cleanup error: {cleanup_e}")


if __name__ == "__main__":
    asyncio.run(test_meta_protocol_coordination())
