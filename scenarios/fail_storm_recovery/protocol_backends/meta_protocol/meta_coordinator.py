"""
Meta-Protocol Coordinator with LLM-based Intelligent Routing

Unified coordinator that uses LLM to intelligently select protocols based on task requirements.
Integrates with fail_storm_recovery protocol implementations and supports dynamic protocol selection
with actual performance data from fail storm recovery testing.
"""

import asyncio
import uuid
import logging
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Core imports
import sys
from pathlib import Path

# Add fail_storm_recovery to path
current_file = Path(__file__).resolve()
fail_storm_path = current_file.parents[2]  # Go up to fail_storm_recovery
project_root = fail_storm_path.parent.parent  # Go up to agent_network
src_path = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(fail_storm_path))
sys.path.insert(0, str(src_path))

# Import from fail_storm_recovery core
try:
    from core.failstorm_metrics import FailStormMetricsCollector
except ImportError as e:
    raise ImportError(f"FailStorm core metrics required but not available: {e}")

# Import Meta-specific performance metrics
from .meta_performance_metrics import MetaPerformanceMetricsCollector

# Meta-protocol agent imports
from .acp_agent import create_acp_meta_worker, ACPMetaAgent
from .anp_agent import create_anp_meta_worker, ANPMetaAgent
from .agora_agent import create_agora_meta_worker, AgoraMetaAgent
from .a2a_agent import create_a2a_meta_worker, A2AMetaAgent

# LLM router import
from .llm_router import LLMIntelligentRouter, RoutingDecision

logger = logging.getLogger(__name__)


class MetaProtocolCoordinator:
    """
    Intelligent meta-protocol coordinator with LLM-based routing for fail_storm_recovery.
    
    This coordinator uses LLM to analyze tasks and intelligently select protocols based on
    actual fail_storm_recovery performance data:
    - ANP: 61.0% success, 22.0% answer rate, 6.76s avg response time
    - Agora: 60.0% success, 20.0% answer rate, 7.10s avg response time  
    - A2A: 59.6% success, 19.1% answer rate, 7.39s avg response time
    - ACP: 59.0% success, 17.9% answer rate, 7.83s avg response time
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, output=None):
        self.config = config or {}
        self.output = output
        
        # Initialize FailStorm metrics
        self.failstorm_metrics_collector = FailStormMetricsCollector(
            protocol_name="meta_protocol",
            config=self.config
        )
        
        # Meta-protocol metrics collector
        self.meta_metrics_collector = MetaPerformanceMetricsCollector()
        qa_cfg = self.config.get("qa", {}) or {}
        network_cfg = qa_cfg.get("network", {}) or {}
        self.meta_metrics_collector.response_timeout = float(network_cfg.get("response_timeout", 60))
        
        # Meta-protocol agent instances
        self.meta_agents: Dict[str, Any] = {}  # agent_id -> MetaAgent instance
        self.protocol_types: Dict[str, str] = {}  # agent_id -> protocol type
        self.worker_ids: List[str] = []
        
        # Performance tracking (legacy compatibility)
        self.protocol_stats: Dict[str, Dict[str, Any]] = {}
        
        # LLM-based intelligent router
        self.llm_router = LLMIntelligentRouter()
        self.llm_client = None
        self._initialize_llm_client()
        
        # Result file path
        self.result_file = self.config.get("qa", {}).get("coordinator", {}).get("result_file", "data/qa_results_meta.json")
        
        logger.info("[META-COORDINATOR] Initialized intelligent meta-protocol coordinator with LLM routing for fail_storm_recovery")
    
    def _initialize_llm_client(self):
        """Initialize LLM client for routing decisions."""
        try:
            # Load config from fail_storm_recovery
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
            if not config_path.exists():
                config_path = Path(__file__).parent.parent.parent / "configs" / "config_meta.yaml"
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # Support both 'llm' (new) and 'core' (legacy) config keys
                llm_config = config.get("llm") or config.get("core", {})
                api_key = llm_config.get("openai_api_key", "")
                base_url = llm_config.get("openai_base_url", "https://api.openai.com/v1")
                model = llm_config.get("model") or llm_config.get("name", "gpt-4o")
            else:
                # Use config from init parameter - support both 'llm' and 'core' keys
                llm_config = self.config.get("llm") or self.config.get("core", {})
                api_key = llm_config.get("openai_api_key", "")
                base_url = llm_config.get("openai_base_url", "https://api.openai.com/v1")
                model = llm_config.get("model") or llm_config.get("name", "gpt-4o")
            
            if not api_key:
                logger.warning("[META-COORDINATOR] No OpenAI API key found, LLM routing disabled")
                return
            
            # Create simple LLM client
            class SimpleLLMClient:
                def __init__(self, api_key, base_url, model):
                    self.api_key = api_key
                    self.base_url = base_url
                    self.model = model
                
                async def ask_tool(self, messages, tools, tool_choice):
                    import aiohttp
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": self.model,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": tool_choice,
                        "temperature": 0.0
                    }
                    endpoint = f"{self.base_url}/chat/completions"
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(endpoint, headers=headers, json=payload) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(f"API call failed: {response.status} - {error_text}")
                            result = await response.json()
                            return result["choices"][0]["message"]
            
            self.llm_client = SimpleLLMClient(api_key, base_url, model)
            self.llm_router.set_llm_client(self.llm_client)
            
            logger.info(f"[META-COORDINATOR] LLM client initialized: {model}")
            
        except Exception as e:
            logger.error(f"[META-COORDINATOR] Failed to initialize LLM client: {e}")
            self.llm_client = None
    
    def _initialize_worker_stats(self, worker_ids: List[str]):
        """Initialize protocol stats for all workers"""
        for worker_id in worker_ids:
            if worker_id not in self.protocol_stats:
                protocol_type = self.protocol_types.get(worker_id, "unknown")
                self.protocol_stats[worker_id] = {
                    "questions_processed": 0,
                    "total_response_time": 0.0,
                    "avg_response_time": 0.0,
                    "errors": 0,
                    "protocol": protocol_type
                }
                
                # Register worker with correct protocol in metrics collector
                if hasattr(self, 'meta_metrics_collector') and self.meta_metrics_collector:
                    self.meta_metrics_collector.register_worker(worker_id, protocol_type)
    
    async def create_agents_with_llm_routing(self, sample_tasks: List[Dict[str, Any]] = None) -> List[str]:
        """
        Use LLM to analyze tasks and create optimal protocol agents.
        
        Args:
            sample_tasks: Sample tasks to analyze for protocol selection
            
        Returns:
            List of created agent IDs
        """
        logger.info("[META-COORDINATOR] Starting LLM-based agent creation")
        
        # If no sample tasks provided, use default fail_storm_recovery task
        if not sample_tasks:
            sample_tasks = [{
                "question": "Fail storm recovery test with diverse question types",
                "context": "High-volume QA processing for fault tolerance testing", 
                "metadata": {"type": "fail_storm_test", "priority": "resilience", "volume": "high"}
            }]
        
        # Use first task as representative for routing decision
        representative_task = sample_tasks[0]
        
        try:
            # Get LLM routing decision for 4 agents
            routing_decision = await self.llm_router.route_task_with_llm(representative_task, num_agents=4)
            
            # Record LLM routing decision in Meta metrics
            self.meta_metrics_collector.record_llm_routing_decision({
                "selected_protocols": routing_decision.selected_protocols,
                "agent_assignments": routing_decision.agent_assignments,
                "reasoning": routing_decision.reasoning,
                "confidence": routing_decision.confidence,
                "strategy": routing_decision.strategy
            })
            
            logger.info(f"[META-COORDINATOR] LLM routing decision:")
            logger.info(f"  Selected protocols: {routing_decision.selected_protocols}")
            logger.info(f"  Agent assignments: {routing_decision.agent_assignments}")
            logger.info(f"  Strategy: {routing_decision.strategy}")
            logger.info(f"  Confidence: {routing_decision.confidence:.2%}")
            logger.info(f"  Reasoning: {routing_decision.reasoning[:100]}...")
            
            # Create agents based on LLM decision
            created_agents = []
            base_port = 10001
            
            for agent_id, protocol in routing_decision.agent_assignments.items():
                try:
                    port = base_port + len(created_agents)
                    agent_url = await self.add_protocol_agent(protocol, agent_id, self.config or {}, port)
                    created_agents.append(agent_id)
                    
                    logger.info(f"[META-COORDINATOR] Created {protocol.upper()} agent: {agent_id} @ {agent_url}")
                    
                except Exception as e:
                    logger.error(f"[META-COORDINATOR] Failed to create {protocol} agent {agent_id}: {e}")
                    continue
            
            logger.info(f"[META-COORDINATOR] Successfully created {len(created_agents)} agents based on LLM routing")
            return created_agents
            
        except Exception as e:
            logger.error(f"[META-COORDINATOR] LLM routing failed, using fallback: {e}")
            # Fallback to default optimal agents based on fail_storm_recovery data
            return await self._create_default_optimal_agents()
    
    async def _create_default_optimal_agents(self) -> List[str]:
        """Create default optimal agents based on fail_storm_recovery performance data when LLM routing fails."""
        logger.info("[META-COORDINATOR] Creating default optimal agents based on fail_storm_recovery data")
        
        # Default assignment based on actual performance data:
        # ANP: Highest success rate (61.0%) and answer rate (22.0%)
        # Agora: Second highest success rate (60.0%) and good answer rate (20.0%)
        # A2A: Good throughput (178 tasks) and fast recovery (6.0s)
        # ACP: Best recovery performance (0.70s avg) and fault tolerance (8.0s recovery)
        default_assignments = {
            "OptimalAgent-1": "anp",    # Best accuracy and answer rate
            "OptimalAgent-2": "agora",  # Best throughput and second-best accuracy
            "OptimalAgent-3": "a2a",    # High volume and fast recovery
            "OptimalAgent-4": "acp"     # Best fault recovery performance
        }
        
        created_agents = []
        base_port = 10001
        
        for agent_id, protocol in default_assignments.items():
            try:
                port = base_port + len(created_agents)
                agent_url = await self.add_protocol_agent(protocol, agent_id, self.config or {}, port)
                created_agents.append(agent_id)
                
                logger.info(f"[META-COORDINATOR] Created default {protocol.upper()} agent: {agent_id}")
                
            except Exception as e:
                logger.error(f"[META-COORDINATOR] Failed to create default {protocol} agent: {e}")
                continue
        
        return created_agents
    
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
                # Try DID-based ANP outbound adapter first (with HTTP fallback URL)
                try:
                    await self._install_anp_outbound_adapter(router_ba, dst_id, url)
                    logger.info("[META-COORDINATOR] Installed ANP DID outbound adapter: %s", dst_id)
                    continue
                except Exception as e:
                    logger.warning("[META-COORDINATOR] ANP DID adapter failed: %s", e)
                    
                    # Fallback to simple HTTP adapter for development/testing (DID missing)
                    try:
                        logger.warning(f"[META-COORDINATOR] ANP DID missing; falling back to simple HTTP adapter for {dst_id} (dev-only)")
                        fallback_adapter = self._create_simple_http_adapter(router_ba, url, dst_id)
                        await fallback_adapter.initialize()
                        router_ba.add_outbound_adapter(dst_id, fallback_adapter)
                        logger.info("[META-COORDINATOR] Installed ANP simple HTTP fallback adapter: %s", dst_id)
                    except Exception as fallback_e:
                        logger.error("[META-COORDINATOR] ANP HTTP fallback also failed for %s: %s", dst_id, fallback_e)
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

    def _create_simple_http_adapter(self, src_base_agent, dst_url: str, dst_id: str):
        """Create an ANP-HTTP adapter for ANP fallback (dev-only)."""
        
        class ANPHTTPAdapter:
            """ANP-HTTP adapter that maintains ANP protocol semantics over HTTP transport."""
            
            def __init__(self, httpx_client, base_url: str, agent_id: str):
                self.httpx_client = httpx_client
                self.base_url = base_url.rstrip('/')
                self.agent_id = agent_id
                self.protocol_name = "anp"  # Use ANP encoding (transport is HTTP fallback)
                
            async def initialize(self):
                """Initialize adapter - check if target ANP agent is reachable."""
                try:
                    response = await self.httpx_client.get(f"{self.base_url}/health", timeout=5.0)
                    if response.status_code == 200:
                        return
                except:
                    pass
                # If /health fails, that's ok for ANP-HTTP adapter
                
            async def send_message(self, dst_id: str, payload):
                """Send message using ANP-compatible format over HTTP."""
                import json
                import time
                import uuid
                
                # Convert to ANP agent expected format (simple content + meta)
                anp_message = {
                    "content": payload,  # ANP agent expects 'content' field
                    "sender": self.agent_id,
                    "meta": {
                        "id": str(uuid.uuid4()),
                        "timestamp": time.time(),
                        "transport": "anp_http_fallback"
                    }
                }
                
                # Try ANP-specific endpoints first, then fallback
                endpoints = ["/anp/message", "/message", "/"]
                
                for endpoint in endpoints:
                    try:
                        headers = {
                            "Content-Type": "application/json",
                            "X-Protocol": "ANP-HTTP",  # Indicate ANP protocol over HTTP
                            "X-Agent-ID": self.agent_id
                        }
                        
                        response = await self.httpx_client.post(
                            f"{self.base_url}{endpoint}",
                            json=anp_message,
                            headers=headers,
                            timeout=30.0
                        )
                        response.raise_for_status()
                        return response.json()
                    except Exception as e:
                        if endpoint != endpoints[-1]:  # Not the last endpoint
                            continue  # Try next endpoint
                        else:
                            raise e  # All endpoints failed
                
                raise ConnectionError(f"All ANP-HTTP endpoints failed for {self.base_url}")
            
            async def receive_message(self):
                """Receive message (not used in outbound adapter)."""
                return {}
            
            def get_agent_card(self):
                """Get agent card (not used in outbound adapter)."""
                return {"agent_id": self.agent_id, "protocol": "anp_http"}
            
            async def cleanup(self):
                """Cleanup adapter resources (no-op for HTTP adapter)."""
                pass
        
        return ANPHTTPAdapter(src_base_agent._httpx_client, dst_url, dst_id)

    async def _install_anp_outbound_adapter(self, router_ba, dst_id: str, base_url: str) -> None:
        """
        Install a DID-based ANP outbound adapter on the router BaseAgent with HTTP fallback.
        
        Args:
            router_ba: Router BaseAgent to install adapter on
            dst_id: Destination agent ID
            base_url: HTTP base URL for fallback (when DID resolution fails)
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
        from agent_connect.utils.did_generate import did_generate
        from agent_connect.utils.crypto_tool import get_pem_from_private_key
        
        local_ws_port = _find_free_port()
        local_ws_endpoint = f"ws://127.0.0.1:{local_ws_port}/ws"
        private_key, _, local_did, did_document_json = did_generate(local_ws_endpoint)
        local_did_info = {
            "private_key_pem": get_pem_from_private_key(private_key),
            "did": local_did,
            "did_document_json": did_document_json
        }

        # 3) build ANPAdapter with target DID + local DID info + HTTP fallback
        from src.agent_adapters.anp_adapter import ANPAdapter
        import os
        
        anp_adp = ANPAdapter(
            httpx_client=router_ba._httpx_client,
            target_did=server_did,
            local_did_info=local_did_info,
            target_http_base_url=base_url,         # HTTP fallback URL
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
    
    async def save_results(self, results: List[Dict[str, Any]]) -> None:
        """Save results with Meta-specific performance metrics"""
        try:
            from pathlib import Path
            import json
            import time
            
            # If是相对路径，相对于 fail_storm_recovery 目录
            if not Path(self.result_file).is_absolute():
                current_file = Path(__file__).resolve()
                fail_storm_dir = current_file.parent.parent.parent  # 从 meta_protocol 回到 fail_storm_recovery
                p = fail_storm_dir / self.result_file
            else:
                p = Path(self.result_file)
            p.parent.mkdir(parents=True, exist_ok=True)

            # Basic statistics
            times = [r["response_time"] for r in results if r.get("response_time")]
            avg_rt = (sum(times) / len(times)) if times else 0.0
            
            # Get comprehensive Meta performance metrics
            performance_report = self.meta_metrics_collector.get_comprehensive_report()
            
            # Include FailStorm metrics
            failstorm_report = self.failstorm_metrics_collector.get_final_results()
            
            payload = {
                "metadata": {
                    "total_questions": len(results),
                    "successful_questions": sum(1 for r in results if r["status"] == "success"),
                    "failed_questions": sum(1 for r in results if r["status"] == "failed"),
                    "average_response_time": avg_rt,
                    "timestamp": time.time(),
                    "network_type": self.__class__.__name__,
                    "scenario": "fail_storm_recovery_meta_protocol"
                },
                "detailed_performance_metrics": performance_report,
                "failstorm_metrics": failstorm_report,
                "meta_protocol_stats": self.protocol_stats,  # Include legacy stats
                "results": results,
            }
            p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            self._o("success", f"Meta results saved to: {p}")
            
            # Print Meta-specific performance summary
            self.meta_metrics_collector.print_meta_performance_summary(self.output)
            
        except Exception as e:
            self._o("error", f"Failed to save Meta results: {e}")
    
    def _o(self, level: str, message: str):
        """Output helper"""
        if self.output and hasattr(self.output, level):
            getattr(self.output, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
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
        config: Configuration dict with protocol-specific settings and llm config
    
    Returns:
        Configured MetaProtocolCoordinator with all agents
    """
    coordinator = MetaProtocolCoordinator(config)
    
    # Extract protocol configurations
    protocols_config = config.get("protocols", {})
    
    # If no protocols specified, enable all protocols with default config
    if not protocols_config:
        protocols_config = {
            "anp": {"enabled": True},
            "agora": {"enabled": True},
            "a2a": {"enabled": True},
            "acp": {"enabled": True}
        }
    
    # Get base port from network or agents config
    base_port = config.get("network", {}).get("base_port") or config.get("agents", {}).get("base_port", 8080)
    
    # Add agents for each configured protocol
    port_offset = 0
    for protocol in ["anp", "agora", "a2a", "acp"]:  # Order by performance from fail_storm_recovery data
        if protocol in protocols_config:
            protocol_conf = protocols_config[protocol]
            
            # Get agent_count for this protocol (default to 1)
            agent_count = protocol_conf.get("agent_count", 1)
            
            # Create multiple agents for this protocol
            for agent_num in range(agent_count):
                # Pass the complete config (includes llm, network, etc.)
                agent_id = f"{protocol.upper()}-Meta-Worker-{uuid.uuid4().hex[:8]}"
                port = base_port + port_offset
                
                try:
                    await coordinator.add_protocol_agent(protocol, agent_id, config, port)
                    port_offset += 1
                except Exception as e:
                    logger.warning(f"[META-NETWORK] Failed to add {protocol} agent {agent_num+1}/{agent_count}: {e}")
    
    logger.info(f"[META-NETWORK] Created meta-protocol network with {len(coordinator.meta_agents)} agents")
    return coordinator
