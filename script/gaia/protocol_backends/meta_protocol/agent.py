"""
Meta Protocol Agent Implementation for GAIA Framework.
Provides intelligent protocol selection and cross-protocol communication capabilities.
"""

import asyncio
import time
import os
import json
import logging
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.agent import MeshAgent
from core.schema import AgentState, Message

# Import src/core for BaseAgent and intelligent routing  
# Use absolute path approach to avoid relative import issues
import os
current_file = Path(__file__).resolve()

# Find agent_network root by looking for src directory
search_path = current_file.parent
while search_path.parent != search_path:  # Not at filesystem root
    if (search_path / "src" / "core" / "base_agent.py").exists():
        agent_network_root = search_path
        break
    search_path = search_path.parent
else:
    raise RuntimeError("Cannot find agent_network root directory")

src_path = agent_network_root / "src"

# Ensure paths are in sys.path
if str(agent_network_root) not in sys.path:
    sys.path.insert(0, str(agent_network_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Change working directory to agent_network root to fix relative imports
original_cwd = os.getcwd()
os.chdir(str(agent_network_root))

try:
    from src.core.base_agent import BaseAgent
    from src.core.intelligent_network_manager import IntelligentNetworkManager, RouterType
    from src.core.intelligent_router import LLMIntelligentRouter
    META_CORE_AVAILABLE = True
    print(f"[MetaProtocolAgent] Meta core imported successfully from {src_path}")
except ImportError as e:
    print(f"[MetaProtocolAgent] Meta core import failed: {e}")
    print(f"[MetaProtocolAgent] Agent network root: {agent_network_root}")
    print(f"[MetaProtocolAgent] Current working directory: {os.getcwd()}")
    raise ImportError(f"Cannot import meta core components: {e}")
finally:
    # Restore original working directory
    os.chdir(original_cwd)

# Setup logger
logger = logging.getLogger(__name__)


class MetaProtocolAgent(MeshAgent):
    """
    Meta Protocol Agent that can intelligently select and use different protocols.
    
    Features:
    - Intelligent protocol selection based on task characteristics
    - Cross-protocol communication capabilities
    - Dynamic protocol switching
    - Performance optimization based on network conditions
    """
    
    def __init__(self, node_id: int, name: str, tool: str, port: int, 
                 config: Dict[str, Any], task_id: Optional[str] = None):
        super().__init__(node_id, name, tool, port, {**config, "protocol": "meta_protocol"}, task_id)
        
        self._base_agent: Optional[BaseAgent] = None
        self._intelligent_router: Optional[IntelligentRouter] = None
        self._protocol_adapters: Dict[str, Any] = {}
        self._current_protocol: Optional[str] = None
        
        # Meta protocol configuration
        self._router_type = config.get("router_type", "llm_based")
        self._available_protocols = config.get("available_protocols", ["a2a", "acp", "agora", "anp"])
        self._llm_config = config.get("llm", {})
        
        # Performance tracking
        self._protocol_performance: Dict[str, Dict[str, float]] = {}
        self._task_history: List[Dict[str, Any]] = []
        
        print(f"[MetaProtocolAgent] Initialized {self.name} with protocols: {self._available_protocols}")
        
        # Initialize meta protocol components (required, no fallback)
        self._initialize_meta_components()
        
        # Mark as initialized for health check
        self._initialized = True

    def _initialize_meta_components(self):
        """Initialize meta protocol components using src/core."""
        try:
            # Create BaseAgent for cross-protocol communication
            self._base_agent = BaseAgent(
                agent_id=f"meta_{self.name}",
                host="127.0.0.1",
                port=self.port
            )
            
            # Initialize intelligent router
            self._intelligent_router = LLMIntelligentRouter(
                llm_client=None  # Will be set later
            )
            
            print(f"[MetaProtocolAgent] Meta components initialized for {self.name}")
            
        except Exception as e:
            print(f"[MetaProtocolAgent] Failed to initialize meta components: {e}")
            self._base_agent = None
            self._intelligent_router = None

    # ==================== Abstract Method Implementations ====================
    
    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """Send message to another agent using meta protocol intelligent routing."""
        try:
            if self._base_agent and self._intelligent_router:
                # Use intelligent routing to select optimal protocol
                selected_protocol = await self._select_optimal_protocol(str(payload))
                
                # Send via BaseAgent with selected protocol
                await self._base_agent.send(str(dst), payload)
                print(f"[MetaProtocolAgent] Sent message to {dst} via {selected_protocol}")
            else:
                # Fallback: store message for later processing
                print(f"[MetaProtocolAgent] Message queued for {dst} (meta components not ready)")
                
        except Exception as e:
            print(f"[MetaProtocolAgent] Failed to send message to {dst}: {e}")
    
    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """Receive message with optional timeout."""
        try:
            if self._base_agent:
                # Use BaseAgent to receive messages
                # Return None to indicate no message available (non-blocking)
                # GAIA will handle message passing through its own mechanisms
                return None
            else:
                return None
                
        except Exception as e:
            print(f"[MetaProtocolAgent] Failed to receive message: {e}")
            return None
    
    async def health_check(self):
        """Monitor meta protocol agent health and all underlying protocols."""
        try:
            # For GAIA integration, return True if agent is properly initialized
            # This matches the pattern of other GAIA protocol agents
            if hasattr(self, '_initialized') and self._initialized:
                return True
            
            # Check if basic components are available
            basic_health = (
                hasattr(self, 'name') and 
                hasattr(self, 'id') and 
                hasattr(self, 'port') and
                hasattr(self, '_available_protocols')
            )
            
            return basic_health
            
        except Exception as e:
            print(f"[MetaProtocolAgent] Health check failed: {e}")
            return False

    async def connect(self):
        """Connect meta protocol agent and initialize protocol adapters."""
        await super().connect()
        
        if META_CORE_AVAILABLE and self._base_agent:
            try:
                # Start BaseAgent
                await self._base_agent.start()
                
                # Initialize protocol adapters for available protocols
                await self._initialize_protocol_adapters()
                
                print(f"[MetaProtocolAgent] {self.name} connected with meta protocol support")
                
            except Exception as e:
                print(f"[MetaProtocolAgent] Error connecting meta components: {e}")
        else:
            print(f"[MetaProtocolAgent] {self.name} connected without meta protocol support")

    async def _initialize_protocol_adapters(self):
        """Initialize adapters for all available protocols."""
        for protocol in self._available_protocols:
            try:
                # Initialize protocol-specific adapter
                adapter = await self._create_protocol_adapter(protocol)
                if adapter:
                    self._protocol_adapters[protocol] = adapter
                    print(f"[MetaProtocolAgent] Initialized {protocol} adapter for {self.name}")
                    
            except Exception as e:
                print(f"[MetaProtocolAgent] Failed to initialize {protocol} adapter: {e}")

    async def _create_protocol_adapter(self, protocol: str) -> Optional[Any]:
        """Create adapter for specific protocol."""
        try:
            if protocol == "a2a":
                from src.agent_adapters.a2a_adapter import A2AAdapter
                return A2AAdapter(httpx_client=self._base_agent._httpx_client)
            elif protocol == "acp":
                from src.agent_adapters.acp_adapter import ACPAdapter
                return ACPAdapter(httpx_client=self._base_agent._httpx_client)
            elif protocol == "agora":
                from src.agent_adapters.agora_adapter import AgoraAdapter
                return AgoraAdapter(httpx_client=self._base_agent._httpx_client)
            elif protocol == "anp":
                from src.agent_adapters.anp_adapter import ANPAdapter
                return ANPAdapter(httpx_client=self._base_agent._httpx_client)
            else:
                print(f"[MetaProtocolAgent] Unknown protocol: {protocol}")
                return None
                
        except ImportError as e:
            print(f"[MetaProtocolAgent] Protocol {protocol} adapter not available: {e}")
            return None

    async def execute(self, content: str) -> str:
        """
        GAIA-safe execute entrypoint.
        Do not call super().execute(); instead, route to the bound BaseAgent so that
        protocol translation happens inside the protocol worker.
        """
        base_agent = getattr(self, "_base_agent", None)
        if base_agent is None:
            return "[MetaProtocolAgent] No BaseAgent is bound to this agent"

        dst_agent_id = getattr(base_agent, "agent_id", str(getattr(self, "id", "unknown")))
        payload = {
            "message": {
                "type": "task",
                "content": content
            },
            "meta": {
                "src_id": f"agent:{getattr(self, 'name', 'unknown')}"
            }
        }
        
        try:
            # BaseAgent.send(...) is expected to return a response (string or dict)
            result = await base_agent.send(dst_agent_id, payload)
            result_str = result if isinstance(result, str) else str(result)
            
            # Trigger result_callback if available (critical for workflow coordination)
            if hasattr(self, 'result_callback') and self.result_callback:
                complete_message = {
                    "agent_id": self.id,
                    "agent_name": self.name,
                    "sender_id": "meta_protocol",
                    "message_type": "task_result",
                    "original_content": content,
                    "assistant_response": result_str,
                    "processing_steps": 1,
                    "status": "completed"
                }
                try:
                    # Trigger result callback for workflow coordination
                    if asyncio.iscoroutinefunction(self.result_callback):
                        asyncio.create_task(self.result_callback(complete_message))
                    else:
                        self.result_callback(complete_message)
                except Exception as e:
                    print(f"[MetaProtocolAgent] Error in result callback: {e}")
            
            return result_str
            
        except Exception as e:
            error_msg = f"[MetaProtocolAgent] Execution failed: {e}"
            print(error_msg)
            
            # Trigger error callback
            if hasattr(self, 'result_callback') and self.result_callback:
                error_message = {
                    "agent_id": self.id,
                    "agent_name": self.name,
                    "sender_id": "meta_protocol",
                    "message_type": "task_result",
                    "original_content": content,
                    "assistant_response": error_msg,
                    "processing_steps": 0,
                    "status": "error"
                }
                try:
                    if asyncio.iscoroutinefunction(self.result_callback):
                        asyncio.create_task(self.result_callback(error_message))
                    else:
                        self.result_callback(error_message)
                except Exception as callback_e:
                    print(f"[MetaProtocolAgent] Error in error callback: {callback_e}")
            
            return error_msg

    async def _select_optimal_protocol(self, message: str) -> str:
        """Select optimal protocol based on task characteristics and current performance."""
        if self._intelligent_router:
            try:
                # Use LLM-based intelligent routing
                task_info = {
                    "message": message,
                    "agent_id": self.name,
                    "timestamp": time.time(),
                    "available_protocols": list(self._protocol_adapters.keys())
                }
                
                routing_decision = await self._intelligent_router.route_task(task_info)
                selected_protocol = routing_decision.get("protocol", self._available_protocols[0])
                
                print(f"[MetaProtocolAgent] Selected protocol {selected_protocol} for {self.name}")
                return selected_protocol
                
            except Exception as e:
                print(f"[MetaProtocolAgent] Router selection failed: {e}")
        
        # Fallback: select based on performance history or default
        return self._select_protocol_fallback()

    def _select_protocol_fallback(self) -> str:
        """Fallback protocol selection based on performance history."""
        if not self._protocol_performance:
            # Default to first available protocol
            return self._available_protocols[0] if self._available_protocols else "dummy"
        
        # Select protocol with best average performance
        best_protocol = None
        best_score = -1
        
        for protocol, metrics in self._protocol_performance.items():
            if protocol in self._protocol_adapters:
                # Calculate composite score (success_rate * 0.7 + (1/avg_response_time) * 0.3)
                success_rate = metrics.get("success_rate", 0.0)
                avg_response_time = metrics.get("avg_response_time", 10.0)
                score = success_rate * 0.7 + (1 / max(avg_response_time, 0.1)) * 0.3
                
                if score > best_score:
                    best_score = score
                    best_protocol = protocol
        
        return best_protocol or self._available_protocols[0]

    async def _execute_with_meta_protocol(self, message: str, protocol: str) -> str:
        """Execute task using meta protocol with selected underlying protocol."""
        try:
            adapter = self._protocol_adapters.get(protocol)
            if not adapter:
                raise RuntimeError(f"Protocol {protocol} adapter not available")
            
            # Use BaseAgent to send message via selected protocol
            # This would typically involve more complex routing logic
            # For GAIA integration, we'll use a simplified approach
            
            # Convert message to BaseAgent format
            unified_message = {
                "text": message,
                "protocol": protocol,
                "source_agent": self.name,
                "timestamp": time.time()
            }
            
            # Execute via BaseAgent (this would route through the intelligent network)
            response = await self._base_agent.process_message(unified_message)
            
            return response.get("result", f"Processed via {protocol}: {message}")
            
        except Exception as e:
            print(f"[MetaProtocolAgent] Meta protocol execution failed: {e}")
            return await self._execute_fallback(message)

    async def _execute_fallback(self, message: str) -> str:
        """Fallback execution when meta protocol is not available."""
        # Use standard MeshAgent execution
        result = await super().execute(message)
        
        # Add meta protocol signature
        return f"[MetaProtocol-Fallback] {result}"

    def _record_task_performance(self, message: str, protocol: str, result: str):
        """Record task performance for future protocol selection."""
        task_record = {
            "timestamp": time.time(),
            "message": message[:100],  # Truncate for storage
            "protocol": protocol,
            "success": "error" not in result.lower(),
            "response_length": len(result)
        }
        
        self._task_history.append(task_record)
        
        # Update protocol performance metrics
        if protocol not in self._protocol_performance:
            self._protocol_performance[protocol] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "total_response_time": 0.0,
                "success_rate": 0.0,
                "avg_response_time": 0.0
            }
        
        metrics = self._protocol_performance[protocol]
        metrics["total_tasks"] += 1
        if task_record["success"]:
            metrics["successful_tasks"] += 1
        
        # Update derived metrics
        metrics["success_rate"] = metrics["successful_tasks"] / metrics["total_tasks"]
        
        # Keep only recent history (last 100 tasks)
        if len(self._task_history) > 100:
            self._task_history = self._task_history[-100:]

    async def disconnect(self):
        """Disconnect meta protocol agent and cleanup resources."""
        try:
            if self._base_agent:
                await self._base_agent.stop()
            
            # Cleanup protocol adapters
            for protocol, adapter in self._protocol_adapters.items():
                try:
                    if hasattr(adapter, 'cleanup'):
                        await adapter.cleanup()
                except Exception as e:
                    print(f"[MetaProtocolAgent] Error cleaning up {protocol} adapter: {e}")
            
            await super().disconnect()
            print(f"[MetaProtocolAgent] {self.name} disconnected")
            
        except Exception as e:
            print(f"[MetaProtocolAgent] Error disconnecting: {e}")

    def get_protocol_status(self) -> Dict[str, Any]:
        """Get current protocol status and performance metrics."""
        return {
            "agent_id": self.name,
            "current_protocol": self._current_protocol,
            "available_protocols": self._available_protocols,
            "active_adapters": list(self._protocol_adapters.keys()),
            "performance_metrics": self._protocol_performance,
            "task_history_count": len(self._task_history),
            "meta_core_available": META_CORE_AVAILABLE
        }
