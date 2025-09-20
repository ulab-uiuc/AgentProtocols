"""
ANP Agent for GAIA Framework.
This agent integrates with ANP (Agent Network Protocol) using the original ANP-SDK.
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.agent import MeshAgent
from core.schema import AgentState

# Suppress noisy logs
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)  
logging.getLogger("starlette").setLevel(logging.ERROR)

# ANP-SDK imports (original ANP components)
try:
    # Import AgentConnect components directly
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    agentconnect_path = os.path.join(ROOT_DIR, 'agentconnect_src')
    if agentconnect_path not in sys.path:
        sys.path.insert(0, agentconnect_path)
    
    from agent_connect.python.simple_node.simple_node import SimpleNode
    from agent_connect.python.simple_node import SimpleNodeSession
    from agent_connect.python.authentication import DIDAllClient
    from agent_connect.python.utils.did_generate import did_generate
    from agent_connect.python.utils.crypto_tool import get_pem_from_private_key
    from agent_connect.python.meta_protocol.meta_protocol import MetaProtocol, ProtocolType
    
    ANP_SDK_AVAILABLE = True
    print("✅ ANP-SDK components imported successfully")
except ImportError:
    ANP_SDK_AVAILABLE = False
    SimpleNode = None
    SimpleNodeSession = None
    DIDAllClient = None
    did_generate = None
    get_pem_from_private_key = None
    MetaProtocol = None
    ProtocolType = None
    print("⚠️ ANP-SDK components not available")

logger = logging.getLogger(__name__)



class ANPAgent(MeshAgent):
    """
    ANP Protocol Agent that inherits from MeshAgent.
    The ANPNetwork is responsible for spawning ANP SimpleNode servers which
    handle ANP protocol communication via WebSocket and DID authentication.
    """

    # Allow setting extra attributes on Pydantic model instances
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}
    
    def __init__(self, node_id: int, name: str, tool: str, port: int, 
                 config: Dict[str, Any], task_id: Optional[str] = None):
        super().__init__(node_id, name, tool, port, config, task_id)

        # Runtime flags
        self._connected = False
        self._simple_node = None
        self._node_task = None
        self._uvicorn_server = None
        
        # ANP specific configuration
        self.anp_config = config.get("anp", {})
        self.host_domain = self.anp_config.get("host_domain", "127.0.0.1")
        self.host_port = self.anp_config.get("host_port", port)
        self.host_ws_path = "/ws"
        self.enable_encryption = self.anp_config.get("enable_encryption", True)
        self.enable_negotiation = self.anp_config.get("enable_negotiation", False)
        
        # DID information (will be set by network)
        self.local_did_info: Dict[str, str] = {}
        self.target_did = f"did:all:agent_{node_id}"  # Default, will be updated
        
        # Message queue for local communication
        self.message_queue = asyncio.Queue()
        
        # ANP state
        self.anp_initialized = False

        # Pretty initialization output
        print(f"[{name}] ANP Agent on port {port}")
        print(f"[ANPAgent] Initialized with ANP-SDK: {ANP_SDK_AVAILABLE}")

        self._log("ANPAgent initialized (network-managed ANP protocol)")

    @property
    def simple_node(self):
        """Expose SimpleNode for network backend compatibility."""
        return self._simple_node

    @simple_node.setter
    def simple_node(self, value):
        self._simple_node = value

    async def connect(self):
        """Start ANP SimpleNode and mark agent as connected."""
        if not self._connected and ANP_SDK_AVAILABLE:
            await self._start_anp_node()
            self._connected = True
            self._log("ANPAgent connected and ANP node started")
        elif not ANP_SDK_AVAILABLE:
            self._connected = True  # Fallback mode
            self._log("ANPAgent connected (fallback mode - no ANP-SDK)")

    async def disconnect(self):
        """Stop ANP SimpleNode and mark agent as disconnected."""
        if self._connected:
            await self._stop_anp_node()
            self._connected = False
            self._log("ANPAgent disconnected and ANP node stopped")

    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """No direct send from agent; ANPNetwork delivers via ANP backend."""
        self._log(f"send_msg called (dst={dst}) - handled by network backend")

    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """Receive messages from local queue (for network communication)."""
        try:
            if timeout == 0.0:
                if self.message_queue.empty():
                    return None
                return self.message_queue.get_nowait()
            else:
                return await asyncio.wait_for(self.message_queue.get(), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return None

    def get_connection_status(self) -> Dict[str, Any]:
        """Basic connection status for diagnostics."""
        return {
            "agent_id": self.id,
            "agent_name": self.name,
            "protocol": "anp",
            "connected": self._connected,
            "anp_sdk_available": ANP_SDK_AVAILABLE,
            "anp_initialized": self.anp_initialized,
            "did": self.local_did_info.get('did', self.target_did),
            "websocket_endpoint": f"ws://{self.host_domain}:{self.host_port}{self.host_ws_path}",
            "node_running": self._node_task is not None and not self._node_task.done() if self._node_task else False,
        }

    async def start(self):
        """Start the agent main loop."""
        await self.connect()
        await super().start()

    async def stop(self):
        """Stop the agent and cleanup resources."""
        await self.disconnect()
        await super().stop()

    # ==================== Execution Entry for ANP Agent ====================
    async def execute(self, message: str) -> str:
        """
        Entry point used by ANP network to process a request.
        Execute only ONE step to align with other protocol behavior.
        Network layer handles the workflow coordination.
        """
        try:
            # Log the incoming message
            self._log(f"🔄 Processing ANP request: {message[:100]}...")
            
            # Simple response generation (can be enhanced with actual tool execution)
            if self.tool_name == "create_chat_completion":
                response = await self._process_with_llm(message)
            elif self.tool_name == "search":
                response = await self._process_with_search(message)
            else:
                response = f"ANP Agent {self.name}: {message}"
            
            self._log(f"✅ ANP request processed, response length: {len(response)}")
            return response
            
        except Exception as e:
            error_msg = f"Error executing ANP request: {e}"
            self._log(error_msg)
            return error_msg
    
    async def _process_with_llm(self, message: str) -> str:
        """Process message using LLM (simplified implementation)."""
        try:
            # This is a placeholder for LLM integration
            return f"LLM response from {self.name}: {message}"
        except Exception as e:
            return f"LLM processing error: {e}"
    
    async def _process_with_search(self, message: str) -> str:
        """Process message using search tool (simplified implementation)."""
        try:
            # This is a placeholder for search tool integration
            return f"Search result from {self.name}: {message}"
        except Exception as e:
            return f"Search processing error: {e}"

    async def health_check(self) -> bool:
        """Check if the ANP agent is healthy and ready to process messages."""
        try:
            # If SDK不可用，仅检查连接标志
            if not ANP_SDK_AVAILABLE:
                return bool(self._connected)
            # SDK可用时，放宽判定，避免启动竞态导致误报
            return bool(self._connected and (self.anp_initialized or self._simple_node is not None))
        except Exception as e:
            self._log(f"Health check failed: {e}")
            return False

    # ==================== ANP Protocol Specific Methods ====================
    async def _start_anp_node(self):
        """Start ANP SimpleNode as a managed asyncio task."""
        if not ANP_SDK_AVAILABLE:
            return
        
        try:
            # Generate DID if not exists
            if not self.local_did_info.get('did'):
                await self._generate_did()
            
            # Create SimpleNode
            self._simple_node = SimpleNode(
                host_domain=self.host_domain,
                host_port=str(self.host_port),
                host_ws_path=self.host_ws_path,
                private_key_pem=self.local_did_info.get('private_key_pem'),
                did=self.local_did_info.get('did'),
                did_document_json=self.local_did_info.get('did_document_json')
            )
            
            # Start SimpleNode (non-blocking)
            self._simple_node.run()
            
            # Wait for node to be ready
            await asyncio.sleep(0.5)
            
            self.anp_initialized = True
            self._log(f"✅ ANP SimpleNode started as a background task on {self.host_domain}:{self.host_port}")
            
        except Exception as e:
            self._log(f"❌ Failed to start ANP node: {e}")
            self.anp_initialized = False
            if self._node_task and not self._node_task.done():
                self._node_task.cancel()
            raise

    async def _stop_anp_node(self):
        """Gracefully and robustly stop the ANP SimpleNode task."""
        if self._simple_node and hasattr(self._simple_node, 'stop'):
            try:
                # The stop() method should handle the Uvicorn server shutdown.
                await self._simple_node.stop()
                self._log("Called SimpleNode.stop()")
            except Exception as e:
                self._log(f"Warning: Error in SimpleNode.stop(): {e}")
        
        if self._node_task and not self._node_task.done():
            self._node_task.cancel()
            try:
                await self._node_task
            except asyncio.CancelledError:
                pass # This is expected
            except Exception as e:
                self._log(f"Warning: Exception during ANP task cleanup: {e}")

        # Give the OS a brief moment to release the network socket.
        # This can help prevent race conditions in rapid start/stop cycles.
        await asyncio.sleep(0.1)

        self._simple_node = None
        self._node_task = None
        self.anp_initialized = False
        self._log(f"ANP Node for port {self.host_port} has been shut down.")

    async def _generate_did(self):
        """Generate DID using ANP-SDK."""
        if not ANP_SDK_AVAILABLE or not did_generate:
            self.local_did_info = {
                'did': f"did:all:agent_{self.id}@{self.host_domain}:{self.host_port}",
                'private_key_pem': '',
                'did_document_json': '{}'
            }
            return
        
        try:
            ws_endpoint = f"ws://{self.host_domain}:{self.host_port}{self.host_ws_path}"
            private_key, _, did, did_document_json = did_generate(ws_endpoint)
            private_key_pem = get_pem_from_private_key(private_key)
            
            # Modify DID format to include domain:port
            if did.startswith("did:all:") and "@" not in did:
                bitcoin_address = did.replace("did:all:", "")
                did = f"did:all:{bitcoin_address}@{self.host_domain}:{self.host_port}"
                if did_document_json:
                    did_document_json = did_document_json.replace(f"did:all:{bitcoin_address}", did)
            
            self.local_did_info = {
                'private_key_pem': private_key_pem,
                'did': did,
                'did_document_json': did_document_json or '{}'
            }
            self.target_did = did
            
            self._log(f"✅ Generated DID: {did}")
            
        except Exception as e:
            self._log(f"❌ Failed to generate DID: {e}")
            # Fallback
            self.local_did_info = {
                'did': f"did:all:agent_{self.id}@{self.host_domain}:{self.host_port}",
                'private_key_pem': '',
                'did_document_json': '{}'
            }

    def get_agent_card(self) -> Dict[str, Any]:
        """Get agent card information."""
        return {
            "id": self.local_did_info.get('did', self.target_did),
            "gaia_id": self.id,
            "gaia_name": self.name,
            "protocol": "ANP",
            "host": f"{self.host_domain}:{self.host_port}",
            "websocket_endpoint": f"ws://{self.host_domain}:{self.host_port}{self.host_ws_path}",
            "did_format": "did:all:address@domain:port",
            "features": {
                "did_auth": True,
                "e2e_encryption": self.enable_encryption,
                "protocol_negotiation": self.enable_negotiation,
                "websocket": True,
                "anp_sdk": ANP_SDK_AVAILABLE,
                "initialized": self.anp_initialized,
                "connected": self._connected
            }
        }

