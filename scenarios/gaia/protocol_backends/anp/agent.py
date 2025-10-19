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
import uvicorn
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
    
    from agent_connect.simple_node import SimpleNode, SimpleNodeSession
    from agent_connect.utils.did_generate import did_generate
    from agent_connect.utils.crypto_tool import get_pem_from_private_key
    from agent_connect.meta_protocol.meta_protocol import MetaProtocol, ProtocolType
    
    print("✅ ANP-SDK components imported successfully")
except ImportError as e:
    raise ImportError(f"ANP-SDK components required but not available: {e}")

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
        self._uvicorn_server: Optional[uvicorn.Server] = None
        # Pretty initialization output
        print(f"[{name}] ANP Agent on port {port}")
        print(f"[ANPAgent] Initialized with ANP-SDK")

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
        if not self._connected:
            await self._start_anp_node()
            self._connected = True
            self._log("ANPAgent connected and ANP node started")

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
            "anp_initialized": self.anp_initialized,
            "did": self.local_did_info.get('did', self.target_did),
            "websocket_endpoint": f"ws://{self.host_domain}:{self.host_port}{self.host_ws_path}",
            "node_running": self._node_task is not None and not self._node_task.done() if self._node_task else False
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
            return bool(self._connected and (self.anp_initialized or self._simple_node is not None))
        except Exception as e:
            self._log(f"Health check failed: {e}")
            return False

    # ==================== ANP Protocol Specific Methods ====================
    async def _start_anp_node(self):
        """
        Start ANP SimpleNode by directly creating and managing the Uvicorn server.
        This provides full control over the server's lifecycle and prevents SystemExit crashes.
        """
        try:
            if not self.local_did_info.get('did'):
                await self._generate_did()
            
            # 1. 实例化 SimpleNode 来准备 FastAPI app（必须提供 new_session_callback）
            self._simple_node = SimpleNode(
                host_domain=self.host_domain,
                new_session_callback=self._on_new_session,
                host_port=str(self.host_port),
                host_ws_path=self.host_ws_path,
                private_key_pem=self.local_did_info.get('private_key_pem'),
                did=self.local_did_info.get('did'),
                did_document_json=self.local_did_info.get('did_document_json')
                # 注意：SimpleNode 里的 SSL 参数在这里没有传递，如果需要请添加
            )
            
            # 2. 手动创建 uvicorn.Config，模仿 SimpleNode._run() 的行为
            #    这样可以避免 SimpleNode 内部调用 uvicorn.run，从而避免 SystemExit
            config = uvicorn.Config(
                app=self._simple_node.app, # 从 SimpleNode 获取 FastAPI 实例
                host="0.0.0.0",
                port=int(self.host_port),
                lifespan="off" # 保持与 SimpleNode 源码一致
                # 注意：SSL 配置也需要在这里添加，如果您的用例需要的话
            )

            # 3. 创建 uvicorn.Server 实例并保存它
            self._uvicorn_server = uvicorn.Server(config)

            # 4. 将服务器的运行作为一个可管理的后台任务
            self._node_task = asyncio.create_task(self._uvicorn_server.serve())
            
            # 等待服务器完成启动。uvicorn.Server.started 是一个标志位
            await asyncio.sleep(0.5)

            if not self._uvicorn_server.started:
                 raise RuntimeError(f"Uvicorn server for port {self.host_port} failed to start.")

            self.anp_initialized = True
            self._log(f"✅ ANP Uvicorn server started directly on {self.host_domain}:{self.host_port}")

        except Exception as e:
            self._log(f"❌ Failed to start ANP node: {e}")
            self.anp_initialized = False
            # 确保即使启动失败，任务也能被清理
            if self._node_task and not self._node_task.done():
                self._node_task.cancel()
            raise

    async def _on_new_session(self, session: SimpleNodeSession) -> None:
        """
        Handle new incoming WebSocket session from SimpleNode.
        This is required by SimpleNode's new_session_callback parameter.
        """
        try:
            remote_did = getattr(session, 'remote_did', 'unknown')
            self._log(f"✅ New ANP session established with DID: {remote_did}")
            # Store session if needed for future use
            if not hasattr(self, '_sessions'):
                self._sessions = {}
            self._sessions[remote_did] = session
        except Exception as e:
            self._log(f"⚠️ Error handling new session: {e}")

    async def _stop_anp_node(self):
        """Gracefully shut down the managed Uvicorn server."""
        # 优先关闭服务器
        if self._uvicorn_server and self._uvicorn_server.started:
            self._log(f"Shutting down Uvicorn server on port {self.host_port}...")
            # 这是 Uvicorn 官方推荐的优雅关闭方式
            self._uvicorn_server.should_exit = True
            
            # 等待服务器任务真正结束
            if self._node_task and not self._node_task.done():
                try:
                    # 等待任务完成，它会在 should_exit=True 后自行退出
                    await asyncio.wait_for(self._node_task, timeout=5.0)
                    self._log(f"✅ Uvicorn server on port {self.host_port} shut down.")
                except asyncio.TimeoutError:
                    self._log(f"⚠️ Uvicorn server on port {self.host_port} did not shut down in time, cancelling task.")
                    self._node_task.cancel()
                except asyncio.CancelledError:
                    pass # 任务已经被取消，是正常情况

        # 清理所有相关状态
        self._simple_node = None
        self._uvicorn_server = None
        self._node_task = None
        self.anp_initialized = False

    async def _generate_did(self):
        """Generate DID using ANP-SDK."""
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
                "initialized": self.anp_initialized,
                "connected": self._connected
            }
        }

