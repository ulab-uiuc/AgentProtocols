"""
BaseAgent v1.0.0 - Dual-role agent with server and multi-client capabilities
"""

import asyncio
import json
import socket
import time
import warnings
from typing import Any, Dict, Optional, Set, Union, List
from urllib.parse import urlparse

import httpx
import uvicorn

try:
    # Try package-style imports first
    from .agent_adapters.base_adapter import BaseProtocolAdapter
    from .agent_adapters.a2a_adapter import A2AAdapter
    from .agent_adapters.agent_protocol_adapter import AgentProtocolAdapter
    from .metrics import REQUEST_LATENCY, REQUEST_FAILURES, MSG_BYTES, MetricsTimer
    from .server_adapters import BaseServerAdapter, A2AServerAdapter, AgentProtocolServerAdapter
except ImportError:
    # Fallback to direct imports for standalone execution
    from agent_adapters.base_adapter import BaseProtocolAdapter
    from agent_adapters.a2a_adapter import A2AAdapter
    from agent_adapters.agent_protocol_adapter import AgentProtocolAdapter
    from metrics import REQUEST_LATENCY, REQUEST_FAILURES, MSG_BYTES, MetricsTimer
    from server_adapters import BaseServerAdapter, A2AServerAdapter, AgentProtocolServerAdapter

# Module-level constants for better reusability
DEFAULT_SERVER_STARTUP_TIMEOUT = 10.0
DEFAULT_SERVER_SHUTDOWN_TIMEOUT = 5.0


def is_sdk_native_executor(obj) -> bool:
    """
    Check if an object implements the SDK native executor interface.
    
    SDK native interface should have:
    async def execute(self, context, event_queue) -> None
    """
    import inspect
    
    if not hasattr(obj, "execute"):
        return False
    
    try:
        sig = inspect.signature(obj.execute)
        param_names = list(sig.parameters.keys())
        
        # Check for: self, context, event_queue (at minimum)
        # Allow additional parameters but require these first 3
        if len(param_names) < 2:  # self is implicit, so we need at least 2 more
            return False
        
        # Check parameter names (excluding 'self' which is implicit)
        return param_names[:2] == ["context", "event_queue"]
        
    except Exception:
        return False



class BaseAgent:
    """
    v1.0.0: Dual-role agent that acts as both server (receives messages) 
    and multi-client (sends messages to different targets using different protocols).
    
    Note: Now only supports A2A SDK native executors with the interface:
    async def execute(context: RequestContext, event_queue: EventQueue) -> None
    """

    def __init__(
        self,
        agent_id: str,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        httpx_client: Optional[httpx.AsyncClient] = None,
        server_adapter: Optional[BaseServerAdapter] = None
    ):
        """
        Initialize BaseAgent with dual-role capabilities.
        
        Parameters
        ----------
        agent_id : str
            Unique identifier for this agent in the network
        host : str
            Server listening host address
        port : Optional[int]
            Server listening port (auto-assigned if None)
        httpx_client : Optional[httpx.AsyncClient]
            Shared HTTP client for connection pooling
        server_adapter : Optional[BaseServerAdapter]
            Protocol-specific server adapter (defaults to A2AServerAdapter)
        """
        self.agent_id = agent_id
        self._host = host
        self._port = port or self._find_free_port()
        self._httpx_client = httpx_client or httpx.AsyncClient(timeout=30.0)
        self._server_adapter = server_adapter or A2AServerAdapter()
        
        # Multi-adapter support: dst_id -> adapter
        self._outbound: Dict[str, BaseProtocolAdapter] = {}
        
        # Server components
        self._server_task: Optional[asyncio.Task] = None
        self._server_instance: Optional[uvicorn.Server] = None
        self._self_agent_card: Optional[Dict[str, Any]] = None
        
        # Compatibility (TODO: remove in v2.0.0)
        self.outgoing_edges: Set[str] = set()  # deprecated but kept for compatibility
        self._initialized = False

    @staticmethod
    def _find_free_port() -> int:
        """Find a free port for server binding."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    @classmethod
    async def create_a2a(
        cls,
        agent_id: str,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        executor: Optional[Any] = None,
        httpx_client: Optional[httpx.AsyncClient] = None,
        server_adapter: Optional[BaseServerAdapter] = None
    ) -> "BaseAgent":
        """
        v1.0.0 factory method: Create BaseAgent with A2A server capability.
        
        Parameters
        ----------
        agent_id : str
            Unique agent identifier
        host : str
            Server listening host
        port : Optional[int]
            Server listening port (auto-assigned if None)
        executor : Optional[Any]
            SDK native executor implementing execute(context, event_queue) interface
        httpx_client : Optional[httpx.AsyncClient]
            Shared HTTP client for connection pooling
        server_adapter : Optional[BaseServerAdapter]
            Server adapter (defaults to A2AServerAdapter for A2A protocol)
        
        Returns
        -------
        BaseAgent
            Initialized BaseAgent with running server
        """
        # Validate executor interface (executor is now required)
        if executor is None:
            raise ValueError("executor parameter is required")
        
        if not is_sdk_native_executor(executor):
            raise TypeError(
                f"Executor {type(executor)} must implement SDK native interface: "
                "async def execute(context: RequestContext, event_queue: EventQueue) -> None"
            )
        
        final_executor = executor
        
        # Create BaseAgent instance
        agent = cls(
            agent_id=agent_id,
            host=host,
            port=port,
            httpx_client=httpx_client,
            server_adapter=server_adapter or A2AServerAdapter()
        )
        
        # Start server with SDK native executor
        await agent._start_server(final_executor)
        
        # Fetch self agent card
        await agent._fetch_self_card()
        
        agent._initialized = True
        return agent

    async def _start_server(self, executor: Any) -> None:
        """Start the internal server using pluggable adapter."""
        # Use server adapter to build server and agent card
        self._server_instance, self._self_agent_card = self._server_adapter.build(
            host=self._host,
            port=self._port,
            agent_id=self.agent_id,
            executor=executor
        )
        
        # Start server in background task
        self._server_task = asyncio.create_task(self._server_instance.serve())
        
        # Wait for server to be ready with health check polling
        await self._wait_for_server_ready()

    async def _wait_for_server_ready(self, timeout: float = DEFAULT_SERVER_STARTUP_TIMEOUT) -> None:
        """Wait for server to be ready by polling health endpoint."""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                url = f"http://{self._host}:{self._port}/health"
                response = await self._httpx_client.get(url, timeout=2.0)
                if response.status_code == 200:
                    return  # Server is ready
            except Exception:
                pass  # Server not ready yet
            
            await asyncio.sleep(0.1)  # Short polling interval
        
        raise RuntimeError(f"Server failed to start within {timeout}s")

    async def _fetch_self_card(self) -> None:
        """Fetch agent card from our own server."""
        try:
            url = f"http://{self._host}:{self._port}/.well-known/agent.json"
            response = await self._httpx_client.get(url)
            response.raise_for_status()
            self._self_agent_card = response.json()
        except Exception as e:
            # Fallback card if server not ready
            self._self_agent_card = {
                "name": f"Agent {self.agent_id}",
                "url": f"http://{self._host}:{self._port}/",
                "error": f"Failed to fetch card: {e}"
            }

    @classmethod
    async def from_a2a(
        cls,
        agent_id: str,
        base_url: str,
        httpx_client: Optional[httpx.AsyncClient] = None
    ) -> "BaseAgent":
        """
        DEPRECATED: v0.x compatibility method (client-only mode).
        Use create_a2a() for full server capability.
        """
        warnings.warn(
            "BaseAgent.from_a2a() is deprecated. Use create_a2a() for full server capability.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Create client-only BaseAgent
        client = httpx_client or httpx.AsyncClient(timeout=30.0)
        
        # Parse the base URL to get host/port (for compatibility)
        parsed = urlparse(base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 8080
        
        agent = cls(
            agent_id=agent_id,
            host=host,
            port=port,
            httpx_client=client
        )
        
        # Add a single outbound adapter to mimic old behavior
        adapter = A2AAdapter(httpx_client=client, base_url=base_url)
        await adapter.initialize()
        agent._outbound["default"] = adapter
        
        agent._initialized = True
        return agent

    @classmethod
    async def from_ioa(
        cls,
        agent_id: str,
        ioa_params: Dict[str, Any],
        httpx_client: Optional[httpx.AsyncClient] = None
    ) -> "BaseAgent":
        """
        Factory method for IoA protocol (placeholder for future implementation).
        """
        raise NotImplementedError("IoA adapter not yet implemented in v1.0.0")

    # --- Connection Management (called by AgentNetwork) ---

    def add_outbound_adapter(self, dst_id: str, adapter: BaseProtocolAdapter) -> None:
        """Add an outbound adapter for connecting to a destination agent."""
        self._outbound[dst_id] = adapter
        self.outgoing_edges.add(dst_id)  # compatibility

    def remove_outbound_adapter(self, dst_id: str) -> None:
        """Remove an outbound adapter."""
        if dst_id in self._outbound:
            del self._outbound[dst_id]
        self.outgoing_edges.discard(dst_id)  # compatibility

    def get_outbound_adapters(self) -> Dict[str, BaseProtocolAdapter]:
        """Get all outbound adapters (for debugging/monitoring)."""
        return self._outbound.copy()

    # --- Message Operations ---

    async def send(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        """
        Send message to destination agent using appropriate outbound adapter.
        
        Parameters
        ----------
        dst_id : str
            Destination agent ID
        payload : Dict[str, Any]
            Message payload to send
        
        Returns
        -------
        Any
            Response from destination agent
        
        Raises
        ------
        RuntimeError
            If no outbound adapter found for destination or send fails
        """
        if not self._initialized:
            raise RuntimeError(f"Agent {self.agent_id} not initialized")
        
        # Find outbound adapter
        if dst_id not in self._outbound:
            raise RuntimeError(f"No outbound adapter found for destination {dst_id}")
        
        adapter = self._outbound[dst_id]
        
        # Add source information to payload
        enriched_payload = payload.copy()
        enriched_payload.setdefault("source", self.agent_id)
        
        # Record metrics
        protocol_name = type(adapter).__name__.replace("Adapter", "").lower()
        
        with MetricsTimer(REQUEST_LATENCY, (self.agent_id, dst_id, protocol_name)):
            try:
                # Delegate to protocol adapter
                response = await adapter.send_message(dst_id, enriched_payload)
                
                # Record successful message bytes (accurate JSON size of enriched payload)
                msg_size = len(json.dumps(enriched_payload).encode('utf-8'))
                MSG_BYTES.labels("out", self.agent_id).inc(msg_size)
                
                return response
                
            except Exception as e:
                # Record failure
                REQUEST_FAILURES.labels(self.agent_id, dst_id).inc()
                raise RuntimeError(f"Failed to send message from {self.agent_id} to {dst_id}: {e}") from e

    async def health_check(self) -> bool:
        """
        Check if the agent's server is healthy and responsive.
        
        Returns
        -------
        bool
            True if agent server is healthy, False otherwise
        """
        if not self._initialized or not self._server_task:
            return False
        
        try:
            # Check our own server health
            url = f"http://{self._host}:{self._port}/health"
            response = await self._httpx_client.get(url, timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def get_card(self) -> Dict[str, Any]:
        """
        Get this agent's card (from local server).
        
        Returns
        -------
        Dict[str, Any]
            Agent card with capabilities and metadata
        """
        if not self._initialized or not self._self_agent_card:
            return {"error": "Agent not initialized"}
        
        # Add BaseAgent-level metadata
        card = self._self_agent_card.copy()
        card.update({
            "agent_id": self.agent_id,
            "server_address": f"http://{self._host}:{self._port}",
            "outbound_connections": len(self._outbound),
            "server_running": self.is_server_running(),
            "initialized": self._initialized
        })
        return card

    # --- Server Information ---

    def get_server_info(self) -> Dict[str, Any]:
        """Get server status and configuration info."""
        return {
            "agent_id": self.agent_id,
            "host": self._host,
            "port": self._port,
            "listening_address": self.get_listening_address(),
            "server_running": self.is_server_running(),
            "outbound_connections": len(self._outbound)
        }

    def is_server_running(self) -> bool:
        """Check if the server task is running."""
        return (
            self._server_task is not None and 
            not self._server_task.done() and 
            not self._server_task.cancelled()
        )

    def get_listening_address(self) -> str:
        """Get the complete listening address."""
        return f"http://{self._host}:{self._port}"

    # --- Lifecycle Management ---

    async def stop(self) -> None:
        """
        Gracefully stop the agent server and clean up all resources.
        """
        # 1. Gracefully shutdown server
        if self._server_instance and self._server_task:
            # Signal server to stop
            self._server_instance.should_exit = True
            
            # Wait for server to shutdown gracefully
            try:
                await asyncio.wait_for(self._server_task, timeout=DEFAULT_SERVER_SHUTDOWN_TIMEOUT)
            except asyncio.TimeoutError:
                # Force cancel if graceful shutdown takes too long
                self._server_task.cancel()
                try:
                    await self._server_task
                except asyncio.CancelledError:
                    pass
            
            self._server_task = None
            self._server_instance = None
        
        # 2. Clean up all adapters
        for adapter in self._outbound.values():
            await adapter.cleanup()
        self._outbound.clear()
        
        # 3. Clear compatibility tracking
        self.outgoing_edges.clear()
        
        # 4. Optionally close HTTP client (but be careful about shared clients)
        # Note: We don't close _httpx_client here since it might be shared
        # The caller is responsible for httpx_client lifecycle management
        
        self._initialized = False

    # --- Legacy compatibility ---

    async def cleanup(self) -> None:
        """
        DEPRECATED: Use stop() instead.
        """
        warnings.warn(
            "BaseAgent.cleanup() is deprecated. Use stop() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        await self.stop()

    @property 
    def adapter(self) -> Optional[BaseProtocolAdapter]:
        """
        DEPRECATED: Legacy single-adapter access.
        Returns first adapter for compatibility.
        """
        warnings.warn(
            "BaseAgent.adapter is deprecated. Use get_outbound_adapters() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        adapters = list(self._outbound.values())
        return adapters[0] if adapters else None

    def __repr__(self) -> str:
        """Debug representation of BaseAgent."""
        return (
            f"BaseAgent(id='{self.agent_id}', "
            f"server={self.get_listening_address()}, "
            f"outbound_adapters={len(self._outbound)}, "
            f"server_running={self.is_server_running()}, "
            f"initialized={self._initialized})"
        )
    
    @classmethod
    async def create_agent_protocol(
        cls,
        agent_id: str,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        executor: Optional[Any] = None,
        httpx_client: Optional[httpx.AsyncClient] = None
    ) -> "BaseAgent":
        """
        创建使用 Agent Protocol v1 的 BaseAgent 实例
        
        Parameters
        ----------
        agent_id : str
            智能体唯一标识符
        host : str
            服务器监听主机
        port : Optional[int]
            服务器监听端口（如果为 None 则自动分配）
        executor : Optional[Any]
            智能体执行器，应实现 execute_step 或相关方法
        httpx_client : Optional[httpx.AsyncClient]
            共享的 HTTP 客户端
        
        Returns
        -------
        BaseAgent
            使用 Agent Protocol 服务器的已初始化 BaseAgent
        """
        # 验证执行器（执行器是必需的）
        if executor is None:
            raise ValueError("executor parameter is required for Agent Protocol")
        
        # 创建 Agent Protocol 服务器适配器
        server_adapter = AgentProtocolServerAdapter()
        
        # 创建 BaseAgent 实例
        agent = cls(
            agent_id=agent_id,
            host=host,
            port=port,
            httpx_client=httpx_client,
            server_adapter=server_adapter
        )
        
        # 启动服务器
        await agent._start_server(executor)
        
        # 获取自身智能体卡片
        await agent._fetch_self_card()
        
        agent._initialized = True
        return agent

    async def add_connection(
        self,
        dst_id: str, 
        base_url: str,
        protocol: Optional[str] = None,
        auth_headers: Optional[Dict[str, str]] = None,
        **adapter_kwargs
    ) -> None:
        """
        智能连接方法：自动检测协议并创建适当的适配器
        
        Parameters
        ----------
        dst_id : str
            目标智能体ID
        base_url : str
            目标智能体的基础URL
        protocol : Optional[str]
            指定协议类型 ("a2a", "agent_protocol", "auto")，默认为 "auto"
        auth_headers : Optional[Dict[str, str]]
            认证头信息
        **adapter_kwargs : dict
            传递给适配器的额外参数
        """
        if not self._initialized:
            raise RuntimeError(f"Agent {self.agent_id} not initialized")
        
        protocol = protocol or "auto"
        
        if protocol == "auto":
            # 自动检测协议
            detected_protocol = await self._detect_protocol(base_url, auth_headers)
            if not detected_protocol:
                raise RuntimeError(f"Failed to detect protocol for {base_url}")
            protocol = detected_protocol
        
        # 创建相应的适配器
        adapter = await self._create_adapter(
            protocol=protocol,
            base_url=base_url,
            auth_headers=auth_headers,
            **adapter_kwargs
        )
        
        # 初始化适配器
        await adapter.initialize()
        
        # 添加到出站适配器列表
        self.add_outbound_adapter(dst_id, adapter)
    
    async def _detect_protocol(
        self, 
        base_url: str, 
        auth_headers: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        检测目标智能体支持的协议
        
        Returns
        -------
        Optional[str]
            检测到的协议类型："a2a" 或 "agent_protocol"
        """
        headers = auth_headers or {}
        
        try:
            # 获取智能体卡片
            response = await self._httpx_client.get(
                f"{base_url.rstrip('/')}/.well-known/agent.json",
                headers=headers,
                timeout=10.0
            )
            
            if response.status_code == 200:
                card = response.json()
                
                # 检查卡片中声明的协议
                if "protocols" in card:
                    protocols = card["protocols"]
                    if isinstance(protocols, list):
                        if "agent_protocol" in protocols:
                            return "agent_protocol"
                        elif "a2a" in protocols:
                            return "a2a"
                
                # 检查端点信息
                if "endpoints" in card:
                    endpoints = card["endpoints"]
                    if "/ap/v1/agent/tasks" in str(endpoints):
                        return "agent_protocol"
                
                # 检查协议版本
                if "protocolVersion" in card:
                    if card["protocolVersion"] == "v1" or "v1" in str(card["protocolVersion"]):
                        # 进一步检查是否有 Agent Protocol 端点
                        if await self._test_agent_protocol_endpoint(base_url, headers):
                            return "agent_protocol"
                
                # 默认假设是 A2A
                return "a2a"
            
        except Exception:
            pass
        
        # 尝试检测 Agent Protocol 端点
        if await self._test_agent_protocol_endpoint(base_url, headers):
            return "agent_protocol"
        
        # 尝试检测 A2A 端点
        if await self._test_a2a_endpoint(base_url, headers):
            return "a2a"
        
        return None
    
    async def _test_agent_protocol_endpoint(
        self, 
        base_url: str, 
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """测试是否存在 Agent Protocol 端点"""
        try:
            # 尝试访问 Agent Protocol 任务端点
            response = await self._httpx_client.get(
                f"{base_url.rstrip('/')}/ap/v1/agent/tasks",
                headers=headers or {},
                timeout=5.0
            )
            # Agent Protocol 应该返回 405 (Method Not Allowed) 或 200
            return response.status_code in [200, 405]
        except Exception:
            return False
    
    async def _test_a2a_endpoint(
        self, 
        base_url: str, 
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """测试是否存在 A2A 端点"""
        try:
            # 尝试访问 A2A 消息端点（用 HEAD 请求避免发送真实消息）
            response = await self._httpx_client.head(
                f"{base_url.rstrip('/')}/message",
                headers=headers or {},
                timeout=5.0
            )
            # A2A 应该返回 405 (Method Not Allowed) 或类似状态
            return response.status_code in [200, 405, 415]
        except Exception:
            return False
    
    async def _create_adapter(
        self,
        protocol: str,
        base_url: str,
        auth_headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> BaseProtocolAdapter:
        """
        根据协议类型创建适配器
        
        Parameters
        ----------
        protocol : str
            协议类型 ("a2a" 或 "agent_protocol")
        base_url : str
            目标URL
        auth_headers : Optional[Dict[str, str]]
            认证头
        **kwargs : dict
            额外参数
        
        Returns
        -------
        BaseProtocolAdapter
            创建的适配器实例
        """
        if protocol == "a2a":
            return A2AAdapter(
                httpx_client=self._httpx_client,
                base_url=base_url,
                auth_headers=auth_headers,
                **kwargs
            )
        elif protocol == "agent_protocol":
            return AgentProtocolAdapter(
                httpx_client=self._httpx_client,
                base_url=base_url,
                auth_headers=auth_headers,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
    
    def get_connection_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有连接信息
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            连接信息字典，键为目标ID，值为连接详情
        """
        connections = {}
        for dst_id, adapter in self._outbound.items():
            connections[dst_id] = {
                "adapter_type": type(adapter).__name__,
                "protocol": type(adapter).__name__.replace("Adapter", "").lower(),
                "base_url": getattr(adapter, "base_url", "unknown"),
                "auth_enabled": bool(getattr(adapter, "auth_headers", {})),
                "agent_card_loaded": bool(getattr(adapter, "agent_card", {}))
            }
        return connections