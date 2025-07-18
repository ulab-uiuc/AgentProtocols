"""
BaseAgent v1.0.0 - Dual-role agent with server and multi-client capabilities
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

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
    from .agent_adapters.agora_adapter import AgoraClientAdapter, AgoraServerAdapter
    from .metrics import REQUEST_LATENCY, REQUEST_FAILURES, MSG_BYTES, MetricsTimer
    from .server_adapters import BaseServerAdapter, A2AServerAdapter
except ImportError:
    # Fallback to direct imports for standalone execution
    from agent_adapters.base_adapter import BaseProtocolAdapter
    from agent_adapters.a2a_adapter import A2AAdapter
    from agent_adapters.agora_adapter import AgoraClientAdapter, AgoraServerAdapter
    from metrics import REQUEST_LATENCY, REQUEST_FAILURES, MSG_BYTES, MetricsTimer
    from server_adapters import BaseServerAdapter, A2AServerAdapter

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
    
    Supports A2A and Agora protocols with SDK native executors.
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

    # @classmethod
    # async def create_a2a(
    #     cls,
    #     agent_id: str,
    #     host: str = "0.0.0.0",
    #     port: Optional[int] = None,
    #     executor: Optional[Any] = None,
    #     httpx_client: Optional[httpx.AsyncClient] = None,
    #     server_adapter: Optional[BaseServerAdapter] = None,
    #     protocol: str = "agora",
    #     openai_api_key: Optional[str] = None,
    #     **kwargs
    # ) -> "BaseAgent":
    #     """
    #     v1.0.0 factory method: Create BaseAgent with A2A or Agora server capability.
        
    #     Parameters
    #     ----------
    #     agent_id : str
    #         Unique agent identifier
    #     host : str
    #         Server listening host
    #     port : Optional[int]
    #         Server listening port (auto-assigned if None)
    #     executor : Optional[Any]
    #         SDK native executor implementing execute(context, event_queue) interface
    #     httpx_client : Optional[httpx.AsyncClient]
    #         Shared HTTP client for connection pooling
    #     server_adapter : Optional[BaseServerAdapter]
    #         Server adapter (defaults to A2AServerAdapter or AgoraServerAdapter based on protocol)
    #     protocol : str
    #         Protocol to use ("a2a" or "agora")
    #     openai_api_key : Optional[str]
    #         API key for Agora toolformer (if protocol is "agora")
    #     **kwargs
    #         Additional parameters for server adapter (e.g., model for Agora)
        
    #     Returns
    #     -------
    #     BaseAgent
    #         Initialized BaseAgent with running server
    #     """
    #     if executor is None:
    #         raise ValueError("executor parameter is required")
        
    #     if not is_sdk_native_executor(executor):
    #         raise TypeError(
    #             f"Executor {type(executor)} must implement SDK native interface: "
    #             "async def execute(context: RequestContext, event_queue: EventQueue) -> None"
    #         )
        
    #     final_executor = executor
  
    #     # Create BaseAgent instance
    #     agent = cls(
    #         agent_id=agent_id,
    #         host=host,
    #         port=port,
    #         httpx_client=httpx_client,
    #         server_adapter=server_adapter
    #     )
        
    #     # Start server with SDK native executor
    #     await agent._start_server(final_executor, toolformer=toolformer, openai_api_key=openai_api_key, **kwargs)
        
    #     # Fetch self agent card
    #     await agent._fetch_self_card()
        
    #     agent._initialized = True
    #     return agent

    @classmethod
    async def create_a2a(
        cls,
        agent_id: str,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        executor: Optional[Any] = None,
        httpx_client: Optional[httpx.AsyncClient] = None,
        server_adapter: Optional[BaseServerAdapter] = None,
        protocol: str = "agora",  # Assuming default protocol is Agora
        openai_api_key: Optional[str] = None,
        **kwargs
    ) -> "BaseAgent":
        """
        v1.0.0 factory method: Create BaseAgent with A2A or Agora server capability.
        
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
            Server adapter (defaults to A2AServerAdapter or AgoraServerAdapter based on protocol)
        protocol : str
            Protocol to use ("a2a" or "agora")
        openai_api_key : Optional[str]
            API key for Agora toolformer (if protocol is "agora")
        **kwargs
            Additional parameters for server adapter (e.g., model for Agora)
        
        Returns
        -------
        BaseAgent
            Initialized BaseAgent with running server
        """
        if executor is None:
            raise ValueError("executor parameter is required")
        
        if not is_sdk_native_executor(executor):
            raise TypeError(
                f"Executor {type(executor)} must implement SDK native interface: "
                "async def execute(context: RequestContext, event_queue: EventQueue) -> None"
            )
        
        final_executor = executor
        
        # Initialize toolformer for Agora protocol
        toolformer = None
        if protocol == "agora":
            try:
                from agora import toolformers  # Assuming the Agora library has toolformers
                toolformer = toolformers.LangChainToolformer(
                    model=kwargs.get('model', 'gpt-4o-mini'),
                )
            except ImportError:
                raise ImportError("Agora library not available or failed to import toolformer")

        # Create BaseAgent instance
        agent = cls(
            agent_id=agent_id,
            host=host,
            port=port,
            httpx_client=httpx_client,
            server_adapter=server_adapter
        )
        
        # Start server with SDK native executor and toolformer (if Agora protocol)
        await agent._start_server(final_executor, toolformer=toolformer, openai_api_key=openai_api_key, **kwargs)
        
        # Fetch self agent card
        await agent._fetch_self_card()
        
        agent._initialized = True
        return agent


    async def _start_server(self, executor: Any, toolformer: Any = None, **kwargs) -> None:
        """Start the internal server using pluggable adapter."""
        # Use server adapter to build server and agent card
        # 删除 openai_api_key，避免传两次
        kwargs.pop("openai_api_key", None)  
        self._server_instance, self._self_agent_card = self._server_adapter.build(
            host=self._host,
            port=self._port,
            agent_id=self.agent_id,
            executor=executor,
            openai_api_key=kwargs.get('openai_api_key'),
            **kwargs
        )
        
        # Start server in background task
        self._server_task = asyncio.create_task(self._server_instance.serve())
        
        # Wait for server to be ready with health check polling
        await self._wait_for_server_ready()

    # async def _wait_for_server_ready(self, timeout: float = DEFAULT_SERVER_STARTUP_TIMEOUT) -> None:
    #     """Wait for server to be ready by polling health endpoint."""
    #     import time
    #     start_time = time.time()
        
    #     while time.time() - start_time < timeout:
    #         try:
    #             url = f"http://{self._host}:{self._port}/health"
    #             response = await self._httpx_client.get(url, timeout=2.0)
    #             if response.status_code == 200:
    #                 return  # Server is ready
    #         except Exception:
    #             pass  # Server not ready yet
            
    #         await asyncio.sleep(0.1)  # Short polling interval
        
    #     raise RuntimeError(f"Server failed to start within {timeout}s")

    async def _wait_for_server_ready(self, timeout: float = DEFAULT_SERVER_STARTUP_TIMEOUT) -> None:
        """Wait for server to be ready by polling health endpoint."""
        import time
        start_time = time.time()

        # Introduce a delay to ensure server starts
        await asyncio.sleep(2)  # Add a 2-second wait before first health check

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
        httpx_client: Optional[httpx.AsyncClient] = None,
        protocol: str = "a2a"
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
        
        # Parse the base URL to get host/port
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
        if protocol == "agora":
            try:
                from agora import toolformers
                toolformer = toolformers.Toolformer(
                    model=kwargs.get('model', 'gpt-4o-mini'),
                    api_key=openai_api_key
                )
                adapter = AgoraClientAdapter(
                    toolformer=toolformer,
                    target_url=base_url
                )
            except ImportError:
                raise ImportError("Agora library not available")
        else:
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
                
                # Record successful message bytes
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