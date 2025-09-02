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
    from ..agent_adapters.base_adapter import BaseProtocolAdapter
    from ..agent_adapters.a2a_adapter import A2AAdapter
    from ..agent_adapters.agent_protocol_adapter import AgentProtocolAdapter
    from ..agent_adapters.agora_adapter import AgoraClientAdapter
    from .metrics import REQUEST_LATENCY, REQUEST_FAILURES, MSG_BYTES, MetricsTimer
    from ..server_adapters import BaseServerAdapter, A2AServerAdapter, AgentProtocolServerAdapter, ANPServerAdapter, ANP_AVAILABLE, ACPServerAdapter
    from .unified_message import UTE
    from .protocol_converter import ENCODE_TABLE, DECODE_TABLE
except ImportError:
    # Fallback to direct imports for standalone execution
    from agent_adapters.base_adapter import BaseProtocolAdapter
    from agent_adapters.a2a_adapter import A2AAdapter
    from agent_adapters.agent_protocol_adapter import AgentProtocolAdapter
    from agent_adapters.agora_adapter import AgoraClientAdapter
    from src.core.metrics import REQUEST_LATENCY, REQUEST_FAILURES, MSG_BYTES, MetricsTimer
    from server_adapters import BaseServerAdapter, A2AServerAdapter, AgentProtocolServerAdapter, ANPServerAdapter, ANP_AVAILABLE, ACPServerAdapter
    from src.core.unified_message import UTE
    from src.core.protocol_converter import ENCODE_TABLE, DECODE_TABLE

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
    
    # ----------- Factory Methods ----------- 
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

    @classmethod
    async def create_acp(
        cls,
        agent_id: str,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        executor: Optional[Any] = None,
        httpx_client: Optional[httpx.AsyncClient] = None,
        server_adapter: Optional[BaseServerAdapter] = None
    ) -> "BaseAgent":
        """
        v1.0.0 factory method: Create BaseAgent with ACP server capability.

        Parameters
        ----------
        agent_id : str
            Unique agent identifier
        host : str
            Server listening host
        port : Optional[int]
            Server listening port (auto-assigned if None)
        executor : Optional[Any]
            ACP SDK native executor implementing async generator interface:
            async def executor(input: list[Message], context: Context) -> AsyncGenerator[RunYield, None]
        httpx_client : Optional[httpx.AsyncClient]
            Shared HTTP client for connection pooling
        server_adapter : Optional[BaseServerAdapter]
            Server adapter (defaults to ACPServerAdapter for ACP protocol)

        Returns
        -------
        BaseAgent
            Initialized BaseAgent with running ACP server
        """
        # Validate executor is provided
        if executor is None:
            raise ValueError("executor parameter is required for ACP agents")

        # Validate ACP executor interface
        if not callable(executor):
            raise TypeError(
                f"ACP executor {type(executor)} must be callable and implement ACP SDK interface: "
                "async def executor(input: list[Message], context: Context) -> AsyncGenerator[RunYield, None]"
            )

        # Create BaseAgent instance
        agent = cls(
            agent_id=agent_id,
            host=host,
            port=port,
            httpx_client=httpx_client,
            server_adapter=server_adapter or ACPServerAdapter()
        )

        # Start server with ACP executor
        await agent._start_server(executor)

        # Fetch self agent card
        await agent._fetch_self_card()

        agent._initialized = True
        return agent

    @classmethod
    async def create_anp(
        cls,
        agent_id: str,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        executor: Optional[Any] = None,
        httpx_client: Optional[httpx.AsyncClient] = None,
        did_info: Optional[Dict[str, str]] = None,
        did_service_url: Optional[str] = None,
        did_api_key: Optional[str] = None,
        host_ws_path: str = "/ws",
        enable_protocol_negotiation: bool = False
    ) -> "BaseAgent":
        """
        Create BaseAgent with ANP (Agent Network Protocol) server capability.
        
        Parameters
        ----------
        agent_id : str
            Unique agent identifier
        host : str
            Server listening host
        port : Optional[int]
            Server listening port (auto-assigned if None)
        executor : Optional[Any]
            ANP-compatible executor (supports A2A SDK, Agent Protocol, or callable interface)
        httpx_client : Optional[httpx.AsyncClient]
            Shared HTTP client for connection pooling
        did_info : Optional[Dict[str, str]]
            Pre-generated DID information containing:
            - private_key_pem: Private key in PEM format
            - did: DID string
            - did_document_json: DID document JSON
        did_service_url : Optional[str]
            DID resolution service URL for remote DID generation
        did_api_key : Optional[str]
            API key for DID service authentication
        host_ws_path : str
            WebSocket path for ANP communication (default: /ws)
        enable_protocol_negotiation : bool
            Enable LLM-based protocol negotiation
        
        Returns
        -------
        BaseAgent
            Initialized BaseAgent with running ANP server
        
        Raises
        ------
        ImportError
            If AgentConnect library is not available
        ValueError
            If executor is None
        """
        # ANP server adapter should be available if we reach this point
        
        # Validate executor (executor is required for ANP)
        if executor is None:
            raise ValueError("executor parameter is required for ANP")
        
        # Create ANP server adapter with additional configuration
        server_adapter = ANPServerAdapter()
        
        # Prepare ANP-specific configuration
        anp_config = {
            "did_info": did_info,
            "did_service_url": did_service_url,
            "did_api_key": did_api_key,
            "host_ws_path": host_ws_path,
            "enable_protocol_negotiation": enable_protocol_negotiation
        }
        
        # Create BaseAgent instance
        agent = cls(
            agent_id=agent_id,
            host=host,
            port=port,
            httpx_client=httpx_client,
            server_adapter=server_adapter
        )
        
        # Start server with executor and ANP configuration
        await agent._start_anp_server(executor, anp_config)
        
        # For ANP, we don't fetch card via HTTP, as it uses WebSocket
        # The card is generated during server setup
        await agent._setup_anp_card()
        
        agent._initialized = True
        return agent

    @classmethod
    async def create_agora(
        cls,
        agent_id: str,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        executor: Optional[Any] = None,
        httpx_client: Optional[httpx.AsyncClient] = None,
        server_adapter: Optional[BaseServerAdapter] = None,
        openai_api_key: Optional[str] = None,
        **kwargs
    ) -> "BaseAgent":
        """
        Create BaseAgent with Agora server capability.

        Parameters
        ----------
        agent_id : str
            Unique agent identifier
        host : str
            Server listening host
        port : Optional[int]
            Server listening port (auto-assigned if None)
        executor : Optional[Any]
            A2A SDK native executor implementing execute(context, event_queue) interface
        httpx_client : Optional[httpx.AsyncClient]
            Shared HTTP client for connection pooling
        server_adapter : Optional[BaseServerAdapter]
            Server adapter (defaults to AgoraServerAdapter)
        openai_api_key : Optional[str]
            OpenAI API key for Agora toolformer
        **kwargs
            Additional configuration for Agora (model name, etc.)

        Returns
        -------
        BaseAgent
            Initialized BaseAgent with running Agora server
        """
        # Validate executor interface (executor is required)
        if executor is None:
            raise ValueError("executor parameter is required for Agora agents")

        if not is_sdk_native_executor(executor):
            raise TypeError(
                f"Executor {type(executor)} must implement SDK native interface: "
                "async def execute(context: RequestContext, event_queue: EventQueue) -> None"
            )

        # Import AgoraServerAdapter
        try:
            from ..server_adapters.agora_adapter import AgoraServerAdapter
        except ImportError:
            from server_adapters.agora_adapter import AgoraServerAdapter

        # Create BaseAgent instance with Agora server adapter
        agent = cls(
            agent_id=agent_id,
            host=host,
            port=port,
            httpx_client=httpx_client,
            server_adapter=server_adapter or AgoraServerAdapter()
        )

        # Configure Agora-specific parameters
        agora_config = kwargs.copy()
        if openai_api_key:
            agora_config['openai_api_key'] = openai_api_key

        # Start server with executor and Agora configuration
        await agent._start_agora_server(executor, agora_config)

        # Fetch self agent card
        await agent._fetch_self_card()

        agent._initialized = True
        return agent

    @classmethod
    async def create_agent_protocol(
        cls,
        agent_id: str,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        executor: Optional[Any] = None,
        httpx_client: Optional[httpx.AsyncClient] = None,
        server_adapter: Optional[BaseServerAdapter] = None
    ) -> "BaseAgent":
        """
        Create BaseAgent with Agent Protocol server capability.

        Parameters
        ----------
        agent_id : str
            Unique agent identifier
        host : str
            Server listening host
        port : Optional[int]
            Server listening port (auto-assigned if None)
        executor : Optional[Any]
            A2A SDK native executor implementing execute(context, event_queue) interface
        httpx_client : Optional[httpx.AsyncClient]
            Shared HTTP client for connection pooling
        server_adapter : Optional[BaseServerAdapter]
            Server adapter (defaults to AgentProtocolServerAdapter)

        Returns
        -------
        BaseAgent
            Initialized BaseAgent with running Agent Protocol server
        """
        # Validate executor interface (executor is required)
        if executor is None:
            raise ValueError("executor parameter is required for Agent Protocol agents")

        if not is_sdk_native_executor(executor):
            raise TypeError(
                f"Executor {type(executor)} must implement SDK native interface: "
                "async def execute(context: RequestContext, event_queue: EventQueue) -> None"
            )

        # Create BaseAgent instance with Agent Protocol server adapter
        agent = cls(
            agent_id=agent_id,
            host=host,
            port=port,
            httpx_client=httpx_client,
            server_adapter=server_adapter or AgentProtocolServerAdapter()
        )

        # Start server with executor
        await agent._start_server(executor)

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

    async def _start_anp_server(self, executor: Any, anp_config: Dict[str, Any]) -> None:
        """Start ANP server with special configuration."""
        # Use server adapter to build ANP server and agent card
        self._server_instance, self._self_agent_card = self._server_adapter.build(
            host=self._host,
            port=self._port,
            agent_id=self.agent_id,
            executor=executor,
            **anp_config
        )
        
        # Start ANP server in background task
        self._server_task = asyncio.create_task(self._server_instance.serve())
        
        # ANP uses WebSocket, so we wait differently
        await self._wait_for_anp_server_ready()

    async def _wait_for_anp_server_ready(self, timeout: float = DEFAULT_SERVER_STARTUP_TIMEOUT) -> None:
        """Wait for ANP server to be ready (WebSocket-based)."""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # ANP server readiness is checked differently
                # We can check if the server task is running without error
                if self._server_task and not self._server_task.done():
                    return  # Server is running
                elif self._server_task and self._server_task.done():
                    # Server task completed, check for exceptions
                    try:
                        await self._server_task
                    except Exception as e:
                        raise RuntimeError(f"ANP server failed to start: {e}")
            except Exception:
                pass  # Server not ready yet
            
            await asyncio.sleep(0.1)  # Short polling interval
        
        raise RuntimeError(f"ANP server failed to start within {timeout}s")

    async def _setup_anp_card(self) -> None:
        """Setup ANP agent card (already created during server build)."""
        # For ANP, the agent card is already set up during server build
        # We just need to ensure it exists
        if not self._self_agent_card:
            self._self_agent_card = {
                "name": f"ANP Agent {self.agent_id}",
                "protocol": "ANP",
                "protocolVersion": "1.0.0",
                "agent_id": self.agent_id,
                "endpoints": {
                    "websocket": f"ws://{self._host}:{self._port}/ws"
                },
                "error": "Failed to setup ANP card during server build"
            }

    async def _wait_for_server_ready(self, timeout: float = DEFAULT_SERVER_STARTUP_TIMEOUT) -> None:
        """Wait for server to be ready by polling health endpoint."""
        import time
        start_time = time.time()
        attempts = 0

        # Use print for debugging since logger might not be available
        print(f"DEBUG: Waiting for server at http://{self._host}:{self._port}/health to be ready...")

        while time.time() - start_time < timeout:
            attempts += 1
            try:
                # Use 127.0.0.1 for health checks when server binds to 0.0.0.0
                health_host = "127.0.0.1" if self._host == "0.0.0.0" else self._host
                url = f"http://{health_host}:{self._port}/health"
                response = await self._httpx_client.get(url, timeout=2.0)
                if response.status_code == 200:
                    print(f"DEBUG: Server ready after {attempts} attempts in {time.time() - start_time:.2f}s")
                    return  # Server is ready
                else:
                    print(f"DEBUG: Attempt {attempts}: Health check returned {response.status_code}")
            except Exception as e:
                if attempts % 20 == 0:  # Log every 2 seconds
                    print(f"DEBUG: Attempt {attempts}: Health check failed: {e}")

            await asyncio.sleep(0.1)  # Short polling interval

        print(f"ERROR: Server failed to start within {timeout}s after {attempts} attempts")
        print(f"ERROR: Server task status: {self._server_task.done() if self._server_task else 'No task'}")
        if self._server_task and self._server_task.done():
            try:
                exc = self._server_task.exception()
                if exc:
                    print(f"ERROR: Server task exception: {exc}")
            except:
                pass
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

    # ----------- Agent Skills and Cards -----------
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
    async def from_ap(
        cls,
        agent_id: str,
        base_url: str,
        httpx_client: Optional[httpx.AsyncClient] = None
    ) -> "BaseAgent":
        """
        Create Agent Protocol client-only agent from existing AP server endpoint.
        
        Parameters
        ----------
        agent_id : str
            Agent identifier
        base_url : str
            Agent Protocol server endpoint URL
        httpx_client : Optional[httpx.AsyncClient]
            Shared HTTP client for connection pooling
            
        Returns
        -------
        BaseAgent
            Created Agent Protocol client agent instance
        """
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
        adapter = AgentProtocolAdapter(httpx_client=client, base_url=base_url)
        await adapter.initialize()
        agent._outbound["default"] = adapter
        
        agent._initialized = True
        return agent
    @classmethod
    async def from_anp(
        cls,
        agent_id: str,
        target_did: str,
        httpx_client: Optional[httpx.AsyncClient] = None,
        local_did_info: Optional[Dict[str, str]] = None,
        host_domain: str = "localhost",
        host_port: Optional[str] = None,
        **kwargs
    ) -> "BaseAgent":
        """
        Create ANP client-only agent for connecting to existing ANP server endpoint.
        
        Parameters
        ----------
        agent_id : str
            Agent identifier
        target_did : str
            Target agent's DID (Decentralized Identifier)
        httpx_client : Optional[httpx.AsyncClient]
            Shared HTTP client for connection pooling
        local_did_info : Optional[Dict[str, str]]
            Local DID information (will be generated if not provided)
        host_domain : str
            Local host domain for WebSocket server
        host_port : Optional[str]
            Local port for WebSocket server
        **kwargs : dict
            Additional ANP configuration parameters
            
        Returns
        -------
        BaseAgent
            Created ANP client agent instance
        
        Raises
        ------
        ImportError
            If AgentConnect library is not available
        """
        try:
            from ..agent_adapters.anp_adapter import ANPAdapter
        except ImportError:
            try:
                from agent_adapters.anp_adapter import ANPAdapter
            except ImportError:
                raise ImportError(
                    "ANP adapter not available. "
                    "Please install AgentConnect library to use ANP protocol."
                )
        
        # Create client-only BaseAgent
        client = httpx_client or httpx.AsyncClient(timeout=30.0)
        
        # Use default ports for compatibility
        host = host_domain
        port = int(host_port) if host_port else 8080
        
        agent = cls(
            agent_id=agent_id,
            host=host,
            port=port,
            httpx_client=client
        )
        
        # Create ANP adapter instance
        adapter = ANPAdapter(
            httpx_client=client,
            target_did=target_did,
            local_did_info=local_did_info,
            host_domain=host_domain,
            host_port=host_port,
            **kwargs
        )
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
        Now uses Unified Transport Envelope (UTE) for protocol-agnostic sending.

        Parameters
        ----------
        dst_id : str
            Destination agent ID
        payload : Dict[str, Any]
            Message payload (business content) to send

        Returns
        -------
        Any
            Response content from destination agent, decoded back into a dict.

        Raises
        ------
        RuntimeError
            If no outbound adapter found for destination or send fails
        """
        if not self._initialized:
            raise RuntimeError(f"Agent {self.agent_id} not initialized")

        # Find outbound adapter
        if dst_id not in self._outbound:
            # Try default if specific not found
            if "default" not in self._outbound:
                raise RuntimeError(f"No outbound adapter found for destination {dst_id}")
            adapter = self._outbound["default"]
        else:
            adapter = self._outbound[dst_id]

        # ★ UTE Conversion Step 1: Create UTE
        ute = UTE.new(src=self.agent_id, dst=dst_id, content=payload, context={})
        
        # ★ UTE Conversion Step 2: Encode UTE to protocol-specific payload
        proto_payload = ENCODE_TABLE[adapter.protocol_name](ute)
        
        # Record metrics
        protocol_name = adapter.protocol_name

        with MetricsTimer(REQUEST_LATENCY, (self.agent_id, dst_id, protocol_name)):
            try:
                # Delegate to protocol adapter with the protocol-specific payload
                response = await adapter.send_message(dst_id, proto_payload)

                # Record successful message bytes (accurate JSON size of encoded payload)
                msg_size = len(json.dumps(proto_payload).encode('utf-8'))
                MSG_BYTES.labels("out", self.agent_id).inc(msg_size)

                # ★ UTE Conversion Step 3: Decode response back to UTE
                ute_response = DECODE_TABLE[adapter.protocol_name](response)
                return ute_response.content  # Return only the business content

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
                    # CancelledError is expected during shutdown
                    pass
            except asyncio.CancelledError:
                # Handle direct cancellation
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

    async def _start_agora_server(self, executor: Any, agora_config: Dict[str, Any]) -> None:
        """Start Agora server with special configuration."""
        # Use server adapter to build Agora server and agent card
        self._server_instance, self._self_agent_card = self._server_adapter.build(
            host=self._host,
            port=self._port,
            agent_id=self.agent_id,
            executor=executor,
            **agora_config
        )
        
        # Start Agora server in background task
        self._server_task = asyncio.create_task(self._server_instance.serve())
        
        # Wait for server to be ready
        await self._wait_for_server_ready()