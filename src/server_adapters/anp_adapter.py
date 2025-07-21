"""
ANP (Agent Network Protocol) Server Adapter - AgentConnect协议服务器适配器
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Tuple, Optional, Callable, Awaitable
from contextlib import asynccontextmanager
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

# Import base adapter
try:
    from .base_adapter import BaseServerAdapter
except ImportError:
    # Fall back to absolute import for standalone usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from server_adapters.base_adapter import BaseServerAdapter

# Import AgentConnect components
# Add AgentConnect to path if not already there
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
agentconnect_path = os.path.join(project_root, 'agentconnect_src')
if agentconnect_path not in sys.path:
    sys.path.insert(0, agentconnect_path)

# Import AgentConnect components
from agent_connect.python.simple_node import SimpleNode, SimpleNodeSession
from agent_connect.python.authentication import (
    DIDAllClient, create_did_wba_document
)
from agent_connect.python.utils.did_generate import did_generate
from agent_connect.python.utils.crypto_tool import get_pem_from_private_key
AGENTCONNECT_AVAILABLE = True


logger = logging.getLogger(__name__)


class ANPExecutorWrapper:
    """
    Wrapper to adapt standard executors to ANP message handling.
    """
    
    def __init__(self, executor: Any):
        """
        Initialize executor wrapper.
        
        Parameters
        ----------
        executor : Any
            The original executor (A2A or Agent Protocol style)
        """
        self.executor = executor
        self.executor_type = self._detect_executor_type(executor)
    
    def _detect_executor_type(self, executor: Any) -> str:
        """Detect the type of executor."""
        if hasattr(executor, 'execute') and hasattr(executor, '__call__'):
            # Check if it's A2A SDK native executor
            import inspect
            try:
                sig = inspect.signature(executor.execute)
                param_names = list(sig.parameters.keys())
                if len(param_names) >= 2 and param_names[:2] == ["context", "event_queue"]:
                    return "a2a_sdk"
            except Exception:
                pass
        
        if hasattr(executor, 'execute_step'):
            return "agent_protocol"
        
        if hasattr(executor, '__call__'):
            return "callable"
        
        return "unknown"
    
    async def handle_anp_message(self, session: SimpleNodeSession, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle ANP message and route to appropriate executor.
        
        Parameters
        ----------
        session : SimpleNodeSession
            The ANP session
        message : Dict[str, Any]
            The received message
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Response message, if any
        """
        try:
            message_type = message.get("type", "unknown")
            
            # Handle ping/pong for health checks
            if message_type == "ping":
                return {
                    "type": "pong",
                    "ping_timestamp": message.get("timestamp"),
                    "pong_timestamp": time.time(),
                    "request_id": message.get("request_id")
                }
            
            # Route to appropriate executor
            if self.executor_type == "a2a_sdk":
                return await self._handle_a2a_message(message)
            elif self.executor_type == "agent_protocol":
                return await self._handle_agent_protocol_message(message)
            elif self.executor_type == "callable":
                return await self._handle_callable_message(message)
            else:
                logger.warning(f"Unknown executor type: {self.executor_type}")
                return {
                    "type": "error",
                    "error": "Unsupported executor type",
                    "request_id": message.get("request_id")
                }
                
        except Exception as e:
            logger.error(f"Error handling ANP message: {e}")
            return {
                "type": "error",
                "error": str(e),
                "request_id": message.get("request_id")
            }
    
    async def _handle_a2a_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle message using A2A SDK executor."""
        try:
            # Import A2A SDK components
            from a2a.server.agent_execution import RequestContext
            from a2a.server.events import EventQueue
            from a2a.types import Message, MessageSendParams
            from a2a.utils import new_agent_text_message
            
            # Extract payload
            payload = message.get("payload", {})
            
            # Create A2A message
            if "content" in payload:
                text_content = payload["content"]
            elif "data" in payload:
                text_content = json.dumps(payload["data"])
            else:
                text_content = json.dumps(payload)
            
            # Create A2A SDK message
            a2a_message = new_agent_text_message(text_content)
            params = MessageSendParams(message=a2a_message)
            ctx = RequestContext(params)
            
            # Create event queue
            queue = EventQueue()
            
            # Execute
            await self.executor.execute(ctx, queue)
            
            # Collect events
            events = []
            try:
                while True:
                    event = await queue.dequeue_event(no_wait=True)
                    events.append(self._event_to_dict(event))
            except asyncio.QueueEmpty:
                pass
            
            return {
                "type": "a2a_response",
                "events": events,
                "request_id": message.get("request_id")
            }
            
        except ImportError:
            # A2A SDK not available, fallback to simple response
            payload = message.get("payload", {})
            content = payload.get("content", str(payload))
            
            return {
                "type": "text_response",
                "content": f"Processed: {content}",
                "request_id": message.get("request_id")
            }
    
    async def _handle_agent_protocol_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle message using Agent Protocol executor."""
        try:
            payload = message.get("payload", {})
            
            # Create a simple step-like object
            class SimpleStep:
                def __init__(self, input_text: str):
                    self.input = input_text
                    self.step_id = message.get("request_id", "unknown")
            
            # Extract input
            if "content" in payload:
                input_text = payload["content"]
            elif "data" in payload:
                input_text = json.dumps(payload["data"])
            else:
                input_text = json.dumps(payload)
            
            step = SimpleStep(input_text)
            
            # Execute step
            result = await self.executor.execute_step(step)
            
            # Format response
            if isinstance(result, dict):
                output = result.get("output", str(result))
                status = result.get("status", "completed")
            else:
                output = str(result)
                status = "completed"
            
            return {
                "type": "agent_protocol_response",
                "output": output,
                "status": status,
                "request_id": message.get("request_id")
            }
            
        except Exception as e:
            return {
                "type": "error",
                "error": f"Agent Protocol execution failed: {e}",
                "request_id": message.get("request_id")
            }
    
    async def _handle_callable_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle message using callable executor."""
        try:
            payload = message.get("payload", {})
            
            # Call executor with payload
            result = await self.executor(payload)
            
            return {
                "type": "callable_response",
                "result": result,
                "request_id": message.get("request_id")
            }
            
        except Exception as e:
            return {
                "type": "error",
                "error": f"Callable execution failed: {e}",
                "request_id": message.get("request_id")
            }
    
    def _event_to_dict(self, event: Any) -> Dict[str, Any]:
        """Convert A2A event to dictionary."""
        try:
            # Try pydantic v2 model_dump() first
            if hasattr(event, 'model_dump'):
                return event.model_dump()
            # Fallback to pydantic v1 dict() method
            elif hasattr(event, 'dict'):
                return event.dict()
            # For dict-like objects
            elif isinstance(event, dict):
                return event
            else:
                return {
                    "type": getattr(event, "type", event.__class__.__name__),
                    "data": str(event)
                }
        except Exception:
            return {
                "type": event.__class__.__name__,
                "data": str(event)
            }


class ANPSimpleNodeWrapper:
    """
    Wrapper around SimpleNode to integrate with uvicorn server interface.
    Provides both WebSocket (via SimpleNode) and HTTP endpoints.
    """
    
    def __init__(self, 
                 agent_card: Dict[str, Any], 
                 executor_wrapper: ANPExecutorWrapper,
                 host_domain: str,
                 host_port: int,
                 host_ws_path: str = "/ws",
                 did_info: Optional[Dict[str, str]] = None):
        """
        Initialize ANP SimpleNode wrapper.
        
        Parameters
        ----------
        agent_card : Dict[str, Any]
            Agent card information
        executor_wrapper : ANPExecutorWrapper
            Wrapped executor for message handling
        host_domain : str
            Host domain
        host_port : int
            Host port
        host_ws_path : str
            WebSocket path
        did_info : Optional[Dict[str, str]]
            DID information (private_key_pem, did, did_document_json)
        """
        self.agent_card = agent_card
        self.executor_wrapper = executor_wrapper
        self.host_domain = host_domain
        self.host_port = host_port
        self.host_ws_path = host_ws_path
        self.did_info = did_info or {}
        
        self.simple_node: Optional[SimpleNode] = None
        self.should_exit = False
        self.starlette_app: Optional[Starlette] = None
        
    def _create_http_app(self) -> Starlette:
        """Create Starlette app for HTTP endpoints."""
        
        async def health_check(request):
            """Health check endpoint."""
            return JSONResponse({"status": "healthy", "protocol": "ANP"})
        
        async def agent_card(request):
            """Agent card endpoint."""
            return JSONResponse(self.agent_card)
        
        @asynccontextmanager
        async def lifespan(app):
            """Handle application lifespan events with proper context manager."""
            # Startup
            try:
                yield
            finally:
                # Shutdown - gracefully handle cleanup
                self.should_exit = True
            
        # Define routes
        routes = [
            Route("/health", health_check, methods=["GET"]),
            Route("/.well-known/agent.json", agent_card, methods=["GET"]),
        ]
        
        app = Starlette(routes=routes, lifespan=lifespan)
        return app
        
    async def _setup_did_info(self) -> None:
        """Setup DID information if not provided."""
        if not self.did_info.get('did'):
            logger.info("Generating DID for ANP server")
            
            # Generate WebSocket endpoint
            ws_endpoint = f"ws://{self.host_domain}:{self.host_port}{self.host_ws_path}"
            
            # Generate DID locally
            private_key, _, did, did_document_json = did_generate(ws_endpoint)
            private_key_pem = get_pem_from_private_key(private_key)
            
            self.did_info = {
                'private_key_pem': private_key_pem,
                'did': did,
                'did_document_json': did_document_json
            }
            
            # Update agent card with DID
            self.agent_card["id"] = did
            self.agent_card["authentication"] = {
                "type": "DID",
                "did": did,
                "verification_methods": ["Ed25519VerificationKey2018"]
            }
    
    async def _on_new_session(self, session: SimpleNodeSession) -> None:
        """Handle new WebSocket session."""
        logger.info(f"New ANP session from DID: {session.remote_did}")
        
        # Start message handling task for this session
        asyncio.create_task(self._handle_session_messages(session))
    
    async def _handle_session_messages(self, session: SimpleNodeSession) -> None:
        """Handle messages from a WebSocket session."""
        try:
            while not self.should_exit:
                try:
                    # Receive message
                    raw_message = await session.receive_message()
                    if raw_message is None:
                        break
                    
                    # Decode and parse message
                    if isinstance(raw_message, bytes):
                        message_text = raw_message.decode('utf-8')
                    else:
                        message_text = str(raw_message)
                    
                    try:
                        message = json.loads(message_text)
                    except json.JSONDecodeError:
                        # Treat as plain text
                        message = {
                            "type": "text",
                            "payload": {"content": message_text}
                        }
                    
                    # Handle message
                    response = await self.executor_wrapper.handle_anp_message(session, message)
                    
                    # Send response if any
                    if response:
                        response_json = json.dumps(response, separators=(',', ':'))
                        await session.send_message(response_json)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error handling session message: {e}")
                    # Send error response
                    error_response = {
                        "type": "error",
                        "error": str(e),
                        "request_id": message.get("request_id") if 'message' in locals() else None
                    }
                    try:
                        await session.send_message(json.dumps(error_response))
                    except Exception:
                        pass
                    
        except Exception as e:
            logger.error(f"Session handling task failed: {e}")
    
    async def serve(self) -> None:
        """Main serve method compatible with uvicorn.Server interface."""
        try:
            # Setup DID information
            await self._setup_did_info()
            
            logger.info(f"Starting ANP hybrid server on {self.host_domain}:{self.host_port}")
            logger.info(f"HTTP endpoints: /health, /.well-known/agent.json")
            logger.info(f"WebSocket endpoint: {self.host_ws_path} (via SimpleNode on port {self.host_port + 1000})")
            logger.info(f"Server DID: {self.did_info.get('did', 'unknown')}")
            
            # Create HTTP app for compatibility endpoints
            self.starlette_app = self._create_http_app()
            
            # Create and configure SimpleNode for WebSocket on a different port
            ws_port = self.host_port + 1000  # Use offset port for WebSocket
            
            logger.info(f"Creating SimpleNode with DID: {self.did_info.get('did', 'unknown')}")
            logger.info(f"SimpleNode WebSocket port: {ws_port}")
            
            try:
                self.simple_node = SimpleNode(
                    host_domain=self.host_domain,
                    new_session_callback=self._on_new_session,
                    host_port=str(ws_port),
                    host_ws_path=self.host_ws_path,
                    private_key_pem=self.did_info.get('private_key_pem'),
                    did=self.did_info.get('did'),
                    did_document_json=self.did_info.get('did_document_json')
                )
                logger.info("SimpleNode created successfully")
            except Exception as e:
                logger.error(f"Failed to create SimpleNode: {e}")
                self.simple_node = None
            
            # Start SimpleNode in background
            simple_node_task = asyncio.create_task(self._run_simple_node())
            
            # Start HTTP server using uvicorn with simple config
            config = uvicorn.Config(
                self.starlette_app,
                host=self.host_domain,
                port=self.host_port,
                log_level="critical",
                access_log=False,
                use_colors=False
            )
            server = uvicorn.Server(config)
            
            # Run both servers concurrently with proper error handling
            try:
                await asyncio.gather(
                    server.serve(),
                    simple_node_task,
                    return_exceptions=False  # Let exceptions bubble up properly
                )
            except asyncio.CancelledError:
                # Handle graceful shutdown
                logger.info("ANP server shutdown requested")
                self.should_exit = True
                # Cancel SimpleNode task if still running
                if not simple_node_task.done():
                    simple_node_task.cancel()
                    try:
                        await simple_node_task
                    except asyncio.CancelledError:
                        pass
                
        except Exception as e:
            logger.error(f"ANP server error: {e}")
            raise
        finally:
            await self.stop()
            
    async def _run_simple_node(self) -> None:
        """Run SimpleNode for WebSocket functionality."""
        try:
            # Check if SimpleNode was created successfully
            if self.simple_node is None:
                logger.error("SimpleNode not initialized")
                return
                
            # Run the SimpleNode
            self.simple_node.run()
            
            # Keep running until should_exit is set
            while not self.should_exit:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"SimpleNode error: {e}")
            # Don't raise to prevent blocking other tasks
    
    async def stop(self) -> None:
        """Stop the ANP server."""
        self.should_exit = True
        if self.simple_node:
            try:
                await self.simple_node.stop()
            except Exception as e:
                # Log but don't raise - we want cleanup to continue
                logger.debug(f"Error stopping SimpleNode (expected during shutdown): {e}")
            self.simple_node = None


class ANPServerAdapter(BaseServerAdapter):
    """ANP (Agent Network Protocol) Server Adapter"""
    
    protocol_name = "ANP"
    
    def build(
        self,
        host: str,
        port: int,
        agent_id: str,
        executor: Any,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Build ANP server using AgentConnect SimpleNode.
        
        Parameters
        ----------
        host : str
            Server host address
        port : int
            Server port
        agent_id : str
            Agent identifier
        executor : Any
            Agent executor instance
        **kwargs : dict
            Additional configuration parameters:
            - did_service_url: DID resolution service URL
            - did_api_key: API key for DID service
            - host_ws_path: WebSocket path (default: /ws)
            - did_info: Pre-generated DID information
            
        Returns
        -------
        Tuple[Any, Dict[str, Any]]
            Server instance and agent card
        """
        if not AGENTCONNECT_AVAILABLE:
            raise ImportError(
                "AgentConnect library not available. "
                "Please install AgentConnect to use ANP server adapter."
            )
        
        # Extract configuration
        did_service_url = kwargs.get('did_service_url')
        did_api_key = kwargs.get('did_api_key')
        host_ws_path = kwargs.get('host_ws_path', '/ws')
        did_info = kwargs.get('did_info', {})
        
        # Generate agent card
        agent_card = self._generate_agent_card(agent_id, host, port, host_ws_path, did_info)
        
        # Create executor wrapper
        executor_wrapper = ANPExecutorWrapper(executor)
        
        # Create ANP server wrapper
        server_wrapper = ANPSimpleNodeWrapper(
            agent_card=agent_card,
            executor_wrapper=executor_wrapper,
            host_domain=host,
            host_port=port,
            host_ws_path=host_ws_path,
            did_info=did_info
        )
        
        return server_wrapper, agent_card
    
    def _generate_agent_card(self, 
                           agent_id: str, 
                           host: str, 
                           port: int, 
                           ws_path: str,
                           did_info: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """Generate ANP agent card."""
        # Ensure did_info is not None
        if did_info is None:
            did_info = {}
        
        # WebSocket runs on offset port
        ws_port = port + 1000
        
        return {
            "id": did_info.get('did', agent_id),
            "name": f"ANP Agent - {agent_id}",
            "description": f"Agent Network Protocol (ANP) server for {agent_id}",
            "version": "1.0.0",
            "url": f"http://{host}:{port}/",  # HTTP endpoint
            "protocol": "ANP",
            "protocolVersion": "1.0.0",
            "capabilities": {
                "did_authentication": True,
                "websocket_transport": True,
                "end_to_end_encryption": True,
                "protocol_negotiation": False,  # Can be enabled later
                "persistent_connections": True,
                "real_time_communication": True
            },
            "endpoints": {
                "websocket": f"ws://{host}:{ws_port}{ws_path}",  # WebSocket on offset port
                "http": f"http://{host}:{port}/",  # HTTP for compatibility
                "did_document": did_info.get('did', ''),
                "health": f"http://{host}:{port}/health"  # Health check via HTTP
            },
            "supportedMessageTypes": [
                "text", "json", "binary"
            ],
            "supportedEncodings": [
                "utf-8", "json"
            ],
            "authentication": {
                "type": "DID",
                "did": did_info.get('did', ''),
                "verification_methods": [
                    "Ed25519VerificationKey2018",
                    "EcdsaSecp256k1VerificationKey2019"
                ],
                "required": True
            },
            "security": {
                "end_to_end_encryption": True,
                "transport_encryption": True,
                "authentication_required": True
            },
            "features": {
                "session_persistence": True,
                "message_ordering": True,
                "delivery_confirmation": False,  # Can be added
                "message_history": False,       # Can be added
                "file_transfer": False          # Can be added
            },
            "limits": {
                "max_message_size": 1048576,    # 1MB
                "max_concurrent_sessions": 100,
                "session_timeout": 3600         # 1 hour
            }
        }



# Optional: Enhanced ANP Server Adapter with more features
class EnhancedANPServerAdapter(ANPServerAdapter):
    """Enhanced ANP Server Adapter with additional features."""
    
    def build(self, host: str, port: int, agent_id: str, executor: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Build enhanced ANP server with additional features."""
        
        # Enable protocol negotiation if LLM is available
        enable_negotiation = kwargs.get('enable_protocol_negotiation', False)
        llm_instance = kwargs.get('llm_instance')
        
        if enable_negotiation and llm_instance:
            # Use SimpleNegotiationNode instead of SimpleNode
            logger.info("Enabling protocol negotiation for ANP server")
            # This would require additional implementation
        
        # Call parent implementation
        return super().build(host, port, agent_id, executor, **kwargs) 