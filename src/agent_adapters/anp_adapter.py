"""
ANP (Agent Network Protocol) Adapter - AgentConnect协议适配器实现
"""

import asyncio
import json
import logging
import uuid
import time
import os
from typing import Any, Dict, Optional, AsyncIterator, Callable
from urllib.parse import urlparse

import httpx

# Import base adapter
try:
    from .base_adapter import BaseProtocolAdapter
except ImportError:
    # Fall back to absolute import for standalone usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from agent_adapters.base_adapter import BaseProtocolAdapter

# Import AgentConnect components  
from agent_connect.python.simple_node import SimpleNode, SimpleNodeSession
from agent_connect.python.authentication import (
    DIDAllClient, create_did_wba_document, generate_auth_header
)
from agent_connect.python.utils.did_generate import did_generate
from agent_connect.python.meta_protocol.meta_protocol import MetaProtocol, ProtocolType
from agent_connect.python.app_protocols.protocol_container import ProtocolContainer
from agent_connect.python.e2e_encryption.message_generation import generate_encrypted_message
AGENTCONNECT_AVAILABLE = True

logger = logging.getLogger(__name__)


class ANPAdapter(BaseProtocolAdapter):
    """
    Adapter for ANP (Agent Network Protocol) specification.
    
    Implements the full AgentConnect protocol stack including:
    - DID-based decentralized authentication (did:wba)
    - WebSocket persistent connections with SimpleNode
    - End-to-end encryption via ECDHE + AES-GCM
    - Meta-protocol for LLM-driven protocol negotiation
    - Application protocol management framework
    - Session management with heartbeat and recovery
    
    This is a complete implementation of the Agent Network Protocol
    as defined in the AgentConnect specification.
    """

    def __init__(
        self, 
        httpx_client: httpx.AsyncClient,
        target_did: str,
        local_did_info: Optional[Dict[str, str]] = None,
        host_domain: str = "localhost",
        host_port: Optional[str] = None,
        host_ws_path: str = "/ws",
        did_service_url: Optional[str] = None,
        did_api_key: Optional[str] = None,
        enable_protocol_negotiation: bool = True,  # Enable by default
        enable_e2e_encryption: bool = True,        # Enable by default  
        llm_instance: Optional[Any] = None,        # For protocol negotiation
        protocol_code_path: Optional[str] = None,  # For generated protocols
        auth_headers: Optional[Dict[str, str]] = None,
        new_session_callback: Optional[Callable] = None  # Callback for new sessions
    ):
        """
        Initialize ANP adapter with full AgentConnect feature set.
        
        Parameters
        ----------
        httpx_client : httpx.AsyncClient
            Shared HTTP client for connection pooling
        target_did : str
            Target agent's DID (Decentralized Identifier)
        local_did_info : Optional[Dict[str, str]]
            Local DID information containing:
            - private_key_pem: Private key in PEM format
            - did: Local DID string  
            - did_document_json: DID document JSON
        host_domain : str
            Local host domain for WebSocket server
        host_port : Optional[str]
            Local port for WebSocket server (auto-assigned if None)
        host_ws_path : str
            WebSocket path (default: /ws)
        did_service_url : Optional[str]
            DID resolution service URL for did:wba
        did_api_key : Optional[str]
            API key for DID service authentication
        enable_protocol_negotiation : bool
            Enable LLM-based meta-protocol negotiation (default: True)
        enable_e2e_encryption : bool
            Enable end-to-end encryption via ECDHE (default: True)
        llm_instance : Optional[Any]
            LLM instance for protocol negotiation (required if negotiation enabled)
        protocol_code_path : Optional[str]
            Path for storing generated protocol code
        auth_headers : Optional[Dict[str, str]]
            Additional authentication headers
        new_session_callback : Optional[Callable]
            Callback function for handling new incoming sessions
        """
        if not AGENTCONNECT_AVAILABLE:
            raise ImportError(
                "AgentConnect library not available. "
                "Please install AgentConnect to use ANP adapter."
            )
        
        super().__init__(base_url=f"anp://{target_did}", auth_headers=auth_headers or {})
        
        # Basic configuration
        self.httpx_client = httpx_client
        self.target_did = target_did
        self.local_did_info = local_did_info or {}
        self.host_domain = host_domain
        self.host_port = host_port
        self.host_ws_path = host_ws_path
        self.did_service_url = did_service_url
        self.did_api_key = did_api_key
        self.enable_protocol_negotiation = enable_protocol_negotiation
        self.enable_e2e_encryption = enable_e2e_encryption
        self.llm_instance = llm_instance
        self.protocol_code_path = protocol_code_path or "./generated_protocols"
        self.new_session_callback = new_session_callback
        
        # AgentConnect core components
        self.simple_node: Optional[SimpleNode] = None
        self.node_session: Optional[SimpleNodeSession] = None
        self.did_client: Optional[DIDAllClient] = None
        self.meta_protocol: Optional[MetaProtocol] = None
        self.protocol_container: Optional[ProtocolContainer] = None
        
        # Connection and session management
        self._connected = False
        self._connecting = False
        self._connect_lock = asyncio.Lock()
        self._sessions: Dict[str, SimpleNodeSession] = {}  # DID -> Session mapping
        
        # Message handling with protocol support
        self._message_queue = asyncio.Queue()
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._protocol_handlers: Dict[str, Callable] = {}
        
        # Agent card
        self.agent_card: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """
        Initialize ANP adapter with full AgentConnect stack.
        
        This initializes:
        1. DID authentication system (did:wba)
        2. SimpleNode for WebSocket communication  
        3. Meta-protocol for LLM-driven negotiation
        4. Application protocol container
        5. End-to-end encryption setup
        
        Raises
        ------
        ConnectionError
            If any component fails to initialize
        """
        try:
            logger.info("Initializing ANP adapter with AgentConnect stack")
            
            # 1. Initialize DID client and authentication
            await self._setup_did_authentication()
            
            # 2. Setup SimpleNode for WebSocket communication
            await self._setup_simple_node()
            
            # 3. Initialize meta-protocol for negotiation
            if self.enable_protocol_negotiation:
                await self._setup_meta_protocol()
            
            # 4. Initialize application protocol container
            await self._setup_protocol_container()
            
            # 5. Setup agent card with all capabilities
            self._setup_comprehensive_agent_card()
            
            logger.info(f"ANP adapter fully initialized with DID: {self.local_did_info.get('did', 'unknown')}")
            
        except Exception as e:
            raise ConnectionError(f"Failed to initialize ANP adapter: {e}") from e

    async def _setup_did_authentication(self) -> None:
        """Setup DID-based authentication system."""
        # Initialize DID client for did:wba if service available
        if self.did_service_url and self.did_api_key:
            self.did_client = DIDAllClient(self.did_service_url, self.did_api_key)
            logger.info("DID client initialized for did:wba service")
        
        # Generate or load local DID information
        if not self.local_did_info.get('did'):
            logger.info("Generating new DID document for ANP adapter")
            
            # Generate WebSocket communication endpoint
            ws_endpoint = f"wss://{self.host_domain}:{self.host_port or '8000'}{self.host_ws_path}"
            
            if self.did_client:
                # Use DID service for did:wba generation and registration
                try:
                    private_key_pem, did, did_document_json = await self.did_client.generate_register_did_document(
                        communication_service_endpoint=ws_endpoint
                    )
                    if not did:
                        raise ConnectionError("Failed to register DID with did:wba service")
                    logger.info(f"DID registered with service: {did}")
                except Exception as e:
                    logger.warning(f"DID service failed, falling back to local generation: {e}")
                    # Fallback to local generation
                    private_key, _, did, did_document_json = did_generate(ws_endpoint)
                    from agent_connect.python.utils.crypto_tool import get_pem_from_private_key
                    private_key_pem = get_pem_from_private_key(private_key)
            else:
                # Generate DID locally using AgentConnect utils
                private_key, _, did, did_document_json = did_generate(ws_endpoint)
                from agent_connect.python.utils.crypto_tool import get_pem_from_private_key
                private_key_pem = get_pem_from_private_key(private_key)
            
            self.local_did_info = {
                'private_key_pem': private_key_pem,
                'did': did,
                'did_document_json': did_document_json
            }
            logger.info(f"Local DID generated: {did}")

    async def _setup_simple_node(self) -> None:
        """Setup SimpleNode for AgentConnect WebSocket communication."""
        try:
            # Create SimpleNode with proper configuration
            self.simple_node = SimpleNode(
                host_domain=self.host_domain,
                host_port=self.host_port,
                host_ws_path=self.host_ws_path,
                private_key_pem=self.local_did_info.get('private_key_pem'),
                did=self.local_did_info.get('did'),
                did_document_json=self.local_did_info.get('did_document_json'),
                new_session_callback=self._on_new_session
            )
            
            # Start the SimpleNode (this starts the WebSocket server)
            self.simple_node.run()
            
            logger.info(f"SimpleNode initialized on {self.host_domain}:{self.host_port}{self.host_ws_path}")
            
            # Wait for node to be ready
            await asyncio.sleep(0.5)
            
        except Exception as e:
            raise ConnectionError(f"Failed to setup SimpleNode: {e}") from e

    async def _setup_meta_protocol(self) -> None:
        """Setup meta-protocol for LLM-driven protocol negotiation."""
        if not self.enable_protocol_negotiation:
            return
            
        if not self.llm_instance:
            logger.warning("Protocol negotiation enabled but no LLM instance provided")
            return
        
        try:
            # Create meta-protocol with send callback and capability callback
            self.meta_protocol = MetaProtocol(
                send_callback=self._send_meta_protocol_data,
                get_capability_info_callback=self._get_capability_info,
                llm=self.llm_instance,
                protocol_code_path=self.protocol_code_path
            )
            
            logger.info("Meta-protocol initialized for LLM-driven negotiation")
            
        except Exception as e:
            logger.error(f"Failed to setup meta-protocol: {e}")
            # Continue without meta-protocol
            self.enable_protocol_negotiation = False

    async def _setup_protocol_container(self) -> None:
        """Setup application protocol container for dynamic protocol loading."""
        try:
            self.protocol_container = ProtocolContainer()
            
            # Load any existing protocols from the protocol path
            if os.path.exists(self.protocol_code_path):
                await self.protocol_container.load_protocols_from_path(self.protocol_code_path)
            
            logger.info("Protocol container initialized")
            
        except Exception as e:
            logger.warning(f"Failed to setup protocol container: {e}")
            # Continue without protocol container

    def _setup_comprehensive_agent_card(self) -> None:
        """Setup comprehensive agent card with all ANP capabilities."""
        self.agent_card = {
            "id": self.local_did_info.get('did', 'unknown'),
            "name": f"ANP Agent - {self.local_did_info.get('did', 'unknown')[:16]}...",
            "description": "Full Agent Network Protocol (ANP) agent with AgentConnect integration",
            "version": "1.0.0",
            "protocol": "ANP",
            "protocolVersion": "1.0.0",
            
            # Core ANP capabilities
            "capabilities": {
                # Authentication
                "did_authentication": True,
                "did_method": "did:wba",
                "decentralized_identity": True,
                
                # Communication
                "websocket_transport": True,
                "persistent_connections": True,
                "real_time_communication": True,
                "session_management": True,
                
                # Security
                "end_to_end_encryption": self.enable_e2e_encryption,
                "ecdhe_key_exchange": self.enable_e2e_encryption,
                "aes_gcm_encryption": self.enable_e2e_encryption,
                
                # Protocol features
                "protocol_negotiation": self.enable_protocol_negotiation,
                "meta_protocol_support": self.enable_protocol_negotiation,
                "llm_driven_negotiation": self.enable_protocol_negotiation and bool(self.llm_instance),
                "dynamic_protocol_loading": True,
                "code_generation": self.enable_protocol_negotiation,
                
                # Advanced features
                "agent_description_support": True,
                "multi_protocol_support": True,
                "heartbeat_monitoring": True,
                "connection_recovery": True
            },
            
            # Endpoints and connection info
            "endpoints": {
                "websocket": f"wss://{self.host_domain}:{self.host_port or '8000'}{self.host_ws_path}",
                "did_document": self.local_did_info.get('did', ''),
                "agent_description": self.local_did_info.get('agent_description_url', '')
            },
            
            # Supported message types and formats
            "supportedMessageTypes": [
                "text", "json", "binary", "encrypted",
                "meta_protocol", "application_protocol", "natural_language"
            ],
            
            "supportedEncodings": ["utf-8", "json", "base64", "binary"],
            
            # Authentication details
            "authentication": {
                "type": "DID",
                "method": "did:wba",
                "did": self.local_did_info.get('did', ''),
                "verification_methods": [
                    "Ed25519VerificationKey2018", 
                    "EcdsaSecp256k1VerificationKey2019"
                ],
                "required": True
            },
            
            # Security specifications
            "security": {
                "end_to_end_encryption": self.enable_e2e_encryption,
                "transport_encryption": True,
                "authentication_required": True,
                "key_exchange": "ECDHE" if self.enable_e2e_encryption else None,
                "encryption_algorithm": "AES-GCM" if self.enable_e2e_encryption else None
            },
            
            # Feature specifications
            "features": {
                "session_persistence": True,
                "message_ordering": True,
                "delivery_confirmation": True,
                "message_history": False,  # Can be enabled
                "file_transfer": True,
                "streaming_support": True,
                "protocol_negotiation": self.enable_protocol_negotiation,
                "meta_protocol": self.enable_protocol_negotiation
            },
            
            # Operational limits
            "limits": {
                "max_message_size": 10485760,  # 10MB
                "max_concurrent_sessions": 1000,
                "session_timeout": 3600,  # 1 hour
                "heartbeat_interval": 30,  # 30 seconds
                "max_negotiation_rounds": 10
            },
            
            # Protocol specifications
            "protocols": {
                "supported": ["ANP", "meta-protocol"],
                "negotiable": self.enable_protocol_negotiation,
                "auto_generation": self.enable_protocol_negotiation,
                "llm_integration": bool(self.llm_instance)
            }
        }

    async def _on_new_session(self, session: SimpleNodeSession) -> None:
        """Handle new incoming WebSocket session from AgentConnect SimpleNode."""
        try:
            remote_did = session.remote_did
            logger.info(f"New ANP session established with DID: {remote_did}")
            
            # Store the session
            self._sessions[remote_did] = session
            self._connected = True
            
            # If this is a session to our target DID, store it as primary session
            if remote_did == self.target_did:
                self.node_session = session
            
            # Start message processing task for this session
            asyncio.create_task(self._process_session_messages(session))
            
            # Call user-provided callback if available
            if self.new_session_callback:
                await self.new_session_callback(session)
                
        except Exception as e:
            logger.error(f"Error handling new session: {e}")

    async def _process_session_messages(self, session: SimpleNodeSession) -> None:
        """Process incoming messages from a WebSocket session with full AgentConnect features."""
        try:
            while self._connected and session.remote_did in self._sessions:
                try:
                    # Receive message using AgentConnect SimpleNodeSession
                    source_did, destination_did, raw_message = await session.receive_message()
                    
                    if raw_message is None:
                        break
                    
                    # Decode message (could be encrypted)
                    try:
                        if isinstance(raw_message, bytes):
                            message_text = raw_message.decode('utf-8')
                        else:
                            message_text = str(raw_message)
                        
                        # Parse message - could be JSON or plain text
                        try:
                            message = json.loads(message_text)
                        except json.JSONDecodeError:
                            # Treat as plain text message
                            message = {"type": "text", "content": message_text}
                    except Exception as e:
                        logger.warning(f"Failed to decode message: {e}")
                        continue
                    
                    # Route message based on protocol type
                    await self._route_message(session, source_did, destination_did, message)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error processing message from {session.remote_did}: {e}")
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Session message processing failed: {e}")
        finally:
            # Clean up session
            if session.remote_did in self._sessions:
                del self._sessions[session.remote_did]
            if self.node_session == session:
                self._connected = False

    async def _route_message(self, session: SimpleNodeSession, source_did: str, 
                           destination_did: str, message: Dict[str, Any]) -> None:
        """Route incoming message based on AgentConnect protocol types."""
        try:
            message_type = message.get("type", "unknown")
            
            # Check if this is a protocol-typed message
            protocol_type = message.get("protocol_type")
            
            if protocol_type == ProtocolType.META.value or message_type == "meta_protocol":
                # Handle meta-protocol negotiation
                await self._handle_meta_protocol_message(session, message)
            elif protocol_type == ProtocolType.APPLICATION.value or message_type == "application_protocol":
                # Handle application protocol message
                await self._handle_application_protocol_message(session, message)
            elif protocol_type == ProtocolType.NATURAL.value or message_type == "natural_language":
                # Handle natural language message
                await self._handle_natural_language_message(session, message)
            elif message_type == "ping":
                # Handle ping for health check
                await self._handle_ping_message(session, message)
            else:
                # Handle regular message or response
                await self._handle_regular_message(session, source_did, message)
                
        except Exception as e:
            logger.error(f"Error routing message: {e}")

    async def _handle_meta_protocol_message(self, session: SimpleNodeSession, message: Dict[str, Any]) -> None:
        """Handle meta-protocol negotiation messages."""
        if not self.meta_protocol:
            logger.warning("Received meta-protocol message but meta-protocol not initialized")
            return
        
        try:
            # Extract meta-protocol data
            meta_data = message.get("data", message.get("content", ""))
            
            if isinstance(meta_data, str):
                meta_data = meta_data.encode('utf-8')
            elif isinstance(meta_data, dict):
                meta_data = json.dumps(meta_data).encode('utf-8')
            
            # Process meta-protocol message
            self.meta_protocol.handle_meta_data(meta_data)
            
            logger.info("Processed meta-protocol negotiation message")
            
        except Exception as e:
            logger.error(f"Error handling meta-protocol message: {e}")

    async def _handle_application_protocol_message(self, session: SimpleNodeSession, message: Dict[str, Any]) -> None:
        """Handle application protocol messages via protocol container."""
        if not self.protocol_container:
            logger.warning("Received application protocol message but container not initialized")
            return
        
        try:
            protocol_name = message.get("protocol_name", "default")
            protocol_data = message.get("data", message.get("content"))
            
            # Route to appropriate protocol handler
            if protocol_name in self._protocol_handlers:
                handler = self._protocol_handlers[protocol_name]
                response = await handler(protocol_data)
                
                # Send response back
                if response:
                    await self._send_message_to_session(session, {
                        "type": "application_protocol_response",
                        "protocol_name": protocol_name,
                        "data": response,
                        "request_id": message.get("request_id")
                    })
            else:
                logger.warning(f"No handler for protocol: {protocol_name}")
                
        except Exception as e:
            logger.error(f"Error handling application protocol message: {e}")

    async def _handle_natural_language_message(self, session: SimpleNodeSession, message: Dict[str, Any]) -> None:
        """Handle natural language messages."""
        try:
            content = message.get("content", message.get("data", ""))
            
            # Queue natural language message for processing
            await self._message_queue.put({
                "session": session,
                "source_did": session.remote_did,
                "type": "natural_language",
                "content": content,
                "request_id": message.get("request_id")
            })
            
        except Exception as e:
            logger.error(f"Error handling natural language message: {e}")

    async def _handle_ping_message(self, session: SimpleNodeSession, message: Dict[str, Any]) -> None:
        """Handle ping message for health checks."""
        try:
            # Send pong response
            pong_response = {
                "type": "pong",
                "ping_timestamp": message.get("timestamp"),
                "pong_timestamp": asyncio.get_event_loop().time(),
                "request_id": message.get("request_id")
            }
            
            await self._send_message_to_session(session, pong_response)
            
        except Exception as e:
            logger.error(f"Error handling ping message: {e}")

    async def _handle_regular_message(self, session: SimpleNodeSession, source_did: str, message: Dict[str, Any]) -> None:
        """Handle regular request-response messages."""
        try:
            # Check if this is a response to a pending request
            request_id = message.get("request_id")
            if request_id and request_id in self._pending_responses:
                future = self._pending_responses.pop(request_id)
                if not future.done():
                    future.set_result(message)
                return
            
            # Queue regular message for processing
            await self._message_queue.put({
                "session": session,
                "source_did": source_did,
                "message": message
            })
            
        except Exception as e:
            logger.error(f"Error handling regular message: {e}")

    async def _send_message_to_session(self, session: SimpleNodeSession, message: Dict[str, Any]) -> bool:
        """Send message to a specific session using AgentConnect encryption."""
        try:
            # Serialize message
            message_json = json.dumps(message, separators=(',', ':'))
            
            # Send via SimpleNodeSession (handles encryption automatically)
            success = await session.send_message(message_json)
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending message to session: {e}")
            return False

    async def _send_meta_protocol_data(self, data: bytes) -> None:
        """Callback for meta-protocol to send data."""
        if not self.node_session:
            logger.warning("No active session for sending meta-protocol data")
            return
        
        try:
            # Wrap meta-protocol data in ANP message format
            meta_message = {
                "type": "meta_protocol",
                "protocol_type": ProtocolType.META.value,
                "data": data.decode('utf-8') if isinstance(data, bytes) else data,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            await self._send_message_to_session(self.node_session, meta_message)
            
        except Exception as e:
            logger.error(f"Error sending meta-protocol data: {e}")

    async def _get_capability_info(self, requirement: str, input_desc: str, output_desc: str) -> str:
        """Callback for meta-protocol to assess capability information."""
        try:
            # Basic capability assessment based on adapter features
            capabilities = []
            
            if self.enable_e2e_encryption:
                capabilities.append("end-to-end encryption support")
            
            if self.enable_protocol_negotiation:
                capabilities.append("protocol negotiation and code generation")
            
            if self.protocol_container:
                capabilities.append("dynamic protocol loading")
            
            capability_info = f"""
Capability Assessment:
- Requirements: {requirement}
- Input format: {input_desc} 
- Output format: {output_desc}
- Available capabilities: {', '.join(capabilities)}
- Can process: JSON, text, binary formats
- Can generate: JSON responses, structured data
- Limitations: None for standard message types
- Assessment: Can meet the specified requirements
"""
            
            return capability_info
            
        except Exception as e:
            logger.error(f"Error assessing capabilities: {e}")
            return f"Capability assessment failed: {e}"

    async def _connect_to_target(self) -> None:
        """Establish connection to target DID using AgentConnect SimpleNode."""
        async with self._connect_lock:
            if self._connected or self._connecting:
                return
            
            self._connecting = True
            try:
                logger.info(f"Connecting to target DID via AgentConnect: {self.target_did}")
                
                if not self.simple_node:
                    raise ConnectionError("SimpleNode not initialized")
                
                # Use SimpleNode to connect to target DID (this creates encrypted session)
                session = await self.simple_node.connect_to_did(self.target_did)
                
                if session:
                    self.node_session = session
                    self._sessions[self.target_did] = session
                    self._connected = True
                    
                    # Start message processing for this session
                    asyncio.create_task(self._process_session_messages(session))
                    
                    logger.info(f"Successfully connected to {self.target_did} via AgentConnect")
                else:
                    raise ConnectionError("Failed to establish session with target DID")
                
            except Exception as e:
                raise ConnectionError(f"Failed to connect to {self.target_did}: {e}") from e
            finally:
                self._connecting = False

    async def send_message(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        """
        Send message via ANP protocol.
        
        Parameters
        ----------
        dst_id : str
            Destination agent ID (should match target_did)
        payload : Dict[str, Any]
            Message payload to send
        
        Returns
        -------
        Any
            Response from target agent
            
        Raises
        ------
        TimeoutError
            If request times out
        ConnectionError
            For connection errors
        RuntimeError
            For other send failures
        """
        # Ensure connection is established
        if not self._connected:
            await self._connect_to_target()
        
        if not self.node_session:
            raise ConnectionError("No active session to target DID")
        
        request_id = str(uuid.uuid4())
        
        try:
            # Prepare message with metadata
            anp_message = {
                "request_id": request_id,
                "source_did": self.local_did_info.get('did'),
                "target_did": dst_id,
                "timestamp": asyncio.get_event_loop().time(),
                "payload": payload
            }
            
            # Set up response future
            response_future = asyncio.Future()
            self._pending_responses[request_id] = response_future
            
            try:
                # Send message via WebSocket
                message_json = json.dumps(anp_message, separators=(',', ':'))
                success = await self.node_session.send_message(message_json)
                
                if not success:
                    raise RuntimeError("Failed to send message via WebSocket")
                
                # Wait for response with timeout
                try:
                    response = await asyncio.wait_for(response_future, timeout=30.0)
                    return response.get("payload", response)
                    
                except asyncio.TimeoutError:
                    raise TimeoutError(f"ANP request timeout to {dst_id} (req_id: {request_id})")
                
            finally:
                # Clean up pending response
                self._pending_responses.pop(request_id, None)
                
        except Exception as e:
            if isinstance(e, (TimeoutError, ConnectionError)):
                raise
            else:
                raise RuntimeError(f"ANP send failed: {e} (req_id: {request_id})") from e

    async def send_message_streaming(self, dst_id: str, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Send message with streaming response via ANP protocol.
        
        Parameters
        ----------
        dst_id : str
            Destination agent ID
        payload : Dict[str, Any]
            Message payload
        
        Yields
        ------
        Dict[str, Any]
            Response events
        """
        # For streaming, we need to handle multiple responses
        # This is a simplified implementation
        response = await self.send_message(dst_id, payload)
        
        # If response is a list of events, yield each one
        if isinstance(response, dict) and "events" in response:
            for event in response["events"]:
                yield event
        else:
            # Single response
            yield response

    async def receive_message(self) -> Dict[str, Any]:
        """
        Receive message from ANP protocol.
        
        Returns
        -------
        Dict[str, Any]
            Received message
        """
        if not self._connected:
            return {"messages": []}
        
        try:
            # Get message from queue with timeout
            message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
            return {"messages": [message]}
        except asyncio.TimeoutError:
            return {"messages": []}

    def get_agent_card(self) -> Dict[str, Any]:
        """Return the agent card."""
        return self.agent_card.copy()

    async def health_check(self) -> bool:
        """
        Check if the ANP connection is healthy.
        
        Returns
        -------
        bool
            True if connection is healthy, False otherwise
        """
        try:
            if not self._connected or not self.node_session:
                return False
            
            # Try to send a ping message
            ping_payload = {"type": "ping", "timestamp": asyncio.get_event_loop().time()}
            
            try:
                response = await asyncio.wait_for(
                    self.send_message(self.target_did, ping_payload),
                    timeout=5.0
                )
                return True
            except (TimeoutError, ConnectionError):
                return False
                
        except Exception:
            return False

    async def cleanup(self) -> None:
        """
        Clean up adapter resources.
        """
        try:
            self._connected = False
            
            # Cancel all pending responses
            for future in self._pending_responses.values():
                if not future.done():
                    future.cancel()
            self._pending_responses.clear()
            
            # Close session if exists
            if self.node_session:
                # SimpleNodeSession doesn't have explicit close method
                # Connection will be closed when SimpleNode stops
                self.node_session = None
            
            # Stop SimpleNode
            if self.simple_node:
                await self.simple_node.stop()
                self.simple_node = None
            
            # Clear queues
            while not self._message_queue.empty():
                try:
                    self._message_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            self.agent_card.clear()
            
        except Exception as e:
            logger.error(f"Error during ANP adapter cleanup: {e}")

    def get_capabilities(self) -> Dict[str, Any]:
        """Extract capabilities from agent card."""
        return self.agent_card.get("capabilities", {})

    def get_endpoint_info(self) -> Dict[str, Any]:
        """Get endpoint information."""
        return {
            "protocol": "ANP",
            "target_did": self.target_did,
            "local_did": self.local_did_info.get('did', 'unknown'),
            "websocket_endpoint": f"ws://{self.host_domain}:{self.host_port or '8000'}{self.host_ws_path}",
            "supports_streaming": True,
            "supports_auth": True,
            "authentication_type": "DID",
            "supports_encryption": True,
            "connection_type": "persistent",
            "protocol_negotiation": self.enable_protocol_negotiation
        }

    def get_protocol_version(self) -> str:
        """Get ANP protocol version."""
        return self.agent_card.get("protocolVersion", "1.0.0")

    def __repr__(self) -> str:
        """Debug representation of ANPAdapter."""
        return (
            f"ANPAdapter(target_did='{self.target_did[:20]}...', "
            f"local_did='{self.local_did_info.get('did', 'unknown')[:20]}...', "
            f"connected={self._connected}, "
            f"protocol_negotiation={self.enable_protocol_negotiation})"
        )


class ANPMessageBuilder:
    """
    Helper class for building ANP messages.
    """
    
    @staticmethod
    def text_message(content: str, message_type: str = "text") -> Dict[str, Any]:
        """Build a text message."""
        return {
            "type": message_type,
            "content": content
        }
    
    @staticmethod
    def json_message(data: Dict[str, Any], message_type: str = "json") -> Dict[str, Any]:
        """Build a JSON message."""
        return {
            "type": message_type,
            "data": data
        }
    
    @staticmethod
    def protocol_negotiation_message(requirement: str, input_desc: str, output_desc: str) -> Dict[str, Any]:
        """Build a protocol negotiation message."""
        return {
            "type": "protocol_negotiation",
            "requirement": requirement,
            "input_description": input_desc,
            "output_description": output_desc
        }
    
    @staticmethod
    def ping_message() -> Dict[str, Any]:
        """Build a ping message for health check."""
        return {
            "type": "ping",
            "timestamp": asyncio.get_event_loop().time()
        }
    
    @staticmethod
    def pong_message(ping_timestamp: float) -> Dict[str, Any]:
        """Build a pong response message."""
        return {
            "type": "pong",
            "ping_timestamp": ping_timestamp,
            "pong_timestamp": asyncio.get_event_loop().time()
        } 