"""
ANP (Agent Network Protocol) Adapter - AgentConnect协议适配器实现
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, Optional, AsyncIterator
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
    DIDAllClient, create_did_wba_document, generate_auth_header
)
from agent_connect.python.utils.did_generate import did_generate
AGENTCONNECT_AVAILABLE = True

logger = logging.getLogger(__name__)


class ANPAdapter(BaseProtocolAdapter):
    """
    Adapter for ANP (Agent Network Protocol) specification.
    
    Implements the AgentConnect protocol for decentralized agent communication
    using DID-based authentication and WebSocket transport.
    
    ANP Features:
    - DID-based decentralized authentication
    - WebSocket persistent connections
    - End-to-end encryption
    - Protocol negotiation via LLM
    - Dynamic protocol loading
    
    Instance granularity: One outbound edge = One Adapter instance.
    Different dst_id / target_did / protocol configuration do not share instances.
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
        protocol_negotiation: bool = False,
        auth_headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize ANP adapter.
        
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
            DID resolution service URL
        did_api_key : Optional[str]
            API key for DID service
        protocol_negotiation : bool
            Enable LLM-based protocol negotiation
        auth_headers : Optional[Dict[str, str]]
            Additional authentication headers
        """
        if not AGENTCONNECT_AVAILABLE:
            raise ImportError(
                "AgentConnect library not available. "
                "Please install AgentConnect to use ANP adapter."
            )
        
        super().__init__(base_url=f"anp://{target_did}", auth_headers=auth_headers or {})
        
        self.httpx_client = httpx_client
        self.target_did = target_did
        self.local_did_info = local_did_info or {}
        self.host_domain = host_domain
        self.host_port = host_port
        self.host_ws_path = host_ws_path
        self.did_service_url = did_service_url
        self.did_api_key = did_api_key
        self.protocol_negotiation = protocol_negotiation
        
        # ANP components
        self.simple_node: Optional[SimpleNode] = None
        self.node_session: Optional[SimpleNodeSession] = None
        self.did_client: Optional[DIDAllClient] = None
        
        # Connection state
        self._connected = False
        self._connecting = False
        self._connect_lock = asyncio.Lock()
        
        # Message handling
        self._message_queue = asyncio.Queue()
        self._pending_responses: Dict[str, asyncio.Future] = {}
        
        # Agent card
        self.agent_card: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """
        Initialize ANP adapter by setting up local DID and SimpleNode.
        
        Raises
        ------
        ConnectionError
            If DID setup or node initialization fails
        """
        try:
            # Initialize DID client if service available
            if self.did_service_url and self.did_api_key:
                self.did_client = DIDAllClient(self.did_service_url, self.did_api_key)
            
            # Generate or load local DID information
            await self._setup_local_did()
            
            # Create SimpleNode for WebSocket communication
            await self._setup_simple_node()
            
            # Set up agent card
            self._setup_agent_card()
            
            logger.info(f"ANP adapter initialized with local DID: {self.local_did_info.get('did', 'unknown')}")
            
        except Exception as e:
            raise ConnectionError(f"Failed to initialize ANP adapter: {e}") from e

    async def _setup_local_did(self) -> None:
        """Setup local DID information."""
        if not self.local_did_info.get('did'):
            # Generate new DID if not provided
            logger.info("Generating new DID for ANP adapter")
            
            # Generate DID with communication endpoint
            ws_endpoint = f"ws://{self.host_domain}:{self.host_port or '8000'}{self.host_ws_path}"
            
            if self.did_client:
                # Use DID service to generate and register
                private_key_pem, did, did_document_json = await self.did_client.generate_register_did_document(
                    communication_service_endpoint=ws_endpoint
                )
                if not did:
                    raise ConnectionError("Failed to register DID with service")
            else:
                # Generate DID locally
                private_key, _, did, did_document_json = did_generate(ws_endpoint)
                from agentconnect_src.agent_connect.python.utils.crypto_tool import get_pem_from_private_key
                private_key_pem = get_pem_from_private_key(private_key)
            
            self.local_did_info = {
                'private_key_pem': private_key_pem,
                'did': did,
                'did_document_json': did_document_json
            }

    async def _setup_simple_node(self) -> None:
        """Setup SimpleNode for WebSocket communication."""
        try:
            # Create SimpleNode with session callback
            self.simple_node = SimpleNode(
                host_domain=self.host_domain,
                new_session_callback=self._on_new_session,
                host_port=self.host_port,
                host_ws_path=self.host_ws_path,
                private_key_pem=self.local_did_info.get('private_key_pem'),
                did=self.local_did_info.get('did'),
                did_document_json=self.local_did_info.get('did_document_json')
            )
            
            # Start the node (non-blocking)
            self.simple_node.run()
            
            # Wait a moment for the node to start
            await asyncio.sleep(0.5)
            
        except Exception as e:
            raise ConnectionError(f"Failed to setup SimpleNode: {e}") from e

    def _setup_agent_card(self) -> None:
        """Setup agent card with ANP capabilities."""
        self.agent_card = {
            "id": self.local_did_info.get('did', 'unknown'),
            "name": f"ANP Agent - {self.local_did_info.get('did', 'unknown')[:16]}...",
            "description": "Agent Network Protocol (ANP) compatible agent with DID-based authentication",
            "version": "1.0.0",
            "protocol": "ANP",
            "protocolVersion": "1.0.0",
            "capabilities": {
                "did_authentication": True,
                "websocket_transport": True,
                "end_to_end_encryption": True,
                "protocol_negotiation": self.protocol_negotiation,
                "persistent_connections": True
            },
            "endpoints": {
                "websocket": f"ws://{self.host_domain}:{self.host_port or '8000'}{self.host_ws_path}",
                "did_document": self.local_did_info.get('did', '')
            },
            "supportedMessageTypes": ["text", "json", "binary"],
            "authentication": {
                "type": "DID",
                "did": self.local_did_info.get('did', ''),
                "verification_methods": ["Ed25519VerificationKey2018", "EcdsaSecp256k1VerificationKey2019"]
            }
        }

    async def _on_new_session(self, session: SimpleNodeSession) -> None:
        """Handle new incoming WebSocket session."""
        logger.info(f"New ANP session established with DID: {session.remote_did}")
        
        # If this is the session we're looking for, store it
        if session.remote_did == self.target_did:
            self.node_session = session
            self._connected = True
            
            # Start message processing task
            asyncio.create_task(self._process_incoming_messages(session))

    async def _connect_to_target(self) -> None:
        """Establish connection to target DID."""
        async with self._connect_lock:
            if self._connected or self._connecting:
                return
            
            self._connecting = True
            try:
                logger.info(f"Connecting to target DID: {self.target_did}")
                
                # Use SimpleNode to connect to target DID
                session = await self.simple_node.connect_to_did(self.target_did)
                self.node_session = session
                self._connected = True
                
                # Start message processing task
                asyncio.create_task(self._process_incoming_messages(session))
                
                logger.info(f"Successfully connected to {self.target_did}")
                
            except Exception as e:
                raise ConnectionError(f"Failed to connect to {self.target_did}: {e}") from e
            finally:
                self._connecting = False

    async def _process_incoming_messages(self, session: SimpleNodeSession) -> None:
        """Process incoming messages from WebSocket session."""
        try:
            while self._connected:
                try:
                    # Receive message from session
                    raw_message = await session.receive_message()
                    if raw_message is None:
                        break
                    
                    # Decode message
                    if isinstance(raw_message, bytes):
                        message_text = raw_message.decode('utf-8')
                    else:
                        message_text = str(raw_message)
                    
                    # Try to parse as JSON
                    try:
                        message = json.loads(message_text)
                    except json.JSONDecodeError:
                        # Treat as plain text message
                        message = {"type": "text", "content": message_text}
                    
                    # Handle response messages
                    if isinstance(message, dict) and "request_id" in message:
                        request_id = message["request_id"]
                        if request_id in self._pending_responses:
                            future = self._pending_responses.pop(request_id)
                            if not future.done():
                                future.set_result(message)
                            continue
                    
                    # Queue regular messages
                    await self._message_queue.put(message)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error processing incoming message: {e}")
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Message processing task failed: {e}")
        finally:
            self._connected = False

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
            "protocol_negotiation": self.protocol_negotiation
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
            f"protocol_negotiation={self.protocol_negotiation})"
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