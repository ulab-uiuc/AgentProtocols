#!/usr/bin/env python3
"""
ANP Agent implementation for Fail-Storm Recovery scenario.

This module provides a self-contained ANP agent implementation that doesn't
depend on external src/ components, making protocol_backends truly independent.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path
import sys

# Add AgentConnect to path
current_dir = Path(__file__).parent.parent.parent
agentconnect_path = current_dir.parent.parent / "agentconnect_src"
sys.path.insert(0, str(agentconnect_path))

from agent_connect.python.simple_node import SimpleNode, SimpleNodeSession
from agent_connect.python.authentication import (
    DIDWbaAuthHeader, verify_auth_header_signature
)
from agent_connect.python.utils.did_generate import did_generate
from agent_connect.python.utils.crypto_tool import get_pem_from_private_key


class ANPAgent:
    """
    Self-contained ANP Agent for Fail-Storm recovery scenarios.
    
    This agent implements ANP protocol capabilities without depending
    on external BaseAgent or src components.
    """
    
    def __init__(self, agent_id: str, host: str, port: int, executor: Any):
        """
        Initialize ANP agent.
        
        Args:
            agent_id: Unique identifier for this agent
            host: Host address to bind to
            port: HTTP port for this agent
            executor: Task executor for handling QA operations
        """
        self.agent_id = agent_id
        self.host = host
        self.port = port
        self.executor = executor
        
        # ANP specific components
        self.http_server = None
        self.websocket_server = None
        self.websocket_port = port + 1000  # WebSocket on HTTP port + 1000
        
        # DID and authentication
        self.did_doc = None
        self.private_keys = None
        self.simple_node = None
        self.node_session = None
        
        # Server management
        self._startup_complete = asyncio.Event()
        self._shutdown_complete = asyncio.Event()
        
        # Compatibility with BaseAgent interface
        self.process = None  # For compatibility with kill operations
        
        self.logger = logging.getLogger(f"ANPAgent.{agent_id}")
    
    async def start(self) -> None:
        """Start the ANP agent with both HTTP and WebSocket servers."""
        try:
            # Generate DID and keys for authentication
            await self._setup_did_authentication()
            
            # Start HTTP server
            await self._start_http_server()
            
            # Start WebSocket server  
            await self._start_websocket_server()
            
            # Initialize ANP communication
            await self._setup_anp_node()
            
            self._startup_complete.set()
            self.logger.info(f"ANP Agent {self.agent_id} started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start ANP agent {self.agent_id}: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the ANP agent and cleanup resources."""
        try:
            if self.http_server:
                self.http_server.should_exit = True
                await self.http_server.shutdown()
            
            if self.websocket_server:
                self.websocket_server.close()
                await self.websocket_server.wait_closed()
            
            if self.node_session:
                await self.node_session.close()
            
            self._shutdown_complete.set()
            self.logger.info(f"ANP Agent {self.agent_id} stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping ANP agent {self.agent_id}: {e}")
    
    async def _setup_did_authentication(self) -> None:
        """Setup DID document and authentication keys."""
        try:
            # Generate DID and private keys
            communication_endpoint = f"http://{self.host}:{self.port}"
            private_key, public_key, did, did_doc = await asyncio.to_thread(
                did_generate, communication_endpoint
            )
            
            self.private_keys = private_key
            self.did_doc = did_doc
            
            self.logger.info(f"Generated DID authentication for {self.agent_id}: {did}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup DID authentication: {e}")
            raise
    
    async def _start_http_server(self) -> None:
        """Start HTTP server for REST API."""
        import uvicorn
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        
        async def health_check(request):
            return JSONResponse({"status": "healthy", "agent_id": self.agent_id})
        
        async def get_did(request):
            return JSONResponse({"did": self.did_doc})
        
        async def handle_message(request):
            """Handle incoming ANP messages."""
            try:
                body = await request.json()
                # Process message through executor if needed
                result = {"status": "received", "agent_id": self.agent_id}
                return JSONResponse(result)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=400)
        
        routes = [
            Route("/health", health_check, methods=["GET"]),
            Route("/did", get_did, methods=["GET"]),
            Route("/message", handle_message, methods=["POST"]),
        ]
        
        app = Starlette(routes=routes)
        
        config = uvicorn.Config(
            app=app,
            host=self.host,
            port=self.port,
            log_level="error"  # Reduce noise
        )
        
        self.http_server = uvicorn.Server(config)
        
        # Start server in background task
        asyncio.create_task(self.http_server.serve())
        
        # Wait a bit for server to start
        await asyncio.sleep(0.5)
    
    async def _start_websocket_server(self) -> None:
        """Start WebSocket server for real-time communication."""
        import websockets
        
        async def handle_websocket(websocket, path):
            """Handle WebSocket connections."""
            try:
                async for message in websocket:
                    # Echo message back for now
                    response = {
                        "agent_id": self.agent_id,
                        "received": message,
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    await websocket.send(json.dumps(response))
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
        
        self.websocket_server = await websockets.serve(
            handle_websocket,
            "0.0.0.0",  # Listen on all interfaces
            self.websocket_port
        )
        
        self.logger.info(f"WebSocket server started on port {self.websocket_port}")
    
    async def _setup_anp_node(self) -> None:
        """Setup ANP SimpleNode for protocol communication."""
        try:
            # Create SimpleNode with our DID
            self.simple_node = SimpleNode(
                did_document=self.did_doc,
                private_key_set=self.private_keys
            )
            
            # Start a session
            self.node_session = SimpleNodeSession(self.simple_node)
            
            self.logger.info(f"ANP node session established for {self.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup ANP node: {e}")
            # Continue without ANP node if it fails
            pass
    
    async def send_message(self, target_agent_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send message to another ANP agent.
        
        Args:
            target_agent_id: ID of target agent
            message: Message to send
            
        Returns:
            Response from target agent or None if failed
        """
        try:
            if self.node_session:
                # Use ANP protocol for communication
                response = await self.node_session.send_message(
                    target_did=f"did:example:{target_agent_id}",
                    message=message
                )
                return response
            else:
                # Fallback to direct HTTP communication
                return await self._send_http_message(target_agent_id, message)
                
        except Exception as e:
            self.logger.error(f"Failed to send message to {target_agent_id}: {e}")
            return None
    
    async def _send_http_message(self, target_agent_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fallback HTTP message sending."""
        import httpx
        
        try:
            # Assume target is on next port
            target_port = self.port + 1 if "0" in target_agent_id else self.port - 1
            url = f"http://{self.host}:{target_port}/message"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=message, timeout=10.0)
                return response.json()
                
        except Exception as e:
            self.logger.error(f"HTTP message failed: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "host": self.host,
            "http_port": self.port,
            "websocket_port": self.websocket_port,
            "did": self.did_doc.get("id") if self.did_doc else None,
            "anp_ready": self.node_session is not None,
            "startup_complete": self._startup_complete.is_set(),
        }


async def create_anp_agent(agent_id: str, host: str, port: int, executor: Any) -> ANPAgent:
    """
    Factory function to create and start an ANP agent.
    
    Args:
        agent_id: Unique identifier for the agent
        host: Host address to bind to  
        port: Port number for HTTP server
        executor: Task executor for QA operations
        
    Returns:
        Started ANP agent instance
    """
    agent = ANPAgent(agent_id, host, port, executor)
    await agent.start()
    return agent
