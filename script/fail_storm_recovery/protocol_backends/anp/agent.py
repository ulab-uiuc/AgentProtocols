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
    
    This agent implements ANP protocol capabilities using official ANP SDK (AgentConnect)
    SimpleNode for server-side and communication.
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
        
        # ANP specific components (using official SDK)
        self.simple_node = None  # Official ANP SDK SimpleNode
        self.node_session = None
        
        # DID and authentication
        self.did_doc = None
        self.private_keys = None
        
        # Server management
        self._startup_complete = asyncio.Event()
        self._shutdown_complete = asyncio.Event()
        
        # Compatibility with BaseAgent interface
        self.process = None  # For compatibility with kill operations
        
        self.logger = logging.getLogger(f"ANPAgent.{agent_id}")
    
    async def start(self) -> None:
        """Start the ANP agent with official ANP SDK SimpleNode server."""
        try:
            # Generate DID and keys for authentication
            await self._setup_did_authentication()
            
            # Start ANP SDK SimpleNode server
            await self._start_anp_server()
            
            self._startup_complete.set()
            self.logger.info(f"ANP Agent {self.agent_id} started successfully with official SDK")
            
        except Exception as e:
            self.logger.error(f"Failed to start ANP agent {self.agent_id}: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the ANP agent and cleanup resources."""
        try:
            # Stop ANP SDK SimpleNode server
            if self.simple_node and hasattr(self.simple_node, 'server_task'):
                if self.simple_node.server_task:
                    self.simple_node.server_task.cancel()
            
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
    
    async def _start_anp_server(self) -> None:
        """Start ANP SDK SimpleNode server."""
        try:
            # Create communication endpoint URL
            communication_endpoint = f"http://{self.host}:{self.port}"
            
            # Create SimpleNode with official ANP SDK
            self.simple_node = SimpleNode(
                host_domain=self.host,
                host_port=str(self.port),
                host_ws_path="/ws",
                private_key_pem=get_pem_from_private_key(self.private_keys),
                did_document_json=json.dumps(self.did_doc)
            )
            
            # Add health check endpoint to the FastAPI app
            @self.simple_node.app.get("/health")
            async def health_check():
                return {"status": "healthy", "agent_id": self.agent_id, "protocol": "ANP"}
            
            @self.simple_node.app.get("/.well-known/agent.json")
            async def agent_card():
                return {
                    "name": f"ANP Agent {self.agent_id}",
                    "url": communication_endpoint,
                    "protocol": "ANP (Official SDK)",
                    "agent_id": self.agent_id,
                    "did": self.did_doc.get("id") if self.did_doc else None
                }
            
            # Add missing /message endpoint for inter-agent communication
            @self.simple_node.app.post("/message")
            async def handle_message(request):
                """Handle incoming messages from other agents."""
                try:
                    from fastapi import Request
                    if isinstance(request, Request):
                        body = await request.json()
                    else:
                        body = request
                    
                    # Handle A2A message format: {messageId, role, parts, meta}
                    if isinstance(body, dict):
                        # Extract content from A2A format
                        if 'parts' in body and isinstance(body['parts'], list) and len(body['parts']) > 0:
                            # A2A format: parts[0]['text']
                            content = body['parts'][0].get('text', '')
                        else:
                            # Simple format: direct content
                            content = body.get('content', str(body))
                        
                        # Extract sender and meta
                        meta = body.get('meta', {})
                        sender = meta.get('sender', body.get('sender', 'unknown'))
                        
                        # Process message through executor
                        if self.executor and hasattr(self.executor, 'worker'):
                            result = await self.executor.worker.process_message(sender, content, meta)
                            return {"status": "success", "response": result}
                        else:
                            return {"status": "error", "message": "No executor available"}
                    else:
                        # Handle string content directly
                        content = str(body)
                        if self.executor and hasattr(self.executor, 'worker'):
                            result = await self.executor.worker.process_message("unknown", content, {})
                            return {"status": "success", "response": result}
                        else:
                            return {"status": "error", "message": "No executor available"}
                        
                except Exception as e:
                    self.logger.error(f"Error handling message: {e}")
                    return {"status": "error", "message": str(e)}
            
            # Start the SimpleNode server using official SDK
            self.simple_node.run()  # This starts the server in background
            
            # Create a session for communication (will be created when needed for specific connections)
            # SimpleNodeSession requires specific connection parameters
            self.node_session = None
            
            # Wait a bit for server to start
            await asyncio.sleep(1)
            
            self.logger.info(f"ANP SDK SimpleNode server started on {self.host}:{self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start ANP server: {e}")
            raise
    
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
            "port": self.port,
            "did": self.did_doc.get("id") if self.did_doc else None,
            "anp_sdk_ready": self.simple_node is not None,
            "session_ready": self.node_session is not None,
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
