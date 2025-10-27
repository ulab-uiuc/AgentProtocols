#!/usr/bin/env python3
"""
ACP Agent implementation for Fail-Storm Recovery scenario.

This module provides ACP agent implementation using real ACP SDK server,
following the same patterns as A2A agent but using ACP native SDK.
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional, List, AsyncGenerator
from pathlib import Path
import sys

# Add agent_network src path
agent_network_src = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(agent_network_src))

# Import ACP SDK (required) - use agent_network environment
from acp_sdk.models import Message, MessagePart, RunCreateRequest
from acp_sdk.server import Context, RunYield
from acp_sdk.server import Server
from acp_sdk.client import Client

# HTTP client for inter-agent communication
import httpx


def serialize_for_json(obj):
    """Helper function to serialize objects containing datetime for JSON."""
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj


class ACPExecutorWrapper:
    """Wrapper to adapt ShardWorkerExecutor to ACP AgentManifest interface."""
    
    def __init__(self, executor):
        self.executor = executor
        self.agent_id = getattr(executor, 'shard_id', 'unknown')
        self.logger = logging.getLogger(f"ACPExecutorWrapper.{self.agent_id}")
    
    async def process_message(self, sender: str, content: str, meta: dict = None) -> str:
        """Process message through the wrapped executor."""
        try:
            if hasattr(self.executor, 'worker') and hasattr(self.executor.worker, 'process_message'):
                return await self.executor.worker.process_message(sender, content, meta or {})
            else:
                return f"Processed message from {sender}: {content}"
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return f"Error processing message: {str(e)}"

class ACPNativeServer:
    """ACP native server following A2A server patterns."""
    
    def __init__(self, executor, host: str, port: int, agent_id: str):
        self.executor = executor
        self.host = host
        self.port = port
        self.agent_id = agent_id
        self.server_thread = None
        self.acp_server = None
        self.logger = logging.getLogger(f"ACPNativeServer.{agent_id}")
        self._started = False  # Prevent duplicate starts
    
    def _create_native_acp_server(self, executor):
        """Create native ACP server using official SDK."""
        acp_server = Server()

        # Register agent using decorator pattern (correct ACP SDK API)
        @acp_server.agent()
        async def agent_handler(messages: List[Message], context: Context) -> AsyncGenerator[RunYield, None]:
            """Handle incoming ACP messages."""
            try:
                # Extract text content from ACP messages
                content = ""
                for message in messages:
                    for part in message.parts:
                        if part.type == "text":
                            content += part.text
                
                # Process through executor
                result = await executor.process_message("acp_client", content, {})
                
                # Return ACP-formatted response
                response_msg = Message(
                    role="assistant",
                    parts=[MessagePart(type="text", text=str(result))]
                )
                
                yield RunYield(output={"messages": [response_msg]})
                
            except Exception as e:
                self.logger.error(f"Error in ACP agent handler: {e}")
                error_msg = Message(
                    role="assistant", 
                    parts=[MessagePart(type="text", text=f"Error: {str(e)}")]
                )
                yield RunYield(output={"messages": [error_msg]})
        
        self.logger.info(f"Registered ACP agent handler for {self.agent_id}")
        return acp_server
    
    def serve(self):
        """Start server in background thread."""
        if self._started:
            self.logger.warning(f"ACP server for {self.agent_id} already started, skipping")
            return
            
        def run_server():
            try:
                # Create native ACP server
                self.acp_server = self._create_native_acp_server(self.executor)
                # Start ACP SDK server
                self.acp_server.run(host=self.host, port=self.port, log_level="error")
            except Exception as e:
                self.logger.error(f"Server error: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self._started = True
        self.logger.info(f"ACP server started on {self.host}:{self.port}")
    
    def shutdown(self):
        """Shutdown server and release resources."""
        try:
            if self.acp_server and hasattr(self.acp_server, 'should_exit'):
                self.acp_server.should_exit = True
            self._started = False
            self.logger.info("ACP server shutdown completed")
        except Exception as e:
            self.logger.error(f"Error shutting down server: {e}")

class ACPAgent:
    """ACP Agent implementation following A2A patterns but using ACP SDK."""
    
    def __init__(self, agent_id: str, host: str, port: int, executor: Any):
        self.agent_id = agent_id
        self.host = host
        self.port = port
        self.executor = executor
        
        # ACP specific components
        self.native_server = None
        self.acp_executor = None
        
        # Endpoint management
        self._endpoints: Dict[str, str] = {}
        
        # Server management
        self._startup_complete = asyncio.Event()
        self._shutdown_complete = asyncio.Event()
        
        # Compatibility with BaseAgent interface
        self.process = None  # For compatibility with kill operations
        
        self.logger = logging.getLogger(f"ACPAgent.{agent_id}")
    
    async def start(self) -> None:
        """Start the ACP agent with official SDK server."""
        if self._startup_complete.is_set():
            self.logger.warning(f"ACP Agent {self.agent_id} already started, skipping")
            return
            
        try:
            # Start ACP SDK server
            await self._start_acp_server()
            
            self._startup_complete.set()
            self.logger.info(f"ACP Agent {self.agent_id} started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start ACP agent {self.agent_id}: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the ACP agent and cleanup resources."""
        try:
            if self.native_server:
                self.native_server.shutdown()
            
            self._shutdown_complete.set()
            self.logger.info(f"ACP Agent {self.agent_id} stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping ACP agent {self.agent_id}: {e}")
    
    async def _start_acp_server(self) -> None:
        """Start ACP SDK server."""
        try:
            # Create ACP executor wrapper
            self.acp_executor = ACPExecutorWrapper(self.executor)
            
            # Create and start native ACP server
            self.native_server = ACPNativeServer(
                executor=self.acp_executor,
                host=self.host,
                port=self.port,
                agent_id=self.agent_id
            )
            
            self.native_server.serve()  # Starts server in background
            await asyncio.sleep(1)  # Give server time to start
            
        except Exception as e:
            self.logger.error(f"Failed to start ACP server: {e}")
            raise
    
    async def register_endpoint(self, agent_id: str, base_url: str) -> None:
        """Register endpoint for another agent."""
        self._endpoints[agent_id] = base_url
        self.logger.info(f"Registered endpoint for {agent_id}: {base_url}")
    
    async def send_message(self, target_agent_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send message to another ACP agent (following A2A pattern)."""
        try:
            return await self.send(self.agent_id, target_agent_id, message)
        except Exception as e:
            self.logger.error(f"Failed to send message to {target_agent_id}: {e}")
            return None
    
    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """Send message using ACP SDK Client."""
        endpoint = self._endpoints.get(dst_id)
        if not endpoint:
            raise RuntimeError(f"unknown dst_id={dst_id}")

        # Build text content
        text_in = (
            payload.get("body")
            or payload.get("content")
            or (json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload))
        )

        # Use ACP SDK Client to send message
        try:
            # Create ACP client pointing to target agent
            acp_client = Client(base_url=endpoint)
            
            # Create message
            msg = Message(
                role="user",
                parts=[MessagePart(type="text", text=str(text_in))]
            )
            
            # Call agent using SDK (agent name is "agent_handler" by default)
            # Use run_sync since we're in async context already
            run = acp_client.run_sync(
                agent="agent_handler",
                input=[msg]
            )
            
            # Extract response
            text_out = ""
            if hasattr(run, 'messages') and run.messages:
                for msg in run.messages:
                    if hasattr(msg, 'parts'):
                        for part in msg.parts:
                            if hasattr(part, 'text'):
                                text_out += part.text
            
            if not text_out:
                text_out = str(run)
            
            return {"raw": run, "text": text_out}

        except Exception as e:
            self.logger.error(f"ACP protocol send failed: {e}")
            raise RuntimeError(f"ACP protocol send failed: {e}")

    async def health_check(self, agent_id: str) -> bool:
        """Check health of target agent."""
        endpoint = self._endpoints.get(agent_id)
        if not endpoint:
            return False
        client = self._clients.get(agent_id)
        try:
            resp = await client.get("/health")
            return resp.status_code == 200
        except Exception:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "host": self.host,
            "port": self.port,
            "acp_ready": self.acp_executor is not None,
            "startup_complete": self._startup_complete.is_set(),
        }

async def create_acp_agent(agent_id: str, host: str, port: int, executor: Any) -> ACPAgent:
    """Factory function to create and start an ACP agent."""
    agent = ACPAgent(agent_id, host, port, executor)
    await agent.start()
    return agent