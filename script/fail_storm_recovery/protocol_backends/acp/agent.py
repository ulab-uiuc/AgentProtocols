#!/usr/bin/env python3
"""
ACP Agent implementation for Fail-Storm Recovery scenario.

This module provides ACP agent implementation using real ACP SDK server,
creating a self-contained agent that uses ACP's native server implementation.
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, Optional, List, AsyncGenerator
from pathlib import Path
import sys

# Import ACP SDK (required)
from acp_sdk.models import Message, MessagePart, RunCreateRequest
from acp_sdk.server import Context, RunYield
# Import ACP SDK server components (correct imports)
from acp_sdk.server import Server, AgentManifest

# HTTP client for inter-agent communication
import httpx

# Flask for additional endpoints
from flask import jsonify


class ACPAgentManifest(AgentManifest):
    """100% Native ACP AgentManifest implementation."""
    
    def __init__(self, executor_wrapper, agent_id: str):
        super().__init__()
        self.executor_wrapper = executor_wrapper
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"ACPAgentManifest.{agent_id}")
    
    async def run(self, messages: List[Message], context: Context) -> AsyncGenerator[RunYield, None]:
        """
        100% Native ACP SDK run method implementation.
        
        This is the official ACP SDK interface that AgentManifest must implement.
        """
        self.logger.debug(f"[ACP] Native AgentManifest.run called with {len(messages)} messages")
        
        # Use the executor wrapper to process messages
        async for result in self.executor_wrapper(messages, context):
            yield result


class ACPNativeServer:
    """100% Native ACP server using only official SDK."""
    
    def __init__(self, executor, host: str, port: int, agent_id: str):
        self.executor = executor
        self.host = host
        self.port = port
        self.agent_id = agent_id
        self.server_thread = None
        self.acp_server = None
        self.agent_manifest = None
        self.logger = logging.getLogger(f"ACPNativeServer.{agent_id}")
    
    def _create_100_native_acp_server(self, executor):
        """Create 100% native ACP server using only official SDK."""
        
        # Create 100% native ACP SDK Server
        acp_server = Server()
        
        # Create native AgentManifest with our executor
        self.agent_manifest = ACPAgentManifest(executor, self.agent_id)
        
        # Register the AgentManifest with ACP SDK (this is the official way)
        acp_server.register(self.agent_manifest)
        
        self.logger.info(f"Registered native AgentManifest with ACP SDK Server")
        
        return acp_server
    
    def serve(self):
        """Start server in background thread."""
        def run_server():
            try:
                # Create 100% native ACP server
                self.acp_server = self._create_100_native_acp_server(self.executor)
                # Use 100% native ACP SDK Server.run() method
                self.acp_server.run(host=self.host, port=self.port, log_level="error")
            except Exception as e:
                self.logger.error(f"Server error: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.logger.info(f"100% Native ACP server started on {self.host}:{self.port}")
    
    def shutdown(self):
        """Shutdown server and release resources."""
        try:
            if self.acp_server:
                # Set should_exit flag to stop the server
                self.acp_server.should_exit = True
                if hasattr(self.acp_server, 'shutdown'):
                    self.acp_server.shutdown()
            
            # Cancel the server thread if it exists
            if self.server_thread and self.server_thread.is_alive():
                # Give it more time for proper port release
                self.server_thread.join(timeout=5.0)
                
            # Additional wait to ensure port is fully released
            import time
            time.sleep(1.0)
                
            self.logger.info(f"ACP server shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


class ACPExecutorWrapper:
    """
    Wrapper to convert ShardWorkerExecutor to ACP SDK native executor interface.
    
    ACP SDK expects: async def executor(messages: List[Message], context: Context) -> AsyncGenerator[RunYield, None]
    """
    
    def __init__(self, shard_worker_executor: Any):
        """
        Initialize with shard worker executor.
        
        Args:
            shard_worker_executor: ShardWorkerExecutor instance
        """
        self.shard_worker_executor = shard_worker_executor
        self.capabilities = ["text_processing", "async_generation", "acp_sdk_1.0.3"]
        self.logger = logging.getLogger("ACPExecutorWrapper")
    
    async def __call__(self, messages: List[Message], context: Context) -> AsyncGenerator[RunYield, None]:
        """
        ACP SDK native executor interface implementation.
        
        Args:
            messages: List of ACP SDK Message objects
            context: ACP SDK Context object
            
        Yields:
            RunYield objects as required by ACP SDK
        """
        self.logger.debug(f"[ACP] ACPExecutorWrapper called with {len(messages)} messages")
        
        try:
            # Process each message (typically just one)
            for i, message in enumerate(messages):
                # Generate run_id for this execution
                run_id = str(uuid.uuid4())
                
                self.logger.debug(f"[ACP] Processing message {i+1}/{len(messages)} with run_id: {run_id}")
                
                # Extract text content from message
                text_content = ""
                if hasattr(message, 'parts') and message.parts:
                    for part in message.parts:
                        if hasattr(part, 'type') and part.type == "text":
                            text_content += getattr(part, 'text', getattr(part, 'content', ""))
                else:
                    text_content = str(message)
                
                # Use shard worker to process the content
                if self.shard_worker_executor and hasattr(self.shard_worker_executor, 'worker'):
                    try:
                        # Call LLM directly through worker.answer() if available
                        if hasattr(self.shard_worker_executor.worker, 'answer'):
                            result = await self.shard_worker_executor.worker.answer(text_content)
                            yield result  # Yield LLM result as string
                            self.logger.debug(f"[ACP] LLM result: {len(str(result))} chars")
                        else:
                            # Fallback to start_task
                            result = await self.shard_worker_executor.worker.start_task(0)
                            yield str(result) if result else "No result"
                            
                    except Exception as e:
                        self.logger.error(f"Error in shard worker execution: {e}")
                        yield f"Execution error: {e}"
                else:
                    yield "Shard worker executor not available"
                    
                self.logger.debug(f"[ACP] Executor yielded result for message {i+1}")
                
        except Exception as e:
            # Yield error as string
            error_msg = f"ACP processing error: {e}"
            yield error_msg
            self.logger.error(f"[ACP] Executor error: {e}", exc_info=True)


class ACPAgent:
    """
    Self-contained ACP Agent for Fail-Storm recovery scenarios.
    
    This agent implements ACP protocol capabilities using official SDK
    for server-side and HTTP client for communication.
    """
    
    def __init__(self, agent_id: str, host: str, port: int, executor: Any):
        """
        Initialize ACP agent.
        
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
        
        # ACP specific components
        self.acp_executor = None
        self.native_server = None
        self._clients = {}
        self._endpoints = {}
        
        # Server management
        self._startup_complete = asyncio.Event()
        self._shutdown_complete = asyncio.Event()
        
        # Compatibility with BaseAgent interface
        self.process = None  # For compatibility with kill operations
        
        self.logger = logging.getLogger(f"ACPAgent.{agent_id}")
    
    async def start(self) -> None:
        """Start the ACP agent with official SDK server."""
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
            # Close HTTP clients
            await self.close()
            
            # Shutdown server and ensure port release
            if self.native_server:
                self.native_server.shutdown()
                # Wait for proper shutdown
                await asyncio.sleep(2.0)
            
            self._shutdown_complete.set()
            self.logger.info(f"ACP Agent {self.agent_id} stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping ACP agent {self.agent_id}: {e}")
    
    async def _start_acp_server(self) -> None:
        """Start ACP SDK server."""
        try:
            # Create ACP executor wrapper
            self.acp_executor = ACPExecutorWrapper(self.executor)
            
            # Create and Run Server using 100% native ACP SDK
            self.native_server = ACPNativeServer(
                executor=self.acp_executor,
                host=self.host,
                port=self.port,
                agent_id=self.agent_id
            )
            
            self.native_server.serve()  # Starts the server in a background thread
            await asyncio.sleep(1)  # Give the server a moment to start
            
        except Exception as e:
            self.logger.error(f"Failed to start ACP server: {e}")
            raise
    
    async def register_endpoint(self, agent_id: str, base_url: str) -> None:
        """Register endpoint for another agent."""
        self._endpoints[agent_id] = base_url
        
        # Create HTTP client for this endpoint
        self._clients[agent_id] = httpx.AsyncClient(base_url=base_url, timeout=10.0)
        
        self.logger.info(f"Registered endpoint for {agent_id}: {base_url}")
    
    async def send_message(self, target_agent_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send message to another ACP agent using HTTP client.
        
        Args:
            target_agent_id: ID of target agent
            message: Message to send
            
        Returns:
            Response from target agent or None if failed
        """
        try:
            return await self.send(self.agent_id, target_agent_id, message)
        except Exception as e:
            self.logger.error(f"Failed to send message to {target_agent_id}: {e}")
            return None
    
    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """
        Send message using ACP SDK protocol format.
        """
        endpoint = self._endpoints.get(dst_id)
        if not endpoint:
            raise RuntimeError(f"unknown dst_id={dst_id}")

        # Create ACP-shaped payload (RunCreateRequest format)
        acp_payload = {
            "input": {
                "messages": [
                    {
                        "parts": [
                            {
                                "type": "text",
                                "text": payload.get("body", str(payload))
                            }
                        ]
                    }
                ]
            }
        }

        # Use protocol-native HTTP POST (official ACP protocol format)
        client = self._clients.get(dst_id)
        try:
            resp = await client.post("/runs", json=acp_payload)
            resp.raise_for_status()
            raw = resp.json()
            # Extract text from ACP response format
            text = ""
            if "output" in raw and "messages" in raw["output"]:
                for msg in raw["output"]["messages"]:
                    if "parts" in msg:
                        for part in msg["parts"]:
                            if part.get("type") == "text":
                                text += part.get("text", "")
            else:
                text = str(raw)
            return {"raw": raw, "text": text}
        except Exception as e:
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
    
    async def close(self) -> None:
        """Close all HTTP clients."""
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "host": self.host,
            "port": self.port,
            "acp_ready": self.acp_executor is not None,
            "startup_complete": self._startup_complete.is_set(),
            "endpoints": list(self._endpoints.keys()),
        }


async def create_acp_agent(agent_id: str, host: str, port: int, executor: Any) -> ACPAgent:
    """
    Factory function to create and start an ACP agent.
    
    Args:
        agent_id: Unique identifier for the agent
        host: Host address to bind to  
        port: Port number for HTTP server
        executor: Task executor for QA operations
        
    Returns:
        Started ACP agent instance
    """
    agent = ACPAgent(agent_id, host, port, executor)
    await agent.start()
    return agent