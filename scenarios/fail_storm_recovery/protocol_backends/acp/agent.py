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
from typing import Any, Dict, Optional, List, AsyncGenerator
from pathlib import Path
import sys

# Add agent_network src path
agent_network_src = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(agent_network_src))

# Import ACP SDK (required) - use agent_network environment
from acp_sdk.models import Message, MessagePart, RunCreateRequest
from acp_sdk.server import Context, RunYield
from acp_sdk.server import Server, AgentManifest

# HTTP client for inter-agent communication
import httpx

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

class ACPAgentManifest(AgentManifest):
    """ACP AgentManifest implementation following A2A patterns."""
    
    def __init__(self, executor_wrapper, agent_id: str):
        super().__init__()
        self.executor_wrapper = executor_wrapper
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"ACPAgentManifest.{agent_id}")
    
    async def run(self, messages: List[Message], context: Context) -> AsyncGenerator[RunYield, None]:
        """Handle incoming ACP messages using official SDK patterns."""
        try:
            # Extract text content from ACP messages
            content = ""
            for message in messages:
                for part in message.parts:
                    if part.type == "text":
                        content += part.text
            
            # Process through executor
            result = await self.executor_wrapper.process_message("acp_client", content, {})
            
            # Return ACP-formatted response
            response_msg = Message(
                role="assistant",
                parts=[MessagePart(type="text", text=str(result))]
            )
            
            yield RunYield(output={"messages": [response_msg]})
            
        except Exception as e:
            self.logger.error(f"Error in ACP manifest run: {e}")
            error_msg = Message(
                role="assistant", 
                parts=[MessagePart(type="text", text=f"Error: {str(e)}")]
            )
            yield RunYield(output={"messages": [error_msg]})

class ACPNativeServer:
    """ACP native server following A2A server patterns."""
    
    def __init__(self, executor, host: str, port: int, agent_id: str):
        self.executor = executor
        self.host = host
        self.port = port
        self.agent_id = agent_id
        self.server_thread = None
        self.acp_server = None
        self.agent_manifest = None
        self.logger = logging.getLogger(f"ACPNativeServer.{agent_id}")
        self._started = False  # Prevent duplicate starts
    
    def _create_native_acp_server(self, executor):
        """Create native ACP server using official SDK."""
        acp_server = Server()

        # Create native AgentManifest with our executor
        self.agent_manifest = ACPAgentManifest(executor, self.agent_id)

        # Register the AgentManifest with ACP SDK
        acp_server.register(self.agent_manifest)
        self.logger.info("Registered native AgentManifest with ACP SDK Server")

        # Add /health endpoint for debugging
        try:
            app = acp_server.app  # FastAPI app
            @app.get("/health")
            async def _health():
                return {"status": "ok", "agent_id": self.agent_id}
        except Exception:
            # Older SDKs may not expose .app; ignore silently
            pass

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
        
        # HTTP client management (following A2A pattern)
        self._endpoints: Dict[str, str] = {}
        self._clients: Dict[str, httpx.AsyncClient] = {}
        
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
            
            # Close HTTP clients
            for client in self._clients.values():
                await client.aclose()
            self._clients.clear()
            
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
        """Register endpoint for another agent (following A2A pattern)."""
        self._endpoints[agent_id] = base_url
        
        # Create HTTP client for this endpoint
        self._clients[agent_id] = httpx.AsyncClient(base_url=base_url, timeout=10.0)
        
        self.logger.info(f"Registered endpoint for {agent_id}: {base_url}")
    
    async def send_message(self, target_agent_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send message to another ACP agent (following A2A pattern)."""
        try:
            return await self.send(self.agent_id, target_agent_id, message)
        except Exception as e:
            self.logger.error(f"Failed to send message to {target_agent_id}: {e}")
            return None
    
    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """Send message using ACP SDK-native RunCreateRequest schema."""
        endpoint = self._endpoints.get(dst_id)
        if not endpoint:
            raise RuntimeError(f"unknown dst_id={dst_id}")

        client = self._clients.get(dst_id)
        if client is None:
            raise RuntimeError(f"http client for {dst_id} not initialized")

        # Build text content
        text_in = (
            payload.get("body")
            or payload.get("content")
            or (json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload))
        )

        # Use ACP SDK models to build request body
        try:
            # Message MUST have a role
            msg = Message(
                role="user",
                parts=[MessagePart(type="text", text=str(text_in))]
            )
            run = RunCreateRequest(
                messages=[msg],
                input={"meta": {"src": src_id, "dst": dst_id}}
            )

            # Pydantic v2 vs v1 compatibility
            try:
                body = run.model_dump(by_alias=True, exclude_none=True)  # pydantic v2
            except AttributeError:
                body = run.dict(by_alias=True, exclude_none=True)        # pydantic v1

            # Send to ACP server
            resp = await client.post("/runs", json=body)

            # For older/newer variants that mount at /v1/runs
            if resp.status_code == 404:
                resp = await client.post("/v1/runs", json=body)

            resp.raise_for_status()
            raw = resp.json()

            # Extract text from ACP response
            text_out = ""
            if isinstance(raw, dict) and "output" in raw and "messages" in raw["output"]:
                for m in raw["output"]["messages"]:
                    for p in m.get("parts", []):
                        if p.get("type") == "text":
                            text_out += p.get("text", "")

            if not text_out:
                text_out = json.dumps(raw, ensure_ascii=False)

            return {"raw": raw, "text": text_out}

        except httpx.HTTPStatusError as e:
            # Surface FastAPI/Pydantic validation details to logs
            details = e.response.text if e.response is not None else ""
            raise RuntimeError(f"ACP protocol send failed: {e} | details={details}")
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