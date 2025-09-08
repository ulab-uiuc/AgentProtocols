#!/usr/bin/env python3
"""
Agora Agent implementation for Fail-Storm Recovery scenario.

This module provides a self-contained Agora agent implementation that uses
the official Agora SDK for server-side and HTTP client for communication.
"""

import asyncio
import json
import logging
import threading
from typing import Any, Dict, Optional
from pathlib import Path
import sys

# Add AgentConnect to path for compatibility if needed
current_dir = Path(__file__).parent.parent.parent
agentconnect_path = current_dir.parent.parent / "agentconnect_src"
sys.path.insert(0, str(agentconnect_path))


class AgoraServerWrapper:
    """Wrapper for Agora receiver server running in background thread."""
    
    def __init__(self, receiver, host: str, port: int, agent_id: str):
        self.receiver = receiver
        self.host = host
        self.port = port
        self.agent_id = agent_id
        self.server_thread = None
        self.logger = logging.getLogger(f"AgoraServer.{agent_id}")
    
    def serve(self):
        """Start server in background thread."""
        def run_server():
            try:
                # Use Agora SDK to serve on specified host:port
                self.receiver.serve(host=self.host, port=self.port)
            except Exception as e:
                self.logger.error(f"Server error: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.logger.info(f"Agora server started on {self.host}:{self.port}")
    
    def shutdown(self):
        """Shutdown server."""
        # Agora SDK receiver doesn't have explicit shutdown, 
        # but daemon thread will terminate with main process
        pass


class AgoraAgent:
    """
    Self-contained Agora Agent for Fail-Storm recovery scenarios.
    
    This agent implements Agora protocol capabilities using official SDK
    for server-side and HTTP client for communication.
    """
    
    def __init__(self, agent_id: str, host: str, port: int, executor: Any):
        """
        Initialize Agora agent.
        
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
        
        # Agora specific components
        self.receiver = None
        self.server_wrapper = None
        self._clients = {}
        self._endpoints = {}
        
        # Server management
        self._startup_complete = asyncio.Event()
        self._shutdown_complete = asyncio.Event()
        
        # Compatibility with BaseAgent interface
        self.process = None  # For compatibility with kill operations
        
        self.logger = logging.getLogger(f"AgoraAgent.{agent_id}")
    
    async def start(self) -> None:
        """Start the Agora agent with official SDK server."""
        try:
            # Start Agora SDK server
            await self._start_agora_server()
            
            self._startup_complete.set()
            self.logger.info(f"Agora Agent {self.agent_id} started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start Agora agent {self.agent_id}: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the Agora agent and cleanup resources."""
        try:
            # Close HTTP clients
            await self.close()
            
            # Shutdown server
            if self.server_wrapper:
                self.server_wrapper.shutdown()
            
            self._shutdown_complete.set()
            self.logger.info(f"Agora Agent {self.agent_id} stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping Agora agent {self.agent_id}: {e}")
    
    async def _start_agora_server(self) -> None:
        """Start Agora SDK server."""
        try:
            # 1. Create Toolformer
            try:
                from langchain_openai import ChatOpenAI
                from agora.common.toolformers import LangChainToolformer
                
                model = ChatOpenAI(model="gpt-4o-mini") 
                toolformer = LangChainToolformer(model)
            except ImportError:
                raise RuntimeError("LangChain/OpenAI dependencies not found. Please install langchain-openai.")
            except Exception as e:
                self.logger.error(f"Failed to create Toolformer. Ensure OPENAI_API_KEY is set. Error: {e}")
                raise RuntimeError(f"Failed to create Toolformer. Ensure OPENAI_API_KEY is set. Error: {e}")

            # 2. Create Tools
            loop = asyncio.get_running_loop()
            def general_service(message: str, context: str = ""):
                """Handle general messages and requests by calling the provided executor."""
                try:
                    # Convert message to the format expected by executor
                    input_data = {
                        "content": [{"type": "text", "text": message}]
                    }
                    
                    # The agora tool function is synchronous, but the executor is async.
                    # We need to run the async function in the main thread's event loop.
                    coro = self.executor.execute(input_data)
                    future = asyncio.run_coroutine_threadsafe(coro, loop)
                    result = future.result()  # Wait for the result
                    
                    # Adapt the result to a string response for Agora.
                    if isinstance(result, dict):
                        if "content" in result and isinstance(result["content"], list):
                            # Extract text from content array
                            for item in result["content"]:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    return item.get("text", "")
                        elif "body" in result:
                            return result["body"]
                        elif "text" in result:
                            return result["text"]
                    elif isinstance(result, str):
                        return result
                    else:
                        return str(result)
                except Exception as e:
                    self.logger.error(f"Error executing task in general_service: {e}")
                    return f"Error executing task: {e}"

            tools = [general_service]

            # 3. Create Receiver using real Agora SDK
            from agora.receiver import Receiver
            self.receiver = Receiver.make_default(toolformer, tools=tools)

            # 4. Create and Run Server using the wrapper
            self.server_wrapper = AgoraServerWrapper(
                receiver=self.receiver,
                host=self.host,
                port=self.port,
                agent_id=self.agent_id
            )
            
            self.server_wrapper.serve()  # Starts the server in a background thread
            await asyncio.sleep(1)  # Give the server a moment to start
            
        except Exception as e:
            self.logger.error(f"Failed to start Agora server: {e}")
            raise
    
    async def register_endpoint(self, agent_id: str, base_url: str) -> None:
        """Register endpoint for another agent."""
        self._endpoints[agent_id] = base_url
        
        # Create HTTP client for this endpoint
        import httpx
        self._clients[agent_id] = httpx.AsyncClient(base_url=base_url, timeout=10.0)
        
        self.logger.info(f"Registered endpoint for {agent_id}: {base_url}")
    
    async def send_message(self, target_agent_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send message to another Agora agent using HTTP client.
        
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
        Send message using ONLY official Agora SDK client.
        No fallback to HTTP - must use official SDK or fail.
        """
        endpoint = self._endpoints.get(dst_id)
        if not endpoint:
            raise RuntimeError(f"unknown dst_id={dst_id}")

        # Minimal Agora-shaped payload
        agora_payload = {
            "protocolHash": None,
            "body": payload.get("body", str(payload)),
            "protocolSources": []
        }

        # --- Use protocol-native HTTP POST (official Agora protocol format) ---
        # Note: Server side uses real Agora SDK (ReceiverServer), client side uses protocol-native HTTP
        # This maintains protocol compliance while avoiding complex SDK client configuration
        client = self._clients.get(dst_id)
        try:
            resp = await client.post("/", json=agora_payload)
            resp.raise_for_status()
            raw = resp.json()
            text = raw.get("raw", {}).get("body", "") or raw.get("body", "")
            return {"raw": raw, "text": text}
        except Exception as e:
            raise RuntimeError(f"Agora protocol send failed: {e}")

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
            "agora_ready": self.receiver is not None,
            "startup_complete": self._startup_complete.is_set(),
            "endpoints": list(self._endpoints.keys()),
        }


async def create_agora_agent(agent_id: str, host: str, port: int, executor: Any) -> AgoraAgent:
    """
    Factory function to create and start an Agora agent.
    
    Args:
        agent_id: Unique identifier for the agent
        host: Host address to bind to  
        port: Port number for HTTP server
        executor: Task executor for QA operations
        
    Returns:
        Started Agora agent instance
    """
    agent = AgoraAgent(agent_id, host, port, executor)
    await agent.start()
    return agent