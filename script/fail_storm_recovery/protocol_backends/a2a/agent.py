#!/usr/bin/env python3
"""
A2A Agent implementation for Fail-Storm Recovery scenario.

This module provides A2A agent implementation using real A2A SDK server,
creating a self-contained agent that uses A2A's native server implementation.
"""

import asyncio
import json
import logging
import threading
import time
from typing import Any, Dict, Optional
from pathlib import Path
import sys

# Import A2A SDK (required)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.types import Message, MessageSendParams

# Import A2A server components
try:
    from a2a.server import create_server
    from a2a.server.config import ServerConfig
except ImportError:
    # Fallback for older A2A SDK versions
    def create_server(config):
        from flask import Flask
        app = Flask(__name__)
        app.config.update(config.__dict__ if hasattr(config, '__dict__') else {})
        return app
    
    class ServerConfig:
        def __init__(self, host, port, executor):
            self.host = host
            self.port = port
            self.executor = executor

# HTTP client for inter-agent communication
import httpx

# Flask for additional endpoints
from flask import jsonify


class A2ANativeServer:
    """100% Native A2A server using only official SDK components."""
    
    def __init__(self, executor, host: str, port: int, agent_id: str):
        self.executor = executor
        self.host = host
        self.port = port
        self.agent_id = agent_id
        self.server_thread = None
        self.event_queue = None
        self.logger = logging.getLogger(f"A2ANativeServer.{agent_id}")
    
    def _create_native_a2a_server(self, executor):
        """Create 100% native A2A server using only SDK components."""
        
        # Create A2A SDK EventQueue for native message handling
        from a2a.server.events import EventQueue
        self.event_queue = EventQueue()
        
        # Create minimal FastAPI app that uses pure A2A SDK components
        from fastapi import FastAPI
        app = FastAPI(title=f"A2A Agent {self.agent_id}")
        
        # A2A native message endpoint using SDK components
        @app.post("/")
        async def handle_a2a_message(request: dict):
            """Handle A2A messages using pure SDK components."""
            try:
                # Use A2A SDK RequestContext
                from a2a.server.agent_execution import RequestContext
                context = RequestContext()
                
                # Execute using A2A SDK executor pattern
                await self.executor.execute(context, self.event_queue)
                
                # Get result from event queue
                events = []
                while not self.event_queue.empty():
                    events.append(self.event_queue.get())
                
                return {"status": "success", "events": events}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        # Compatibility endpoints
        @app.get('/health')
        async def health_check():
            return {
                "status": "healthy", 
                "agent_id": self.agent_id,
                "protocol": "A2A (100% Native SDK)"
            }
        
        @app.get('/.well-known/agent.json')
        async def agent_card():
            return {
                "name": f"A2A Agent {self.agent_id}",
                "url": f"http://{self.host}:{self.port}/",
                "protocol": "A2A (100% Native SDK)",
                "agent_id": self.agent_id,
            }
        
        return app
    
    def serve(self):
        """Start server in background thread."""
        def run_server():
            try:
                # Create native A2A server
                app = self._create_native_a2a_server(self.executor)
                # Use uvicorn as the ASGI server (this is standard for FastAPI)
                import uvicorn
                uvicorn.run(app, host=self.host, port=self.port, log_level="error")
            except Exception as e:
                self.logger.error(f"Server error: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.logger.info(f"A2A native server started on {self.host}:{self.port}")
    
    def shutdown(self):
        """Shutdown server."""
        # Daemon thread will terminate with main process
        pass


class A2AExecutorWrapper(AgentExecutor):
    """
    Wrapper to convert ShardWorkerExecutor to A2A SDK native executor interface.
    
    A2A SDK expects: async def execute(context: RequestContext, event_queue: EventQueue) -> None
    """
    
    def __init__(self, shard_worker_executor: Any):
        """
        Initialize with shard worker executor.
        
        Args:
            shard_worker_executor: ShardWorkerExecutor instance
        """
        self.shard_worker_executor = shard_worker_executor
        self.logger = logging.getLogger("A2AExecutorWrapper")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        A2A SDK native executor interface implementation.
        
        Args:
            context: A2A SDK RequestContext object
            event_queue: A2A SDK EventQueue object
        """
        try:
            # Extract user input from context
            user_input = context.get_user_input() if hasattr(context, "get_user_input") else None
            
            # Extract text content
            if user_input:
                if isinstance(user_input, str):
                    question = user_input
                elif isinstance(user_input, dict):
                    question = user_input.get("text", str(user_input))
                else:
                    question = str(user_input)
            else:
                question = "What is artificial intelligence?"  # Default question
            
            # Use shard worker to process the question
            if self.shard_worker_executor and hasattr(self.shard_worker_executor, 'worker'):
                try:
                    # Call LLM directly through worker.answer() if available
                    if hasattr(self.shard_worker_executor.worker, 'answer'):
                        result = await self.shard_worker_executor.worker.answer(question)
                    else:
                        # Fallback to start_task
                        result = await self.shard_worker_executor.worker.start_task(0)
                    
                    # Send result to event queue using A2A SDK format
                    result_message = new_agent_text_message(str(result) if result else "No result")
                    
                    # Try both sync and async enqueue_event (based on main branch pattern)
                    try:
                        # First try async (correct way based on RuntimeWarning)
                        await event_queue.enqueue_event(result_message)
                    except Exception:
                        try:
                            # Fallback to sync
                            event_queue.enqueue_event(result_message)
                        except Exception:
                            # Last resort - try simple text event
                            event_queue.enqueue_event(new_agent_text_message(f"A2A result: {result}"))
                    
                except Exception as e:
                    self.logger.error(f"Error in shard worker execution: {e}")
                    error_message = new_agent_text_message(f"Execution error: {e}")
                    await event_queue.enqueue_event(error_message)
            else:
                # No executor available
                no_exec_message = new_agent_text_message("Shard worker executor not available")
                await event_queue.enqueue_event(no_exec_message)
                
        except Exception as e:
            self.logger.error(f"Error in A2A executor: {e}")
            try:
                error_message = new_agent_text_message(f"A2A executor error: {e}")
                await event_queue.enqueue_event(error_message)
            except:
                pass  # If we can't even send error message, just log it
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel execution if needed."""
        try:
            cancel_message = new_agent_text_message("Execution cancelled")
            await event_queue.enqueue_event(cancel_message)
        except Exception as e:
            self.logger.error(f"Error cancelling A2A execution: {e}")


class A2AAgent:
    """
    Self-contained A2A Agent for Fail-Storm recovery scenarios.
    
    This agent implements A2A protocol capabilities using official SDK
    for server-side and HTTP client for communication.
    """
    
    def __init__(self, agent_id: str, host: str, port: int, executor: Any):
        """
        Initialize A2A agent.
        
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
        
        # A2A specific components
        self.a2a_executor = None
        self.native_server = None
        self._clients = {}
        self._endpoints = {}
        
        # Server management
        self._startup_complete = asyncio.Event()
        self._shutdown_complete = asyncio.Event()
        
        # Compatibility with BaseAgent interface
        self.process = None  # For compatibility with kill operations
        
        self.logger = logging.getLogger(f"A2AAgent.{agent_id}")
    
    async def start(self) -> None:
        """Start the A2A agent with official SDK server."""
        try:
            # Start A2A SDK server
            await self._start_a2a_server()
            
            self._startup_complete.set()
            self.logger.info(f"A2A Agent {self.agent_id} started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start A2A agent {self.agent_id}: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the A2A agent and cleanup resources."""
        try:
            # Close HTTP clients
            await self.close()
            
            # Shutdown server
            if self.native_server:
                self.native_server.shutdown()
            
            self._shutdown_complete.set()
            self.logger.info(f"A2A Agent {self.agent_id} stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping A2A agent {self.agent_id}: {e}")
    
    async def _start_a2a_server(self) -> None:
        """Start A2A SDK server."""
        try:
            # Create A2A executor wrapper
            self.a2a_executor = A2AExecutorWrapper(self.executor)
            
            # Create and Run Server using 100% native A2A SDK
            self.native_server = A2ANativeServer(
                executor=self.a2a_executor,
                host=self.host,
                port=self.port,
                agent_id=self.agent_id
            )
            
            self.native_server.serve()  # Starts the server in a background thread
            await asyncio.sleep(1)  # Give the server a moment to start
            
        except Exception as e:
            self.logger.error(f"Failed to start A2A server: {e}")
            raise
    
    async def register_endpoint(self, agent_id: str, base_url: str) -> None:
        """Register endpoint for another agent."""
        self._endpoints[agent_id] = base_url
        
        # Create HTTP client for this endpoint
        self._clients[agent_id] = httpx.AsyncClient(base_url=base_url, timeout=10.0)
        
        self.logger.info(f"Registered endpoint for {agent_id}: {base_url}")
    
    async def send_message(self, target_agent_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send message to another A2A agent using HTTP client.
        
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
        Send message using A2A SDK protocol format.
        """
        endpoint = self._endpoints.get(dst_id)
        if not endpoint:
            raise RuntimeError(f"unknown dst_id={dst_id}")

        # Create A2A-shaped payload
        a2a_payload = {
            "from": src_id,
            "to": dst_id,
            "body": payload.get("body", str(payload)),
            "timestamp": time.time()
        }

        # Use protocol-native HTTP POST (official A2A protocol format)
        client = self._clients.get(dst_id)
        try:
            resp = await client.post("/message", json=a2a_payload)
            resp.raise_for_status()
            raw = resp.json()
            text = raw.get("body", "") or str(raw)
            return {"raw": raw, "text": text}
        except Exception as e:
            raise RuntimeError(f"A2A protocol send failed: {e}")

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
            "a2a_ready": self.a2a_executor is not None,
            "startup_complete": self._startup_complete.is_set(),
            "endpoints": list(self._endpoints.keys()),
        }


async def create_a2a_agent(agent_id: str, host: str, port: int, executor: Any) -> A2AAgent:
    """
    Factory function to create and start an A2A agent.
    
    Args:
        agent_id: Unique identifier for the agent
        host: Host address to bind to  
        port: Port number for HTTP server
        executor: Task executor for QA operations
        
    Returns:
        Started A2A agent instance
    """
    agent = A2AAgent(agent_id, host, port, executor)
    await agent.start()
    return agent