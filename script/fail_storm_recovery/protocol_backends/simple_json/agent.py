#!/usr/bin/env python3
"""
Simple JSON Agent implementation for Fail-Storm Recovery scenario.

This module provides a self-contained Simple JSON agent implementation that doesn't
depend on external src/ components, enabling pure protocol-native communication.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path
import sys

# Simple HTTP server imports
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route


class SimpleJSONAgent:
    """
    Self-contained Simple JSON Agent for Fail-Storm recovery scenarios.
    
    This agent implements simple HTTP JSON communication without depending
    on external BaseAgent or src components.
    """
    
    def __init__(self, agent_id: str, host: str, port: int, executor: Any):
        """
        Initialize Simple JSON agent.
        
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
        
        # Simple JSON specific components
        self.http_server = None
        
        # Server management
        self._startup_complete = asyncio.Event()
        self._shutdown_complete = asyncio.Event()
        
        # Compatibility with BaseAgent interface
        self.process = None  # For compatibility with kill operations
        
        self.logger = logging.getLogger(f"SimpleJSONAgent.{agent_id}")
    
    async def start(self) -> None:
        """Start the Simple JSON agent with HTTP server."""
        try:
            # Start HTTP server
            await self._start_http_server()
            
            self._startup_complete.set()
            self.logger.info(f"Simple JSON Agent {self.agent_id} started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start Simple JSON agent {self.agent_id}: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the Simple JSON agent and cleanup resources."""
        try:
            if self.http_server:
                self.http_server.should_exit = True
                await self.http_server.shutdown()
            
            self._shutdown_complete.set()
            self.logger.info(f"Simple JSON Agent {self.agent_id} stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping Simple JSON agent {self.agent_id}: {e}")
    
    async def _start_http_server(self) -> None:
        """Start HTTP server for JSON API."""
        async def health_check(request):
            return JSONResponse({"status": "healthy", "agent_id": self.agent_id})
        
        async def get_info(request):
            return JSONResponse({
                "agent_id": self.agent_id,
                "protocol": "simple_json",
                "host": self.host,
                "port": self.port
            })
        
        async def handle_message(request):
            """Handle incoming JSON messages."""
            try:
                body = await request.json()
                # Process message through executor if needed
                result = {
                    "status": "received", 
                    "agent_id": self.agent_id,
                    "timestamp": asyncio.get_event_loop().time()
                }
                return JSONResponse(result)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=400)
        
        async def handle_search(request):
            """Handle search requests."""
            try:
                body = await request.json()
                query = body.get("query", "")
                
                # Simple mock search response
                result = {
                    "agent_id": self.agent_id,
                    "query": query,
                    "results": [
                        {"id": 1, "content": f"Mock result 1 for: {query}"},
                        {"id": 2, "content": f"Mock result 2 for: {query}"}
                    ],
                    "timestamp": asyncio.get_event_loop().time()
                }
                return JSONResponse(result)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=400)
        
        routes = [
            Route("/health", health_check, methods=["GET"]),
            Route("/info", get_info, methods=["GET"]),
            Route("/message", handle_message, methods=["POST"]),
            Route("/search", handle_search, methods=["POST"]),
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
    
    async def send_message(self, target_agent_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send message to another Simple JSON agent.
        
        Args:
            target_agent_id: ID of target agent
            message: Message to send
            
        Returns:
            Response from target agent or None if failed
        """
        try:
            return await self._send_http_message(target_agent_id, message)
                
        except Exception as e:
            self.logger.error(f"Failed to send message to {target_agent_id}: {e}")
            return None
    
    async def _send_http_message(self, target_agent_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send HTTP JSON message to target agent."""
        import httpx
        
        try:
            # Simple port calculation (assumes consecutive ports)
            if "0" in target_agent_id:
                target_port = 9000
            elif "1" in target_agent_id:
                target_port = 9001
            elif "2" in target_agent_id:
                target_port = 9002
            else:
                # Fallback to port + 1
                target_port = self.port + 1
                
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
            "protocol": "simple_json",
            "host": self.host,
            "port": self.port,
            "startup_complete": self._startup_complete.is_set(),
        }


async def create_simple_json_agent(agent_id: str, host: str, port: int, executor: Any) -> SimpleJSONAgent:
    """
    Factory function to create and start a Simple JSON agent.
    
    Args:
        agent_id: Unique identifier for the agent
        host: Host address to bind to  
        port: Port number for HTTP server
        executor: Task executor for QA operations
        
    Returns:
        Started Simple JSON agent instance
    """
    agent = SimpleJSONAgent(agent_id, host, port, executor)
    await agent.start()
    return agent
