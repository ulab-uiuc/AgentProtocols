#!/usr/bin/env python3
"""
A simple test A2A agent for Fail-Storm recovery testing.

This is a minimal implementation for demonstrating the A2A protocol integration.
Replace this with your actual A2A agent implementation.
"""

import asyncio
import json
import argparse
from pathlib import Path
from typing import Dict, Any
from aiohttp import web
import aiohttp_cors
import logging

# Reduce logging noise
logging.getLogger("aiohttp").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)


class TestA2AAgent:
    """A minimal test A2A agent implementation."""
    
    def __init__(self, agent_id: str, port: int, ws_port: int, workspace: str):
        self.agent_id = agent_id
        self.port = port
        self.ws_port = ws_port
        self.workspace = Path(workspace)
        self.peers = {}  # {peer_url: last_seen}
        self.mesh_data = {}
        
    async def start(self):
        """Start the HTTP server."""
        app = web.Application()
        
        # Add CORS
        cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Routes
        app.router.add_get('/healthz', self.health_check)
        app.router.add_post('/mesh/add_peer', self.add_peer)
        app.router.add_post('/mesh/broadcast', self.broadcast)
        app.router.add_post('/qa/submit', self.qa_submit)
        app.router.add_get('/mesh/peers', self.get_peers)
        
        # Apply CORS to all routes
        for route in list(app.router.routes()):
            cors.add(route)
        
        print(f"🚀 [Test A2A Agent] {self.agent_id} starting on port {self.port}")
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', self.port)
        await site.start()
        print(f"✅ [Test A2A Agent] {self.agent_id} listening on http://127.0.0.1:{self.port}")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print(f"🛑 [Test A2A Agent] {self.agent_id} shutting down...")
        finally:
            await runner.cleanup()
    
    async def health_check(self, request):
        """Health check endpoint."""
        return web.json_response({"status": "ok", "agent_id": self.agent_id})
    
    async def add_peer(self, request):
        """Add a peer to the mesh."""
        try:
            data = await request.json()
            peer_url = data.get("peer")
            if peer_url:
                self.peers[peer_url] = asyncio.get_event_loop().time()
                print(f"🔗 [Test A2A Agent] {self.agent_id} added peer: {peer_url}")
                return web.json_response({"status": "ok", "peer_added": peer_url})
            else:
                return web.json_response({"status": "error", "message": "Missing peer URL"}, status=400)
        except Exception as e:
            print(f"❌ [Test A2A Agent] {self.agent_id} add_peer error: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)
    
    async def broadcast(self, request):
        """Receive broadcast message."""
        try:
            data = await request.json()
            doc = data.get("doc", {})
            self.mesh_data["last_broadcast"] = doc
            print(f"📡 [Test A2A Agent] {self.agent_id} received broadcast: {doc.get('title', 'Unknown')}")
            return web.json_response({"status": "ok", "received": True})
        except Exception as e:
            print(f"❌ [Test A2A Agent] {self.agent_id} broadcast error: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)
    
    async def qa_submit(self, request):
        """Handle QA task submission."""
        try:
            data = await request.json()
            mode = data.get("mode", "unknown")
            window_s = data.get("window_s", 0)
            print(f"🔍 [Test A2A Agent] {self.agent_id} QA task: mode={mode}, window={window_s}s")
            
            # Simulate QA processing
            await asyncio.sleep(0.1)
            
            return web.json_response({
                "status": "ok", 
                "mode": mode,
                "agent_id": self.agent_id,
                "simulated": True
            })
        except Exception as e:
            print(f"❌ [Test A2A Agent] {self.agent_id} qa_submit error: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)
    
    async def get_peers(self, request):
        """Get current peer list."""
        return web.json_response({
            "status": "ok",
            "agent_id": self.agent_id,
            "peers": list(self.peers.keys()),
            "peer_count": len(self.peers)
        })


async def main():
    parser = argparse.ArgumentParser(description="Test A2A Agent")
    parser.add_argument("--port", type=int, required=True, help="HTTP port")
    parser.add_argument("--ws-port", type=int, required=True, help="WebSocket port")
    parser.add_argument("--id", required=True, help="Agent ID")
    parser.add_argument("--workspace", required=True, help="Workspace directory")
    parser.add_argument("--data", help="Data file (optional)")
    
    args = parser.parse_args()
    
    agent = TestA2AAgent(args.id, args.port, args.ws_port, args.workspace)
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())

