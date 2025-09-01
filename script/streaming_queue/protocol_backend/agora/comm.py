# script/streaming_queue/protocol_backend/Agora/comm.py
"""
Agora (Agent Network Protocol) Communication Backend
"""
import asyncio
import json
import yaml
import time
import sys
import threading
from pathlib import Path
from typing import Dict, List, Any

import httpx
import agora
from flask import jsonify

from ...comm.base import BaseCommBackend


class AgoraServerWrapper:
    """
    Wrapper to make official Agora ReceiverServer compatible with the Agent communication backend.
    Bridges the gap between the asyncio-based framework and Agora's thread-based server.
    """
    
    def __init__(self, receiver, host: str, port: int, agent_id: str):
        self.receiver = receiver
        self.host = host
        self.port = port
        self.agent_id = agent_id
        self.server_thread = None
        
        # Create official Agora server with additional endpoints
        self.agora_server = self._create_enhanced_agora_server(receiver)
    
    def _create_enhanced_agora_server(self, receiver):
        """Create Agora ReceiverServer with additional health and agent card endpoints."""
        
        original_server = agora.ReceiverServer(receiver)
        
        @original_server.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint for AgentNetwork compatibility."""
            return jsonify({
                "status": "healthy",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }), 200
        
        @original_server.app.route('/.well-known/agent.json', methods=['GET'])
        def agent_card():
            """Agent card endpoint for AgentNetwork compatibility."""
            return jsonify({
                "name": f"Agora Agent {self.agent_id}",
                "url": f"http://{self.host}:{self.port}/",
                "protocol": "Agora (Official)",
                "agent_id": self.agent_id,
            }), 200
        
        return original_server
    
    def serve(self):
        """Start the official Agora server in a background thread."""
        self.server_thread = threading.Thread(
            target=self._run_agora_server,
            daemon=True
        )
        self.server_thread.start()
    
    def _run_agora_server(self):
        """Run official Agora server in background thread."""
        try:
            print(f"📡 Agora ReceiverServer starting on {self.host}:{self.port}")
            self.agora_server.run(host=self.host, port=self.port, debug=False)
        except Exception as e:
            print(f"❌ Agora ReceiverServer error: {e}")
    
    def shutdown(self):
        """
        Shutdown the server.
        Note: Flask's built-in server doesn't have a graceful public shutdown method.
        Using a daemon thread means it will be forcefully terminated when the main process exits.
        For production, a proper WSGI server like Gunicorn should be used.
        """
        print(f"🛑 Shutdown signal received for Agora Server {self.agent_id}. The daemon thread will exit with the main process.")


class AgoraCommBackend(BaseCommBackend):
    def __init__(self, **kwargs):
        self._endpoints: Dict[str, str] = {}  # agent_id -> endpoint uri
        self._clients: Dict[str, httpx.AsyncClient] = {}  # HTTP clients for each agent
        self._servers: Dict[str, Any] = {}  # For spawned local servers
    
    async def register_endpoint(self, agent_id: str, address: str) -> None:
        """注册Agora agent的endpoint"""
        self._endpoints[agent_id] = address
        self._clients[agent_id] = httpx.AsyncClient(base_url=address, timeout=30.0)

    async def connect(self, src_id: str, dst_id: str) -> None:
        """Agora协议不需要显式连接建立"""
        pass

    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """
        Send message using official Agora protocol.
        """
        endpoint = self._endpoints.get(dst_id)
        if not endpoint:
            raise RuntimeError(f"unknown dst_id={dst_id}")

        # This is a simplified payload structure for Agora.
        agora_payload = {
            "protocolHash": None,
            "body": payload.get("body", str(payload)),
            "protocolSources": []
        }

        client = self._clients.get(dst_id)
        try:
            resp = await client.post("/", json=agora_payload)
            resp.raise_for_status()
            raw = resp.json()
            # The response from an agora server is typically in the 'body' of the 'output'
            text = raw.get("raw", {}).get("body", "")

            return {"raw": raw, "text": text}
        except Exception as e:
            raise RuntimeError(f"Agora send failed: {e}")

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
        for client in self._clients.values():
            await client.aclose()
        for srv in self._servers.values():
            if hasattr(srv, "shutdown"):
                srv.shutdown()

    async def spawn_local_agent(self, agent_id: str, host: str, port: int, executor: Any) -> Any:
        """启动本地Agora Agent的HTTP服务"""
        
        # 1. Create Toolformer
        try:
            from langchain_openai import ChatOpenAI
            import agora
            # This requires OPENAI_API_KEY to be set in the environment.
            model = ChatOpenAI(model="gpt-4o-mini") 
            toolformer = agora.toolformers.LangChainToolformer(model)
        except ImportError:
            raise RuntimeError("LangChain/OpenAI dependencies not found. Please install langchain-openai.")
        except Exception as e:
            print(f"[-] Failed to create Toolformer. Ensure OPENAI_API_KEY is set. Error: {e}")
            raise RuntimeError(f"Failed to create Toolformer. Ensure OPENAI_API_KEY is set. Error: {e}")

        # 2. Create Tools
        loop = asyncio.get_running_loop()
        def general_service(message: str, context: str = ""):
            """Handle general messages and requests by calling the provided executor."""
            try:
                # The agora tool function is synchronous, but the executor is async.
                # We need to run the async function in the main thread's event loop.
                coro = executor.execute(message)
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                result = future.result()  # Wait for the result
                
                # Adapt the result to a string response for Agora.
                if isinstance(result, dict) and "body" in result:
                    return result["body"]
                elif isinstance(result, str):
                    return result
                else:
                    return str(result)
            except Exception as e:
                print(f"Error executing task in general_service: {e}")
                return f"Error executing task: {e}"

        tools = [general_service]

        # 3. Create Receiver
        receiver = agora.Receiver.make_default(toolformer, tools=tools)

        # 4. Create and Run Server using the wrapper
        server_wrapper = AgoraServerWrapper(
            receiver=receiver,
            host=host,
            port=port,
            agent_id=agent_id
        )
        
        server_wrapper.serve()  # Starts the server in a background thread
        
        self._servers[agent_id] = server_wrapper
        await asyncio.sleep(1)  # Give the server a moment to start

        base_url = f"http://{host}:{port}"
        await self.register_endpoint(agent_id, base_url)
        
        return type("AgentHandle", (object,), {"base_url": base_url, "server": server_wrapper})
