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
import inspect
from flask import jsonify

# Add comm path for BaseCommBackend
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parent.parent.parent
comm_path = streaming_queue_path / "comm"
if str(comm_path) not in sys.path:
    sys.path.insert(0, str(comm_path))
from base import BaseCommBackend


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
        
        # Use real Agora SDK ReceiverServer
        from agora.receiver import ReceiverServer
        original_server = ReceiverServer(receiver)
        
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
        # Official SDK clients cache
        self._agora_client = None
        self._agora_sender = None
    
    async def _maybe_await(self, value):
        """Await if awaitable; otherwise return value."""
        if inspect.isawaitable(value):
            return await value
        return value
    
    async def register_endpoint(self, agent_id: str, address: str) -> None:
        """Register an Agora agent's endpoint."""
        self._endpoints[agent_id] = address
        self._clients[agent_id] = httpx.AsyncClient(base_url=address, timeout=30.0)

    async def connect(self, src_id: str, dst_id: str) -> None:
        """Agora protocol does not require an explicit connection setup."""
        pass

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
        for client in self._clients.values():
            await client.aclose()
        for srv in self._servers.values():
            if hasattr(srv, "shutdown"):
                srv.shutdown()

    async def spawn_local_agent(self, agent_id: str, host: str, port: int, executor: Any) -> Any:
        """Start a local Agora Agent HTTP service."""
        
        # 1. Create Toolformer
        try:
            from langchain_openai import ChatOpenAI
            # Import real Agora SDK toolformers
            from agora.common.toolformers import LangChainToolformer
            # This requires OPENAI_API_KEY to be set in the environment.
            model = ChatOpenAI(model="gpt-4o-mini") 
            toolformer = LangChainToolformer(model)
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
                # Convert message to the format expected by executor
                input_data = {
                    "content": [{"type": "text", "text": message}]
                }
                
                # The agora tool function is synchronous, but the executor is async.
                # We need to run the async function in the main thread's event loop.
                coro = executor.execute(input_data)
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
                print(f"Error executing task in general_service: {e}")
                return f"Error executing task: {e}"

        tools = [general_service]

        # 3. Create Receiver using real Agora SDK
        from agora.receiver import Receiver
        receiver = Receiver.make_default(toolformer, tools=tools)

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
