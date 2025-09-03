# -*- coding: utf-8 -*-
"""
Agora Communication Backend for Privacy Testing
Implements Agora protocol communication for the privacy testing framework.
"""

from __future__ import annotations

import asyncio
import json
import time
import threading
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

# Import base communication interface
try:
    from ...comm.base import BaseCommBackend
except ImportError:
    from comm.base import BaseCommBackend

# Agora Protocol imports
try:
    import agora
    from langchain_openai import ChatOpenAI
    AGORA_AVAILABLE = True
except ImportError:
    AGORA_AVAILABLE = False
    print("Warning: Agora Protocol SDK not available. Install with: pip install agora-protocol")


@dataclass
class AgoraAgentHandle:
    """Handle for locally spawned Agora agent."""
    agent_id: str
    host: str
    port: int
    base_url: str
    server_thread: Optional[threading.Thread] = None


class AgoraCommBackend(BaseCommBackend):
    """Agora protocol communication backend for privacy testing."""

    def __init__(self, **kwargs):
        self._endpoints: Dict[str, str] = {}  # agent_id -> endpoint uri
        self._client = httpx.AsyncClient(timeout=30.0)
        self._local_agents: Dict[str, AgoraAgentHandle] = {}  # For locally spawned agents

    async def register_endpoint(self, agent_id: str, address: str) -> None:
        """Register Agora agent endpoint."""
        self._endpoints[agent_id] = address
        print(f"[AgoraCommBackend] Registered {agent_id} @ {address}")

    async def connect(self, src_id: str, dst_id: str) -> None:
        """Agora doesn't require explicit connection setup."""
        return None

    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """Send message via Agora protocol."""
        endpoint = self._endpoints.get(dst_id)
        if not endpoint:
            raise RuntimeError(f"Unknown destination agent: {dst_id}")

        # Convert payload to Agora message format
        agora_message = self._to_agora_message(payload)
        
        try:
            # Send HTTP request to Agora agent endpoint
            response = await self._client.post(
                f"{endpoint}/",
                json=agora_message,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            raw_response = response.json()
            text_content = self._extract_text_from_agora_response(raw_response)
            
            return {
                "raw": raw_response,
                "text": text_content
            }
            
        except Exception as e:
            print(f"[AgoraCommBackend] Send failed {src_id} -> {dst_id}: {e}")
            return {"raw": None, "text": ""}

    async def health_check(self, agent_id: str) -> bool:
        """Check Agora agent health."""
        endpoint = self._endpoints.get(agent_id)
        if not endpoint:
            return False
            
        # For simulation mode with mock endpoints, consider agents always healthy
        if endpoint.startswith("agora://"):
            return True
            
        try:
            # For real HTTP endpoints, do actual health check
            response = await self._client.get(f"{endpoint}/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            print(f"[AgoraCommBackend] Health check failed for {agent_id}: {e}")
            return False

    async def close(self) -> None:
        """Close Agora communication backend."""
        await self._client.aclose()
        
        # Stop any locally spawned agents
        for handle in self._local_agents.values():
            if handle.server_thread and handle.server_thread.is_alive():
                print(f"Stopping Agora agent {handle.agent_id}...")
                # Note: Flask server doesn't have graceful shutdown in threading mode

    async def spawn_local_agent(self, agent_id: str, host: str, port: int, executor: Any) -> AgoraAgentHandle:
        """Spawn local Agora agent server using Agora Protocol SDK."""
        if not AGORA_AVAILABLE:
            raise RuntimeError("Agora Protocol SDK not available. Install with: pip install agora-protocol")
        
        # Ensure OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        base_url = f"http://{host}:{port}"
        
        # 1. Create Toolformer with ChatOpenAI
        try:
            model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
            toolformer = agora.toolformers.LangChainToolformer(model)
        except Exception as e:
            raise RuntimeError(f"Failed to create Agora Toolformer: {e}")

        # 2. Create tool function that wraps the executor
        loop = asyncio.get_running_loop()
        
        def privacy_agent_service(message: str, context: str = "") -> str:
            """
            隐私测试智能体服务工具
            这是Agora Protocol的标准工具接口
            """
            try:
                # Agora工具函数是同步的，但executor是异步的
                # 需要在主线程的事件循环中运行异步函数
                coro = executor.execute(message)
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                result = future.result(timeout=30.0)
                
                # 确保返回字符串
                if isinstance(result, str):
                    return result
                else:
                    return str(result)
                    
            except Exception as e:
                error_msg = f"Error in privacy_agent_service: {e}"
                print(f"[AgoraCommBackend] {error_msg}")
                return error_msg

        # 3. Create Receiver with the tool
        receiver = agora.Receiver.make_default(toolformer, tools=[privacy_agent_service])

        # 4. Create and start ReceiverServer
        server = agora.ReceiverServer(receiver)

        # 4.1 Inject health and agent card endpoints for compatibility
        try:
            from flask import jsonify  # ReceiverServer.app is a Flask app
            app = getattr(server, "app", None)
            if app is not None:
                @app.route('/health', methods=['GET'])
                def health_check():
                    """Health check endpoint for AgentNetwork compatibility."""
                    return jsonify({
                        "status": "healthy",
                        "agent_id": agent_id,
                        "timestamp": time.time()
                    }), 200

                @app.route('/.well-known/agent.json', methods=['GET'])
                def agent_card():
                    """Agent card endpoint for AgentNetwork compatibility."""
                    return jsonify({
                        "name": f"Agora Agent {agent_id}",
                        "url": f"http://{host}:{port}/",
                        "protocol": "Agora (Official)",
                        "agent_id": agent_id,
                    }), 200
            else:
                print("[AgoraCommBackend] ReceiverServer has no Flask app attribute; skipping extra endpoints")
        except Exception as e:
            print(f"[AgoraCommBackend] Failed to add health/agent endpoints: {e}")
        
        # Start server in background thread
        def run_server():
            try:
                print(f"Starting Agora Privacy Agent {agent_id} on {host}:{port}")
                server.run(host=host, port=port, debug=False)
            except Exception as e:
                print(f"Agora server error for {agent_id}: {e}")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        await asyncio.sleep(2)

        handle = AgoraAgentHandle(
            agent_id=agent_id,
            host=host,
            port=port,
            base_url=base_url,
            server_thread=server_thread
        )
        
        self._local_agents[agent_id] = handle
        
        return handle

    # ---------------------- Helper Methods ----------------------
    
    def _to_agora_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convert standard payload to Agora message format."""
        if "body" in payload:
            return {
                "protocolHash": payload.get("protocolHash"),
                "body": payload["body"],
                "protocolSources": payload.get("protocolSources", [])
            }
        elif "text" in payload:
            return {
                "protocolHash": None,
                "body": payload["text"],
                "protocolSources": []
            }
        else:
            return {
                "protocolHash": None,
                "body": str(payload),
                "protocolSources": []
            }

    def _extract_text_from_agora_response(self, response: Dict[str, Any]) -> str:
        """Extract text content from Agora response."""
        try:
            # Agora Protocol response typically contains the result in different places
            if isinstance(response, str):
                return response
            elif isinstance(response, dict):
                # Try different possible response formats
                if "body" in response:
                    return response["body"]
                elif "text" in response:
                    return response["text"]
                elif "result" in response:
                    result = response["result"]
                    if isinstance(result, str):
                        return result
                    elif isinstance(result, dict) and "body" in result:
                        return result["body"]
                elif "data" in response:
                    return str(response["data"])
                else:
                    # Fallback: convert entire response to string
                    return str(response)
            
            return ""
        except Exception:
            return ""