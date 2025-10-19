"""
Agora Protocol Agent Implementation for GAIA Framework.
Integrates with Agora Protocol using agora SDK for multi-agent communication.
"""

import asyncio
import time
import os
import threading
from typing import Dict, Any, Optional
import sys

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.agent import MeshAgent
from core.schema import AgentState, Message

# Agora Protocol imports
try:
    import agora
    from langchain_openai import ChatOpenAI
except ImportError as e:
    raise ImportError(f"Agora SDK components required but not available: {e}")


class AgoraAgent(MeshAgent):
    """Agora Protocol Agent that integrates with Agora SDK."""
    
    def __init__(self, node_id: int, name: str, tool: str, port: int, 
                 config: Dict[str, Any], task_id: Optional[str] = None):
        super().__init__(node_id, name, tool, port, {**config, "protocol": "agora"}, task_id)
        
        self._connected = False
        self._endpoint: Optional[str] = None
        self._server_thread: Optional[threading.Thread] = None
        
        # Model configuration
        self._openai_model = config.get("openai_model", "gpt-4o-mini")
        self._openai_temperature = config.get("openai_temperature", 0.1)
        
        print(f"[AgoraAgent] Initialized agent {self.name} (ID {self.id}) on port {self.port}")

    async def connect(self):
        """Start agent and Agora server."""
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        self._start_agora_server()
        self._connected = True
        self._log("AgoraAgent connected and server started")

    async def disconnect(self):
        """Stop agent and server."""
        self._connected = False
        self._log("AgoraAgent disconnected")

    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """Message sending handled by network backend."""
        self._log(f"send_msg called (dst={dst}) - handled by network backend")

    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """Message receiving via Agora server callbacks."""
        if timeout:
            await asyncio.sleep(min(timeout, 0.01))
        return None

    def get_connection_status(self) -> Dict[str, Any]:
        """Get agent connection status."""
        return {
            "agent_id": self.id,
            "agent_name": self.name,
            "protocol": "agora",
            "connected": self._connected,
            "endpoint": self._endpoint,
            "server_running": self._server_thread.is_alive() if self._server_thread else False
        }

    async def start(self):
        """Start agent."""
        await self.connect()
        await super().start()

    async def stop(self):
        """Stop agent."""
        await self.disconnect()
        await super().stop()

    async def execute(self, message: str) -> str:
        """Execute agent processing for incoming requests."""
        try:
            self.messages.clear()
            self.state = AgentState.IDLE
            self.current_step = 0

            user_msg = Message.user_message(message)
            self.messages.append(user_msg)
            self.memory.add_message(user_msg)

            # Run agent steps
            while self.current_step < self.max_steps and self.state != AgentState.FINISHED:
                await self.step()
                self.current_step += 1
                result = self._extract_final_result()
                if result and result.strip():
                    return result

            final_result = self._extract_final_result()
            return final_result or "No result generated"
            
        except Exception as e:
            return f"Error executing request: {e}"

    def _start_agora_server(self) -> None:
        """Start Agora ReceiverServer in background thread."""
        if self._server_thread and self._server_thread.is_alive():
            return

        # Get current event loop for async callbacks
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        def agora_tool(message: str, context: str = "") -> str:
            """Agora tool that calls agent's execute method."""
            try:
                future = asyncio.run_coroutine_threadsafe(self.execute(message), loop)
                result = future.result(timeout=30.0)
                return result if isinstance(result, str) else str(result)
            except Exception as e:
                error_msg = f"Error in agora tool: {e}"
                print(f"[AgoraAgent] {error_msg}")
                return error_msg

        # Create Agora components
        model = ChatOpenAI(model=self._openai_model, temperature=self._openai_temperature)
        toolformer = agora.toolformers.LangChainToolformer(model)
        receiver = agora.Receiver.make_default(toolformer, tools=[agora_tool])
        server = agora.ReceiverServer(receiver)

        # Add health endpoints
        self._add_health_endpoints(server)

        def run_server():
            try:
                print(f"[AgoraAgent] Starting Agora server for {self.name} on port {self.port}")
                server.run(host="localhost", port=self.port, debug=False)
            except Exception as e:
                print(f"[AgoraAgent] Server error: {e}")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        print(f"[AgoraAgent] Server started for {self.name} at http://localhost:{self.port}")

    def _add_health_endpoints(self, server) -> None:
        """Add health check endpoints to Agora server."""
        try:
            from flask import jsonify
            import logging
            app = getattr(server, "app", None)
            if app:
                # 禁用Flask/Werkzeug的访问日志来避免健康检查刷屏
                werkzeug_logger = logging.getLogger('werkzeug')
                werkzeug_logger.setLevel(logging.ERROR)
                
                @app.route('/health', methods=['GET'])
                def health_check():
                    return jsonify({
                        "status": "healthy",
                        "agent_id": str(self.id),
                        "timestamp": time.time()
                    }), 200

                @app.route('/.well-known/agent.json', methods=['GET'])
                def agent_card():
                    return jsonify({
                        "name": f"GAIA Agora Agent {self.id}",
                        "url": f"http://localhost:{self.port}/",
                        "protocol": "Agora",
                        "agent_id": str(self.id)
                    }), 200
        except Exception as e:
            print(f"[AgoraAgent] Failed to add health endpoints: {e}")

    async def health_check(self, agent_id: Optional[str] = None) -> bool:
        """Check agent health by testing server connection."""
        try:
            if not self._connected or not self._server_thread:
                return False
            
            if not self._server_thread.is_alive():
                return False
                
            # If endpoint is available, try HTTP health check
            if self._endpoint:
                import httpx
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self._endpoint}/health")
                    return response.status_code == 200
                    
            return True
        except Exception:
            return False
