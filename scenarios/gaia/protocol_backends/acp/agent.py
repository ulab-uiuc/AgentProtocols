"""
ACP Protocol Agent Implementation for GAIA Framework.
Communication is managed by ACP Server/Client HTTP APIs (acp_sdk),
which are different from dummy/agora implementations.
Follow the acp_sdk server demo pattern for request handling.
"""

import asyncio
import os
import sys
import time
import re
import threading
import socket
from typing import Any, Dict, Optional, AsyncGenerator

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.agent import MeshAgent
from core.schema import Message, AgentState

# ACP SDK imports
try:
    from acp_sdk.server import Server, Context, RunYield, RunYieldResume
    from acp_sdk.models import Message as ACPMessage, MessagePart as ACPMessagePart
except ImportError as e:
    raise ImportError(f"ACP SDK components required but not available: {e}")


class ACPAgent(MeshAgent):
    """
    ACP Agent for GAIA. This agent exposes an ACP server endpoint that
    runs an agent's think/act loop per incoming Run, and streams yields
    like the demo (thought -> final message).

    - No local router queues like Dummy
    - No SDK receiver like Agora
    - Pure acp_sdk HTTP server for request handling
    """

    def __init__(self, node_id: int, name: str, tool: str, port: int,
                 config: Dict[str, Any], task_id: Optional[str] = None):
        super().__init__(node_id, name, tool, port, {**config, "protocol": "acp"}, task_id)
        self._connected = False
        self._endpoint: Optional[str] = None  # populated by network
        self._server: Optional[Any] = None
        self._server_task: Optional[asyncio.Task] = None  # deprecated: kept for backward compatibility
        self._server_started = asyncio.Event()
        self._server_thread: Optional[threading.Thread] = None
        # 保存 ACP 侧的 agent 名（用于 server 注册与 role 输出）
        self._acp_agent_name = name

        # Pretty init message (align with other backends style)
        print(f"[ACPAgent] Initialized agent {self.name} (ID {self.id}) on port {self.port}")

    async def connect(self):
        self._connected = True
        self._log("ACPAgent ready")

    async def disconnect(self):
        self._connected = False
        self._log("ACPAgent disconnected")

    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        # Network delivers messages via ACP client; agent doesn't send directly
        self._log(f"send_msg called (dst={dst}) - handled by ACPNetwork backend")

    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        # Requests are HTTP-driven via ACP Server; no inbox polling here
        if timeout:
            try:
                await asyncio.sleep(min(timeout, 0.01))
            except Exception:
                pass
        return None

    def get_connection_status(self) -> Dict[str, Any]:
        return {
            "agent_id": self.id,
            "agent_name": self.name,
            "protocol": "acp",
            "connected": self._connected,
            "endpoint": self._endpoint,
        }

    async def start(self):
        await self.connect()
        await super().start()

    async def stop(self):
        await self.disconnect()
        # try to stop background server
        try:
            self._stop_server()
        except Exception:
            pass
        await super().stop()

    # ==================== ACP Server integration ====================
    def _create_server(self) -> Optional[Any]:
        server = Server()

        # 允许 name 动态传入（若 SDK 不支持 name 形参则降级为默认 echo）
        def _sanitize_role(name: str) -> str:
            # 仅保留字母、数字、下划线、短横线
            return re.sub(r"[^a-zA-Z0-9_\-]", "_", name or "agent")

        decorator = getattr(server, "agent")
        try:
            agent_decorator = decorator(name=self._acp_agent_name)
            role_suffix = _sanitize_role(self._acp_agent_name)
        except TypeError:
            agent_decorator = decorator()
            role_suffix = "echo"

        @agent_decorator
        async def echo(input: list[ACPMessage], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:  # type: ignore
            """Minimal echo agent compatible with demo, but backed by GAIA logic."""
            # Reset per-run state
            self.messages.clear()
            self.state = AgentState.IDLE
            self.current_step = 0

            # Extract plain text from ACP messages and push into GAIA
            user_texts = []
            for m in input:
                # m.parts is a list of MessagePart
                for p in getattr(m, "parts", []) or []:
                    content = getattr(p, "content", None)
                    ctype = getattr(p, "content_type", "text/plain")
                    if content is None:
                        continue
                    if ctype == "text/plain":
                        user_texts.append(str(content))
                    else:
                        user_texts.append(str(content))

            request_text = "\n\n".join(user_texts) if user_texts else ""

            # Stream a thought (as demo)
            await asyncio.sleep(0.2)
            yield {"thought": "I should process the request"}

            # Feed into GAIA messages/memory
            if request_text:
                user_msg = Message.user_message(request_text)
                self.messages.append(user_msg)
                self.memory.add_message(user_msg)

            # Run a few think/act steps (bounded)
            steps = 0
            final_text: str = ""
            max_local_steps = min(self.max_steps, 8)
            while steps < max_local_steps and self.state != AgentState.FINISHED:
                _ = await self.step()
                steps += 1
                last = self._extract_final_result()
                if last and last.strip():
                    final_text = last.strip()
                    break

            if not final_text:
                final_text = "No result generated by ACP agent"

            await asyncio.sleep(0.2)
            yield ACPMessage(
                role=f"agent/{role_suffix}",
                parts=[ACPMessagePart(content=final_text, content_type="text/plain")],
            )

        return server

    def run_server_background(self) -> None:
        """Start ACP server in a background asyncio task and wait for port readiness."""
        # If already running, do nothing
        if self._server_thread and self._server_thread.is_alive():
            return

        self._server = self._create_server()
        if self._server is None:
            return

        host = "127.0.0.1"
        port = self.port

        def _thread_runner():
            try:
                if hasattr(self._server, "run"):
                    try:
                        self._server.run(host=host, port=port)  # type: ignore[arg-type]
                    except TypeError:
                        os.environ.setdefault("ACP_HOST", host)
                        os.environ.setdefault("ACP_PORT", str(port))
                        self._server.run()  # type: ignore[misc]
                else:
                    # Fallback
                    self._server.run()  # type: ignore[misc]
            except Exception as e:
                print(f"[ACPAgent] Server run error: {e}")

        # Spin up server thread
        t = threading.Thread(target=_thread_runner, name=f"ACPServer-{self.id}", daemon=True)
        t.start()
        self._server_thread = t

        # Wait for port readiness (up to 5 seconds)
        ready = self._wait_port_ready(host, port, timeout=5.0)
        if ready:
            self._endpoint = f"http://{host}:{port}"
            self._server_started.set()
            print(f"[ACPAgent] Server ready at {self._endpoint}")
        else:
            print(f"[ACPAgent] Warning: server port {host}:{port} not ready within timeout")

    def _wait_port_ready(self, host: str, port: int, timeout: float = 5.0) -> bool:
        """Poll the TCP port until it accepts connections or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=0.2):
                    return True
            except Exception:
                time.sleep(0.1)
        return False

    async def health_check(self, agent_id: str | None = None) -> bool:
        """Agent-level health check with real socket probing when possible."""
        try:
            if self._endpoint and self._endpoint.startswith("acp://"):
                return True
            # Prefer checking TCP connectivity
            host = "127.0.0.1"
            port = self.port
            try:
                with socket.create_connection((host, port), timeout=0.2):
                    return True
            except Exception:
                return self._server_thread is not None and self._server_thread.is_alive()
        except Exception:
            return False
