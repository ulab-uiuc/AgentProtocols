# script/streaming_queue/protocol_backend/acp/comm.py
from __future__ import annotations
from typing import Any, Dict, Optional
import httpx
from ...comm.base import BaseCommBackend
# Optional FastAPI imports - handle compatibility issues gracefully
try:
    from fastapi import FastAPI, Request
    from uvicorn import Config, Server
    FASTAPI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: FastAPI not available due to compatibility issue: {e}")
    FastAPI = None
    Request = None
    Config = None
    Server = None
    FASTAPI_AVAILABLE = False
import asyncio

class ACPCommBackend(BaseCommBackend):
    def __init__(self, **kwargs):
        self._endpoints: Dict[str, str] = {}  # agent_id -> endpoint uri
        self._clients: Dict[str, httpx.AsyncClient] = {}  # Optional per-agent clients
        self._servers: Dict[str, Any] = {}  # For spawned local servers

    async def register_endpoint(self, agent_id: str, address: str) -> None:
        self._endpoints[agent_id] = address
        self._clients[agent_id] = httpx.AsyncClient(base_url=address)

    async def connect(self, src_id: str, dst_id: str) -> None:
        pass  # ACP doesn't require explicit connect

    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        endpoint = self._endpoints.get(dst_id)
        if not endpoint:
            raise RuntimeError(f"unknown dst_id={dst_id}")

        acp_payload = {
            "input": {
                "content": [{"type": "text", "text": payload.get("text", str(payload))}]
            }
        }

        client = self._clients.get(dst_id)
        try:
            resp = await client.post("/runs", json=acp_payload)
            resp.raise_for_status()
            raw = resp.json()
            text = raw.get("output", {}).get("content", [{}])[0].get("text", "")
            return {"raw": raw, "text": text}
        except Exception as e:
            raise RuntimeError(f"ACP send failed: {e}")

    async def health_check(self, agent_id: str) -> bool:
        endpoint = self._endpoints.get(agent_id)
        if not endpoint:
            return False
        client = self._clients.get(agent_id)
        try:
            resp = await client.get("/agents")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        for srv in self._servers.values():
            if hasattr(srv, "shutdown"):
                try:
                    # Handle different uvicorn versions
                    if hasattr(srv, "should_exit"):
                        srv.should_exit = True
                    await srv.shutdown()
                except AttributeError:
                    # Fallback for different uvicorn versions
                    if hasattr(srv, "should_exit"):
                        srv.should_exit = True

    async def spawn_local_agent(self, agent_id: str, host: str, port: int, executor: Any) -> Any:
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI is not available due to compatibility issues. Cannot spawn local agent.")
        
        app = FastAPI()

        # ACP /agents endpoint (for discovery/health)
        @app.get("/agents")
        async def get_agents():
            return {
                "agents": [{
                    "id": agent_id,
                    "name": f"Agent {agent_id}",
                    "url": f"http://{host}:{port}/",
                    "protocolVersion": "1.0.0",
                    "capabilities": ["text_processing"]
                }]
            }

        # ACP /runs endpoint (for task execution)
        @app.post("/runs")
        async def run_task(request: Request):
            try:
                body = await request.json()
                result = await executor.execute(body.get("input", {}))

                # Extract text from executor result (which is in ACP format)
                text = ""
                if isinstance(result, dict):
                    content = result.get("content", [])
                    if content and isinstance(content[0], dict):
                        text = content[0].get("text", "")

                return {
                    "output": {
                        "content": [{"type": "text", "text": text}]
                    },
                    "status": "success"
                }
            except Exception as e:
                return {
                    "output": {"content": [{"type": "text", "text": f"Error: {str(e)}"}]},
                    "status": "error"
                }

        config = Config(app=app, host=host, port=port, log_level="error")
        srv = Server(config)
        self._servers[agent_id] = srv

        asyncio.create_task(srv.serve())
        await asyncio.sleep(1)  # Wait for startup

        base_url = f"http://{host}:{port}"
        return type("Handle", (), {"base_url": base_url})