# script/streaming_queue/protocol_backend/anp/comm.py
"""
ANP (Agent Network Protocol) Communication Backend

ANP协议特点：
- 简洁的JSON消息格式
- 支持同步和异步通信
- 内置负载均衡和故障恢复
- 轻量级网络开销

消息格式：
{
    "protocol": "ANP",
    "version": "1.0",
    "message_id": "unique_id",
    "sender": "agent_id", 
    "receiver": "agent_id",
    "message_type": "request|response|broadcast",
    "payload": {
        "action": "execute|status|health_check",
        "data": "actual_content"
    },
    "timestamp": 1234567890.123
}
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import httpx
import time
import uuid
try:
    from ...comm.base import BaseCommBackend
except ImportError:
    try:
        from comm.base import BaseCommBackend
    except ImportError:
        from script.streaming_queue.comm.base import BaseCommBackend
from fastapi import FastAPI, Request
from uvicorn import Config, Server
import asyncio
import json

class ANPCommBackend(BaseCommBackend):
    def __init__(self, **kwargs):
        self._endpoints: Dict[str, str] = {}  # agent_id -> endpoint uri
        self._clients: Dict[str, httpx.AsyncClient] = {}  # HTTP clients for each agent
        self._servers: Dict[str, Any] = {}  # For spawned local servers
        self.protocol_version = "1.0"

    async def register_endpoint(self, agent_id: str, address: str) -> None:
        """注册ANP agent的endpoint"""
        self._endpoints[agent_id] = address
        self._clients[agent_id] = httpx.AsyncClient(base_url=address, timeout=30.0)

    async def connect(self, src_id: str, dst_id: str) -> None:
        """ANP协议不需要显式连接建立"""
        pass

    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """发送ANP消息"""
        endpoint = self._endpoints.get(dst_id)
        if not endpoint:
            raise RuntimeError(f"ANP: unknown dst_id={dst_id}")

        # 构造ANP消息格式
        anp_message = {
            "protocol": "ANP",
            "version": self.protocol_version,
            "message_id": str(uuid.uuid4()),
            "sender": src_id,
            "receiver": dst_id,
            "message_type": "request",
            "payload": {
                "action": "execute",
                "data": payload.get("text", str(payload))
            },
            "timestamp": time.time()
        }

        client = self._clients.get(dst_id)
        try:
            resp = await client.post("/anp/message", json=anp_message)
            resp.raise_for_status()
            raw = resp.json()
            
            # 从ANP响应中提取文本
            text = ""
            if isinstance(raw, dict):
                anp_payload = raw.get("payload", {})
                text = anp_payload.get("data", "")
            
            return {"raw": raw, "text": text}
        except Exception as e:
            raise RuntimeError(f"ANP send failed: {e}")

    async def health_check(self, agent_id: str) -> bool:
        """ANP健康检查"""
        endpoint = self._endpoints.get(agent_id)
        if not endpoint:
            return False
        
        client = self._clients.get(agent_id)
        try:
            # 发送ANP health check消息
            health_message = {
                "protocol": "ANP",
                "version": self.protocol_version,
                "message_id": str(uuid.uuid4()),
                "sender": "system",
                "receiver": agent_id,
                "message_type": "request",
                "payload": {
                    "action": "health_check",
                    "data": ""
                },
                "timestamp": time.time()
            }
            
            resp = await client.post("/anp/health", json=health_message)
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """关闭所有连接和服务器"""
        for client in self._clients.values():
            await client.aclose()
        for srv in self._servers.values():
            if hasattr(srv, "shutdown"):
                await srv.shutdown()

    async def spawn_local_agent(self, agent_id: str, host: str, port: int, executor: Any) -> Any:
        """启动本地ANP agent服务"""
        app = FastAPI()

        # ANP健康检查端点
        @app.post("/anp/health")
        async def health_check(request: Request):
            try:
                body = await request.json()
                return {
                    "protocol": "ANP",
                    "version": self.protocol_version,
                    "message_id": str(uuid.uuid4()),
                    "sender": agent_id,
                    "receiver": body.get("sender", "system"),
                    "message_type": "response",
                    "payload": {
                        "action": "health_check",
                        "data": "healthy"
                    },
                    "timestamp": time.time()
                }
            except Exception as e:
                return {
                    "protocol": "ANP",
                    "version": self.protocol_version,
                    "message_id": str(uuid.uuid4()),
                    "sender": agent_id,
                    "receiver": "system",
                    "message_type": "response",
                    "payload": {
                        "action": "health_check",
                        "data": f"error: {str(e)}"
                    },
                    "timestamp": time.time()
                }

        # ANP消息处理端点
        @app.post("/anp/message")
        async def handle_message(request: Request):
            try:
                anp_message = await request.json()
                
                # 验证ANP消息格式
                if anp_message.get("protocol") != "ANP":
                    raise ValueError("Invalid protocol")
                
                payload = anp_message.get("payload", {})
                action = payload.get("action", "")
                data = payload.get("data", "")
                
                # 调用executor处理消息
                if hasattr(executor, 'execute'):
                    # 对于coordinator或worker，传递相应的输入格式
                    if action == "execute":
                        result = await executor.execute({"text": data})
                    else:
                        result = await executor.execute({"text": action})
                else:
                    result = {"text": f"Unknown action: {action}"}
                
                # 返回ANP响应格式
                response_data = result.get("text", str(result)) if isinstance(result, dict) else str(result)
                
                return {
                    "protocol": "ANP",
                    "version": self.protocol_version,
                    "message_id": str(uuid.uuid4()),
                    "sender": agent_id,
                    "receiver": anp_message.get("sender", "unknown"),
                    "message_type": "response",
                    "payload": {
                        "action": "execute",
                        "data": response_data
                    },
                    "timestamp": time.time()
                }
                
            except Exception as e:
                return {
                    "protocol": "ANP",
                    "version": self.protocol_version,
                    "message_id": str(uuid.uuid4()),
                    "sender": agent_id,
                    "receiver": "system",
                    "message_type": "response",
                    "payload": {
                        "action": "error",
                        "data": f"Error: {str(e)}"
                    },
                    "timestamp": time.time()
                }

        # 启动服务器
        config = Config(app=app, host=host, port=port, log_level="error")
        srv = Server(config)
        self._servers[agent_id] = srv

        asyncio.create_task(srv.serve())
        await asyncio.sleep(1)  # 等待服务器启动

        base_url = f"http://{host}:{port}"
        return type("ANPHandle", (), {"base_url": base_url})
