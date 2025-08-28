# -*- coding: utf-8 -*-
"""
A2A Comm Backend (HTTP + 原生事件格式)
放置位置：agent_network/script/streaming_queue/protocol_backend/a2a/comm.py

说明：
- 直接走 A2A Server 的 /message 接口与事件流结构
- 集成“轻量 Host”，可在本进程内启动一个 FastAPI + Uvicorn 服务来承载任意 AgentExecutor
- 不再使用自定义 adapter
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

# A2A 的执行器接口（必需依赖）
from a2a.server.agent_execution import AgentExecutor  # type: ignore

# ------------------- Comm Base 引用 -------------------
import sys
from pathlib import Path

# Add streaming_queue to path for imports
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parent.parent.parent  # Go up from a2a -> protocol_backend -> streaming_queue
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))

try:
    from comm.base import BaseCommBackend  # type: ignore
except ImportError as e:
    raise ImportError(f"Cannot import BaseCommBackend from comm.base: {e}")


# ==========================
# 内嵌 Host (FastAPI/Uvicorn)
# ==========================

# 为了最小依赖/快速启动，我们把 FastAPI、Uvicorn 的导入放在用到时再做（延迟导入）。
# 这样不用 Host 功能也不会强依赖它们。

# 移除简化类，使用真正的 A2A 组件

@dataclass
class A2AAgentHandle:
    agent_id: str
    host: str
    port: int
    base_url: str
    _server: Any | None
    _task: asyncio.Task | None

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass


async def _start_a2a_host(agent_id: str, host: str, port: int, executor: AgentExecutor) -> A2AAgentHandle:
    """
    启动一个极简 A2A Host（/message + /health），使用 Starlette 纯 JSON，
    不依赖 Pydantic/BaseModel。
    """
    # 延迟导入（仅在需要 host 时依赖这些库）
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.requests import Request
    from starlette.routing import Route
    import uvicorn

    async def message_endpoint(request: Request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        # 兼容 A2A 入参结构：{"id": "...", "params": {"message": {...}}}
        msg = (payload.get("params") or {}).get("message") or {}
        parts = msg.get("parts") or []
        text = ""
        if parts and isinstance(parts, list) and isinstance(parts[0], dict):
            # 兼容 {"kind":"text","text":"..."} / {"kind":"text","data":"..."}
            text = parts[0].get("text") or parts[0].get("data") or ""

        # 导入真正的 A2A 组件
        from a2a.server.agent_execution import RequestContext
        from a2a.server.events import EventQueue
        from a2a.types import Message, MessageSendParams, Role, TextPart
        
        # 创建真正的 A2A Message 对象
        if text:
            # 手动创建用户消息
            message = Message(
                role=Role.user,
                parts=[TextPart(text=text)],
                messageId=str(time.time_ns())
            )
        else:
            # 尝试从原始 payload 构造 Message
            msg_data = payload.get("params", {}).get("message", {})
            if msg_data:
                # 从字典构造 Message 对象
                message = Message.model_validate(msg_data)
            else:
                # 默认状态消息
                message = Message(
                    role=Role.user,
                    parts=[TextPart(text="status")],
                    messageId=str(time.time_ns())
                )
        
        # 创建 MessageSendParams 和 RequestContext
        params = MessageSendParams(message=message)
        ctx = RequestContext(params)
        eq = EventQueue()
        
        await executor.execute(context=ctx, event_queue=eq)

        # 从 A2A EventQueue 中获取所有事件
        serializable_events = []
        try:
            while not eq.queue.empty():
                event = await eq.dequeue_event()
                if event:
                    if hasattr(event, 'model_dump'):
                        # Pydantic v2 方式 - 使用 mode='json' 确保枚举等被正确序列化
                        serializable_events.append(event.model_dump(mode='json'))
                    elif hasattr(event, 'dict'):
                        # Pydantic v1 方式
                        serializable_events.append(event.dict())
                    else:
                        # 如果已经是字典，直接使用
                        serializable_events.append(event)
        except:
            # 如果队列为空或出现异常，使用空列表
            pass

        return JSONResponse({"events": serializable_events})

    async def health_endpoint(_request: Request):
        return JSONResponse({"ok": True, "agent_id": agent_id})

    routes = [
        Route("/message", message_endpoint, methods=["POST"]),
        Route("/health", health_endpoint, methods=["GET"]),
    ]
    app = Starlette(routes=routes)

    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    async def _serve():
        await server.serve()

    task = asyncio.create_task(_serve())
    await asyncio.sleep(0.3)  # 等端口起来

    return A2AAgentHandle(
        agent_id=agent_id,
        host=host,
        port=port,
        base_url=f"http://{host}:{port}",
        _server=server,
        _task=task,
    )


# ==========================
# 通信后端（含 Host 管理）
# ==========================

class A2ACommBackend(BaseCommBackend):
    """
    - 维护 agent_id -> base_url
    - 负责 /message 调用（发送/接收）
    - 可在本进程 spawn 本地 A2A host（spawn_local_agent）
    """

    def __init__(self, httpx_client: httpx.AsyncClient | None = None, request_timeout: float = 60.0):
        self._client = httpx_client or httpx.AsyncClient(timeout=request_timeout)
        self._own_client = httpx_client is None
        self._addr: Dict[str, str] = {}        # agent_id -> base_url
        self._hosts: Dict[str, A2AAgentHandle] = {}  # 若由本进程启动，则保存句柄（便于关闭）

    # ---------- endpoint 注册 ----------
    async def register_endpoint(self, agent_id: str, address: str) -> None:
        self._addr[agent_id] = address.rstrip("/")

    # ---------- 本地 Host 管理 ----------
    async def spawn_local_agent(self, agent_id: str, host: str, port: int, executor: AgentExecutor) -> A2AAgentHandle:
        """
        启动一个本地 A2A Host，并自动 register_endpoint。
        """
        if agent_id in self._hosts:
            raise RuntimeError(f"[A2ACommBackend] local agent already exists: {agent_id}")
        handle = await _start_a2a_host(agent_id, host, port, executor)
        self._hosts[agent_id] = handle
        await self.register_endpoint(agent_id, handle.base_url)
        return handle

    async def stop_local_agent(self, agent_id: str) -> None:
        h = self._hosts.pop(agent_id, None)
        if h:
            await h.stop()
        self._addr.pop(agent_id, None)

    # ---------- 发送消息 ----------
    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """
        发送一条消息到 dst 的 A2A /message。
        payload 支持：
          1) {"text":"..."} 或 {"parts":[{"kind":"text","text":"..."}]}
          2) 已是完整 A2A message 结构（含 role/parts），则直接透传到 params.message
        """
        base = self._addr.get(dst_id)
        if not base:
            raise RuntimeError(f"[A2ACommBackend] unknown dst_id={dst_id}. Did you register_endpoint()?")

        msg = self._to_a2a_message(payload)
        req = {
            "id": str(time.time_ns()),
            "params": {
                "message": msg
            }
        }
        resp = await self._client.post(f"{base}/message", json=req)
        resp.raise_for_status()
        data = resp.json()
        return {
            "raw": data,
            "text": self._extract_text_from_events(data)
        }

    # ---------- 健康检查 ----------
    async def health_check(self, agent_id: str) -> bool:
        base = self._addr.get(agent_id)
        if not base:
            return False
        # 先试 /health
        try:
            r = await self._client.get(f"{base}/health")
            if r.status_code == 200:
                return True
        except Exception:
            pass
        # 退化：发一条 status 作为 message
        try:
            req = {
                "id": str(time.time_ns()),
                "params": {
                    "message": {
                        "role": "user",
                        "messageId": str(time.time_ns()),
                        "parts": [{"kind": "text", "text": "status"}],
                    }
                },
            }
            r = await self._client.post(f"{base}/message", json=req)
            return r.status_code == 200
        except Exception:
            return False

    # ---------- 关闭 ----------
    async def close(self) -> None:
        # 关闭由本进程启动的 Host
        for aid in list(self._hosts.keys()):
            try:
                await self.stop_local_agent(aid)
            except Exception:
                pass
        if self._own_client:
            await self._client.aclose()

    # -------------------- helpers --------------------
    def _to_a2a_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # 已经是 A2A message 结构
        if all(k in payload for k in ("role", "parts")):
            return payload

        if "parts" in payload and isinstance(payload["parts"], list):
            parts = payload["parts"]
        else:
            text = payload.get("text") or payload.get("content") or ""
            parts = [{"kind": "text", "text": text}]

        return {
            "role": "user",
            "parts": parts,
            "messageId": str(time.time_ns()),
        }

    def _extract_text_from_events(self, data: Dict[str, Any]) -> str:
        events = data.get("events") or []
        for ev in events:
            # 与 a2a.utils.new_agent_text_message 对齐
            if ev.get("type") == "agent_text_message":
                return ev.get("data") or ev.get("text") or ""
            # 某些实现是 {"kind":"message","parts":[{"type":"text","text":"..."}]}
            if ev.get("kind") == "message":
                parts = ev.get("parts") or []
                if parts and isinstance(parts[0], dict):
                    t = parts[0].get("text")
                    if t:
                        return t
        return ""


# ==========================
# A2A Network (concrete)
# ==========================
from typing import Optional

try:
    from core.network_base import NetworkBase  # type: ignore
except ImportError as e:
    raise ImportError(f"Cannot import NetworkBase from core.network_base: {e}")


class A2ANetwork(NetworkBase):
    """
    NetworkBase 的 A2A 具体实现：
      - 用 A2ACommBackend 注入通信能力
      - 额外提供 spawn_local_agent() 语法糖，用于在本进程启动执行器 HTTP 服务并自动 register
    """
    def __init__(self, httpx_client: Optional[httpx.AsyncClient] = None, request_timeout: float = 60.0):
        backend = A2ACommBackend(httpx_client=httpx_client, request_timeout=request_timeout)
        super().__init__(comm_backend=backend)

    async def spawn_local_agent(self, agent_id: str, host: str, port: int, executor: AgentExecutor) -> A2AAgentHandle:
        """
        语法糖：在本进程启动一个 FastAPI+Uvicorn Host 承载该 executor，
        并自动 register 到当前网络。
        """
        # self._comm 类型即 A2ACommBackend
        handle = await self._comm.spawn_local_agent(agent_id, host, port, executor)  # type: ignore[attr-defined]
        await self.register_agent(agent_id, handle.base_url)
        return handle
