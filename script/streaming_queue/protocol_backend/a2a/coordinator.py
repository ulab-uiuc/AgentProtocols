# -*- coding: utf-8 -*-
"""
A2A Coordinator:
- 继承 QACoordinatorBase
- 用 A2A 原生 SDK 实现 send_to_worker()
- 同时提供 QACoordinator A2A Executor，便于把协调器暴露为 A2A Agent
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, Any, Optional, List

import sys
from pathlib import Path

# Add streaming_queue to path for imports
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parent.parent.parent  # Go up from a2a -> protocol_backend -> streaming_queue
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))

from core.qa_coordinator_base import QACoordinatorBase

# --- A2A 原生 SDK（必需依赖） ---
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
A2A_AVAILABLE = True


# ============ A2A Coordinator ============

class A2AQACoordinator(QACoordinatorBase):
    """
    使用 A2A 协议与 worker 通信。
    约定 self.agent_network 暴露一个协议信道方法：
        await agent_network.route_message(src_id, dst_id, payload) -> Dict
    其中 payload 为 A2A 的标准消息（见实现）。
    """

    async def send_to_worker(self, worker_id: str, question: str) -> Dict[str, Any]:
        if not self.agent_network:
            raise RuntimeError("No A2A network/router set. Call set_network() first.")

        # A2A 用户消息（最简 text）
        payload = {
            "messageId": str(int(time.time() * 1000)),
            "role": "user",
            "parts": [
                {"type": "text", "text": question}
            ],
        }

        # 通过 A2A 路由发送（接口与原代码保持一致）
        response = await self.agent_network.route_message(self.coordinator_id, worker_id, payload)

        # 解析 A2A 事件流，提取文本答案
        answer: Optional[str] = None
        try:
            # 处理嵌套的响应结构，检查是否有 "raw" 键
            if isinstance(response, dict):
                actual_response = response.get("raw", response)
                events = actual_response.get("events") if isinstance(actual_response, dict) else None
                
                if events:
                    for ev in events:
                        if ev.get("kind") == "message" and "parts" in ev:
                            parts = ev.get("parts") or []
                            if parts and isinstance(parts[0], dict) and "text" in parts[0]:
                                answer = parts[0]["text"]
                                break
        except Exception:
            answer = None

        return {
            "answer": answer or "No answer received",
            "raw": response,
        }


# ============ A2A Executor（把 Coordinator 暴露为 A2A Agent） ============

class QACoordinatorExecutor(AgentExecutor):
    """
    A2A 原生 Executor，用于对话式控制协调器。
    支持命令：
      - "dispatch" / "start_dispatch"
      - "status"
    """

    def __init__(self, config: Optional[dict] = None, output=None, coordinator: Optional[A2AQACoordinator] = None):
        self.config = config or {}
        self.output = output
        self.coordinator = coordinator or A2AQACoordinator(self.config, self.output)

    # ---- A2A entrypoint ----
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:  # type: ignore[override]
        try:
            raw = context.get_user_input() if hasattr(context, "get_user_input") else "status"
            # 兼容 A2A 结构化输入/字符串输入
            cmd = self._extract_command(raw)

            if cmd in ("dispatch", "start_dispatch"):
                result = await self.coordinator.dispatch_round()
            elif cmd == "status":
                result = await self.coordinator.get_status()
            else:
                result = f"Unknown command: {cmd}. Available commands: dispatch, status"

            # A2A 事件 - 使用真正的 A2A SDK
            await event_queue.enqueue_event(new_agent_text_message(result))

        except Exception as e:
            msg = f"QA Coordinator execution failed: {e}"
            await event_queue.enqueue_event(new_agent_text_message(msg))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:  # type: ignore[override]
        await event_queue.enqueue_event(new_agent_text_message("QA Coordinator operations cancelled."))

    # ---- helpers ----
    def _extract_command(self, user_input: Any) -> str:
        """从 A2A 上下文中抽取命令文本"""
        if isinstance(user_input, dict):
            # A2A 规范里常见：{"parts":[{"type":"text","text":"dispatch"}], ...}
            if "parts" in user_input:
                for part in user_input["parts"]:
                    if part.get("type") == "text" or part.get("kind") == "text":
                        return (part.get("text") or "status").strip().lower()
            if "text" in user_input:
                return str(user_input["text"]).strip().lower()
            return "status"
        if not user_input:
            return "status"
        return str(user_input).strip().lower()
