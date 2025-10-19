# -*- coding: utf-8 -*-
"""
A2A Worker (protocol-specific)

- 继承 QAWorkerBase，使用 A2A 原生 SDK 将其包装成一个可执行的 Agent
- 仅负责“通信适配层”：提取用户输入文本 → 调用基类 answer() → 通过 A2A 事件返回
"""

from __future__ import annotations

from typing import Any, Optional

import sys
from pathlib import Path

# Add streaming_queue to path for imports
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parent.parent.parent  # Go up from a2a -> protocol_backend -> streaming_queue
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))

# Add core path
streaming_queue_path = Path(__file__).resolve().parent.parent.parent
core_path = streaming_queue_path / "core"
if str(core_path) not in sys.path:
    sys.path.insert(0, str(core_path))
from qa_worker_base import QAWorkerBase

# A2A SDK（必需依赖）
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message


class A2AQAWorker(QAWorkerBase):
    """目前不需要额外逻辑，保留扩展点。"""
    pass


class QAAgentExecutor(AgentExecutor):
    """
    A2A 原生 Executor 外壳：
      - 从 RequestContext 中抽取文本（支持多种 A2A 结构）
      - 调用 A2AQAWorker.answer()
      - 通过 EventQueue 返回文本消息
    """

    def __init__(self, config: Optional[dict] = None, output=None):
        self.worker = A2AQAWorker(config=config, output=output)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:  # type: ignore[override]
        try:
            user_input = context.get_user_input() if hasattr(context, "get_user_input") else None
            question = self._extract_text(user_input) or "What is artificial intelligence?"

            result = await self.worker.answer(question)
            
            # Try both sync and async enqueue_event
            try:
                # First try async (correct way based on RuntimeWarning)
                await event_queue.enqueue_event(new_agent_text_message(result))
            except Exception as async_error:
                try:
                    # Fallback to sync
                    event_queue.enqueue_event(new_agent_text_message(result))
                except Exception:
                    # Last resort - try simple text event
                    event_queue.enqueue_event(new_agent_text_message(f"A2A result: {result}"))
        except Exception as e:
            event_queue.enqueue_event(new_agent_text_message(f"[QAAgentExecutor] failed: {e}"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:  # type: ignore[override]
        event_queue.enqueue_event(new_agent_text_message("cancel not supported"))

    # ------------- helpers -------------
    def _extract_text(self, payload: Any) -> str:
        """
        Accept A2A shapes:
          1) str
          2) {"text": "..."} or {"content":"..."}
          3) {"parts":[{"type":"text","text":"..."}]}
          4) {"parts":[{"type":"json","text":{...}}]}  # serialize to string if dict/list
        """
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload

        if isinstance(payload, dict):
            # Prefer parts
            parts = payload.get("parts") or []
            for p in parts:
                if not isinstance(p, dict):
                    continue
                if "text" in p:
                    t = p.get("text")
                    # text could be str or dict/list if type=="json"
                    if isinstance(t, (dict, list)):
                        try:
                            import json
                            return json.dumps(t, ensure_ascii=False)
                        except Exception:
                            return str(t)
                    if isinstance(t, str):
                        return t
            # Fallbacks
            if isinstance(payload.get("text"), str):
                return payload["text"]
            if isinstance(payload.get("content"), str):
                return payload["content"]

        try:
            return str(payload)
        except Exception:
            return ""
