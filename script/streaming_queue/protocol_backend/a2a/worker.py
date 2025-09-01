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

from core.qa_worker_base import QAWorkerBase

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
            # A2A enqueue_event 是同步调用，不需要 await
            event_queue.enqueue_event(new_agent_text_message(result))
        except Exception as e:
            event_queue.enqueue_event(new_agent_text_message(f"[QAAgentExecutor] failed: {e}"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:  # type: ignore[override]
        event_queue.enqueue_event(new_agent_text_message("cancel not supported"))

    # ------------- helpers -------------
    def _extract_text(self, payload: Any) -> str:
        """
        尽量兼容多种可能的 A2A 输入结构：
          1) 直接是 str
          2) {"text": "..."}
          3) {"parts":[{"type":"text","text":"..."}]}
          4) {"parts":[{"kind":"text","text":"..."}]}
        """
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload

        if isinstance(payload, dict):
            # 常见的 A2A 结构
            parts = payload.get("parts") or []
            for p in parts:
                if isinstance(p, dict) and ("text" in p) and (p.get("type") == "text" or p.get("kind") == "text" or "type" not in p):
                    return p.get("text") or ""
            # 退化结构
            if "text" in payload and isinstance(payload["text"], str):
                return payload["text"]
            # chat 格式
            if "content" in payload and isinstance(payload["content"], str):
                return payload["content"]

        # 其他情况直接转字符串
        try:
            return str(payload)
        except Exception:
            return ""
