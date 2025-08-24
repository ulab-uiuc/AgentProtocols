# script/streaming_queue/protocol_backend/acp/worker.py
"""
ACP Worker (protocol-specific)

- 继承 QAWorkerBase，使用 ACP 原生 SDK 将其包装成一个可执行的 Agent
- 仅负责"通信适配层"：提取用户输入文本 → 调用基类 answer() → 通过 ACP 返回
"""

from __future__ import annotations
from typing import Any, Optional, Dict

# 允许两种导入相对路径（避免运行根目录不同导致的导入失败）
try:
    from ...core.qa_worker_base import QAWorkerBase  # when run as package
except Exception:
    from script.streaming_queue.core.qa_worker_base import QAWorkerBase  # type: ignore

# ACP SDK (optional)
try:
    from acp_sdk import Client
    from acp_sdk.models import RunCreateRequest, Input
    ACP_AVAILABLE = True
except ImportError:
    ACP_AVAILABLE = False
    Client = None
    RunCreateRequest = None
    Input = None


class ACPQAWorker(QAWorkerBase):
    """目前不需要额外逻辑，保留扩展点。"""
    pass


class ACPWorkerExecutor:
    """
    ACP 原生 Executor 外壳：
      - 从 input_data 中抽取文本（支持 ACP 结构）
      - 调用 ACPQAWorker.answer()
      - 返回 ACP 格式的响应
    """

    def __init__(self, config: Optional[Dict] = None, output=None):
        self.worker = ACPQAWorker(config=config, output=output)

    async def execute(self, input_data: Dict) -> Dict:
        try:
            # Extract text from ACP input
            parts = input_data.get("content", [])
            text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text") or ""

            if not text:
                text = "What is artificial intelligence?"  # fallback question

            # Add timeout to prevent hanging
            import asyncio
            answer = await asyncio.wait_for(self.worker.answer(text), timeout=30.0)

            # Return ACP output format
            return {"content": [{"type": "text", "text": answer}]}

        except asyncio.TimeoutError:
            return {"content": [{"type": "text", "text": "Error: Request timed out after 30 seconds"}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error: {str(e)}"}]}
