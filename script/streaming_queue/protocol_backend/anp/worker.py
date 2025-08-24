# script/streaming_queue/protocol_backend/anp/worker.py
"""
ANP Worker (protocol-specific)

- 继承 QAWorkerBase，使用 ANP 协议包装成可执行的 Agent
- 仅负责"通信适配层"：提取用户输入文本 → 调用基类 answer() → 通过 ANP 返回
"""

from __future__ import annotations
from typing import Any, Optional, Dict

# 允许多种导入相对路径（避免运行根目录不同导致的导入失败）
try:
    from ...core.qa_worker_base import QAWorkerBase  # when run as package
except ImportError:
    try:
        from core.qa_worker_base import QAWorkerBase
    except ImportError:
        from script.streaming_queue.core.qa_worker_base import QAWorkerBase  # type: ignore


class ANPQAWorker(QAWorkerBase):
    """ANP专用的QA Worker，目前不需要额外逻辑，保留扩展点。"""
    pass


class ANPWorkerExecutor:
    """
    ANP 原生 Executor 外壳：
      - 从 input_data 中抽取文本（支持 ANP 结构或简单文本）
      - 调用 ANPQAWorker.answer()
      - 返回文本结果
    """

    def __init__(self, config: Optional[Dict] = None, output=None):
        self.worker = ANPQAWorker(config=config, output=output)

    async def execute(self, input_data: Dict) -> Dict:
        """
        处理ANP消息并返回答案
        输入格式可能是：
        1. {"text": "question"}  # 简单格式
        2. ANP消息格式的payload
        """
        try:
            # 提取文本内容
            text = ""
            if isinstance(input_data, dict):
                # 支持简单的text格式
                if "text" in input_data:
                    text = input_data["text"]
                # 支持ANP消息格式
                elif "payload" in input_data:
                    payload = input_data.get("payload", {})
                    text = payload.get("data", "")
                # 直接是字符串内容
                else:
                    text = str(input_data)
            else:
                text = str(input_data)

            if not text:
                text = "What is artificial intelligence?"  # fallback question

            # 添加超时防止挂起
            import asyncio
            answer = await asyncio.wait_for(self.worker.answer(text), timeout=30.0)

            # 返回简单的文本格式供ANP CommBackend包装
            return {"text": answer}

        except asyncio.TimeoutError:
            return {"text": "Error: Request timed out after 30 seconds"}
        except Exception as e:
            return {"text": f"Error: {str(e)}"}
