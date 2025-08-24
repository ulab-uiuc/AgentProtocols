# -*- coding: utf-8 -*-
"""
QA Worker Base (protocol-agnostic)

- 提供统一的问答接口：await answer(question: str) -> str
- 内部可使用 utils.core.Core（若不可用则自动降级 mock）
- 不包含任何 A2A/HTTP 相关逻辑，方便复用到其他协议
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any


class QAWorkerBase:
    """协议无关的 QA 工作者：封装 LLM 调用与降级策略。"""

    def __init__(self, config: Optional[dict] = None, output=None) -> None:
        self.output = output
        self.use_mock = False
        self.core = None

        # 允许从项目根下的 src/utils/core.py 引入 Core
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # …/Multiagent-Protocol
            from src.utils.core import Core  # type: ignore

            cfg = config or self._get_default_config()
            self.core = Core(cfg)
            self._log(f"[QAWorkerBase] Core LLM ready: {cfg.get('model', {}).get('name', 'unknown')}")
        except Exception as e:
            self._log(f"[QAWorkerBase] Core init failed, fallback to mock. reason={e}")
            self.use_mock = True

    # --------------------- utils ---------------------
    def _log(self, msg: str) -> None:
        if self.output and hasattr(self.output, "info"):
            try:
                self.output.info(msg)
                return
            except Exception:
                pass
        print(msg)

    def _get_default_config(self) -> dict:
        """默认模型配置。优先使用 OPENAI_API_KEY，否则走本地/代理。"""
        openai_key = os.getenv("OPENAI_API_KEY")

        if openai_key and openai_key != "test-key":
            return {
                "model": {
                    "type": "openai",
                    "name": "gpt-3.5-turbo",
                    "openai_api_key": openai_key,
                    "temperature": 0.3,
                }
            }
        else:
            return {
                "model": {
                    "type": "local",
                    "name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                    "temperature": 0.3,
                },
                "base_url": "http://localhost:8000/v1",
                "port": 8000,
            }

    # --------------------- public API ---------------------
    async def answer(self, question: str) -> str:
        """
        对外统一问答接口。内部若有 Core 则调用 Core，否则返回 mock。
        """
        try:
            if self.use_mock or self.core is None:
                await asyncio.sleep(0.05)
                q = (question or "").strip()
                return f"Mock answer: {q[:80]}{'...' if len(q) > 80 else ''}"

            messages: List[Dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Provide concise, accurate answers to questions. "
                        "Keep responses under 150 words."
                    ),
                },
                {"role": "user", "content": question},
            ]

            loop = asyncio.get_event_loop()
            # 假设 Core.execute 是同步阻塞方法，这里丢到线程池执行
            result: str = await loop.run_in_executor(None, self.core.execute, messages)  # type: ignore
            return (result or "Unable to generate response").strip()
        except Exception as e:
            self._log(f"[QAWorkerBase] error: {e}")
            return f"Error: {str(e)[:100]}..."

