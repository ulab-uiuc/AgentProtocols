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
from loguru import logger

class QAWorkerBase:
    """协议无关的 QA 工作者：封装 LLM 调用与降级策略。"""

    def __init__(self, config: Optional[dict] = None, output=None) -> None:
        self.output = output
        self.use_mock = False
        self.core = None

        # 使用 agent_network/src/utils/core.py 中的 Core
        try:
            agent_network_root = Path(__file__).resolve().parents[3]  # …/agent_network
            src_path = agent_network_root / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            # Try different import paths
            try:
                from utils.core import Core  # type: ignore
            except ImportError:
                from src.utils.core import Core  # type: ignore

            cfg = config or self._get_default_config()
            self._log(f"[QAWorkerBase] Initializing Core with config: {cfg}")
            
            # 确保配置格式正确
            if not cfg or "model" not in cfg:
                raise ValueError("Invalid config: missing 'model' section")
                
            self.core = Core(cfg)
            self.use_mock = False  # 确保不使用 mock
            self._log(f"[QAWorkerBase] Core LLM ready: {cfg.get('model', {}).get('name', 'unknown')}")
        except Exception as e:
            self._log(f"[QAWorkerBase] Core init failed, will NOT use mock. Error: {e}")
            # 不要设置 use_mock = True，而是抛出异常，强制解决问题
            raise RuntimeError(f"Core initialization failed: {e}. Mock answers are not allowed.")

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
        # openai_key = os.getenv("OPENAI_API_KEY")

        # if openai_key and openai_key != "test-key":
        #     return {
        #         "model": {
        #             "type": "openai",
        #             "name": "gpt-3.5-turbo",
        #             "openai_api_key": openai_key,
        #             "temperature": 0.3,
        #         }
        #     }
        # else:
        #     return {
        #         "model": {
        #             "type": "local",
        #             "name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        #             "temperature": 0.3,
        #         },
        #         "base_url": "http://localhost:8000/v1",
        #         "port": 8000,
        #     }
        # use local model
        return {
            "model": {
                "type": "local",
                "name": "Qwen2.5-VL-72B-Instruct",
                "temperature": 0.3,
            },
            "base_url": "http://localhost:8000/v1",
            "port": 8000,
        }

    # --------------------- public API ---------------------
    async def answer(self, question: str) -> str:
        """
        对外统一问答接口。必须使用真正的 Core LLM，禁止 mock。
        """
        if self.core is None:
            raise RuntimeError("Core is not initialized. Mock answers are not allowed.")
            
        try:
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
            # Core.execute 是同步方法，放到线程池执行
            result: str = await loop.run_in_executor(None, self.core.execute, messages)
            
            if not result or result.strip() == "":
                raise RuntimeError("Core returned empty response")
                
            return result.strip()
        except Exception as e:
            self._log(f"[QAWorkerBase] Core execution error: {e}")
            raise RuntimeError(f"LLM execution failed: {e}. Mock answers are not allowed.")

