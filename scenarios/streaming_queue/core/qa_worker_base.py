# -*- coding: utf-8 -*-
"""
QA Worker Base (protocol-agnostic)

- Provides a unified QA interface: await answer(question: str) -> str
- Internally uses utils.core.Core (no mock fallback is allowed)
- Contains no A2A/HTTP logic so it can be reused by other protocols
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger

class QAWorkerBase:
    """Protocol-agnostic QA worker: wraps LLM invocation without mock fallbacks."""

    def __init__(self, config: Optional[dict] = None, output=None) -> None:
        self.output = output
        self.use_mock = False
        self.core = None

    # Use Core from agent_network/src/utils/core.py
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
            
            # Log config with API key masking for security
            import copy
            cfg_safe = copy.deepcopy(cfg)
            if "model" in cfg_safe and "openai_api_key" in cfg_safe["model"]:
                cfg_safe["model"]["openai_api_key"] = "***"
            self._log(f"[QAWorkerBase] Initializing Core with config: {cfg_safe}")
            
            # Ensure config format is correct
            if not cfg or "model" not in cfg:
                raise ValueError("Invalid config: missing 'model' section")
                
            self.core = Core(cfg)
            self.use_mock = False  # Ensure mock is never used
            self._log(f"[QAWorkerBase] Core LLM ready: {cfg.get('model', {}).get('name', 'unknown')}")
        except Exception as e:
            self._log(f"[QAWorkerBase] Core init failed, will NOT use mock. Error: {e}")
            # Do not set use_mock=True; raise and force a proper fix instead
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
        """Default model configuration. Prefer OPENAI_API_KEY; otherwise use local/proxy."""
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
    async def answer(self, question: str) -> Dict[str, Any]:
        """Unified QA interface. Must use the real Core LLM; mock is forbidden.
        Returns dict with 'answer' and 'llm_timing' information.
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
            # Core.execute is synchronous; run it in a thread pool
            # Record LLM execution time
            llm_start = asyncio.get_event_loop().time()
            result: str = await loop.run_in_executor(None, self.core.execute, messages)
            llm_end = asyncio.get_event_loop().time()
            
            if not result or result.strip() == "":
                raise RuntimeError("Core returned empty response")
            
            return {
                "answer": result.strip(),
                "llm_timing": {
                    "llm_start": llm_start,
                    "llm_end": llm_end,
                    "llm_execution_time": llm_end - llm_start
                }
            }
        except Exception as e:
            self._log(f"[QAWorkerBase] Core execution error: {e}")
            raise RuntimeError(f"LLM execution failed: {e}. Mock answers are not allowed.")

