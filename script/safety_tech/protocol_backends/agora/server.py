# -*- coding: utf-8 -*-
"""
Agora 原生服务端（ReceiverServer），使用基于 llm_wrapper 的 Toolformer。
禁止mock/fallback；严格依赖 agora-protocol。
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, Callable, Optional

import uvicorn


def _build_receiver_with_llm_wrapper(agent_name: str):
    # 严格导入官方SDK
    import agora  # type: ignore
    try:
        from script.safety_tech.core.llm_wrapper import generate_doctor_reply
    except Exception:
        from core.llm_wrapper import generate_doctor_reply

    # 构造与 LangChain 兼容的最小模型（invoke）
    class _LLMWrapperModel:
        def __init__(self, role_hint: str) -> None:
            self._role = role_hint

        def invoke(self, messages: Any, **kwargs: Any):  # noqa: ANN401
            try:
                texts = []
                for m in messages or []:
                    content = getattr(m, "content", None)
                    if isinstance(content, str):
                        texts.append(content)
                    elif isinstance(m, dict):
                        c = m.get("content") or m.get("text")
                        if isinstance(c, str):
                            texts.append(c)
                prompt = "\n".join(texts)
            except Exception:
                prompt = str(messages)

            reply = generate_doctor_reply(self._role, prompt)
            class _Msg:
                def __init__(self, content: str) -> None:
                    self.content = content
            return _Msg(reply)

    # doctor_a / doctor_b 角色
    role = 'doctor_a' if agent_name.endswith('_A') else 'doctor_b'

    # 基于 llm_wrapper 的 Toolformer
    try:
        from agora.toolformers.langchain import LangChainToolformer  # type: ignore
        toolformer = LangChainToolformer(_LLMWrapperModel(role))
    except Exception:
        # 尝试备用导入路径
        from agora import toolformers  # type: ignore
        toolformer = toolformers.LangChainToolformer(_LLMWrapperModel(role))

    # 提供一个简单工具（可选），此处不用任何外部依赖
    def echo_tool(text: str) -> str:
        return f"echo:{text}"

    receiver = agora.Receiver.make_default(toolformer, tools=[echo_tool])
    server = agora.ReceiverServer(receiver)
    return server


def spawn_doctor(agent_name: str, port: int) -> None:
    """启动基于 llm_wrapper 的 Agora ReceiverServer。"""
    server = _build_receiver_with_llm_wrapper(agent_name)
    # ReceiverServer 自带 run(port=)，这里通过其run启动
    server.run(port=port)


if __name__ == "__main__":
    import os
    name = os.environ.get("AGORA_AGENT_NAME", "Agora_Doctor_A")
    port = int(os.environ.get("AGORA_PORT", "9302"))
    spawn_doctor(name, port)


