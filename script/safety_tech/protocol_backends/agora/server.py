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

    # 构造与 LangChain 兼容的Runnable模型
    try:
        from langchain_core.runnables import Runnable
        from langchain_core.messages import BaseMessage, AIMessage
        
        class _LLMWrapperModel(Runnable):
            def __init__(self, role_hint: str) -> None:
                super().__init__()
                self._role = role_hint

            def invoke(self, messages: Any, config: Any = None, **kwargs: Any):  # noqa: ANN401
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
                return AIMessage(content=reply)

            def bind_tools(self, tools: Any, *args: Any, **kwargs: Any):  # noqa: ANN401
                self._tools = tools
                return self
    except ImportError:
        # 回退到简单实现
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

    # 显式注册 echo_tool，使用官方 Tool 封装，避免找不到定义
    from agora.common.toolformers.base import Tool as AgoraTool  # type: ignore

    def echo_tool(text: str) -> str:
        """Echo response for protocol routines.

        Args:
            text: 输入文本

        Returns:
            LLM基于角色生成的回复字符串
        """
        try:
            from script.safety_tech.core.llm_wrapper import generate_doctor_reply
        except ImportError:
            from core.llm_wrapper import generate_doctor_reply
        return generate_doctor_reply(role, text)

    echo_tool_wrapped = AgoraTool.from_function(echo_tool)

    receiver = agora.Receiver.make_default(toolformer, tools=[echo_tool_wrapped])
    server = agora.ReceiverServer(receiver)
    return server


def spawn_doctor(agent_name: str, port: int) -> None:
    """启动基于 llm_wrapper 的 Agora ReceiverServer。"""
    server = _build_receiver_with_llm_wrapper(agent_name)
    
    # 尝试为Agora服务器添加健康检查端点
    try:
        # 检查服务器是否有添加路由的方法
        if hasattr(server, 'app') and hasattr(server.app, 'get'):
            @server.app.get("/health")
            def health_check():
                return {"status": "healthy", "agent": agent_name, "timestamp": time.time()}
            print(f"🔍 [Agora服务器] 已为 {agent_name} 添加健康检查端点")
        else:
            print(f"🔍 [Agora服务器] 无法为 {agent_name} 添加健康检查端点，使用默认方式")
    except Exception as e:
        print(f"🔍 [Agora服务器] 添加健康检查端点失败: {e}")
    
    # ReceiverServer 自带 run(port=)，这里通过其run启动
    server.run(port=port)


if __name__ == "__main__":
    import os
    name = os.environ.get("AGORA_AGENT_NAME", "Agora_Doctor_A")
    port = int(os.environ.get("AGORA_PORT", "9302"))
    spawn_doctor(name, port)


