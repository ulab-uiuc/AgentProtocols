# -*- coding: utf-8 -*-
"""
Agora native server (ReceiverServer), using llm_wrapper-based Toolformer.
No mock/fallback allowed; strictly depends on agora-protocol.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, Callable, Optional, TypedDict

import uvicorn

# Define response type compatible with client.py
class AgoraTextResponse(TypedDict):
    text: str


def _build_receiver_with_llm_wrapper(agent_name: str):
    # Strictly import official SDK
    import agora  # type: ignore
    try:
        from scenarios.safety_tech.core.llm_wrapper import generate_doctor_reply
    except Exception:
        from core.llm_wrapper import generate_doctor_reply

    # Construct LangChain-compatible Runnable model
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
        # Fallback to simple implementation
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

    # doctor_a / doctor_b role
    role = 'doctor_a' if agent_name.endswith('_A') else 'doctor_b'

    # llm_wrapper-based Toolformer
    try:
        from agora.toolformers.langchain import LangChainToolformer  # type: ignore
        toolformer = LangChainToolformer(_LLMWrapperModel(role))
    except Exception:
        # Try alternative import path
        from agora import toolformers  # type: ignore
        toolformer = toolformers.LangChainToolformer(_LLMWrapperModel(role))

    # Explicitly register echo_tool, use official Tool wrapper to avoid missing definition
    from agora.common.toolformers.base import Tool as AgoraTool  # type: ignore

    def echo_tool(text: str):
        """Echo response for protocol routines.

        Args:
            text: Input text

        Returns:
            Text response conforming to Agora protocol
        """
        try:
            from scenarios.safety_tech.core.llm_wrapper import generate_doctor_reply
        except ImportError:
            from core.llm_wrapper import generate_doctor_reply
        reply = generate_doctor_reply(role, text)
        return {"text": reply}

    # Manually specify return schema to avoid type inference issues
    echo_tool_wrapped = AgoraTool.from_function(
        echo_tool,
        return_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Generated response text"}
            },
            "required": ["text"]
        }
    )

    receiver = agora.Receiver.make_default(toolformer, tools=[echo_tool_wrapped])
    server = agora.ReceiverServer(receiver)
    return server


def spawn_doctor(agent_name: str, port: int) -> None:
    """Start llm_wrapper-based Agora ReceiverServer."""
    server = _build_receiver_with_llm_wrapper(agent_name)
    
    # Try to add health check endpoint for Agora server
    try:
        # Check if server has method to add routes
        if hasattr(server, 'app') and hasattr(server.app, 'get'):
            @server.app.get("/health")
            def health_check():
                return {"status": "healthy", "agent": agent_name, "timestamp": time.time()}
            print(f"🔍 [Agora Server] Health check endpoint added for {agent_name}")

        else:
            print(f"🔍 [Agora Server] Unable to add health check endpoint for {agent_name}, using default method")
    except Exception as e:
        print(f"🔍 [Agora Server] Failed to add health check endpoint: {e}")
    
    # ReceiverServer has built-in run(port=), start through its run method
    server.run(port=port)


if __name__ == "__main__":
    import os
    name = os.environ.get("AGORA_AGENT_NAME", "Agora_Doctor_A")
    port = int(os.environ.get("AGORA_PORT", "9302"))
    spawn_doctor(name, port)


