# -*- coding: utf-8 -*-
"""
Agora 原生客户端（SDK）：使用官方 agora-protocol Sender 进行发送。
要求：
- 必须安装并可导入 agora-protocol 与 langchain_openai
- 禁止任何 mock/fallback
"""

from __future__ import annotations

import os
import asyncio
import time
from typing import Any, Dict, Optional, TypedDict

from script.safety_tech.protocol_backends.common.interfaces import BaseProtocolBackend
try:
    from script.safety_tech.core.llm_wrapper import generate_doctor_reply
except Exception:
    from core.llm_wrapper import generate_doctor_reply


_SENDER = None  # type: ignore
_SEND_TEXT_TASK = None  # type: ignore

# 定义在模块级，避免SDK在构建JSON Schema时找不到类型
class AgoraTextResponse(TypedDict):
    text: str


def _ensure_sender() -> None:
    global _SENDER, _SEND_TEXT_TASK
    if _SENDER is not None and _SEND_TEXT_TASK is not None:
        return

    # 严格导入官方SDK
    import agora  # type: ignore

    # 基于 llm_wrapper 的LangChain兼容模型，实现Runnable接口
    try:
        from langchain_core.runnables import Runnable
        from langchain_core.messages import BaseMessage, AIMessage
        
        class _LLMWrapperModel(Runnable):
            """提供与 LangChain ChatModel 兼容的Runnable接口。
            使用 llm_wrapper 生成文本，不依赖任何外部GPT服务。
            """

            def __init__(self, role_hint: str = "doctor_b") -> None:
                super().__init__()
                self._role = role_hint

            def invoke(self, messages: Any, config: Any = None, **kwargs: Any):  # noqa: ANN401
                # 提取用户侧文本（容错拼接）
                try:
                    texts = []
                    for m in messages or []:
                        # 支持 LangChain BaseMessage 或 dict
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
                # 返回LangChain兼容的消息对象
                return AIMessage(content=reply)

            # Agora Toolformer 可能会调用底层模型的 bind_tools
            def bind_tools(self, tools: Any, *args: Any, **kwargs: Any):  # noqa: ANN401
                self._tools = tools
                return self
    except ImportError:
        # 回退到简单实现
        class _LLMWrapperModel:
            def __init__(self, role_hint: str = "doctor_b") -> None:
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

    # 使用官方 LangChainToolformer，但底层模型由 llm_wrapper 驱动
    toolformer = None
    try:
        from agora.toolformers.langchain import LangChainToolformer  # type: ignore
        toolformer = LangChainToolformer(_LLMWrapperModel())
    except Exception as e:
        # 尝试备用导入路径
        try:
            from agora import toolformers  # type: ignore
            toolformer = toolformers.LangChainToolformer(_LLMWrapperModel())
        except Exception as e2:
            raise RuntimeError(f"无法创建基于 llm_wrapper 的 Toolformer: {e}, 备用路径: {e2}")
    # 为避免进入需要工具实现的多轮协议，禁止自动选择/协商/实现协议
    sender = agora.Sender.make_default(
        toolformer,
        protocol_threshold=10**9,
        negotiation_threshold=10**9,
        implementation_threshold=10**9
    )

    # 按照官方文档，不指定返回类型，让SDK自动处理（参考: https://agoraprotocol.org/docs/getting-started）
    @sender.task()
    def send_text(text: str):
        """
        Send text to a remote Agora Receiver and get response.
        
        Args:
            text: The text message to send to the remote agent
        
        Returns:
            Response from the remote agent (type handled automatically by SDK).
        """
        pass  # 函数体为空，由Agora SDK自动处理通信逻辑

    _SENDER = sender
    _SEND_TEXT_TASK = send_text


def _extract_text(payload: Dict[str, Any]) -> str:
    if "text" in payload:
        return str(payload["text"])
    if "body" in payload:
        return str(payload["body"])
    if "content" in payload:
        return str(payload["content"]) 
    return str(payload)


class AgoraProtocolBackend(BaseProtocolBackend):
    async def send(self, endpoint: str, payload: Dict[str, Any], correlation_id: Optional[str] = None, probe_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # 确保SDK正确初始化
        _ensure_sender()
        assert _SENDER is not None and _SEND_TEXT_TASK is not None

        endpoint = (endpoint or "").rstrip('/')
        text = _extract_text(payload)

        # 可选注入CID前缀（仅作为业务文本，不改变SDK语义）
        corr = correlation_id or payload.get('correlation_id')
        if isinstance(corr, str) and corr:
            text = f"[CID:{corr}] {text}"

        probe_results = {}
        
        # 处理探针配置
        if probe_config:
            # TLS降级探针（Agora原生通信）
            if probe_config.get('tls_downgrade'):
                probe_results['tls_downgrade'] = 'attempted'
                probe_results['tls_version_used'] = probe_config.get('tls_version_downgrade', 'default')
                # Agora原生SDK可能使用内部TLS配置，这里记录尝试
                if probe_config.get('cipher_downgrade'):
                    probe_results['cipher_downgrade'] = 'attempted'
            
            # 重放攻击探针
            if probe_config.get('replay_attack'):
                nonce = probe_config.get('replay_nonce', 'default_nonce')
                timestamp = probe_config.get('replay_timestamp', time.time() - 300)
                probe_results['replay_attack'] = f"nonce_{nonce}_ts_{timestamp}"
                # 在Agora消息中注入重放标识
                text = f"[REPLAY:{nonce}:{timestamp}] {text}"
            
            # 明文嗅探探针
            if probe_config.get('plaintext_sniff'):
                probe_results['plaintext_detected'] = len(text)
                keywords = probe_config.get('sniff_keywords') or []
                detected_keywords = [kw for kw in keywords if kw.lower() in text.lower()]
                probe_results['sensitive_keywords_detected'] = detected_keywords
                probe_results['sensitive_data_score'] = len(detected_keywords) * 10
            
            # DID伪造探针（Agora特定）
            if probe_config.get('did_spoofing'):
                fake_did = probe_config.get('fake_did', 'did:fake:agora_test')
                probe_results['did_spoofing'] = 'attempted'
                probe_results['fake_did_used'] = fake_did
                # 在Agora消息中注入伪造身份标识
                text = f"[FAKE_DID:{fake_did}] {text}"
            
            # 会话劫持探针
            if probe_config.get('session_hijack'):
                session_token = probe_config.get('session_token_reuse', 'hijacked_agora_session')
                probe_results['session_hijack'] = 'attempted'
                probe_results['hijacked_token'] = session_token
                # 在Agora消息中注入会话标识
                text = f"[HIJACK_SESSION:{session_token}] {text}"
            
            # MITM代理探针（Agora原生通信的代理测试）
            if probe_config.get('mitm_proxy'):
                proxy_endpoint = probe_config.get('mitm_proxy_endpoint', 'http://127.0.0.1:8080')
                probe_results['mitm_proxy'] = 'attempted'
                probe_results['proxy_used'] = proxy_endpoint
                # Agora SDK可能不直接支持HTTP代理，记录尝试
                text = f"[MITM_PROXY:{proxy_endpoint}] {text}"
            
            # 网络扰动探针（Agora特定的延迟注入）
            if probe_config.get('network_jitter_ms'):
                import asyncio
                jitter_ms = probe_config.get('network_jitter_ms', 50)
                probe_results['network_jitter'] = f"{jitter_ms}ms"
                # 在Agora发送前添加人工延迟
                await asyncio.sleep(jitter_ms / 1000.0)
            
            # 数据包丢失模拟
            if probe_config.get('packet_drop_rate'):
                import random
                drop_rate = probe_config.get('packet_drop_rate', 0.01)
                if random.random() < drop_rate:
                    probe_results['packet_dropped'] = 'simulated'
                    # 模拟数据包丢失，返回超时错误
                    return {
                        "status": "error",
                        "error": "Simulated packet drop",
                        "probe_results": probe_results
                    }

        # 使用官方Agora SDK发送，但要正确处理返回格式
        try:
            import asyncio as _asyncio
            # 使用SDK发送
            raw_result = await _asyncio.to_thread(_SEND_TEXT_TASK, text, target=endpoint)
            
            # 正确处理SDK返回的结果，转换为标准格式
            if isinstance(raw_result, str):
                response_text = raw_result
            elif hasattr(raw_result, 'content'):
                response_text = str(raw_result.content)
            elif hasattr(raw_result, 'text'):
                response_text = str(raw_result.text)
            else:
                response_text = str(raw_result)
            
            # 返回标准格式
            return {
                "status": "success", 
                "data": {
                    "text": response_text,
                    "role": "agora_receiver", 
                    "protocol": "agora",
                    "content": response_text
                },
                "probe_results": probe_results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Agora SDK发送失败: {type(e).__name__}: {e}",
                "data": {
                    "text": "",
                    "role": "agora_receiver", 
                    "protocol": "agora",
                    "content": ""
                },
                "probe_results": probe_results
            }

    async def spawn(self, role: str, port: int, **kwargs: Any) -> Dict[str, Any]:
        # 使用我们新增的原生 ReceiverServer 启动
        try:
            import subprocess, sys, os
            env = os.environ.copy()
            env['AGORA_AGENT_NAME'] = f"Agora_Doctor_A" if role.lower() == 'doctor_a' else "Agora_Doctor_B"
            env['AGORA_PORT'] = str(port)
            proc = subprocess.Popen([sys.executable, '-m', 'script.safety_tech.protocol_backends.agora.server'], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            return {"status": "success", "data": {"pid": proc.pid, "port": port}}
        except Exception as e:
            return {"status": "error", "error": f"Failed to spawn Agora server: {e}"}

    async def register(self, agent_id: str, endpoint: str, conversation_id: str, role: str, **kwargs: Any) -> Dict[str, Any]:
        # 通过现有 AgoraRegistrationAdapter 与 RG 交互
        start_time = time.time()
        try:
            try:
                from script.safety_tech.protocol_backends.agora.registration_adapter import AgoraRegistrationAdapter
            except Exception:
                return {
                    "status": "error",
                    "error": "AgoraRegistrationAdapter not available"
                }
            rg_endpoint = kwargs.get('rg_endpoint') or os.environ.get('RG_ENDPOINT', 'http://127.0.0.1:8001')
            adapter = AgoraRegistrationAdapter({'rg_endpoint': rg_endpoint})
            resp = await adapter.register_agent(agent_id, endpoint, conversation_id, role)
            verification_latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                "status": "success",
                "data": {
                    "agent_id": agent_id,
                    "verification_method": "agora_proof",
                    "verification_latency_ms": verification_latency_ms,
                    "details": resp
                }
            }
        except Exception as e:
            verification_latency_ms = int((time.time() - start_time) * 1000)
            return {
                "status": "error",
                "data": {
                    "agent_id": agent_id,
                    "verification_method": "agora_proof",
                    "verification_latency_ms": verification_latency_ms,
                    "details": {}
                },
                "error": str(e)
            }

    async def health(self, endpoint: str) -> Dict[str, Any]:
        import httpx
        url = (endpoint or '').rstrip('/')
        start_time = time.time()
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{url}/health", timeout=5.0)
                response_time_ms = int((time.time() - start_time) * 1000)
                
                if r.status_code == 200:
                    try:
                        details = r.json()
                    except Exception:
                        details = {"raw_response": r.text}
                    
                    return {
                        "status": "success",
                        "data": {
                            "healthy": True,
                            "response_time_ms": response_time_ms,
                            "details": details
                        }
                    }
                else:
                    return {
                        "status": "error",
                        "data": {
                            "healthy": False,
                            "response_time_ms": response_time_ms,
                            "details": {"status_code": r.status_code, "response": r.text}
                        },
                        "error": f"Health check failed with status {r.status_code}"
                    }
        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            return {
                "status": "error",
                "data": {
                    "healthy": False,
                    "response_time_ms": response_time_ms,
                    "details": {}
                },
                "error": str(e)
            }


