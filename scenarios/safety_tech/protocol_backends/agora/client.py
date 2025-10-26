# -*- coding: utf-8 -*-
"""
Agora native client (SDK): uses official agora-protocol Sender for sending.
Requirements:
- Must have agora-protocol and langchain_openai installed and importable
- No mock/fallback allowed
"""

from __future__ import annotations

import os
import asyncio
import time
from typing import Any, Dict, Optional, TypedDict

from scenarios.safety_tech.protocol_backends.common.interfaces import BaseProtocolBackend
try:
    from scenarios.safety_tech.core.llm_wrapper import generate_doctor_reply
except Exception:
    from core.llm_wrapper import generate_doctor_reply


_SENDER = None  # type: ignore
_SEND_TEXT_TASK = None  # type: ignore

# Define at module level to avoid SDK not finding type when building JSON Schema
class AgoraTextResponse(TypedDict):
    text: str


def _ensure_sender() -> None:
    global _SENDER, _SEND_TEXT_TASK
    if _SENDER is not None and _SEND_TEXT_TASK is not None:
        return

    # Strictly import official SDK
    import agora  # type: ignore

    # llm_wrapper-based LangChain-compatible model, implements Runnable interface
    try:
        from langchain_core.runnables import Runnable
        from langchain_core.messages import BaseMessage, AIMessage
        
        class _LLMWrapperModel(Runnable):
            """Provides Runnable interface compatible with LangChain ChatModel.
            Uses llm_wrapper to generate text, does not depend on any external GPT service.
            """

            def __init__(self, role_hint: str = "doctor_b") -> None:
                super().__init__()
                self._role = role_hint

            def invoke(self, messages: Any, config: Any = None, **kwargs: Any):  # noqa: ANN401
                # Extract user-side text (fault-tolerant concatenation)
                try:
                    texts = []
                    for m in messages or []:
                        # Support LangChain BaseMessage or dict
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
                # Return LangChain-compatible message object
                return AIMessage(content=reply)

            # Agora Toolformer may call underlying model's bind_tools
            def bind_tools(self, tools: Any, *args: Any, **kwargs: Any):  # noqa: ANN401
                self._tools = tools
                return self
    except ImportError:
        # Fallback to simple implementation
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

    # Use official LangChainToolformer, but underlying model driven by llm_wrapper
    toolformer = None
    try:
        from agora.toolformers.langchain import LangChainToolformer  # type: ignore
        toolformer = LangChainToolformer(_LLMWrapperModel())
    except Exception as e:
        # Try alternative import path
        try:
            from agora import toolformers  # type: ignore
            toolformer = toolformers.LangChainToolformer(_LLMWrapperModel())
        except Exception as e2:
            raise RuntimeError(f"Unable to create llm_wrapper-based Toolformer: {e}, alternative path: {e2}")
    # To avoid entering multi-round protocols that require tool implementation, disable automatic selection/negotiation/implementation of protocols
    sender = agora.Sender.make_default(
        toolformer,
        protocol_threshold=10**9,
        negotiation_threshold=10**9,
        implementation_threshold=10**9
    )

    # According to official documentation, do not specify return type, let SDK handle automatically (reference: https://agoraprotocol.org/docs/getting-started)
    try:
        @sender.task()
        def send_text(text: str, target: str = None):
            """
            Send text to a remote Agora Receiver and get response.
            
            Args:
                text: The text message to send to the remote agent
                target: Target endpoint (optional, for compatibility)
            
            Returns:
                Response from the remote agent (type handled automatically by SDK).
            """
            pass  # Function body is empty, communication logic automatically handled by Agora SDK

        print(f"🔍 [Agora Client] Task created successfully: {type(send_text)}")
        
        # Global variables already declared at function start
        _SENDER = sender
        _SEND_TEXT_TASK = send_text
        print(f"🔍 [Agora Client] Global variables set: _SENDER={type(_SENDER)}, _SEND_TEXT_TASK={type(_SEND_TEXT_TASK)}")
    except Exception as e:
        print(f"❌ [Agora Client] Task creation failed: {type(e).__name__}: {e}")
        raise RuntimeError(f"Unable to create Agora send task: {type(e).__name__}: {e}")


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
        # Ensure SDK is properly initialized
        _ensure_sender()
        assert _SENDER is not None and _SEND_TEXT_TASK is not None

        endpoint = (endpoint or "").rstrip('/')
        text = _extract_text(payload)

        # Optionally inject CID prefix (only as business text, does not change SDK semantics)
        corr = correlation_id or payload.get('correlation_id')
        if isinstance(corr, str) and corr:
            text = f"[CID:{corr}] {text}"

        probe_results = {}
        
        # Process probe configuration
        if probe_config:
            # TLS downgrade probe (Agora native communication)
            if probe_config.get('tls_downgrade'):
                probe_results['tls_downgrade'] = 'attempted'
                probe_results['tls_version_used'] = probe_config.get('tls_version_downgrade', 'default')
                # Agora native SDK may use internal TLS configuration, record attempt here
                if probe_config.get('cipher_downgrade'):
                    probe_results['cipher_downgrade'] = 'attempted'
            
            # Replay attack probe
            if probe_config.get('replay_attack'):
                nonce = probe_config.get('replay_nonce', 'default_nonce')
                timestamp = probe_config.get('replay_timestamp', time.time() - 300)
                probe_results['replay_attack'] = f"nonce_{nonce}_ts_{timestamp}"
                # Inject replay identifier in Agora message
                text = f"[REPLAY:{nonce}:{timestamp}] {text}"
            
            # Plaintext sniffing probe
            if probe_config.get('plaintext_sniff'):
                probe_results['plaintext_detected'] = len(text)
                keywords = probe_config.get('sniff_keywords') or []
                detected_keywords = [kw for kw in keywords if kw.lower() in text.lower()]
                probe_results['sensitive_keywords_detected'] = detected_keywords
                probe_results['sensitive_data_score'] = len(detected_keywords) * 10
            
            # DID spoofing probe (Agora specific)
            if probe_config.get('did_spoofing'):
                fake_did = probe_config.get('fake_did', 'did:fake:agora_test')
                probe_results['did_spoofing'] = 'attempted'
                probe_results['fake_did_used'] = fake_did
                # Inject forged identity identifier in Agora message
                text = f"[FAKE_DID:{fake_did}] {text}"
            
            # Session hijacking probe
            if probe_config.get('session_hijack'):
                session_token = probe_config.get('session_token_reuse', 'hijacked_agora_session')
                probe_results['session_hijack'] = 'attempted'
                probe_results['hijacked_token'] = session_token
                # Inject session identifier in Agora message
                text = f"[HIJACK_SESSION:{session_token}] {text}"
            
            # MITM proxy probe (Agora native communication proxy test)
            if probe_config.get('mitm_proxy'):
                proxy_endpoint = probe_config.get('mitm_proxy_endpoint', 'http://127.0.0.1:8080')
                probe_results['mitm_proxy'] = 'attempted'
                probe_results['proxy_used'] = proxy_endpoint
                # Agora SDK may not directly support HTTP proxy, record attempt
                text = f"[MITM_PROXY:{proxy_endpoint}] {text}"
            
            # Network disturbance probe (Agora-specific delay injection)
            if probe_config.get('network_jitter_ms'):
                import asyncio
                jitter_ms = probe_config.get('network_jitter_ms', 50)
                probe_results['network_jitter'] = f"{jitter_ms}ms"
                # Add artificial delay before Agora send
                await asyncio.sleep(jitter_ms / 1000.0)
            
            # Packet loss simulation
            if probe_config.get('packet_drop_rate'):
                import random
                drop_rate = probe_config.get('packet_drop_rate', 0.01)
                if random.random() < drop_rate:
                    probe_results['packet_dropped'] = 'simulated'
                    # Simulate packet loss, return timeout error
                    return {
                        "status": "error",
                        "error": "Simulated packet drop",
                        "probe_results": probe_results
                    }
            
            # E2E payload encryption detection probe (newly added)
            if probe_config.get('e2e_payload_detection'):
                from scenarios.safety_tech.core.e2e_detector import E2EEncryptionDetector
                e2e_detector = E2EEncryptionDetector(probe_config.get('e2e_watermark', 'AGORA_E2E_WATERMARK'))
                
                # Modify original payload to inject watermark
                original_payload = {"text": text, "sender_id": "agora_probe", "receiver_id": "agora_target"}
                modified_payload = e2e_detector.inject_watermark_payload(original_payload)
                text = modified_payload.get('text', text)
                
                # Create plaintext detection payload
                if probe_config.get('e2e_probe_payload'):
                    probe_payload = e2e_detector.create_plaintext_probe_payload()
                    text += f" [PROBE_PAYLOAD: {probe_payload['probe_markers']['credit_card']}]"
                
                probe_results['e2e_watermark_injected'] = True
                probe_results['e2e_detector_active'] = True
            
            # Clock skew matrix probe (newly added)
            if probe_config.get('time_skew_matrix'):
                import random
                skew_levels = probe_config.get('time_skew_levels', [30, 120, 300, 600])
                probe_results['time_skew_matrix_active'] = True
                probe_results['skew_levels_tested'] = skew_levels
                
                # Randomly select a skew level for testing
                if skew_levels:
                    selected_skew = random.choice(skew_levels)
                    # Randomly select positive or negative skew
                    skew_direction = random.choice([-1, 1])
                    actual_skew = selected_skew * skew_direction
                    old_timestamp = time.time() + actual_skew
                    
                    # Inject timestamp into Agora message
                    nonce_id = f"skew_{abs(actual_skew)}_{int(time.time())}"
                    text = f"[TIME_SKEW:{old_timestamp}][NONCE:{nonce_id}] {text}"
                    
                    probe_results['time_skew_applied'] = actual_skew
                    probe_results['skew_direction'] = 'future' if actual_skew > 0 else 'past'
                    probe_results['skew_nonce'] = nonce_id
                
                # Window test: inject repeat/disorder markers
                if probe_config.get('time_skew_window_test'):
                    window_markers = [
                        f"[WINDOW_REPEAT:{int(time.time())}]",
                        f"[WINDOW_DISORDER:{random.randint(1000, 9999)}]",
                        f"[WINDOW_DUPLICATE:MSG_{random.randint(100, 999)}]"
                    ]
                    text = f"{random.choice(window_markers)} {text}"
                    probe_results['window_test_marker'] = True

        # Use native Agora SDK for communication
        try:
            import asyncio
            print(f"🔄 [Agora Client] Using native Agora SDK to send message")
            # Directly use Agora SDK to send
            raw_result = await asyncio.to_thread(_SEND_TEXT_TASK, text, target=endpoint)
            
            # Correctly handle SDK returned result, convert to standard format
            if isinstance(raw_result, str):
                response_text = raw_result
            elif hasattr(raw_result, 'content'):
                response_text = str(raw_result.content)
            elif hasattr(raw_result, 'text'):
                response_text = str(raw_result.text)
            else:
                response_text = str(raw_result)
            
            # Return standard format
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
                "error": f"Agora SDK send failed: {type(e).__name__}: {e}",
                "data": {
                    "text": "",
                    "role": "agora_receiver", 
                    "protocol": "agora",
                    "content": ""
                },
                "probe_results": probe_results
            }

    async def spawn(self, role: str, port: int, **kwargs: Any) -> Dict[str, Any]:
        # Use our newly added native ReceiverServer to start
        try:
            import subprocess, sys, os
            from pathlib import Path
            
            # Setup working directory to project root directory
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent.parent  # 5 levels up
            
            env = os.environ.copy()
            env['AGORA_AGENT_NAME'] = f"Agora_Doctor_A" if role.lower() == 'doctor_a' else "Agora_Doctor_B"
            env['AGORA_PORT'] = str(port)
            
            # Capture stderr for debugging
            proc = subprocess.Popen(
                [sys.executable, '-m', 'scenario.safety_tech.protocol_backends.agora.server'], 
                env=env,
                cwd=str(project_root),
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Check if process exits immediately (indicating startup failure)
            try:
                stdout, stderr = proc.communicate(timeout=1.0)
                if proc.returncode != 0:
                    return {
                        "status": "error", 
                        "error": f"Agora server startup failed, exit code: {proc.returncode}, stderr: {stderr}, stdout: {stdout}"
                    }
            except subprocess.TimeoutExpired:
                # Process did not exit immediately, indicating successful startup
                pass
            
            return {"status": "success", "data": {"pid": proc.pid, "port": port}}
        except Exception as e:
            return {"status": "error", "error": f"Failed to spawn Agora server: {e}"}

    async def register(self, agent_id: str, endpoint: str, conversation_id: str, role: str, **kwargs: Any) -> Dict[str, Any]:
        # Interact with RG through existing AgoraRegistrationAdapter
        start_time = time.time()
        try:
            try:
                from scenarios.safety_tech.protocol_backends.agora.registration_adapter import AgoraRegistrationAdapter
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


