# -*- coding: utf-8 -*-
"""
ACP 原生客户端：基于 acp-sdk 暴露的 /runs 端点发送消息。
禁止任何 mock/fallback；仅按原生接口调用。
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import httpx
import subprocess
import sys
import os
import time

try:
    from script.safety_tech.protocol_backends.acp.registration_adapter import ACPRegistrationAdapter
except Exception:
    ACPRegistrationAdapter = None  # type: ignore

from script.safety_tech.protocol_backends.common.interfaces import BaseProtocolBackend


class ACPProtocolBackend(BaseProtocolBackend):
    async def send(self, endpoint: str, payload: Dict[str, Any], correlation_id: Optional[str] = None, probe_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        endpoint = (endpoint or '').rstrip('/')

        # 提取文本与关联ID，按ACP SDK RunCreateRequest(messages)格式
        text = _extract_text(payload)
        corr = correlation_id or payload.get('correlation_id')
        if isinstance(corr, str) and corr:
            text = f"[CID:{corr}] {text}"

        probe_results = {}
        client_kwargs = {}
        
        # 处理探针配置
        if probe_config:
            # TLS降级探针
            if probe_config.get('tls_downgrade'):
                probe_results['tls_downgrade'] = 'attempted'
                # 配置弱SSL上下文
                import ssl
                ssl_context = ssl.create_default_context()
                if probe_config.get('tls_version_downgrade') == '1.1':
                    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_1
                    ssl_context.maximum_version = ssl.TLSVersion.TLSv1_1
                elif probe_config.get('tls_version_downgrade') == '1.0':
                    ssl_context.minimum_version = ssl.TLSVersion.TLSv1
                    ssl_context.maximum_version = ssl.TLSVersion.TLSv1
                
                if probe_config.get('cipher_downgrade'):
                    ssl_context.set_ciphers('DES-CBC3-SHA:RC4-MD5:NULL-MD5')
                
                client_kwargs['verify'] = ssl_context
                probe_results['tls_version_used'] = probe_config.get('tls_version_downgrade', 'default')
            
            # 重放攻击探针
            if probe_config.get('replay_attack'):
                nonce = probe_config.get('replay_nonce', 'default_nonce')
                timestamp = probe_config.get('replay_timestamp', time.time() - 300)
                probe_results['replay_attack'] = f"nonce_{nonce}_ts_{timestamp}"
                # 在ACP消息中注入重放标识
                text = f"[REPLAY:{nonce}:{timestamp}] {text}"
            
            # 明文嗅探探针
            if probe_config.get('plaintext_sniff'):
                probe_results['plaintext_detected'] = len(text)
                keywords = probe_config.get('sniff_keywords') or []
                detected_keywords = [kw for kw in keywords if kw.lower() in text.lower()]
                probe_results['sensitive_keywords_detected'] = detected_keywords
                probe_results['sensitive_data_score'] = len(detected_keywords) * 10
            
            # DID伪造探针（ACP特定）
            if probe_config.get('did_spoofing'):
                fake_did = probe_config.get('fake_did', 'did:fake:acp_test')
                probe_results['did_spoofing'] = 'attempted'
                probe_results['fake_did_used'] = fake_did
                # 在ACP请求头中注入伪造身份
                client_kwargs.setdefault('headers', {})['X-ACP-Agent-DID'] = fake_did
                client_kwargs['headers']['X-ACP-Signature'] = probe_config.get('fake_signature', 'fake_acp_sig')
            
            # 会话劫持探针
            if probe_config.get('session_hijack'):
                session_token = probe_config.get('session_token_reuse', 'hijacked_acp_session')
                probe_results['session_hijack'] = 'attempted'
                probe_results['hijacked_token'] = session_token
                client_kwargs.setdefault('headers', {})['Authorization'] = f"Bearer {session_token}"
            
            # MITM代理探针
            if probe_config.get('mitm_proxy'):
                proxy_endpoint = probe_config.get('mitm_proxy_endpoint', 'http://127.0.0.1:8080')
                probe_results['mitm_proxy'] = 'attempted'
                probe_results['proxy_used'] = proxy_endpoint
                client_kwargs['proxies'] = {
                    'http://': proxy_endpoint,
                    'https://': proxy_endpoint
                }

        req = {
            "input": {
                "content": [
                    {"type": "text", "text": text}
                ]
            }
        }
        
        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                resp = await client.post(f"{endpoint}/runs", json=req, timeout=30.0)
                if resp.status_code in (200, 202):
                    try:
                        result = resp.json()
                        return {
                            "status": "success",
                            "data": result,
                            "probe_results": probe_results
                        }
                    except Exception:
                        return {
                            "status": "success",
                            "data": {"status": "ok"},
                            "probe_results": probe_results
                        }
                else:
                    return {
                        "status": "error",
                        "error": f"ACP endpoint returned {resp.status_code}: {resp.text}",
                        "probe_results": probe_results
                    }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "probe_results": probe_results
            }

    async def spawn(self, role: str, port: int, **kwargs: Any) -> Dict[str, Any]:
        """启动原生 acp-sdk 服务器。
        改为通过内联 -c 显式调用 create_doctor_*_server(port).run()，确保名称与端口绑定正确。
        """
        try:
            env = os.environ.copy()
            coord = kwargs.get('coord_endpoint') or os.environ.get('COORD_ENDPOINT', 'http://127.0.0.1:8888')
            env['COORD_ENDPOINT'] = coord
            role_l = role.lower()
            if role_l not in ('doctor_a', 'doctor_b'):
                return {"status": "error", "error": f"unknown role: {role}"}
            if role_l == 'doctor_a':
                code = (
                    "import sys;"
                    "from script.safety_tech.protocol_backends.acp.server import create_doctor_a_server;"
                    f"server = create_doctor_a_server({port});"
                    "server.run()"
                )
            else:  # doctor_b
                code = (
                    "import sys;"
                    "from script.safety_tech.protocol_backends.acp.server import create_doctor_b_server;"
                    f"server = create_doctor_b_server({port});"
                    "server.run()"
                )
            proc = subprocess.Popen([sys.executable, '-c', code], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            return {"status": "success", "data": {"pid": proc.pid, "port": port}}
        except Exception as e:
            return {"status": "error", "error": f"Failed to spawn ACP server: {e}"}

    async def register(self, agent_id: str, endpoint: str, conversation_id: str, role: str, **kwargs: Any) -> Dict[str, Any]:
        """使用 ACPRegistrationAdapter 向RG注册。"""
        start_time = time.time()
        try:
            if ACPRegistrationAdapter is None:
                return {
                    "status": "error",
                    "error": "ACPRegistrationAdapter not available"
                }
            rg_endpoint = kwargs.get('rg_endpoint') or os.environ.get('RG_ENDPOINT', 'http://127.0.0.1:8001')
            adapter = ACPRegistrationAdapter({'rg_endpoint': rg_endpoint})
            resp = await adapter.register_agent(agent_id, endpoint, conversation_id, role, acp_probe_endpoint=endpoint)
            verification_latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                "status": "success",
                "data": {
                    "agent_id": agent_id,
                    "verification_method": "acp_proof",
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
                    "verification_method": "acp_proof",
                    "verification_latency_ms": verification_latency_ms,
                    "details": {}
                },
                "error": str(e)
            }

    async def health(self, endpoint: str) -> Dict[str, Any]:
        url = (endpoint or '').rstrip('/')
        start_time = time.time()
        try:
            async with httpx.AsyncClient() as client:
                # 优先 /agents
                r = await client.get(f"{url}/agents", timeout=5.0)
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


def _extract_text(payload: Dict[str, Any]) -> str:
    if 'text' in payload:
        return str(payload['text'])
    if 'body' in payload:
        return str(payload['body'])
    if 'content' in payload:
        return str(payload['content'])
    return str(payload)


