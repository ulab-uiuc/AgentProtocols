# -*- coding: utf-8 -*-
"""
ANP 原生客户端：Safety-Tech 中通过本地 HTTP shim 的 /message 入站，
内部由 SimpleNode 使用原生 WS 通道向对端传递并回投 /deliver。
禁止 mock/fallback；仅按真实部署调用。
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import httpx
import subprocess, sys, os, time

try:
    from script.safety_tech.protocol_backends.anp.server import create_doctor_a_server, create_doctor_b_server
except Exception:
    create_doctor_a_server = create_doctor_b_server = None  # type: ignore

try:
    from script.safety_tech.protocol_backends.anp.registration_adapter import ANPRegistrationAdapter
except Exception:
    ANPRegistrationAdapter = None  # type: ignore

from script.safety_tech.protocol_backends.common.interfaces import BaseProtocolBackend


class ANPProtocolBackend(BaseProtocolBackend):
    async def send(self, endpoint: str, payload: Dict[str, Any], correlation_id: Optional[str] = None, probe_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        endpoint = (endpoint or '').rstrip('/')
        txt = _extract_text(payload)
        
        # 优先使用显式correlation_id参数，fallback到payload中的值
        corr = correlation_id or payload.get('correlation_id')
        
        probe_results = {}
        client_kwargs = {}
        
        # 处理探针配置
        if probe_config:
            # TLS降级探针
            if probe_config.get('tls_downgrade'):
                probe_results['tls_downgrade'] = 'attempted'
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
                # 在ANP消息中注入重放标识
                txt = f"[REPLAY:{nonce}:{timestamp}] {txt}"
                
            # 明文嗅探探针
            if probe_config.get('plaintext_sniff'):
                probe_results['plaintext_detected'] = len(txt)
                keywords = probe_config.get('sniff_keywords') or []
                detected_keywords = [kw for kw in keywords if kw.lower() in txt.lower()]
                probe_results['sensitive_keywords_detected'] = detected_keywords
                probe_results['sensitive_data_score'] = len(detected_keywords) * 10
            
            # DID伪造探针（ANP特定）
            if probe_config.get('did_spoofing'):
                fake_did = probe_config.get('fake_did', 'did:fake:anp_test')
                probe_results['did_spoofing'] = 'attempted'
                probe_results['fake_did_used'] = fake_did
                # ANP使用WebSocket，在HTTP握手头中注入伪造身份
                client_kwargs.setdefault('headers', {})['X-ANP-Agent-DID'] = fake_did
                client_kwargs['headers']['X-ANP-Signature'] = probe_config.get('fake_signature', 'fake_anp_sig')
            
            # 会话劫持探针
            if probe_config.get('session_hijack'):
                session_token = probe_config.get('session_token_reuse', 'hijacked_anp_session')
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
            
            # 网络扰动探针（ANP特有的WebSocket连接测试）
            if probe_config.get('network_jitter_ms'):
                import asyncio
                jitter_ms = probe_config.get('network_jitter_ms', 50)
                probe_results['network_jitter'] = f"{jitter_ms}ms"
                # 在发送前添加人工延迟模拟网络抖动
                await asyncio.sleep(jitter_ms / 1000.0)
            
            # E2E负载加密检测探针（新增）
            if probe_config.get('e2e_payload_detection'):
                from script.safety_tech.core.e2e_detector import E2EEncryptionDetector
                e2e_detector = E2EEncryptionDetector(probe_config.get('e2e_watermark', 'ANP_E2E_WATERMARK'))
                
                # 修改原始payload注入水印
                original_payload = {"text": txt, "sender_id": "anp_probe", "receiver_id": "anp_target"}
                modified_payload = e2e_detector.inject_watermark_payload(original_payload)
                txt = modified_payload.get('text', txt)
                
                # 创建明文探测payload
                if probe_config.get('e2e_probe_payload'):
                    probe_payload = e2e_detector.create_plaintext_probe_payload()
                    txt += f" [PROBE_PAYLOAD: {probe_payload['probe_markers']['credit_card']}]"
                
                probe_results['e2e_watermark_injected'] = True
                probe_results['e2e_detector_active'] = True
            
            # 时钟漂移矩阵探针（新增）
            if probe_config.get('time_skew_matrix'):
                import random
                skew_levels = probe_config.get('time_skew_levels', [30, 120, 300, 600])
                probe_results['time_skew_matrix_active'] = True
                probe_results['skew_levels_tested'] = skew_levels
                
                # 随机选择一个漂移档位进行测试
                if skew_levels:
                    selected_skew = random.choice(skew_levels)
                    # 随机选择正负漂移
                    skew_direction = random.choice([-1, 1])
                    actual_skew = selected_skew * skew_direction
                    old_timestamp = time.time() + actual_skew
                    
                    # 时间戳注入到ANP消息中
                    nonce_id = f"skew_{abs(actual_skew)}_{int(time.time())}"
                    txt = f"[TIME_SKEW:{old_timestamp}][NONCE:{nonce_id}] {txt}"
                    
                    probe_results['time_skew_applied'] = actual_skew
                    probe_results['skew_direction'] = 'future' if actual_skew > 0 else 'past'
                    probe_results['skew_nonce'] = nonce_id
                
                # 窗口测试：注入重复/乱序标记
                if probe_config.get('time_skew_window_test'):
                    window_markers = [
                        f"[WINDOW_REPEAT:{int(time.time())}]",
                        f"[WINDOW_DISORDER:{random.randint(1000, 9999)}]",
                        f"[WINDOW_DUPLICATE:MSG_{random.randint(100, 999)}]"
                    ]
                    txt = f"{random.choice(window_markers)} {txt}"
                    probe_results['window_test_marker'] = True
        
        # 使用ANP标准的/runs端点和格式
        anp_payload = {
            "input": {
                "content": [
                    {"type": "text", "text": txt}
                ]
            }
        }
        
        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                # 如果有correlation_id，优先发送到/message端点（兼容现有逻辑）
                if isinstance(corr, str) and corr:
                    message_payload = {
                        "text": txt,
                        "correlation_id": corr
                    }
                    resp = await client.post(f"{endpoint}/message", json=message_payload, timeout=30.0)
                else:
                    # 使用标准ANP /runs端点
                    resp = await client.post(f"{endpoint}/runs", json=anp_payload, timeout=30.0)
                
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
                        "error": f"ANP endpoint returned {resp.status_code}: {resp.text}",
                        "probe_results": probe_results
                    }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "probe_results": probe_results
            }

    async def spawn(self, role: str, port: int, **kwargs: Any) -> Dict[str, Any]:
        """启动ANP服务器进程"""
        try:
            role_l = role.lower()
            
            # 使用Python直接启动服务器，需要设置正确的路径
            import os
            from pathlib import Path
            
            # 找到项目根目录（包含script目录的地方）
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent.parent  # 从protocol_backends/anp/client.py回到项目根目录
            
            code = (
                f"import sys; sys.path.insert(0, '{project_root}');"
                "from script.safety_tech.protocol_backends.anp.server import create_doctor_a_server, create_doctor_b_server;"
                f"server = create_doctor_a_server({port}) if '{role_l}'=='doctor_a' else create_doctor_b_server({port});"
                "server.run()"
            )
            
            # 设置环境变量和工作目录为项目根目录
            env = os.environ.copy()
            
            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                cwd=str(project_root),
                env=env
            )
            
            return {
                "status": "success",
                "data": {
                    "pid": proc.pid,
                    "port": port
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to spawn ANP server: {e}"
            }

    async def register(self, agent_id: str, endpoint: str, conversation_id: str, role: str, **kwargs: Any) -> Dict[str, Any]:
        """注册ANP Agent到注册网关"""
        rg_endpoint = kwargs.get('rg_endpoint') or os.environ.get('RG_ENDPOINT', 'http://127.0.0.1:8001')
        endpoint = (endpoint or '').rstrip('/')
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient() as client:
                # 从服务端获取真实的注册证明
                proof_resp = await client.get(f"{endpoint}/registration_proof", timeout=5.0)
                if proof_resp.status_code != 200:
                    return {
                        "status": "error",
                        "error": f"Failed to get registration proof from {endpoint}: {proof_resp.status_code}"
                    }
                
                proof = proof_resp.json()
                
                # 确保包含所有必需字段
                proof["agent_name"] = agent_id
                
                req = {
                    'protocol': 'anp',
                    'agent_id': agent_id,
                    'endpoint': endpoint,
                    'conversation_id': conversation_id,
                    'role': role,
                    'proof': proof
                }
                
                rr = await client.post(f"{rg_endpoint}/register", json=req, timeout=10.0)
                verification_latency_ms = int((time.time() - start_time) * 1000)
                
                if rr.status_code == 200:
                    result = rr.json()
                    return {
                        "status": "success",
                        "data": {
                            "agent_id": agent_id,
                            "verification_method": "anp_did_proof",
                            "verification_latency_ms": verification_latency_ms,
                            "details": result
                        }
                    }
                else:
                    return {
                        "status": "error",
                        "data": {
                            "agent_id": agent_id,
                            "verification_method": "anp_did_proof",
                            "verification_latency_ms": verification_latency_ms,
                            "details": {"status_code": rr.status_code, "response": rr.text}
                        },
                        "error": f"RG register(anp) failed: {rr.status_code} - {rr.text}"
                    }
        except Exception as e:
            verification_latency_ms = int((time.time() - start_time) * 1000)
            return {
                "status": "error",
                "data": {
                    "agent_id": agent_id,
                    "verification_method": "anp_did_proof",
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


def _extract_text(payload: Dict[str, Any]) -> str:
    if 'text' in payload:
        return str(payload['text'])
    if 'body' in payload:
        return str(payload['body'])
    if 'content' in payload:
        return str(payload['content'])
    return str(payload)


