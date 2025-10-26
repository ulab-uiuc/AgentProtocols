# -*- coding: utf-8 -*-
"""
A2A native client: sends through a2a server's /message endpoint.
No mock/fallback allowed; maintains a2a payload structure.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import httpx
import subprocess, sys, os, time

try:
    from scenarios.safety_tech.protocol_backends.a2a.registration_adapter import A2ARegistrationAdapter
except Exception:
    A2ARegistrationAdapter = None  # type: ignore

from scenarios.safety_tech.protocol_backends.common.interfaces import BaseProtocolBackend


class A2AProtocolBackend(BaseProtocolBackend):
    async def send(self, endpoint: str, payload: Dict[str, Any], correlation_id: Optional[str] = None, probe_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        endpoint = (endpoint or '').rstrip('/')
        txt = _extract_text(payload)
        corr = correlation_id or payload.get('correlation_id')
        if isinstance(corr, str) and corr:
            txt = f"[CID:{corr}] {txt}"

        probe_results = {}
        client_kwargs = {}
        
        # First create basic message structure
        send_payload: Dict[str, Any] = {
            "params": {
                "message": {
                    "text": txt,
                    "role": "user",
                    "parts": [{"type": "text", "text": txt}],
                    "meta": {"sender_id": payload.get('sender_id')}
                }
            }
        }
        
        # Process probe configuration
        if probe_config:
            # TLS downgrade probe
            if probe_config.get('tls_downgrade'):
                probe_results['tls_downgrade'] = 'attempted'
                # Configure weak SSL context
                import ssl
                ssl_context = ssl.create_default_context()
                if probe_config.get('tls_version_downgrade') == '1.1':
                    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_1
                    ssl_context.maximum_version = ssl.TLSVersion.TLSv1_1
                elif probe_config.get('tls_version_downgrade') == '1.0':
                    ssl_context.minimum_version = ssl.TLSVersion.TLSv1
                    ssl_context.maximum_version = ssl.TLSVersion.TLSv1
                
                # Setup weak cipher suite
                if probe_config.get('cipher_downgrade'):
                    ssl_context.set_ciphers('DES-CBC3-SHA:RC4-MD5:NULL-MD5')
                
                client_kwargs['verify'] = ssl_context
                probe_results['tls_version_used'] = probe_config.get('tls_version_downgrade', 'default')
            
            # Replay attack probe
            if probe_config.get('replay_attack'):
                nonce = probe_config.get('replay_nonce', 'default_nonce')
                timestamp = probe_config.get('replay_timestamp', time.time() - 300)
                probe_results['replay_attack'] = f"nonce_{nonce}_ts_{timestamp}"
                # Inject replay identifier in message
                txt = f"[REPLAY:{nonce}:{timestamp}] {txt}"
            
            # Plaintext sniffing probe
            if probe_config.get('plaintext_sniff'):
                probe_results['plaintext_detected'] = len(txt)
                keywords = probe_config.get('sniff_keywords') or []
                detected_keywords = [kw for kw in keywords if kw.lower() in txt.lower()]
                probe_results['sensitive_keywords_detected'] = detected_keywords
                probe_results['sensitive_data_score'] = len(detected_keywords) * 10
            
            # DID spoofing probe
            if probe_config.get('did_spoofing'):
                fake_did = probe_config.get('fake_did', 'did:fake:a2a_test')
                fake_signature = probe_config.get('fake_signature', 'fake_sig_data')
                probe_results['did_spoofing'] = 'attempted'
                probe_results['fake_did_used'] = fake_did
                # Inject forged identity in message metadata
                send_payload["params"]["message"]["meta"]["fake_did"] = fake_did
                send_payload["params"]["message"]["meta"]["fake_signature"] = fake_signature
            
            # Session hijacking probe
            if probe_config.get('session_hijack'):
                session_token = probe_config.get('session_token_reuse', 'hijacked_session_token')
                probe_results['session_hijack'] = 'attempted'
                probe_results['hijacked_token'] = session_token
                # Inject session token in request header
                client_kwargs.setdefault('headers', {})['Authorization'] = f"Bearer {session_token}"
            
            # E2E payload encryption detection probe (newly added)
            if probe_config.get('e2e_payload_detection'):
                from scenarios.safety_tech.core.e2e_detector import E2EEncryptionDetector
                e2e_detector = E2EEncryptionDetector(probe_config.get('e2e_watermark', 'A2A_E2E_WATERMARK'))
                
                # Modify original payload to inject watermark
                original_payload = {"text": txt, "sender_id": "a2a_probe", "receiver_id": "a2a_target"}
                modified_payload = e2e_detector.inject_watermark_payload(original_payload)
                txt = modified_payload.get('text', txt)
                
                # Create plaintext detection payload
                if probe_config.get('e2e_probe_payload'):
                    probe_payload = e2e_detector.create_plaintext_probe_payload()
                    txt += f" [PROBE_PAYLOAD: {probe_payload['probe_markers']['credit_card']}]"
                
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
                    
                    # Inject timestamp into A2A message
                    nonce_id = f"skew_{abs(actual_skew)}_{int(time.time())}"
                    txt = f"[TIME_SKEW:{old_timestamp}][NONCE:{nonce_id}] {txt}"
                    
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
                    txt = f"{random.choice(window_markers)} {txt}"
                    probe_results['window_test_marker'] = True

        # Update message text (may have been modified by probes)
        send_payload["params"]["message"]["text"] = txt
        send_payload["params"]["message"]["parts"] = [{"type": "text", "text": txt}]

        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                resp = await client.post(f"{endpoint}/message", json=send_payload, timeout=30.0)
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
                        "error": f"A2A endpoint returned {resp.status_code}: {resp.text}",
                        "probe_results": probe_results
                    }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "probe_results": probe_results
            }

    async def spawn(self, role: str, port: int, **kwargs: Any) -> Dict[str, Any]:
        """Start A2A server process"""
        try:
            env = os.environ.copy()
            env['COORD_ENDPOINT'] = kwargs.get('coord_endpoint') or os.environ.get('COORD_ENDPOINT', 'http://127.0.0.1:8888')
            
            # Use direct code execution method, similar to ANP fix
            from pathlib import Path
            
            # Find project root directory (where script directory is located)
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent.parent  # From protocol_backends/a2a/client.py back to project root directory
            
            code = (
                f"import sys; sys.path.insert(0, '{project_root}');"
                "from scenarios.safety_tech.protocol_backends.a2a.server import run_server;"
                f"run_server('A2A_Doctor_A' if '{role.lower()}'=='doctor_a' else 'A2A_Doctor_B', {port})"
            )
            
            if role.lower() == 'doctor_a':
                env['A2A_A_PORT'] = str(port)
            else:
                env['A2A_B_PORT'] = str(port)
            
            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                cwd=str(project_root),
                env=env
            )
            return {"status": "success", "data": {"pid": proc.pid, "port": port}}
        except Exception as e:
            return {"status": "error", "error": f"Failed to spawn A2A server: {e}"}

    async def register(self, agent_id: str, endpoint: str, conversation_id: str, role: str, **kwargs: Any) -> Dict[str, Any]:
        start_time = time.time()
        try:
            if A2ARegistrationAdapter is None:
                return {
                    "status": "error",
                    "error": "A2ARegistrationAdapter not available"
                }
            rg_endpoint = kwargs.get('rg_endpoint') or os.environ.get('RG_ENDPOINT', 'http://127.0.0.1:8001')
            adapter = A2ARegistrationAdapter({'rg_endpoint': rg_endpoint})
            resp = await adapter.register_agent(agent_id, endpoint, conversation_id, role)
            verification_latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                "status": "success",
                "data": {
                    "agent_id": agent_id,
                    "verification_method": "a2a_proof",
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
                    "verification_method": "a2a_proof",
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


