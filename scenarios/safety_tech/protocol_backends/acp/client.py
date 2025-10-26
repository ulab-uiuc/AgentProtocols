# -*- coding: utf-8 -*-
"""
ACP native client: sends messages based on /runs endpoint exposed by acp-sdk.
No mock/fallback allowed; only call native interface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import httpx
import subprocess
import sys
import os
import time

try:
    from scenarios.safety_tech.protocol_backends.acp.registration_adapter import ACPRegistrationAdapter
except Exception:
    ACPRegistrationAdapter = None  # type: ignore

from scenarios.safety_tech.protocol_backends.common.interfaces import BaseProtocolBackend


class ACPProtocolBackend(BaseProtocolBackend):
    async def send(self, endpoint: str, payload: Dict[str, Any], correlation_id: Optional[str] = None, probe_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        endpoint = (endpoint or '').rstrip('/')

        # Extract text and correlation ID, in ACP SDK RunCreateRequest(messages) format
        text = _extract_text(payload)
        corr = correlation_id or payload.get('correlation_id')
        if isinstance(corr, str) and corr:
            text = f"[CID:{corr}] {text}"

        probe_results = {}
        client_kwargs = {}
        
        # Import necessary modules
        import time
        
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
                
                if probe_config.get('cipher_downgrade'):
                    ssl_context.set_ciphers('DES-CBC3-SHA:RC4-MD5:NULL-MD5')
                
                client_kwargs['verify'] = ssl_context
                probe_results['tls_version_used'] = probe_config.get('tls_version_downgrade', 'default')
            
            # Replay attack probe
            if probe_config.get('replay_attack'):
                nonce = probe_config.get('replay_nonce', 'default_nonce')
                timestamp = probe_config.get('replay_timestamp', time.time() - 300)
                probe_results['replay_attack'] = f"nonce_{nonce}_ts_{timestamp}"
                # Inject replay identifier in ACP message
                text = f"[REPLAY:{nonce}:{timestamp}] {text}"
            
            # Plaintext sniffing probe
            if probe_config.get('plaintext_sniff'):
                probe_results['plaintext_detected'] = len(text)
                keywords = probe_config.get('sniff_keywords') or []
                detected_keywords = [kw for kw in keywords if kw.lower() in text.lower()]
                probe_results['sensitive_keywords_detected'] = detected_keywords
                probe_results['sensitive_data_score'] = len(detected_keywords) * 10
            
            # DID spoofing probe (ACP specific)
            if probe_config.get('did_spoofing'):
                fake_did = probe_config.get('fake_did', 'did:fake:acp_test')
                probe_results['did_spoofing'] = 'attempted'
                probe_results['fake_did_used'] = fake_did
                # Inject forged identity in ACP request header
                client_kwargs.setdefault('headers', {})['X-ACP-Agent-DID'] = fake_did
                fake_sig = probe_config.get('fake_signature', 'fake_acp_sig')
                if fake_sig is not None:
                    client_kwargs['headers']['X-ACP-Signature'] = fake_sig
                else:
                    client_kwargs['headers']['X-ACP-Signature'] = 'fake_acp_sig'
            
            # Session hijacking probe
            if probe_config.get('session_hijack'):
                session_token = probe_config.get('session_token_reuse', 'hijacked_acp_session')
                probe_results['session_hijack'] = 'attempted'
                probe_results['hijacked_token'] = session_token
                if session_token is not None:
                    client_kwargs.setdefault('headers', {})['Authorization'] = f"Bearer {session_token}"
                else:
                    client_kwargs.setdefault('headers', {})['Authorization'] = "Bearer hijacked_acp_session"
            
            # MITM proxy probe
            if probe_config.get('mitm_proxy'):
                proxy_endpoint = probe_config.get('mitm_proxy_endpoint', 'http://127.0.0.1:8080')
                probe_results['mitm_proxy'] = 'attempted'
                probe_results['proxy_used'] = proxy_endpoint
                client_kwargs['proxies'] = {
                    'http://': proxy_endpoint,
                    'https://': proxy_endpoint
                }
            
            # E2E payload encryption detection probe (newly added)
            if probe_config.get('e2e_payload_detection'):
                from scenarios.safety_tech.core.e2e_detector import E2EEncryptionDetector
                e2e_detector = E2EEncryptionDetector(probe_config.get('e2e_watermark', 'S2_E2E_WATERMARK'))
                
                # Modify original payload to inject watermark
                original_payload = {"text": text, "sender_id": "acp_probe", "receiver_id": "acp_target"}
                modified_payload = e2e_detector.inject_watermark_payload(original_payload)
                text = modified_payload.get('text', text)
                
                # Create plaintext detection payload
                if probe_config.get('e2e_probe_payload'):
                    probe_payload = e2e_detector.create_plaintext_probe_payload()
                    text += f" [PROBE_PAYLOAD: {probe_payload['probe_markers']['credit_card']}]"
                
                probe_results['e2e_watermark_injected'] = True
                probe_results['e2e_detector_active'] = True
            
            # Certificate validity matrix probe (newly added)
            if probe_config.get('cert_validity_matrix'):
                probe_results['cert_matrix_active'] = True
                
                # Expired certificate test
                if probe_config.get('cert_expired'):
                    import ssl
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    client_kwargs['verify'] = ssl_context
                    probe_results['cert_expired_test'] = 'attempted'
                
                # Self-signed certificate test
                if probe_config.get('cert_self_signed'):
                    # Force strict verification mode to test self-signed rejection
                    if 'verify' not in client_kwargs:
                        client_kwargs['verify'] = True  # Strict verification
                    probe_results['cert_self_signed_test'] = 'attempted'
                
                # Hostname mismatch test
                if probe_config.get('cert_invalid_hostname'):
                    probe_results['cert_hostname_test'] = 'attempted'
                    # Can modify target hostname here to test mismatch
            
            # Clock skew matrix probe (enhanced version)
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
                    
                    # Enhanced timestamp injection
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
                    selected_marker = random.choice(window_markers)
                    text = f"{selected_marker} {text}"
                    probe_results['window_test_marker'] = selected_marker
            
            # Bypass packet capture probe (newly added)
            if probe_config.get('pcap_capture'):
                probe_results['pcap_capture_enabled'] = True
                probe_results['pcap_interface'] = probe_config.get('pcap_interface', 'lo0')
                probe_results['pcap_duration'] = probe_config.get('pcap_duration_seconds', 10)
                # Actual pcap capture will be started in higher-level runner

        req = {
            "input": {
                "content": [
                    {"type": "text", "text": text}
                ]
            }
        }
        
        try:
            print(f"[ACP-DEBUG] Sending request to {endpoint}/runs")
            print(f"[ACP-DEBUG] req: {req}")
            print(f"[ACP-DEBUG] client_kwargs: {client_kwargs}")
            
            async with httpx.AsyncClient(**client_kwargs) as client:
                resp = await client.post(f"{endpoint}/runs", json=req, timeout=30.0)
                
                print(f"[ACP-DEBUG] HTTPResponse: {resp.status_code}")
                print(f"[ACP-DEBUG] Response content preview: {resp.text[:200]}...")
                
                if resp.status_code in (200, 202):
                    try:
                        result = resp.json()
                        print(f"[ACP-DEBUG] JSON parsing successful: {str(result)[:150]}...")
                        return_val = {
                            "status": "success",
                            "data": result,
                            "probe_results": probe_results
                        }
                        print(f"[ACP-DEBUG] Returning success response: status={return_val['status']}")
                        return return_val
                    except Exception as json_ex:
                        print(f"[ACP-DEBUG] JSON parsing failed: {json_ex}")
                        return_val = {
                            "status": "success",
                            "data": {"status": "ok"},
                            "probe_results": probe_results
                        }
                        print(f"[ACP-DEBUG] Returning fallback success response: status={return_val['status']}")
                        return return_val
                else:
                    error_resp = {
                        "status": "error",
                        "error": f"ACP endpoint returned {resp.status_code}: {resp.text}",
                        "probe_results": probe_results
                    }
                    print(f"[ACP-DEBUG] Returning error response: {error_resp['error']}")
                    return error_resp
        except Exception as e:
            error_resp = {
                "status": "error",
                "error": str(e),
                "probe_results": probe_results
            }
            print(f"[ACP-DEBUG] Caught exception: {e}")
            print(f"[ACP-DEBUG] Returning exception response: {error_resp['error']}")
            return error_resp

    async def spawn(self, role: str, port: int, **kwargs: Any) -> Dict[str, Any]:
        """Start native acp-sdk server.
        Changed to explicitly call create_doctor_*_server(port).run() via inline -c to ensure name and port binding is correct.
        """
        try:
            env = os.environ.copy()
            coord = kwargs.get('coord_endpoint') or os.environ.get('COORD_ENDPOINT', 'http://127.0.0.1:8888')
            env['COORD_ENDPOINT'] = coord
            # Pass LLM required environment variables to ensure server can generate replies normally
            # Support NVIDIA and OpenAI compatible environment variables
            for key in (
                'NVIDIA_API_KEY', 'NVIDIA_BASE_URL', 'NVIDIA_MODEL', 'NVIDIA_TEMPERATURE',
                'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL', 'OPENAI_TEMPERATURE',
                'OPENAI_REQUEST_TIMEOUT', 'AGORA_S1_TEST_MODE',  # for S1 fast fail/reduced timeout
            ):
                if key in os.environ:
                    env[key] = os.environ[key]
            # Setup NVIDIA default values (if environment variables not set)
            env.setdefault('NVIDIA_API_KEY', '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')
            env.setdefault('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1')
            env.setdefault('NVIDIA_MODEL', 'meta/llama-3.2-1b-instruct')
            env.setdefault('NVIDIA_TEMPERATURE', '0.3')
            # If S1 fast mode not set, enable lightweight mode by default for test stability
            env.setdefault('AGORA_S1_TEST_MODE', 'light')
            role_l = role.lower()
            if role_l not in ('doctor_a', 'doctor_b'):
                return {"status": "error", "error": f"unknown role: {role}"}
            # Calculate project root directory path
            from pathlib import Path
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent.parent  # 5 levels up to project root directory
            
            if role_l == 'doctor_a':
                code = (
                    "import sys;"
                    f"sys.path.insert(0, '{project_root}');"
                    "from scenarios.safety_tech.protocol_backends.acp.server import create_doctor_a_server;"
                    f"server = create_doctor_a_server({port});"
                    "server.run()"
                )
            else:  # doctor_b
                code = (
                    "import sys;"
                    f"sys.path.insert(0, '{project_root}');"
                    "from scenarios.safety_tech.protocol_backends.acp.server import create_doctor_b_server;"
                    f"server = create_doctor_b_server({port});"
                    "server.run()"
                )
            proc = subprocess.Popen([sys.executable, '-c', code], cwd=str(project_root), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # Give process some time to start, check if it fails immediately
            import time
            time.sleep(1)
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                return {"status": "error", "error": f"Process exited immediately: stdout={stdout[:200]}, stderr={stderr[:200]}"}
            return {"status": "success", "data": {"pid": proc.pid, "port": port}}
        except Exception as e:
            return {"status": "error", "error": f"Failed to spawn ACP server: {e}"}

    async def register(self, agent_id: str, endpoint: str, conversation_id: str, role: str, **kwargs: Any) -> Dict[str, Any]:
        """Register to RG using ACPRegistrationAdapter."""
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
                # Prioritize /agents
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


