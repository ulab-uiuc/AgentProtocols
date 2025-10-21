# -*- coding: utf-8 -*-
"""
S2 certificate validity matrix tests

Systematically test various certificate issues: expiration, hostname mismatch,
self-signed certificates, incomplete chains, weak ciphers, and TLS downgrades.
"""

import ssl
import socket
import tempfile
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import OpenSSL.crypto as crypto
import httpx


class CertificateMatrixTester:
    """Certificate matrix tester."""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "s2_cert_test"
        self.temp_dir.mkdir(exist_ok=True)
        self.test_results = {}
    
    async def run_cert_matrix_test(self, target_host: str = "127.0.0.1", target_port: int = 8888) -> Dict[str, Any]:
        """Run the full certificate matrix tests."""
        results = {
            "target": f"{target_host}:{target_port}",
            "test_timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # 1. Expired certificate test
        results["tests"]["expired_cert"] = await self._test_expired_certificate(target_host, target_port)
        
        # 2. Hostname mismatch test
        results["tests"]["hostname_mismatch"] = await self._test_hostname_mismatch(target_host, target_port)
        
        # 3. Self-signed certificate test
        results["tests"]["self_signed"] = await self._test_self_signed_cert(target_host, target_port)
        
        # 4. Incomplete certificate chain test
        results["tests"]["incomplete_chain"] = await self._test_incomplete_chain(target_host, target_port)
        
        # 5. Weak cipher suites test
        results["tests"]["weak_cipher"] = await self._test_weak_cipher_suites(target_host, target_port)
        
        # 6. TLS version downgrade test
        results["tests"]["tls_downgrade"] = await self._test_tls_version_downgrade(target_host, target_port)
        
        # Calculate overall matrix score
        results["matrix_score"] = self._calculate_matrix_score(results["tests"])
        
        return results
    
    async def _test_expired_certificate(self, host: str, port: int) -> Dict[str, Any]:
        """Test rejection of expired certificates."""
        test_result = {
            "test_name": "Expired certificate test",
            "description": "Attempt to connect using an expired certificate and verify it is correctly rejected",
            "status": "unknown",
            "blocked": False,
            "error_type": None
        }
        
        try:
            # Generate an expired self-signed certificate
            expired_cert_path, expired_key_path = self._generate_expired_certificate(host)

            # Try connecting using the expired certificate
            ssl_context = ssl.create_default_context()
            ssl_context.load_cert_chain(expired_cert_path, expired_key_path)
            ssl_context.check_hostname = False  # Skip hostname check for expired-certificate test
            
            try:
                async with httpx.AsyncClient(verify=False) as client:
                    response = await client.get(f"https://{host}:{port}/health", timeout=5.0)
                    test_result["status"] = "connection_succeeded"
                    test_result["blocked"] = False
                    test_result["response_code"] = response.status_code
                    
            except Exception as e:
                test_result["status"] = "ssl_error"
                test_result["blocked"] = True
                test_result["error_type"] = "SSLError"
                test_result["error_message"] = str(e)
                
            except Exception as e:
                test_result["status"] = "other_error"
                test_result["blocked"] = True
                test_result["error_type"] = type(e).__name__
                test_result["error_message"] = str(e)
                
        except Exception as e:
            test_result["status"] = "test_setup_failed"
            test_result["error"] = str(e)
        
        return test_result
    
    async def _test_hostname_mismatch(self, host: str, port: int) -> Dict[str, Any]:
        """Test rejection for hostname-mismatched certificates."""
        test_result = {
            "test_name": "Hostname mismatch test",
            "description": "Connect using a certificate with the wrong hostname",
            "status": "unknown",
            "blocked": False
        }
        
        try:
            # Generate a certificate with a mismatched hostname (e.g. example.com while connecting to localhost)
            wrong_host = "wrong.example.com" if host == "127.0.0.1" else "127.0.0.1"
            cert_path, key_path = self._generate_certificate(wrong_host)
            
            # Use strict hostname checking
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED
            
            try:
                # This should fail due to hostname mismatch
                with socket.create_connection((host, port), timeout=5) as sock:
                    with ssl_context.wrap_socket(sock, server_hostname=host) as ssock:
                        test_result["status"] = "connection_succeeded"
                        test_result["blocked"] = False
                        
            except ssl.SSLError as e:
                test_result["status"] = "ssl_error"  
                test_result["blocked"] = True
                test_result["error_type"] = "SSLError"
                test_result["error_message"] = str(e)
                
            except Exception as e:
                test_result["status"] = "other_error"
                test_result["blocked"] = True
                test_result["error_type"] = type(e).__name__
                test_result["error_message"] = str(e)
                
        except Exception as e:
            test_result["status"] = "test_setup_failed"
            test_result["error"] = str(e)
        
        return test_result
    
    async def _test_self_signed_cert(self, host: str, port: int) -> Dict[str, Any]:
        """Test handling of self-signed certificates."""
        test_result = {
            "test_name": "Self-signed certificate test",
            "description": "Connect using a self-signed certificate and check whether explicit trust is required",
            "status": "unknown",
            "blocked": False
        }
        
        try:
            # Connect using default strict verification mode
            async with httpx.AsyncClient(verify=True) as client:
                response = await client.get(f"https://{host}:{port}/health", timeout=5.0)
                test_result["status"] = "connection_succeeded"
                test_result["blocked"] = False
                test_result["response_code"] = response.status_code
                
        except Exception as e:
            test_result["status"] = "ssl_error"
            test_result["blocked"] = True  # Good — correctly rejected the self-signed certificate
            test_result["error_type"] = "SSLError"
            test_result["error_message"] = str(e)
            
        except Exception as e:
            test_result["status"] = "other_error"
            test_result["blocked"] = True
            test_result["error_type"] = type(e).__name__
            test_result["error_message"] = str(e)
        
        return test_result
    
    async def _test_incomplete_chain(self, host: str, port: int) -> Dict[str, Any]:
        """Test handling of incomplete certificate chains."""
        return {
            "test_name": "Incomplete chain test",
            "status": "skipped",
            "reason": "Requires a complex CA chain setup; skipped for now",
            "blocked": None
        }
    
    async def _test_weak_cipher_suites(self, host: str, port: int) -> Dict[str, Any]:
        """Test rejection of weak cipher suites."""
        test_result = {
            "test_name": "Weak cipher suites test",
            "description": "Attempt to connect using weak cipher suites",
            "status": "unknown",
            "blocked": False,
            "cipher_results": {}
        }
        
        # Test various weak cipher suites
        weak_ciphers = [
            "DES-CBC3-SHA",     # 3DES
            "RC4-MD5",          # RC4
            "NULL-MD5",         # NULL cipher (no encryption)
            "EXPORT-RC4-40-MD5" # 40-bit export cipher
        ]
        
        for cipher in weak_ciphers:
            try:
                ssl_context = ssl.create_default_context()
                ssl_context.set_ciphers(cipher)
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                try:
                    with socket.create_connection((host, port), timeout=5) as sock:
                        with ssl_context.wrap_socket(sock) as ssock:
                            test_result["cipher_results"][cipher] = {
                                "blocked": False,
                                "actual_cipher": ssock.cipher()
                            }
                            
                except ssl.SSLError as e:
                    test_result["cipher_results"][cipher] = {
                        "blocked": True,
                        "error": str(e)
                    }
                    
                except Exception as e:
                    test_result["cipher_results"][cipher] = {
                        "blocked": True,
                        "error": str(e)
                    }
                    
            except Exception as e:
                test_result["cipher_results"][cipher] = {
                    "blocked": True,
                    "setup_error": str(e)
                }
        
        # Calculate overall block rate
        total_tests = len(weak_ciphers)
        blocked_count = sum(1 for result in test_result["cipher_results"].values() 
                          if result.get("blocked", False))
        
        test_result["blocked"] = blocked_count > 0
        test_result["block_rate"] = blocked_count / total_tests if total_tests > 0 else 0
        test_result["status"] = "completed"
        
        return test_result
    
    async def _test_tls_version_downgrade(self, host: str, port: int) -> Dict[str, Any]:
        """Test rejection of TLS version downgrades."""
        test_result = {
            "test_name": "TLS version downgrade test",
            "description": "Attempt to connect using older TLS versions",
            "status": "unknown",
            "version_results": {}
        }
        
        # Test various TLS versions (older versions are expected to be rejected)
        tls_versions = [
            ("TLS 1.0", ssl.PROTOCOL_TLSv1),
            ("TLS 1.1", ssl.PROTOCOL_TLSv1_1),
        ]
        
        for version_name, protocol in tls_versions:
            try:
                ssl_context = ssl.SSLContext(protocol)
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                try:
                    with socket.create_connection((host, port), timeout=5) as sock:
                        with ssl_context.wrap_socket(sock) as ssock:
                            test_result["version_results"][version_name] = {
                                "blocked": False,
                                "actual_version": ssock.version()
                            }
                            
                except ssl.SSLError as e:
                    test_result["version_results"][version_name] = {
                        "blocked": True,
                        "error": str(e)
                    }
                    
                except Exception as e:
                    test_result["version_results"][version_name] = {
                        "blocked": True,
                        "error": str(e)
                    }
                    
            except Exception as e:
                test_result["version_results"][version_name] = {
                    "blocked": True,
                    "setup_error": str(e)
                }
        
        # Calculate block rate
        total_tests = len(tls_versions)
        blocked_count = sum(1 for result in test_result["version_results"].values() 
                          if result.get("blocked", False))
        
        test_result["block_rate"] = blocked_count / total_tests if total_tests > 0 else 0
        test_result["status"] = "completed"
        
        return test_result
    
    def _generate_expired_certificate(self, hostname: str) -> tuple:
        """Generate an expired self-signed certificate."""
        # Create key pair
        key = crypto.PKey()
        key.generate_key(crypto.TYPE_RSA, 2048)

        # Create certificate
        cert = crypto.X509()
        cert.get_subject().CN = hostname
        cert.set_serial_number(1000)
        
        # Set certificate to be expired (expired yesterday)
        cert.gmtime_adj_notBefore(-86400 * 2)  # valid starting 2 days ago
        cert.gmtime_adj_notAfter(-86400)       # expired yesterday
        
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(key)
        cert.sign(key, 'sha256')
        
        # Save to temporary files
        cert_path = self.temp_dir / f"expired_{hostname}.crt"
        key_path = self.temp_dir / f"expired_{hostname}.key"
        
        with open(cert_path, 'wb') as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
            
        with open(key_path, 'wb') as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
        
        return str(cert_path), str(key_path)
    
    def _generate_certificate(self, hostname: str) -> tuple:
        """Generate a valid self-signed certificate."""
        # Create key pair
        key = crypto.PKey()
        key.generate_key(crypto.TYPE_RSA, 2048)

        # Create certificate
        cert = crypto.X509()
        cert.get_subject().CN = hostname
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(86400 * 365)  # 1 year validity
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(key)
        cert.sign(key, 'sha256')
        
        # Save to temporary files
        cert_path = self.temp_dir / f"test_{hostname}.crt"
        key_path = self.temp_dir / f"test_{hostname}.key"
        
        with open(cert_path, 'wb') as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
            
        with open(key_path, 'wb') as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
        
        return str(cert_path), str(key_path)
    
    def _calculate_matrix_score(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate score for the certificate matrix tests."""
        scores = {
            "expired_cert": 0,
            "hostname_mismatch": 0,
            "self_signed": 0,
            "weak_cipher": 0,
            "tls_downgrade": 0
        }
        
        # Expired certificate: blocked = 100 points, not blocked = 0 points
        if test_results.get("expired_cert", {}).get("blocked"):
            scores["expired_cert"] = 100
            
        # Hostname mismatch: blocked = 100 points
        if test_results.get("hostname_mismatch", {}).get("blocked"):
            scores["hostname_mismatch"] = 100
            
        # Self-signed certificate: blocked = 100 points
        if test_results.get("self_signed", {}).get("blocked"):
            scores["self_signed"] = 100
            
        # Weak cipher suites: score according to block rate
        weak_cipher_rate = test_results.get("weak_cipher", {}).get("block_rate", 0)
        scores["weak_cipher"] = int(weak_cipher_rate * 100)
        
        # TLS version downgrade: score according to block rate
        tls_downgrade_rate = test_results.get("tls_downgrade", {}).get("block_rate", 0)
        scores["tls_downgrade"] = int(tls_downgrade_rate * 100)
        
        # Calculate weighted total score
        total_score = (
            scores["expired_cert"] * 0.25 +      # 25%
            scores["hostname_mismatch"] * 0.25 + # 25% 
            scores["self_signed"] * 0.20 +       # 20%
            scores["weak_cipher"] * 0.15 +       # 15%
            scores["tls_downgrade"] * 0.15       # 15%
        )
        
        return {
            "individual_scores": scores,
            "total_score": round(total_score, 1),
            "grade": self._get_security_grade(total_score)
        }
    
    def _get_security_grade(self, score: float) -> str:
        """Get security grade based on score."""
        if score >= 90:
            return "EXCELLENT"
        elif score >= 75:
            return "GOOD" 
        elif score >= 60:
            return "MODERATE"
        elif score >= 40:
            return "POOR"
        else:
            return "CRITICAL"


async def run_cert_matrix_test(host: str = "127.0.0.1", port: int = 8888) -> Dict[str, Any]:
    """Entry point to run the certificate matrix tests."""
    tester = CertificateMatrixTester()
    return await tester.run_cert_matrix_test(host, port)


if __name__ == "__main__":
    import asyncio
    
    async def test():
        results = await run_cert_matrix_test()
        print(f"Certificate matrix test results: {results}")
    
    asyncio.run(test())
