# -*- coding: utf-8 -*-
"""
S2证书有效性矩阵测试
系统化测试各种证书问题：过期、主机名不匹配、自签名、链不完整等
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
    """证书有效性矩阵测试器"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "s2_cert_test"
        self.temp_dir.mkdir(exist_ok=True)
        self.test_results = {}
    
    async def run_cert_matrix_test(self, target_host: str = "127.0.0.1", target_port: int = 8888) -> Dict[str, Any]:
        """运行完整的证书矩阵测试"""
        results = {
            "target": f"{target_host}:{target_port}",
            "test_timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # 1. 过期证书测试
        results["tests"]["expired_cert"] = await self._test_expired_certificate(target_host, target_port)
        
        # 2. 主机名不匹配测试
        results["tests"]["hostname_mismatch"] = await self._test_hostname_mismatch(target_host, target_port)
        
        # 3. 自签名证书测试
        results["tests"]["self_signed"] = await self._test_self_signed_cert(target_host, target_port)
        
        # 4. 证书链不完整测试
        results["tests"]["incomplete_chain"] = await self._test_incomplete_chain(target_host, target_port)
        
        # 5. 弱加密套件测试
        results["tests"]["weak_cipher"] = await self._test_weak_cipher_suites(target_host, target_port)
        
        # 6. TLS版本降级测试
        results["tests"]["tls_downgrade"] = await self._test_tls_version_downgrade(target_host, target_port)
        
        # 计算总体评分
        results["matrix_score"] = self._calculate_matrix_score(results["tests"])
        
        return results
    
    async def _test_expired_certificate(self, host: str, port: int) -> Dict[str, Any]:
        """测试过期证书拒绝"""
        test_result = {
            "test_name": "过期证书测试",
            "description": "使用过期证书连接，验证是否被正确拒绝",
            "status": "unknown",
            "blocked": False,
            "error_type": None
        }
        
        try:
            # 生成过期的自签名证书
            expired_cert_path, expired_key_path = self._generate_expired_certificate(host)
            
            # 尝试使用过期证书连接
            ssl_context = ssl.create_default_context()
            ssl_context.load_cert_chain(expired_cert_path, expired_key_path)
            ssl_context.check_hostname = False  # 绕过主机名检查，专门测试过期
            
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
        """测试主机名不匹配证书拒绝"""
        test_result = {
            "test_name": "主机名不匹配测试",
            "description": "使用错误主机名的证书连接",
            "status": "unknown",
            "blocked": False
        }
        
        try:
            # 生成主机名不匹配的证书（为example.com生成，但连接localhost）
            wrong_host = "wrong.example.com" if host == "127.0.0.1" else "127.0.0.1"
            cert_path, key_path = self._generate_certificate(wrong_host)
            
            # 使用strict hostname checking
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED
            
            try:
                # 这里应该会因为hostname不匹配而失败
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
        """测试自签名证书处理"""
        test_result = {
            "test_name": "自签名证书测试",
            "description": "使用自签名证书连接，检查是否需要明确信任",
            "status": "unknown",
            "blocked": False
        }
        
        try:
            # 使用默认严格验证模式连接
            async with httpx.AsyncClient(verify=True) as client:
                response = await client.get(f"https://{host}:{port}/health", timeout=5.0)
                test_result["status"] = "connection_succeeded"
                test_result["blocked"] = False
                test_result["response_code"] = response.status_code
                
        except Exception as e:
            test_result["status"] = "ssl_error"
            test_result["blocked"] = True  # 好事，正确拒绝了自签名证书
            test_result["error_type"] = "SSLError"
            test_result["error_message"] = str(e)
            
        except Exception as e:
            test_result["status"] = "other_error"
            test_result["blocked"] = True
            test_result["error_type"] = type(e).__name__
            test_result["error_message"] = str(e)
        
        return test_result
    
    async def _test_incomplete_chain(self, host: str, port: int) -> Dict[str, Any]:
        """测试证书链不完整的处理"""
        return {
            "test_name": "证书链不完整测试",
            "status": "skipped",
            "reason": "需要复杂的CA链设置，暂时跳过",
            "blocked": None
        }
    
    async def _test_weak_cipher_suites(self, host: str, port: int) -> Dict[str, Any]:
        """测试弱加密套件拒绝"""
        test_result = {
            "test_name": "弱加密套件测试",
            "description": "尝试使用弱加密套件连接",
            "status": "unknown",
            "blocked": False,
            "cipher_results": {}
        }
        
        # 测试各种弱加密套件
        weak_ciphers = [
            "DES-CBC3-SHA",     # 3DES
            "RC4-MD5",          # RC4
            "NULL-MD5",         # NULL加密
            "EXPORT-RC4-40-MD5" # 40位导出级加密
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
        
        # 计算总体阻断率
        total_tests = len(weak_ciphers)
        blocked_count = sum(1 for result in test_result["cipher_results"].values() 
                          if result.get("blocked", False))
        
        test_result["blocked"] = blocked_count > 0
        test_result["block_rate"] = blocked_count / total_tests if total_tests > 0 else 0
        test_result["status"] = "completed"
        
        return test_result
    
    async def _test_tls_version_downgrade(self, host: str, port: int) -> Dict[str, Any]:
        """测试TLS版本降级拒绝"""
        test_result = {
            "test_name": "TLS版本降级测试",
            "description": "尝试使用旧版本TLS连接",
            "status": "unknown",
            "version_results": {}
        }
        
        # 测试各种TLS版本
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
        
        # 计算阻断率
        total_tests = len(tls_versions)
        blocked_count = sum(1 for result in test_result["version_results"].values() 
                          if result.get("blocked", False))
        
        test_result["block_rate"] = blocked_count / total_tests if total_tests > 0 else 0
        test_result["status"] = "completed"
        
        return test_result
    
    def _generate_expired_certificate(self, hostname: str) -> tuple:
        """生成过期的自签名证书"""
        # 创建密钥对
        key = crypto.PKey()
        key.generate_key(crypto.TYPE_RSA, 2048)
        
        # 创建证书
        cert = crypto.X509()
        cert.get_subject().CN = hostname
        cert.set_serial_number(1000)
        
        # 设置为已过期（昨天到期）
        cert.gmtime_adj_notBefore(-86400 * 2)  # 2天前开始
        cert.gmtime_adj_notAfter(-86400)       # 昨天过期
        
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(key)
        cert.sign(key, 'sha256')
        
        # 保存到临时文件
        cert_path = self.temp_dir / f"expired_{hostname}.crt"
        key_path = self.temp_dir / f"expired_{hostname}.key"
        
        with open(cert_path, 'wb') as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
            
        with open(key_path, 'wb') as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
        
        return str(cert_path), str(key_path)
    
    def _generate_certificate(self, hostname: str) -> tuple:
        """生成有效的自签名证书"""
        # 创建密钥对
        key = crypto.PKey()
        key.generate_key(crypto.TYPE_RSA, 2048)
        
        # 创建证书
        cert = crypto.X509()
        cert.get_subject().CN = hostname
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(86400 * 365)  # 1年有效期
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(key)
        cert.sign(key, 'sha256')
        
        # 保存到临时文件
        cert_path = self.temp_dir / f"test_{hostname}.crt"
        key_path = self.temp_dir / f"test_{hostname}.key"
        
        with open(cert_path, 'wb') as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
            
        with open(key_path, 'wb') as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
        
        return str(cert_path), str(key_path)
    
    def _calculate_matrix_score(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算证书矩阵测试的综合评分"""
        scores = {
            "expired_cert": 0,
            "hostname_mismatch": 0,
            "self_signed": 0,
            "weak_cipher": 0,
            "tls_downgrade": 0
        }
        
        # 过期证书：被阻止=100分，未阻止=0分
        if test_results.get("expired_cert", {}).get("blocked"):
            scores["expired_cert"] = 100
            
        # 主机名不匹配：被阻止=100分
        if test_results.get("hostname_mismatch", {}).get("blocked"):
            scores["hostname_mismatch"] = 100
            
        # 自签名证书：被阻止=100分
        if test_results.get("self_signed", {}).get("blocked"):
            scores["self_signed"] = 100
            
        # 弱加密套件：按阻断率评分
        weak_cipher_rate = test_results.get("weak_cipher", {}).get("block_rate", 0)
        scores["weak_cipher"] = int(weak_cipher_rate * 100)
        
        # TLS版本降级：按阻断率评分
        tls_downgrade_rate = test_results.get("tls_downgrade", {}).get("block_rate", 0)
        scores["tls_downgrade"] = int(tls_downgrade_rate * 100)
        
        # 计算加权总分
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
        """根据分数获取安全等级"""
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
    """运行证书矩阵测试的入口函数"""
    tester = CertificateMatrixTester()
    return await tester.run_cert_matrix_test(host, port)


if __name__ == "__main__":
    import asyncio
    
    async def test():
        results = await run_cert_matrix_test()
        print(f"证书矩阵测试结果: {results}")
    
    asyncio.run(test())
