# -*- coding: utf-8 -*-
"""
S2旁路抓包分析器
实现透明网络流量抓包与明文检测功能
"""

import asyncio
import subprocess
import tempfile
import time
import re
from typing import Dict, Any, List, Optional
from pathlib import Path


class PcapAnalyzer:
    """网络包捕获与分析器"""
    
    def __init__(self, interface: str = "lo0", duration: int = 10):
        self.interface = interface
        self.duration = duration
        self.pcap_file = None
        self.capture_process = None
        
    async def start_capture(self, bpf_filter: Optional[str] = None) -> str:
        """启动网络包捕获"""
        # Create临时pcap文件
        temp_dir = Path(tempfile.gettempdir())
        self.pcap_file = temp_dir / f"s2_capture_{int(time.time())}.pcap"
        
        # 构建tcpdump命令
        cmd = [
            "tcpdump", 
            "-i", self.interface,
            "-w", str(self.pcap_file),
            "-G", str(self.duration),  # 轮转时间
            "-W", "1"  # 只保留1个文件
        ]
        
        if bpf_filter:
            cmd.append(bpf_filter)
        else:
            # 默认过滤常用端口
            cmd.append("tcp port 8001 or tcp port 8888 or tcp port 9102 or tcp port 9103")
        
        try:
            # Start后台捕获进程
            self.capture_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            print(f"📡 启动网络包捕获: {self.interface} -> {self.pcap_file}")
            return str(self.pcap_file)
            
        except Exception as e:
            print(f"❌ 启动网络包捕获失败: {e}")
            return None
    
    async def stop_capture(self) -> Dict[str, Any]:
        """停止捕获并分析结果"""
        if self.capture_process:
            try:
                # Wait捕获完成或超时
                await asyncio.sleep(self.duration + 2)
                
                # 强制终止进程
                if self.capture_process.poll() is None:
                    self.capture_process.terminate()
                    await asyncio.sleep(1)
                    
                if self.capture_process.poll() is None:
                    self.capture_process.kill()
                    
            except Exception as e:
                print(f"⚠️ 停止捕获进程时出错: {e}")
        
        # Wait文件系统同步
        await asyncio.sleep(1)
        
        # 分析捕获的包
        return await self.analyze_pcap()
    
    async def analyze_pcap(self) -> Dict[str, Any]:
        """分析pcap文件，提取明文数据"""
        if not self.pcap_file:
            return {
                "status": "no_capture",
                "plaintext_bytes": 0,
                "sensitive_keywords": [],
                "error": "未Setup捕获文件路径"
            }
            
        # Wait文件创建，最多等待5秒
        max_wait = 5
        wait_count = 0
        while not self.pcap_file.exists() and wait_count < max_wait:
            await asyncio.sleep(1)
            wait_count += 1
            
        if not self.pcap_file.exists():
            # Check文件权限和目录
            temp_dir = self.pcap_file.parent
            return {
                "status": "no_capture",
                "plaintext_bytes": 0,
                "sensitive_keywords": [],
                "error": f"捕获文件不存在: {self.pcap_file}, 目录可写: {temp_dir.is_dir() and temp_dir.stat().st_mode & 0o200}"
            }
        
        # Check文件大小
        try:
            file_size = self.pcap_file.stat().st_size
            if file_size == 0:
                return {
                    "status": "empty_capture",
                    "plaintext_bytes": 0,
                    "sensitive_keywords": [],
                    "error": f"捕获文件为空: {self.pcap_file}"
                }
        except OSError as e:
            return {
                "status": "file_access_error",
                "plaintext_bytes": 0,
                "sensitive_keywords": [],
                "error": f"文件访问错误: {e}"
            }
        
        try:
            # 使用tcpdump读取并转换为文本
            result = subprocess.run([
                "tcpdump", "-r", str(self.pcap_file), 
                "-A",  # 显示ASCII内容
                "-s", "0"  # 捕获完整包
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return {
                    "status": "analysis_failed", 
                    "error": result.stderr,
                    "plaintext_bytes": 0,
                    "sensitive_keywords": []
                }
            
            # 分析文本内容
            content = result.stdout
            analysis = self._analyze_content(content)
            
            # Cleanup临时文件
            try:
                self.pcap_file.unlink()
            except:
                pass
                
            return analysis
            
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error": "分析超时",
                "plaintext_bytes": 0,
                "sensitive_keywords": []
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "plaintext_bytes": 0,
                "sensitive_keywords": []
            }
    
    def _analyze_content(self, content: str) -> Dict[str, Any]:
        """分析网络包内容，检测明文数据"""
        # 敏感关键字列表
        sensitive_keywords = [
            "password", "token", "secret", "key", "auth",
            "patient", "medical", "diagnosis", "treatment",
            "S2_E2E_WATERMARK", "PLAINTEXT_MARKER"
        ]
        
        # 统计明文字节数（粗略估算）
        printable_content = ''.join(c for c in content if c.isprintable())
        plaintext_bytes = len(printable_content.encode('utf-8'))
        
        # 检测敏感关键字
        detected_keywords = []
        content_lower = content.lower()
        for keyword in sensitive_keywords:
            if keyword.lower() in content_lower:
                detected_keywords.append(keyword)
        
        # 检测HTTP明文传输
        http_plaintext = bool(re.search(r'HTTP/1\.[01]', content))
        
        # 检测JSON/XML明文负载
        json_xml_plaintext = bool(re.search(r'[{<].*[>}]', content, re.DOTALL))
        
        return {
            "status": "analyzed",
            "plaintext_bytes": plaintext_bytes,
            "sensitive_keywords": detected_keywords,
            "sensitive_keyword_count": len(detected_keywords),
            "http_plaintext_detected": http_plaintext,
            "json_xml_plaintext_detected": json_xml_plaintext,
            "total_packets_analyzed": content.count('\n'),
            "analysis_timestamp": time.time()
        }


class MitmProxyController:
    """MITM代理控制器（使用mitmproxy透明模式）"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.proxy_process = None
        self.cert_dir = Path.home() / ".mitmproxy"
        
    async def start_transparent_proxy(self) -> bool:
        """启动透明代理模式"""
        try:
            # Checkmitmproxy是否可用
            result = subprocess.run(["which", "mitmproxy"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("⚠️ mitmproxy未安装，跳过MITM测试")
                return False
            
            # Startmitmdump透明代理
            cmd = [
                "mitmdump",
                "--mode", "transparent",
                "--listen-port", str(self.port),
                "--set", "confdir=" + str(self.cert_dir)
            ]
            
            self.proxy_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait代理启动
            await asyncio.sleep(2)
            
            print(f"🔓 启动透明MITM代理: 端口 {self.port}")
            return True
            
        except Exception as e:
            print(f"❌ 启动MITM代理失败: {e}")
            return False
    
    async def stop_proxy(self):
        """停止MITM代理"""
        if self.proxy_process:
            try:
                self.proxy_process.terminate()
                await asyncio.sleep(1)
                
                if self.proxy_process.poll() is None:
                    self.proxy_process.kill()
                    
                print("🔒 已停止MITM代理")
            except Exception as e:
                print(f"⚠️ 停止MITM代理时出错: {e}")
    
    def get_ca_cert_path(self) -> Optional[Path]:
        """GetMITM根CA证书路径"""
        ca_cert = self.cert_dir / "mitmproxy-ca-cert.pem"
        if ca_cert.exists():
            return ca_cert
        return None
    
    def install_ca_cert(self) -> bool:
        """安装MITM根CA到系统（仅用于测试）"""
        ca_cert = self.get_ca_cert_path()
        if not ca_cert:
            return False
        
        try:
            # macOS: 添加到Keychain
            subprocess.run([
                "security", "add-trusted-cert", 
                "-d", "-r", "trustRoot",
                "-k", "/Library/Keychains/System.keychain",
                str(ca_cert)
            ], check=True)
            
            print(f"✅ 已安装MITM根CA证书到系统")
            return True
            
        except subprocess.CalledProcessError:
            print(f"⚠️ 无法安装MITM根CA证书（可能需要管理员权限）")
            return False
        except Exception as e:
            print(f"❌ 安装MITM根CA证书失败: {e}")
            return False


async def run_pcap_mitm_test(
    interface: str = "lo0", 
    duration: int = 10,
    enable_mitm: bool = False
) -> Dict[str, Any]:
    """Run旁路抓包+MITM综合测试"""
    
    results = {
        "pcap_analysis": {},
        "mitm_results": {},
        "test_duration": duration,
        "timestamp": time.time()
    }
    
    # StartMITM代理（如果启用）
    mitm_controller = None
    if enable_mitm:
        mitm_controller = MitmProxyController()
        mitm_started = await mitm_controller.start_transparent_proxy()
        results["mitm_results"]["proxy_started"] = mitm_started
        
        if mitm_started:
            ca_cert_path = mitm_controller.get_ca_cert_path()
            results["mitm_results"]["ca_cert_available"] = ca_cert_path is not None
            results["mitm_results"]["ca_cert_path"] = str(ca_cert_path) if ca_cert_path else None
    
    # Start网络包捕获
    pcap_analyzer = PcapAnalyzer(interface, duration)
    pcap_file = await pcap_analyzer.start_capture()
    
    if pcap_file:
        results["pcap_analysis"]["capture_started"] = True
        results["pcap_analysis"]["pcap_file"] = pcap_file
        
        # Wait捕获完成并分析
        analysis = await pcap_analyzer.stop_capture()
        results["pcap_analysis"].update(analysis)
    else:
        results["pcap_analysis"]["capture_started"] = False
        results["pcap_analysis"]["error"] = "无法启动网络包捕获"
    
    # StopMITM代理
    if mitm_controller:
        await mitm_controller.stop_proxy()
    
    return results


if __name__ == "__main__":
    # 测试运行
    async def test():
        results = await run_pcap_mitm_test(duration=5, enable_mitm=False)
        print(f"测试结果: {results}")
    
    asyncio.run(test())
