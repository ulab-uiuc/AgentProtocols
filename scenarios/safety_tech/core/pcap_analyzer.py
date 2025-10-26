# -*- coding: utf-8 -*-
"""
S2 Bypass Packet Capture Analyzer
Implements transparent network traffic capture and plaintext detection functionality
"""

import asyncio
import subprocess
import tempfile
import time
import re
from typing import Dict, Any, List, Optional
from pathlib import Path


class PcapAnalyzer:
    """Network packet capture and analyzer"""
    
    def __init__(self, interface: str = "lo0", duration: int = 10):
        self.interface = interface
        self.duration = duration
        self.pcap_file = None
        self.capture_process = None
        
    async def start_capture(self, bpf_filter: Optional[str] = None) -> str:
        """Start network packet capture"""
        # Create temporary pcap file
        temp_dir = Path(tempfile.gettempdir())
        self.pcap_file = temp_dir / f"s2_capture_{int(time.time())}.pcap"
        
        # Build tcpdump command
        cmd = [
            "tcpdump", 
            "-i", self.interface,
            "-w", str(self.pcap_file),
            "-G", str(self.duration),  # Rotation time
            "-W", "1"  # Keep only 1 file
        ]
        
        if bpf_filter:
            cmd.append(bpf_filter)
        else:
            # Default filter for common ports
            cmd.append("tcp port 8001 or tcp port 8888 or tcp port 9102 or tcp port 9103")
        
        try:
            # Start background capture process
            self.capture_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            print(f"📡 Started network packet capture: {self.interface} -> {self.pcap_file}")
            return str(self.pcap_file)
            
        except Exception as e:
            print(f"❌ Failed to start network packet capture: {e}")
            return None
    
    async def stop_capture(self) -> Dict[str, Any]:
        """Stop capture and analyze results"""
        if self.capture_process:
            try:
                # Wait for capture completion or timeout
                await asyncio.sleep(self.duration + 2)
                
                # Force terminate process
                if self.capture_process.poll() is None:
                    self.capture_process.terminate()
                    await asyncio.sleep(1)
                    
                if self.capture_process.poll() is None:
                    self.capture_process.kill()
                    
            except Exception as e:
                print(f"⚠️ Error stopping capture process: {e}")
        
        # Wait for filesystem sync
        await asyncio.sleep(1)
        
        # Analyze captured packets
        return await self.analyze_pcap()
    
    async def analyze_pcap(self) -> Dict[str, Any]:
        """Analyze pcap file and extract plaintext data"""
        if not self.pcap_file:
            return {
                "status": "no_capture",
                "plaintext_bytes": 0,
                "sensitive_keywords": [],
                "error": "Capture file path not set up"
            }
            
        # Wait for file creation, max 5 seconds
        max_wait = 5
        wait_count = 0
        while not self.pcap_file.exists() and wait_count < max_wait:
            await asyncio.sleep(1)
            wait_count += 1
            
        if not self.pcap_file.exists():
            # Check file permissions and directory
            temp_dir = self.pcap_file.parent
            return {
                "status": "no_capture",
                "plaintext_bytes": 0,
                "sensitive_keywords": [],
                "error": f"Capture file does not exist: {self.pcap_file}, directory writable: {temp_dir.is_dir() and temp_dir.stat().st_mode & 0o200}"
            }
        
        # Check file size
        try:
            file_size = self.pcap_file.stat().st_size
            if file_size == 0:
                return {
                    "status": "empty_capture",
                    "plaintext_bytes": 0,
                    "sensitive_keywords": [],
                    "error": f"Capture file is empty: {self.pcap_file}"
                }
        except OSError as e:
            return {
                "status": "file_access_error",
                "plaintext_bytes": 0,
                "sensitive_keywords": [],
                "error": f"File access error: {e}"
            }
        
        try:
            # Use tcpdump to read and convert to text
            result = subprocess.run([
                "tcpdump", "-r", str(self.pcap_file), 
                "-A",  # Display ASCII content
                "-s", "0"  # Capture full packets
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return {
                    "status": "analysis_failed", 
                    "error": result.stderr,
                    "plaintext_bytes": 0,
                    "sensitive_keywords": []
                }
            
            # Analyze text content
            content = result.stdout
            analysis = self._analyze_content(content)
            
            # Cleanup temporary file
            try:
                self.pcap_file.unlink()
            except:
                pass
                
            return analysis
            
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error": "Analysis timeout",
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
        """Analyze network packet content and detect plaintext data"""
        # Sensitive keyword list
        sensitive_keywords = [
            "password", "token", "secret", "key", "auth",
            "patient", "medical", "diagnosis", "treatment",
            "S2_E2E_WATERMARK", "PLAINTEXT_MARKER"
        ]
        
        # Count plaintext bytes (rough estimate)
        printable_content = ''.join(c for c in content if c.isprintable())
        plaintext_bytes = len(printable_content.encode('utf-8'))
        
        # Detect sensitive keywords
        detected_keywords = []
        content_lower = content.lower()
        for keyword in sensitive_keywords:
            if keyword.lower() in content_lower:
                detected_keywords.append(keyword)
        
        # Detect HTTP plaintext transmission
        http_plaintext = bool(re.search(r'HTTP/1\.[01]', content))
        
        # Detect JSON/XML plaintext payload
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
    """MITM proxy controller (using mitmproxy transparent mode)"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.proxy_process = None
        self.cert_dir = Path.home() / ".mitmproxy"
        
    async def start_transparent_proxy(self) -> bool:
        """Start transparent proxy mode"""
        try:
            # Check if mitmproxy is available
            result = subprocess.run(["which", "mitmproxy"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("⚠️ mitmproxy not installed, skipping MITM test")
                return False
            
            # Start mitmdump transparent proxy
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
            
            # Wait for proxy startup
            await asyncio.sleep(2)
            
            print(f"🔓 Started transparent MITM proxy: port {self.port}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start MITM proxy: {e}")
            return False
    
    async def stop_proxy(self):
        """Stop MITM proxy"""
        if self.proxy_process:
            try:
                self.proxy_process.terminate()
                await asyncio.sleep(1)
                
                if self.proxy_process.poll() is None:
                    self.proxy_process.kill()
                    
                print("🔒 Stopped MITM proxy")
            except Exception as e:
                print(f"⚠️ Error stopping MITM proxy: {e}")
    
    def get_ca_cert_path(self) -> Optional[Path]:
        """Get MITM root CA certificate path"""
        ca_cert = self.cert_dir / "mitmproxy-ca-cert.pem"
        if ca_cert.exists():
            return ca_cert
        return None
    
    def install_ca_cert(self) -> bool:
        """Install MITM root CA to system (for testing only)"""
        ca_cert = self.get_ca_cert_path()
        if not ca_cert:
            return False
        
        try:
            # macOS: Add to Keychain
            subprocess.run([
                "security", "add-trusted-cert", 
                "-d", "-r", "trustRoot",
                "-k", "/Library/Keychains/System.keychain",
                str(ca_cert)
            ], check=True)
            
            print(f"✅ Installed MITM root CA certificate to system")
            return True
            
        except subprocess.CalledProcessError:
            print(f"⚠️ Unable to install MITM root CA certificate (may require administrator privileges)")
            return False
        except Exception as e:
            print(f"❌ Failed to install MITM root CA certificate: {e}")
            return False


async def run_pcap_mitm_test(
    interface: str = "lo0", 
    duration: int = 10,
    enable_mitm: bool = False
) -> Dict[str, Any]:
    """Run bypass packet capture + MITM comprehensive test"""
    
    results = {
        "pcap_analysis": {},
        "mitm_results": {},
        "test_duration": duration,
        "timestamp": time.time()
    }
    
    # Start MITM proxy (if enabled)
    mitm_controller = None
    if enable_mitm:
        mitm_controller = MitmProxyController()
        mitm_started = await mitm_controller.start_transparent_proxy()
        results["mitm_results"]["proxy_started"] = mitm_started
        
        if mitm_started:
            ca_cert_path = mitm_controller.get_ca_cert_path()
            results["mitm_results"]["ca_cert_available"] = ca_cert_path is not None
            results["mitm_results"]["ca_cert_path"] = str(ca_cert_path) if ca_cert_path else None
    
    # Start network packet capture
    pcap_analyzer = PcapAnalyzer(interface, duration)
    pcap_file = await pcap_analyzer.start_capture()
    
    if pcap_file:
        results["pcap_analysis"]["capture_started"] = True
        results["pcap_analysis"]["pcap_file"] = pcap_file
        
        # Wait for capture completion and analyze
        analysis = await pcap_analyzer.stop_capture()
        results["pcap_analysis"].update(analysis)
    else:
        results["pcap_analysis"]["capture_started"] = False
        results["pcap_analysis"]["error"] = "Unable to start network packet capture"
    
    # Stop MITM proxy
    if mitm_controller:
        await mitm_controller.stop_proxy()
    
    return results


if __name__ == "__main__":
    # Test run
    async def test():
        results = await run_pcap_mitm_test(duration=5, enable_mitm=False)
        print(f"Test results: {results}")
    
    asyncio.run(test())
