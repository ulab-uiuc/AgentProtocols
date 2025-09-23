# -*- coding: utf-8 -*-
"""
S2保密性探针配置Schema与工厂
统一定义probe_config格式，供runner与协议client使用
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ProbeConfig:
    """S2保密性探针配置"""
    
    # TLS/加密降级探针
    tls_downgrade: bool = False
    tls_version_downgrade: Optional[str] = None  # "1.0", "1.1", "1.2"
    cipher_downgrade: Optional[str] = None  # 指定弱加密套件
    
    # 重放攻击探针
    replay_attack: bool = False
    replay_nonce: Optional[str] = None  # 重放的nonce值
    replay_timestamp: Optional[float] = None  # 重放的时间戳
    
    # 明文嗅探探针
    plaintext_sniff: bool = False
    sniff_keywords: Optional[list] = None  # 敏感关键字列表
    sniff_patterns: Optional[list] = None  # 正则表达式模式
    
    # MITM/中间人攻击探针
    mitm_proxy: bool = False
    mitm_proxy_endpoint: Optional[str] = None  # 代理端点
    mitm_cert_replacement: bool = False  # 证书替换测试
    mitm_transparent_mode: bool = False  # 透明代理模式
    mitm_root_ca_path: Optional[str] = None  # 测试根CA证书路径
    
    # 旁路抓包探针（新增）
    pcap_capture: bool = False
    pcap_interface: str = "lo0"  # 抓包网卡接口
    pcap_filter: Optional[str] = None  # BPF过滤器
    pcap_duration_seconds: int = 10  # 抓包持续时间
    
    # 会话劫持探针
    session_hijack: bool = False
    session_token_reuse: Optional[str] = None  # 重用的会话token
    
    # DID/身份伪造探针
    did_spoofing: bool = False
    fake_did: Optional[str] = None  # 伪造的DID
    fake_signature: Optional[str] = None  # 伪造的签名
    
    # 证书有效性矩阵探针（新增）
    cert_validity_matrix: bool = False
    cert_expired: bool = False  # 过期证书测试
    cert_invalid_hostname: bool = False  # 主机名不匹配测试
    cert_self_signed: bool = False  # 自签名证书测试
    cert_chain_incomplete: bool = False  # 证书链不完整测试
    cert_revoked: bool = False  # 撤销证书测试（OCSP/CRL）
    cert_weak_cipher: bool = False  # 弱加密套件测试
    cert_pinning_bypass: bool = False  # 证书固定绕过测试
    
    # E2E负载加密检测探针（新增）
    e2e_payload_detection: bool = False
    e2e_watermark: Optional[str] = None  # 水印标记
    e2e_probe_payload: bool = False  # 探测payload可读性
    
    # 时钟漂移矩阵探针（新增）
    time_skew_matrix: bool = False
    time_skew_levels: Optional[list] = None  # 漂移档位列表，如[30, 120, 300, 600]秒
    time_skew_window_test: bool = False  # 窗口内外重复测试
    
    # 网络层扰动（S1扩展）
    network_jitter_ms: Optional[int] = None  # 网络抖动
    packet_drop_rate: Optional[float] = None  # 丢包率 0.0-1.0
    packet_reorder_prob: Optional[float] = None  # 乱序概率
    bandwidth_limit_kbps: Optional[int] = None  # 带宽限制
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProbeConfig':
        """从字典创建配置"""
        return cls(**data)


def create_s2_tls_downgrade_config() -> ProbeConfig:
    """创建TLS降级测试配置"""
    return ProbeConfig(
        tls_downgrade=True,
        tls_version_downgrade="1.1",
        plaintext_sniff=True,
        sniff_keywords=["password", "token", "secret", "key"]
    )


def create_s2_replay_attack_config(nonce: str = "replay_test") -> ProbeConfig:
    """创建重放攻击测试配置"""
    import time
    return ProbeConfig(
        replay_attack=True,
        replay_nonce=nonce,
        replay_timestamp=time.time() - 300,  # 5分钟前的时间戳
        plaintext_sniff=True
    )


def create_s2_mitm_config(proxy_endpoint: str = "http://127.0.0.1:8080") -> ProbeConfig:
    """创建MITM测试配置"""
    return ProbeConfig(
        mitm_proxy=True,
        mitm_proxy_endpoint=proxy_endpoint,
        mitm_cert_replacement=True,
        plaintext_sniff=True,
        sniff_keywords=["patient", "medical", "diagnosis", "treatment"]
    )


def create_s2_did_spoofing_config(fake_did: str = "did:fake:test") -> ProbeConfig:
    """创建DID伪造测试配置"""
    return ProbeConfig(
        did_spoofing=True,
        fake_did=fake_did,
        fake_signature="fake_signature_data",
        plaintext_sniff=True
    )


def create_s1_network_disturbance_config() -> ProbeConfig:
    """创建S1网络扰动测试配置"""
    return ProbeConfig(
        network_jitter_ms=50,
        packet_drop_rate=0.01,  # 1%丢包
        packet_reorder_prob=0.005,  # 0.5%乱序
        bandwidth_limit_kbps=1000  # 1Mbps限速
    )


def create_s2_pcap_mitm_config(interface: str = "lo0", duration: int = 10) -> ProbeConfig:
    """创建旁路抓包+MITM测试配置（优先级1）"""
    return ProbeConfig(
        pcap_capture=True,
        pcap_interface=interface,
        pcap_duration_seconds=duration,
        pcap_filter="tcp port 8001 or tcp port 8888 or tcp port 9102 or tcp port 9103",
        mitm_transparent_mode=True,
        mitm_cert_replacement=True,
        plaintext_sniff=True,
        sniff_keywords=["password", "token", "patient", "medical", "diagnosis"]
    )


def create_s2_cert_matrix_config() -> ProbeConfig:
    """创建证书有效性矩阵测试配置（优先级2）"""
    return ProbeConfig(
        cert_validity_matrix=True,
        cert_expired=True,
        cert_invalid_hostname=True, 
        cert_self_signed=True,
        cert_chain_incomplete=True,
        cert_revoked=True,
        cert_weak_cipher=True,
        cert_pinning_bypass=True,
        tls_downgrade=True,
        tls_version_downgrade="1.0"
    )


def create_s2_e2e_detection_config(watermark: str = "S2_E2E_WATERMARK") -> ProbeConfig:
    """创建E2E负载加密检测配置（优先级3）"""
    return ProbeConfig(
        e2e_payload_detection=True,
        e2e_watermark=watermark,
        e2e_probe_payload=True,
        plaintext_sniff=True,
        sniff_keywords=[watermark, "PLAINTEXT_MARKER", "SENSITIVE_DATA"]
    )


def create_s2_time_skew_config(levels: list = None) -> ProbeConfig:
    """创建时钟漂移矩阵测试配置（优先级4）"""
    if levels is None:
        levels = [30, 120, 300, 600]  # ±30s, ±2m, ±5m, ±10m
    
    import time
    return ProbeConfig(
        time_skew_matrix=True,
        time_skew_levels=levels,
        time_skew_window_test=True,
        replay_attack=True,
        replay_nonce="skew_test",
        replay_timestamp=time.time() - levels[0]  # 使用第一个漂移档位
    )


def create_comprehensive_probe_config() -> ProbeConfig:
    """创建综合探针配置（用于全面S2测试）"""
    import time
    return ProbeConfig(
        tls_downgrade=True,
        tls_version_downgrade="1.1",
        replay_attack=True,
        replay_nonce="comprehensive_test",
        replay_timestamp=time.time() - 300,  # 5分钟前的时间戳
        plaintext_sniff=True,
        sniff_keywords=["password", "token", "patient", "medical", "diagnosis"],
        mitm_proxy=False,  # MITM需要额外基础设施，默认关闭
        did_spoofing=True,
        fake_did="did:fake:comprehensive_test",
        # 新增高级探针
        pcap_capture=True,  # 启用旁路抓包
        cert_validity_matrix=True,  # 启用证书矩阵测试
        cert_expired=True,
        cert_invalid_hostname=True,
        cert_self_signed=True,
        e2e_payload_detection=True,  # 启用E2E检测
        e2e_watermark="S2_E2E_WATERMARK_TEST",
        time_skew_matrix=True,  # 启用时钟漂移矩阵
        time_skew_levels=[30, 120, 300, 600],  # ±30s, ±2m, ±5m, ±10m
        session_hijack=True  # 启用会话劫持测试
    )
