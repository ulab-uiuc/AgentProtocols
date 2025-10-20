# -*- coding: utf-8 -*-
"""
S2 confidentiality probe configuration schema and factory
Defines a unified probe_config format for runners and protocol clients.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ProbeConfig:
    """S2 confidentiality probe configuration"""
    
    # TLS / encryption downgrade probes
    tls_downgrade: bool = False
    tls_version_downgrade: Optional[str] = None  # "1.0", "1.1", "1.2"
    cipher_downgrade: Optional[str] = None  # Specify weak cipher suites
    
    # Replay attack probes
    replay_attack: bool = False
    replay_nonce: Optional[str] = None  # Nonce value used for replay
    replay_timestamp: Optional[float] = None  # Timestamp to replay
    
    # Plaintext sniffing probes
    plaintext_sniff: bool = False
    sniff_keywords: Optional[list] = None  # List of sensitive keywords
    sniff_patterns: Optional[list] = None  # Regex patterns
    
    # MITM / man-in-the-middle probes
    mitm_proxy: bool = False
    mitm_proxy_endpoint: Optional[str] = None  # Proxy endpoint
    mitm_cert_replacement: bool = False  # Certificate replacement test
    mitm_transparent_mode: bool = False  # Transparent proxy mode
    mitm_root_ca_path: Optional[str] = None  # Path to root CA certificate for testing
    
    # Bypass packet capture probe (new)
    pcap_capture: bool = False
    pcap_interface: str = "lo0"  # Network interface for packet capture
    pcap_filter: Optional[str] = None  # BPF filter
    pcap_duration_seconds: int = 10  # Capture duration in seconds
    
    # Session hijack probes
    session_hijack: bool = False
    session_token_reuse: Optional[str] = None  # Reused session token
    
    # DID / identity spoofing probes
    did_spoofing: bool = False
    fake_did: Optional[str] = None  # Fake DID
    fake_signature: Optional[str] = None  # Fake signature
    
    # Certificate validity matrix probe (new)
    cert_validity_matrix: bool = False
    cert_expired: bool = False  # Expired certificate test
    cert_invalid_hostname: bool = False  # Hostname mismatch test
    cert_self_signed: bool = False  # Self-signed certificate test
    cert_chain_incomplete: bool = False  # Incomplete certificate chain test
    cert_revoked: bool = False  # Certificate revocation test (OCSP/CRL)
    cert_weak_cipher: bool = False  # Weak cipher suite test
    cert_pinning_bypass: bool = False  # Certificate pinning bypass test
    
    # E2E payload encryption detection probe (new)
    e2e_payload_detection: bool = False
    e2e_watermark: Optional[str] = None  # Watermark marker
    e2e_probe_payload: bool = False  # Probe whether payload is readable
    
    # Time skew matrix probe (new)
    time_skew_matrix: bool = False
    time_skew_levels: Optional[list] = None  # List of skew levels, e.g., [30, 120, 300, 600] seconds
    time_skew_window_test: bool = False  # Windowed repeatability test
    
    # Network-layer disturbances (S1 extension)
    network_jitter_ms: Optional[int] = None  # Network jitter (ms)
    packet_drop_rate: Optional[float] = None  # Packet drop rate between 0.0 and 1.0
    packet_reorder_prob: Optional[float] = None  # Packet reordering probability
    bandwidth_limit_kbps: Optional[int] = None  # Bandwidth limit (kbps)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProbeConfig':
        """Create config from a dictionary"""
        return cls(**data)


def create_s2_tls_downgrade_config() -> ProbeConfig:
    """Create TLS downgrade test configuration"""
    return ProbeConfig(
        tls_downgrade=True,
        tls_version_downgrade="1.1",
        plaintext_sniff=True,
        sniff_keywords=["password", "token", "secret", "key"]
    )


def create_s2_replay_attack_config(nonce: str = "replay_test") -> ProbeConfig:
    """Create replay attack test configuration"""
    import time
    return ProbeConfig(
        replay_attack=True,
        replay_nonce=nonce,
        replay_timestamp=time.time() - 300,  # Timestamp for 5 minutes ago
        plaintext_sniff=True
    )


def create_s2_mitm_config(proxy_endpoint: str = "http://127.0.0.1:8080") -> ProbeConfig:
    """Create MITM test configuration"""
    return ProbeConfig(
        mitm_proxy=True,
        mitm_proxy_endpoint=proxy_endpoint,
        mitm_cert_replacement=True,
        plaintext_sniff=True,
        sniff_keywords=["patient", "medical", "diagnosis", "treatment"]
    )


def create_s2_did_spoofing_config(fake_did: str = "did:fake:test") -> ProbeConfig:
    """Create DID spoofing test configuration"""
    return ProbeConfig(
        did_spoofing=True,
        fake_did=fake_did,
        fake_signature="fake_signature_data",
        plaintext_sniff=True
    )


def create_s1_network_disturbance_config() -> ProbeConfig:
    """Create S1 network disturbance test configuration"""
    return ProbeConfig(
        network_jitter_ms=50,
        packet_drop_rate=0.01,  # 1% packet loss
        packet_reorder_prob=0.005,  # 0.5% reordering
        bandwidth_limit_kbps=1000  # 1 Mbps bandwidth limit
    )


def create_s2_pcap_mitm_config(interface: str = "lo0", duration: int = 10) -> ProbeConfig:
    """Create bypass packet capture + MITM test configuration (priority 1)"""
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
    """Create certificate validity matrix test configuration (priority 2)"""
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
    """Create E2E payload encryption detection configuration (priority 3)"""
    return ProbeConfig(
        e2e_payload_detection=True,
        e2e_watermark=watermark,
        e2e_probe_payload=True,
        plaintext_sniff=True,
        sniff_keywords=[watermark, "PLAINTEXT_MARKER", "SENSITIVE_DATA"]
    )


def create_s2_time_skew_config(levels: list = None) -> ProbeConfig:
    """Create time skew matrix test configuration (priority 4)"""
    if levels is None:
        levels = [30, 120, 300, 600]  # ±30s, ±2m, ±5m, ±10m
    
    import time
    return ProbeConfig(
        time_skew_matrix=True,
        time_skew_levels=levels,
        time_skew_window_test=True,
        replay_attack=True,
        replay_nonce="skew_test",
        replay_timestamp=time.time() - levels[0]  # Use the first skew level
    )


def create_comprehensive_probe_config() -> ProbeConfig:
    """Create comprehensive probe configuration (for full S2 testing)"""
    import time
    return ProbeConfig(
        tls_downgrade=True,
        tls_version_downgrade="1.1",
        replay_attack=True,
        replay_nonce="comprehensive_test",
        replay_timestamp=time.time() - 300,  # Timestamp for 5 minutes ago
        plaintext_sniff=True,
        sniff_keywords=["password", "token", "patient", "medical", "diagnosis"],
        mitm_proxy=False,  # MITM requires additional infrastructure; disabled by default
        did_spoofing=True,
        fake_did="did:fake:comprehensive_test",
        fake_signature="fake_comprehensive_signature",
        # New advanced probes
        pcap_capture=True,  # Enable bypass packet capture
        cert_validity_matrix=True,  # Enable certificate matrix testing
        cert_expired=True,
        cert_invalid_hostname=True,
        cert_self_signed=True,
        e2e_payload_detection=True,  # Enable E2E detection
        e2e_watermark="S2_E2E_WATERMARK_TEST",
        time_skew_matrix=True,  # Enable time skew matrix
        time_skew_levels=[30, 120, 300, 600],  # ±30s, ±2m, ±5m, ±10m
        session_hijack=True  # Enable session hijack testing
    )
