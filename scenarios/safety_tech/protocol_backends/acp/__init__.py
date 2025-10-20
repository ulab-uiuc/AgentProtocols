# -*- coding: utf-8 -*-
"""
ACP Protocol Backend for Privacy Testing
Complete ACP implementation for the privacy testing framework.
"""

from .comm import ACPCommBackend, ACPAgentHandle
from .agents import (
    ACPReceptionistAgent,
    ACPNosyDoctorAgent, 
    ACPReceptionistExecutor,
    ACPNosyDoctorExecutor,
    ACPPrivacySimulator
)
from .analyzer import ACPPrivacyAnalyzer, analyze_acp_privacy

# Auto-register发送后端
try:
    from scenarios.safety_tech.protocol_backends.common.interfaces import register_backend
    from .client import ACPProtocolBackend
    register_backend('acp', ACPProtocolBackend())
except Exception:
    # 注册失败不应阻塞module导入；运行时会显式报错
    pass

__all__ = [
    "ACPCommBackend",
    "ACPAgentHandle", 
    "ACPReceptionistAgent",
    "ACPNosyDoctorAgent",
    "ACPReceptionistExecutor", 
    "ACPNosyDoctorExecutor",
    "ACPPrivacySimulator",
    "ACPPrivacyAnalyzer",
    "analyze_acp_privacy"
]

