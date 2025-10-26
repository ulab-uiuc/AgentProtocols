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

# Auto-register send backend
try:
    from scenarios.safety_tech.protocol_backends.common.interfaces import register_backend
    from .client import ACPProtocolBackend
    register_backend('acp', ACPProtocolBackend())
except Exception:
    # Registration failure should not block module import; will explicitly error at runtime
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

