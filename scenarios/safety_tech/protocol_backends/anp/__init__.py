# -*- coding: utf-8 -*-
"""
ANP Protocol Backend for Privacy Protection Testing
Complete ANP implementation for the privacy testing framework using AgentConnect.
"""

from .comm import ANPCommBackend, ANPPrivacyAgent
from .agents import (
    ANPReceptionistAgent,
    ANPDoctorAgent, 
    ANPReceptionistExecutor,
    ANPDoctorExecutor,
    ANPPrivacySimulator
)
from .analyzer import ANPPrivacyAnalyzer, analyze_anp_privacy

__all__ = [
    "ANPCommBackend",
    "ANPPrivacyAgent", 
    "ANPReceptionistAgent",
    "ANPDoctorAgent",
    "ANPReceptionistExecutor", 
    "ANPDoctorExecutor",
    "ANPPrivacySimulator",
    "ANPPrivacyAnalyzer",
    "analyze_anp_privacy"
]

# Auto-register sending backend
try:
    from scenarios.safety_tech.protocol_backends.common.interfaces import register_backend
    from .client import ANPProtocolBackend
    register_backend('anp', ANPProtocolBackend())
except Exception:
    pass