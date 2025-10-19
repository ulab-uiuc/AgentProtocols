# -*- coding: utf-8 -*-
"""
A2A Protocol Backend for Privacy Protection Testing
Complete A2A implementation for the privacy testing framework.
"""

from .comm import A2ACommBackend, A2APrivacyAgentHandle
from .agents import (
    A2AReceptionistAgent,
    A2ADoctorAgent, 
    A2AReceptionistExecutor,
    A2ADoctorExecutor,
    A2APrivacySimulator
)
from .analyzer import A2APrivacyAnalyzer, analyze_a2a_privacy

__all__ = [
    "A2ACommBackend",
    "A2APrivacyAgentHandle", 
    "A2AReceptionistAgent",
    "A2ADoctorAgent",
    "A2AReceptionistExecutor", 
    "A2ADoctorExecutor",
    "A2APrivacySimulator",
    "A2APrivacyAnalyzer",
    "analyze_a2a_privacy"
]


# 自动注册发送后端
try:
    from scenarios.safety_tech.protocol_backends.common.interfaces import register_backend
    from .client import A2AProtocolBackend
    register_backend('a2a', A2AProtocolBackend())
except Exception:
    pass

