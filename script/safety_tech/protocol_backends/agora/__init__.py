# -*- coding: utf-8 -*-
"""
Agora Protocol Backend for Privacy Testing
Complete Agora implementation for the privacy testing framework.
"""

from .comm import AgoraCommBackend
from .agents import (
    AgoraReceptionistAgent,
    AgoraNosyDoctorAgent, 
    AgoraReceptionistExecutor,
    AgoraNosyDoctorExecutor,
    AgoraPrivacySimulator
)
from .analyzer import AgoraPrivacyAnalyzer, analyze_agora_privacy

__all__ = [
    "AgoraCommBackend",
    "AgoraReceptionistAgent",
    "AgoraNosyDoctorAgent",
    "AgoraReceptionistExecutor", 
    "AgoraNosyDoctorExecutor",
    "AgoraPrivacySimulator",
    "AgoraPrivacyAnalyzer",
    "analyze_agora_privacy"
]

# 自动注册发送后端
try:
    from script.safety_tech.protocol_backends.common.interfaces import register_backend
    from .client import AgoraProtocolBackend
    register_backend('agora', AgoraProtocolBackend())
except Exception:
    pass