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

