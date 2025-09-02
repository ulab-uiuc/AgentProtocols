"""
Safety Tech Scenario - Multi-Agent Task Allocation for Safety-Critical Applications

This module implements a distributed task allocation benchmark focusing on safe,
collaborative AI for processing safety-critical tasks like medical dialogues.

Key Components:
- Multi-protocol interoperability (ANP, A2A, etc.)
- Real-time fault-tolerant agent collaboration
- Safety-specific metrics and monitoring
- Dynamic data integration via Kaggle
- Secure task handoffs with AgentConnect
"""

__version__ = "1.0.0"
__author__ = "Safety Tech Team"

from . import core
from . import protocol_backends
from . import runners

__all__ = ["core", "protocol_backends", "runners"]
