"""
Meta Protocol - Integration of streaming_queue protocols with src/core/base_agent.py

This module creates BaseAgent instances that integrate the native SDK functionality
from streaming_queue protocols into the src/core architecture.
"""

# Only import what exists
try:
    from .a2a_agent import A2AMetaAgent
    __all__ = ["A2AMetaAgent"]
except ImportError:
    __all__ = []
