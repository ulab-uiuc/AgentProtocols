"""
A2A (Agent-to-Agent) protocol backend for MAPF.

Implements BaseRobot and BaseNet communication methods using A2A SDK.
"""

from .agent import A2ARobot
from .network import A2ANet

__all__ = ["A2ARobot", "A2ANet"] 