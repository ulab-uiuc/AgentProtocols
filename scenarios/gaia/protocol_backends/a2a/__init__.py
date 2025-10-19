"""
A2A Protocol Backend for GAIA Framework.
This module provides A2A protocol integration.
"""

from .agent import A2AAgent, A2AExecutor
from .network import A2ANetwork

__all__ = ['A2AAgent', 'A2AExecutor', 'A2ANetwork']
