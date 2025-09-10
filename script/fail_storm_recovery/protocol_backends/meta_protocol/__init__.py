"""
Meta Protocol Integration for Fail-Storm Recovery

This module provides meta-protocol integration for all fail-storm protocol backends,
allowing them to work together through a unified BaseAgent interface.
"""

from .meta_coordinator import FailStormMetaCoordinator
from .a2a_meta_agent import A2AMetaAgent, create_a2a_meta_worker
from .anp_meta_agent import ANPMetaAgent, create_anp_meta_worker
from .acp_meta_agent import ACPMetaAgent, create_acp_meta_worker
from .agora_meta_agent import AgoraMetaAgent, create_agora_meta_worker

__all__ = [
    'FailStormMetaCoordinator',
    'A2AMetaAgent', 'create_a2a_meta_worker',
    'ANPMetaAgent', 'create_anp_meta_worker', 
    'ACPMetaAgent', 'create_acp_meta_worker',
    'AgoraMetaAgent', 'create_agora_meta_worker'
]
