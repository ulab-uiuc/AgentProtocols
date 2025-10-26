# -*- coding: utf-8 -*-
"""
Meta Protocol Backend for Safety Testing
Unified meta-protocol system that wraps existing protocol backends for privacy testing.
"""

from .meta_coordinator import SafetyMetaCoordinator
from .acp_meta_agent import ACPSafetyMetaAgent
from .anp_meta_agent import ANPSafetyMetaAgent
from .agora_meta_agent import AgoraSafetyMetaAgent
from .a2a_meta_agent import A2ASafetyMetaAgent

# Note: Meta Protocol only acts as orchestration layer, does not register to data plane backend registry

__all__ = [
    "SafetyMetaCoordinator",
    "ACPSafetyMetaAgent", 
    "ANPSafetyMetaAgent",
    "AgoraSafetyMetaAgent",
    "A2ASafetyMetaAgent"
]
