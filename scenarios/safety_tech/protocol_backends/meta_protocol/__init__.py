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

# 说明：Meta Protocol 仅作为编排层，不注册到数据面后端注册表

__all__ = [
    "SafetyMetaCoordinator",
    "ACPSafetyMetaAgent", 
    "ANPSafetyMetaAgent",
    "AgoraSafetyMetaAgent",
    "A2ASafetyMetaAgent"
]
