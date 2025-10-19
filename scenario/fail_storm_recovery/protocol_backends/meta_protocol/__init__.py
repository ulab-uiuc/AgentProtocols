"""
Meta Protocol - Integration of fail_storm_recovery protocols with src/core/base_agent.py

This module creates BaseAgent instances that integrate the native SDK functionality
from fail_storm_recovery protocols into the src/core architecture.
"""

# Only import what exists
try:
    from .a2a_agent import A2AMetaAgent
    __all__ = ["A2AMetaAgent"]
except ImportError:
    __all__ = []

try:
    from .acp_agent import ACPMetaAgent
    __all__.append("ACPMetaAgent")
except ImportError:
    pass

try:
    from .agora_agent import AgoraMetaAgent
    __all__.append("AgoraMetaAgent")
except ImportError:
    pass

try:
    from .anp_agent import ANPMetaAgent
    __all__.append("ANPMetaAgent")
except ImportError:
    pass

try:
    from .meta_coordinator import MetaProtocolCoordinator
    __all__.append("MetaProtocolCoordinator")
except ImportError:
    pass

try:
    from .llm_router import LLMIntelligentRouter
    __all__.append("LLMIntelligentRouter")
except ImportError:
    pass

try:
    from .meta_performance_metrics import MetaPerformanceMetricsCollector
    __all__.append("MetaPerformanceMetricsCollector")
except ImportError:
    pass
