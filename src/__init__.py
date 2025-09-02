"""
Agent Network - Unified multi-protocol agent network infrastructure
"""

# Lazy imports to avoid circular import issues
__version__ = "0.1.0"

def get_base_agent():
    """Lazy import for BaseAgent"""
    from .core.base_agent import BaseAgent
    return BaseAgent

def get_agent_network():
    """Lazy import for AgentNetwork"""
    from .core.network import AgentNetwork
    return AgentNetwork

def get_adapters():
    """Lazy import for adapters"""
    from .agent_adapters import BaseProtocolAdapter, A2AAdapter
    return BaseProtocolAdapter, A2AAdapter

def get_metrics():
    """Lazy import for metrics"""
    from .core.metrics import setup_prometheus_metrics, sample_system_metrics
    return setup_prometheus_metrics, sample_system_metrics

__all__ = [
    "get_base_agent",
    "get_agent_network", 
    "get_adapters",
    "get_metrics"
]