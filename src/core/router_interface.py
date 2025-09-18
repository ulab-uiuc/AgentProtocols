"""
Router Interface for Agent Network SDK

Provides standardized routing capabilities that can be implemented by different backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class RouterType(Enum):
    """Types of routers available."""
    LLM_BASED = "llm_based"
    RULE_BASED = "rule_based"
    LOAD_BALANCED = "load_balanced"
    RANDOM = "random"


@dataclass
class ProtocolCapability:
    """Protocol capability description for routing decisions."""
    name: str
    agent_id: str
    strengths: List[str]
    best_for: List[str]
    performance_metrics: Dict[str, float]  # response_time, success_rate, etc.
    current_load: int
    max_capacity: int


@dataclass
class RoutingRequest:
    """Request for routing decision."""
    task: Dict[str, Any]
    num_agents: int
    constraints: Dict[str, Any]  # Optional constraints like max_latency, requires_security
    context: Dict[str, Any]  # Additional context information


@dataclass
class RoutingResult:
    """Result of routing decision."""
    selected_protocols: List[str]
    agent_assignments: Dict[str, str]  # agent_id -> protocol_name
    reasoning: str
    confidence: float
    strategy: str
    metadata: Dict[str, Any]  # Additional routing metadata


class RouterInterface(ABC):
    """Abstract interface for all routers in the SDK."""
    
    @abstractmethod
    async def route(self, request: RoutingRequest) -> RoutingResult:
        """
        Make a routing decision based on the request.
        
        Args:
            request: RoutingRequest containing task and requirements
            
        Returns:
            RoutingResult with protocol selections and reasoning
        """
        pass
    
    @abstractmethod
    def register_protocol(self, capability: ProtocolCapability) -> None:
        """
        Register a protocol with its capabilities.
        
        Args:
            capability: ProtocolCapability describing the protocol
        """
        pass
    
    @abstractmethod
    def update_metrics(self, agent_id: str, metrics: Dict[str, float]) -> None:
        """
        Update performance metrics for an agent.
        
        Args:
            agent_id: Agent identifier
            metrics: Performance metrics dict
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics and performance data."""
        pass


class BaseRouter(RouterInterface):
    """Base implementation of RouterInterface with common functionality."""
    
    def __init__(self, router_type: RouterType):
        self.router_type = router_type
        self.protocols: Dict[str, ProtocolCapability] = {}
        self.routing_history: List[Dict[str, Any]] = []
    
    def register_protocol(self, capability: ProtocolCapability) -> None:
        """Register a protocol capability."""
        self.protocols[capability.name] = capability
        print(f"📝 Registered protocol: {capability.name} (Agent: {capability.agent_id})")
    
    def update_metrics(self, agent_id: str, metrics: Dict[str, float]) -> None:
        """Update agent performance metrics."""
        for protocol_name, capability in self.protocols.items():
            if capability.agent_id == agent_id:
                # Update performance metrics
                for metric_name, value in metrics.items():
                    capability.performance_metrics[metric_name] = value
                break
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic routing statistics."""
        return {
            "router_type": self.router_type.value,
            "total_decisions": len(self.routing_history),
            "registered_protocols": list(self.protocols.keys()),
            "protocol_stats": {
                name: {
                    "agent_id": cap.agent_id,
                    "current_load": cap.current_load,
                    "max_capacity": cap.max_capacity,
                    "performance_metrics": cap.performance_metrics
                }
                for name, cap in self.protocols.items()
            }
        }
    
    def _record_decision(self, request: RoutingRequest, result: RoutingResult) -> None:
        """Record routing decision for analysis."""
        record = {
            "timestamp": __import__("time").time(),
            "request": {
                "task": request.task,
                "num_agents": request.num_agents,
                "constraints": request.constraints
            },
            "result": {
                "selected_protocols": result.selected_protocols,
                "agent_assignments": result.agent_assignments,
                "confidence": result.confidence,
                "strategy": result.strategy
            }
        }
        self.routing_history.append(record)


# Router Factory
class RouterFactory:
    """Factory for creating router instances."""
    
    _registered_routers: Dict[RouterType, type] = {}
    
    @classmethod
    def register_router(cls, router_type: RouterType, router_class: type) -> None:
        """Register a router implementation."""
        cls._registered_routers[router_type] = router_class
        print(f"Registered router type: {router_type.value}")
    
    @classmethod
    def create_router(cls, router_type: RouterType, **kwargs) -> RouterInterface:
        """Create a router instance."""
        if router_type not in cls._registered_routers:
            raise ValueError(f"Router type {router_type.value} not registered")
        
        router_class = cls._registered_routers[router_type]
        return router_class(**kwargs)
    
    @classmethod
    def get_available_routers(cls) -> List[RouterType]:
        """Get list of available router types."""
        return list(cls._registered_routers.keys())


# Convenience functions
def create_router(router_type: RouterType, **kwargs) -> RouterInterface:
    """Convenience function to create a router."""
    return RouterFactory.create_router(router_type, **kwargs)


def register_router_type(router_type: RouterType, router_class: type) -> None:
    """Convenience function to register a router type."""
    RouterFactory.register_router(router_type, router_class)
