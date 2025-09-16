"""
Load Balanced Router Implementation for Agent Network SDK

This router distributes tasks across protocols based on current load and performance metrics.
"""

from typing import Dict, List, Any

try:
    from .router_interface import BaseRouter, RouterType, RoutingRequest, RoutingResult
except ImportError:
    from src.core.router_interface import BaseRouter, RouterType, RoutingRequest, RoutingResult


class LoadBalancedRouter(BaseRouter):
    """
    Load-balanced router that distributes tasks based on current load and performance.
    
    This router selects protocols based on:
    - Current load levels
    - Historical performance metrics
    - Protocol capabilities
    """
    
    def __init__(self, **kwargs):
        """Initialize load balanced router."""
        super().__init__(RouterType.LOAD_BALANCED)
        self.config = kwargs
        print("⚖️ Load Balanced Router initialized")
    
    async def route(self, request: RoutingRequest) -> RoutingResult:
        """
        Make routing decision based on load balancing.
        
        Args:
            request: RoutingRequest with task and requirements
            
        Returns:
            RoutingResult with load-balanced protocol selections
        """
        if not self.protocols:
            return RoutingResult(
                selected_protocols=[],
                agent_assignments={},
                reasoning="No protocols available",
                confidence=0.0,
                strategy="none",
                metadata={"router": "load_balanced"}
            )
        
        # Calculate load scores for each protocol
        protocol_scores = []
        for protocol_name, capability in self.protocols.items():
            load_factor = capability.current_load / max(1, capability.max_capacity)
            response_time = capability.performance_metrics.get("response_time", 1.0)
            success_rate = capability.performance_metrics.get("success_rate", 1.0)
            
            # Lower score is better (less load, faster response, higher success)
            score = load_factor + (response_time / 10.0) + (1.0 - success_rate)
            
            protocol_scores.append((protocol_name, score, capability))
        
        # Sort by score (ascending - lower is better)
        protocol_scores.sort(key=lambda x: x[1])
        
        # Assign agents based on load balancing
        agent_assignments = {}
        selected_protocols = []
        
        for i in range(request.num_agents):
            agent_id = f"agent_{i+1}"
            # Use round-robin through sorted protocols for load balancing
            protocol_name = protocol_scores[i % len(protocol_scores)][0]
            agent_assignments[agent_id] = protocol_name
            
            if protocol_name not in selected_protocols:
                selected_protocols.append(protocol_name)
            
            # Update load for next iteration
            for j, (pname, score, cap) in enumerate(protocol_scores):
                if pname == protocol_name:
                    cap.current_load += 1
                    # Recalculate score
                    load_factor = cap.current_load / max(1, cap.max_capacity)
                    response_time = cap.performance_metrics.get("response_time", 1.0)
                    success_rate = cap.performance_metrics.get("success_rate", 1.0)
                    new_score = load_factor + (response_time / 10.0) + (1.0 - success_rate)
                    protocol_scores[j] = (pname, new_score, cap)
                    break
            
            # Re-sort after load update
            protocol_scores.sort(key=lambda x: x[1])
        
        # Create reasoning
        reasoning_parts = [
            f"Load-balanced assignment for {request.num_agents} agents",
            f"Selected protocols: {', '.join(selected_protocols)}",
            f"Distribution strategy: Round-robin based on load and performance"
        ]
        
        # Add protocol details
        for protocol_name in selected_protocols:
            capability = self.protocols[protocol_name]
            load_pct = (capability.current_load / max(1, capability.max_capacity)) * 100
            reasoning_parts.append(f"{protocol_name}: {load_pct:.1f}% load")
        
        result = RoutingResult(
            selected_protocols=selected_protocols,
            agent_assignments=agent_assignments,
            reasoning=" | ".join(reasoning_parts),
            confidence=0.8,  # High confidence for load balancing
            strategy="load_balanced",
            metadata={
                "router": "load_balanced",
                "load_distribution": {
                    name: cap.current_load for name, cap in self.protocols.items()
                }
            }
        )
        
        # Record decision
        self._record_decision(request, result)
        
        return result
    
    async def _fallback_route(self, request: RoutingRequest) -> RoutingResult:
        """Fallback to simple round-robin when load balancing fails."""
        available_protocols = list(self.protocols.keys())
        
        if not available_protocols:
            return RoutingResult(
                selected_protocols=[],
                agent_assignments={},
                reasoning="No protocols available",
                confidence=0.0,
                strategy="none",
                metadata={"fallback": True}
            )
        
        # Simple round-robin
        agent_assignments = {}
        selected_protocols = []
        
        for i in range(request.num_agents):
            agent_id = f"agent_{i+1}"
            protocol = available_protocols[i % len(available_protocols)]
            agent_assignments[agent_id] = protocol
            if protocol not in selected_protocols:
                selected_protocols.append(protocol)
        
        return RoutingResult(
            selected_protocols=selected_protocols,
            agent_assignments=agent_assignments,
            reasoning="Fallback round-robin assignment",
            confidence=0.5,
            strategy="fallback",
            metadata={"fallback": True}
        )


# Register load balanced router with the factory
from .router_interface import RouterFactory
RouterFactory.register_router(RouterType.LOAD_BALANCED, LoadBalancedRouter)
