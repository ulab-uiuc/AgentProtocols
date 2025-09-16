"""
LLM-based Router Implementation for Agent Network SDK

This router uses Large Language Models to make intelligent protocol selection decisions.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional

try:
    from .router_interface import BaseRouter, RouterType, RoutingRequest, RoutingResult
except ImportError:
    from src.core.router_interface import BaseRouter, RouterType, RoutingRequest, RoutingResult


class LLMRouter(BaseRouter):
    """
    LLM-based router that uses language models for intelligent protocol selection.
    
    This router analyzes task requirements and selects optimal protocols using LLM reasoning.
    """
    
    def __init__(self, llm_client=None, **kwargs):
        """
        Initialize LLM router.
        
        Args:
            llm_client: LLM client instance for making decisions
            **kwargs: Additional configuration
        """
        super().__init__(RouterType.LLM_BASED)
        self.llm_client = llm_client
        self.routing_config = kwargs
        
        print("🧠 LLM Router initialized")
    
    def set_llm_client(self, llm_client) -> None:
        """Set or update the LLM client."""
        self.llm_client = llm_client
        print("🔗 LLM client configured for routing")
    
    async def route(self, request: RoutingRequest) -> RoutingResult:
        """
        Make routing decision using LLM analysis.
        
        Args:
            request: RoutingRequest with task and requirements
            
        Returns:
            RoutingResult with protocol selections
        """
        if not self.llm_client:
            return await self._fallback_route(request)
        
        try:
            # Prepare protocol information for LLM
            protocol_info = []
            for protocol_name, capability in self.protocols.items():
                info = {
                    "name": protocol_name,
                    "strengths": capability.strengths,
                    "best_for": capability.best_for,
                    "current_load": capability.current_load,
                    "max_capacity": capability.max_capacity,
                    "performance_metrics": capability.performance_metrics
                }
                protocol_info.append(info)
            
            # Create LLM prompt
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(request, protocol_info)
            
            # Get LLM decision using tool calling
            routing_result = await self._get_llm_decision(system_prompt, user_prompt, request.num_agents)
            
            # Record decision
            self._record_decision(request, routing_result)
            
            return routing_result
            
        except Exception as e:
            print(f"⚠️ LLM routing failed: {e}")
            return await self._fallback_route(request)
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for LLM routing."""
        return """You are an intelligent protocol router for a multi-agent network system.
Your job is to analyze tasks and select the most appropriate protocols for optimal execution.

Key principles:
1. Analyze task requirements (complexity, security, performance needs)
2. Consider protocol capabilities and current loads
3. Optimize for the specified constraints (speed, security, etc.)
4. Provide clear reasoning for your decisions

Use tool calling to provide structured protocol selection decisions."""
    
    def _create_user_prompt(self, request: RoutingRequest, protocol_info: List[Dict]) -> str:
        """Create user prompt with task and protocol information."""
        task = request.task
        constraints = request.constraints
        
        prompt = f"""
TASK TO ROUTE:
Question: {task.get('question', '')}
Context: {task.get('context', '')}
Metadata: {task.get('metadata', {})}

CONSTRAINTS:
{json.dumps(constraints, indent=2)}

AVAILABLE PROTOCOLS:
{json.dumps(protocol_info, indent=2)}

REQUIREMENTS:
- Must assign exactly {request.num_agents} agents
- Consider task complexity and performance requirements
- Balance load across protocols when appropriate
- Prioritize based on constraints (speed, security, etc.)

Use the select_protocols tool to make your routing decision.
"""
        return prompt
    
    async def _get_llm_decision(self, system_prompt: str, user_prompt: str, 
                              num_agents: int) -> RoutingResult:
        """Get routing decision from LLM using tool calling."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        tools = [{
            "type": "function",
            "function": {
                "name": "select_protocols",
                "description": "Select optimal protocols for task execution",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_protocols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of selected protocol names"
                        },
                        "agent_assignments": {
                            "type": "object",
                            "description": "Mapping of agent IDs to protocol names"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Detailed explanation for the selection"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Confidence score (0-1)"
                        },
                        "strategy": {
                            "type": "string",
                            "enum": ["single_protocol", "multi_protocol", "load_balanced"],
                            "description": "Routing strategy used"
                        }
                    },
                    "required": ["selected_protocols", "agent_assignments", "reasoning", "confidence", "strategy"]
                }
            }
        }]
        
        # Make LLM call
        if hasattr(self.llm_client, 'ask_tool'):
            response = await self.llm_client.ask_tool(
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "select_protocols"}}
            )
            
            if response and response.get("tool_calls"):
                tool_call = response["tool_calls"][0]
                if tool_call.get("function"):
                    result = json.loads(tool_call["function"]["arguments"])
                    
                    # Ensure correct number of agents
                    agent_assignments = result.get("agent_assignments", {})
                    if len(agent_assignments) != num_agents:
                        agent_assignments = self._ensure_agent_count(
                            result.get("selected_protocols", []), num_agents
                        )
                    
                    return RoutingResult(
                        selected_protocols=result.get("selected_protocols", []),
                        agent_assignments=agent_assignments,
                        reasoning=result.get("reasoning", "LLM routing decision"),
                        confidence=result.get("confidence", 0.5),
                        strategy=result.get("strategy", "single_protocol"),
                        metadata={"llm_used": True, "tool_call": True}
                    )
        
        # Fallback to simple ask
        response = await self.llm_client.ask(messages=messages)
        return self._parse_llm_response(response, num_agents)
    
    def _ensure_agent_count(self, selected_protocols: List[str], num_agents: int) -> Dict[str, str]:
        """Ensure we have exactly the required number of agents."""
        agent_assignments = {}
        available_protocols = selected_protocols or list(self.protocols.keys())
        
        for i in range(num_agents):
            agent_id = f"agent_{i+1}"
            protocol = available_protocols[i % len(available_protocols)] if available_protocols else "default"
            agent_assignments[agent_id] = protocol
        
        return agent_assignments
    
    def _parse_llm_response(self, response: str, num_agents: int) -> RoutingResult:
        """Parse LLM response when tool calling is not available."""
        try:
            # Try to extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response[json_start:json_end])
                
                agent_assignments = result.get("agent_assignments", {})
                if len(agent_assignments) != num_agents:
                    agent_assignments = self._ensure_agent_count(
                        result.get("selected_protocols", []), num_agents
                    )
                
                return RoutingResult(
                    selected_protocols=result.get("selected_protocols", []),
                    agent_assignments=agent_assignments,
                    reasoning=result.get("reasoning", "Parsed from LLM response"),
                    confidence=result.get("confidence", 0.5),
                    strategy=result.get("strategy", "single_protocol"),
                    metadata={"llm_used": True, "tool_call": False}
                )
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
        
        # Ultimate fallback
        return self._fallback_route_sync(num_agents)
    
    async def _fallback_route(self, request: RoutingRequest) -> RoutingResult:
        """Fallback routing when LLM is not available."""
        return self._fallback_route_sync(request.num_agents)
    
    def _fallback_route_sync(self, num_agents: int) -> RoutingResult:
        """Synchronous fallback routing."""
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
        
        # Simple round-robin assignment
        agent_assignments = {}
        selected_protocols = []
        
        for i in range(num_agents):
            agent_id = f"agent_{i+1}"
            protocol = available_protocols[i % len(available_protocols)]
            agent_assignments[agent_id] = protocol
            if protocol not in selected_protocols:
                selected_protocols.append(protocol)
        
        return RoutingResult(
            selected_protocols=selected_protocols,
            agent_assignments=agent_assignments,
            reasoning="Fallback round-robin assignment (LLM not available)",
            confidence=0.3,
            strategy="fallback",
            metadata={"fallback": True}
        )


# Register LLM router with the factory
from .router_interface import RouterFactory
RouterFactory.register_router(RouterType.LLM_BASED, LLMRouter)
