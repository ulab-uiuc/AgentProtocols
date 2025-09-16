"""
LLM-based Intelligent Router for Multi-Protocol Agent Network

This router uses LLM to analyze tasks and make intelligent protocol selection decisions.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ProtocolInfo:
    """Protocol information for LLM decision making."""
    name: str
    agent_id: str
    strengths: List[str]
    best_for: List[str]
    current_load: int
    avg_response_time: float
    success_rate: float


@dataclass
class RoutingDecision:
    """LLM routing decision result."""
    selected_protocols: List[str]  # List of protocol names to use
    agent_assignments: Dict[str, str]  # agent_id -> protocol_name
    reasoning: str
    confidence: float
    strategy: str


class LLMIntelligentRouter:
    """
    LLM-based intelligent router that uses language models to make 
    protocol selection decisions.
    """
    
    def __init__(self, llm_client=None):
        self.protocols: Dict[str, ProtocolInfo] = {}
        self.task_history: List[Dict[str, Any]] = []
        self.llm_client = llm_client  # Will be set when available
        
        # Initialize default protocol information
        self._initialize_protocol_info()
    
    def _initialize_protocol_info(self):
        """Initialize protocol information for LLM decision making."""
        
        # A2A Protocol - High throughput, structured communication
        self.protocols["a2a"] = ProtocolInfo(
            name="a2a",
            agent_id="A2A-Worker",
            strengths=[
                "High throughput message processing",
                "Structured data handling",
                "Real-time communication",
                "Event-driven architecture"
            ],
            best_for=[
                "High-volume data processing",
                "Real-time applications",
                "Structured workflows",
                "Event streaming"
            ],
            current_load=0,
            avg_response_time=0.8,
            success_rate=0.95
        )
        
        # ACP Protocol - Secure, enterprise-grade
        self.protocols["acp"] = ProtocolInfo(
            name="acp",
            agent_id="ACP-Worker", 
            strengths=[
                "Enterprise-grade security",
                "Reliable message delivery",
                "Compliance features",
                "Fault tolerance"
            ],
            best_for=[
                "Sensitive data processing",
                "Enterprise workflows",
                "Compliance-required tasks",
                "Mission-critical operations"
            ],
            current_load=0,
            avg_response_time=1.2,
            success_rate=0.98
        )
        
        # ANP Protocol - Decentralized, privacy-focused
        self.protocols["anp"] = ProtocolInfo(
            name="anp",
            agent_id="ANP-Worker",
            strengths=[
                "Decentralized architecture",
                "Privacy protection",
                "DID-based authentication",
                "End-to-end encryption"
            ],
            best_for=[
                "Privacy-critical tasks",
                "Decentralized applications",
                "Identity verification",
                "Confidential communications"
            ],
            current_load=0,
            avg_response_time=1.0,
            success_rate=0.92
        )
        
        # Agora Protocol - Flexible, research-oriented
        self.protocols["agora"] = ProtocolInfo(
            name="agora",
            agent_id="Agora-Worker",
            strengths=[
                "Complex reasoning support",
                "Flexible workflows",
                "Research capabilities",
                "Academic integrations"
            ],
            best_for=[
                "Complex reasoning tasks",
                "Research projects",
                "Academic workflows",
                "Experimental features"
            ],
            current_load=0,
            avg_response_time=1.5,
            success_rate=0.90
        )
    
    def set_llm_client(self, llm_client):
        """Set the LLM client for making routing decisions."""
        self.llm_client = llm_client
        print("🧠 LLM client configured for intelligent routing")
    
    def register_protocol_agent(self, agent_id: str, protocol_name: str, 
                              strengths: List[str] = None, best_for: List[str] = None):
        """Register a new protocol agent."""
        if protocol_name in self.protocols:
            # Update existing protocol info
            self.protocols[protocol_name].agent_id = agent_id
            if strengths:
                self.protocols[protocol_name].strengths = strengths
            if best_for:
                self.protocols[protocol_name].best_for = best_for
        else:
            # Create new protocol info
            self.protocols[protocol_name] = ProtocolInfo(
                name=protocol_name,
                agent_id=agent_id,
                strengths=strengths or [f"{protocol_name} protocol capabilities"],
                best_for=best_for or [f"{protocol_name} tasks"],
                current_load=0,
                avg_response_time=1.0,
                success_rate=1.0
            )
        print(f"🔧 Registered protocol agent: {agent_id} ({protocol_name})")
    
    def update_agent_metrics(self, agent_id: str, response_time: float, 
                           success: bool, load_change: int = 0):
        """Update agent performance metrics based on task execution."""
        for protocol_info in self.protocols.values():
            if protocol_info.agent_id == agent_id:
                # Update response time (exponential moving average)
                alpha = 0.3
                protocol_info.avg_response_time = (alpha * response_time + 
                                                 (1 - alpha) * protocol_info.avg_response_time)
                
                # Update success rate (simple moving average)
                if agent_id not in self.task_history:
                    # Initialize tracking
                    pass
                
                # Update load
                protocol_info.current_load = max(0, protocol_info.current_load + load_change)
                break
    
    async def route_task_with_llm(self, task: Dict[str, Any], 
                                num_agents: int = 1) -> RoutingDecision:
        """
        Use LLM to make intelligent protocol selection decisions.
        
        Args:
            task: Task dictionary with question, context, metadata
            num_agents: Number of agents to create (will select protocols for each)
            
        Returns:
            RoutingDecision with protocol selections
        """
        if not self.llm_client:
            # Fallback to simple selection if no LLM available
            return await self._fallback_routing(task, num_agents)
        
        # Prepare protocol information for LLM
        protocol_descriptions = []
        for protocol_name, info in self.protocols.items():
            protocol_desc = {
                "name": protocol_name,
                "strengths": info.strengths,
                "best_for": info.best_for,
                "current_load": info.current_load,
                "avg_response_time": f"{info.avg_response_time:.1f}s",
                "success_rate": f"{info.success_rate:.1%}"
            }
            protocol_descriptions.append(protocol_desc)
        
        # Create LLM prompt for protocol selection
        system_prompt = """You are an intelligent protocol router for a multi-agent system. 
Your job is to analyze tasks and select the most appropriate protocols for handling them.

You have access to the following protocols:
- A2A: High throughput, structured data, real-time processing
- ACP: Enterprise security, compliance, fault tolerance
- ANP: Privacy-focused, decentralized, DID authentication  
- Agora: Complex reasoning, research tasks, flexible workflows

Respond with a JSON object containing your protocol selection decision."""

        user_prompt = f"""
Task to analyze:
Question: {task.get('question', '')}
Context: {task.get('context', '')}
Metadata: {task.get('metadata', {})}

Available Protocols:
{json.dumps(protocol_descriptions, indent=2)}

Number of agents needed: {num_agents}

Please select the best protocol(s) for this task. Consider:
1. Task complexity and type
2. Security/privacy requirements  
3. Performance needs
4. Current protocol loads

Respond with JSON in this format:
{{
    "selected_protocols": ["protocol1", "protocol2", ...],
    "agent_assignments": {{"agent_1": "protocol1", "agent_2": "protocol2", ...}},
    "reasoning": "Detailed explanation of why these protocols were chosen",
    "confidence": 0.95,
    "strategy": "single_protocol" or "multi_protocol"
}}
"""

        try:
            # Call LLM for routing decision
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Use tool calling for structured output
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
                                "enum": ["single_protocol", "multi_protocol"],
                                "description": "Routing strategy used"
                            }
                        },
                        "required": ["selected_protocols", "agent_assignments", "reasoning", "confidence", "strategy"]
                    }
                }
            }]
            
            # Make LLM call with tool calling
            if hasattr(self.llm_client, 'ask_tool'):
                response = await self.llm_client.ask_tool(
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "select_protocols"}}
                )
                
                # Extract tool call result
                if response and response.get("tool_calls"):
                    tool_call = response["tool_calls"][0]
                    if tool_call.get("function"):
                        result = json.loads(tool_call["function"]["arguments"])
                        
                        # Validate and create routing decision
                        selected_protocols = result.get("selected_protocols", [])
                        agent_assignments = result.get("agent_assignments", {})
                        reasoning = result.get("reasoning", "LLM routing decision")
                        confidence = result.get("confidence", 0.5)
                        strategy = result.get("strategy", "single_protocol")
                        
                        # Ensure we have the right number of agents
                        if len(agent_assignments) != num_agents:
                            # Auto-generate agent assignments if needed
                            agent_assignments = {}
                            for i in range(num_agents):
                                agent_id = f"agent_{i+1}"
                                protocol = selected_protocols[i % len(selected_protocols)] if selected_protocols else "a2a"
                                agent_assignments[agent_id] = protocol
                        
                        return RoutingDecision(
                            selected_protocols=selected_protocols,
                            agent_assignments=agent_assignments,
                            reasoning=reasoning,
                            confidence=confidence,
                            strategy=strategy
                        )
            
            # Fallback if tool calling fails
            response = await self.llm_client.ask(messages=messages)
            return self._parse_llm_response(response, num_agents)
            
        except Exception as e:
            print(f"⚠️ LLM routing failed: {e}")
            return await self._fallback_routing(task, num_agents)
    
    def _parse_llm_response(self, response: str, num_agents: int) -> RoutingDecision:
        """Parse LLM response when tool calling is not available."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                
                selected_protocols = result.get("selected_protocols", ["a2a"])
                agent_assignments = result.get("agent_assignments", {})
                reasoning = result.get("reasoning", "Parsed from LLM response")
                confidence = result.get("confidence", 0.5)
                strategy = result.get("strategy", "single_protocol")
                
                # Ensure we have the right number of agents
                if len(agent_assignments) != num_agents:
                    agent_assignments = {}
                    for i in range(num_agents):
                        agent_id = f"agent_{i+1}"
                        protocol = selected_protocols[i % len(selected_protocols)] if selected_protocols else "a2a"
                        agent_assignments[agent_id] = protocol
                
                return RoutingDecision(
                    selected_protocols=selected_protocols,
                    agent_assignments=agent_assignments,
                    reasoning=reasoning,
                    confidence=confidence,
                    strategy=strategy
                )
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
        
        # Ultimate fallback
        return await self._fallback_routing({}, num_agents)
    
    async def _fallback_routing(self, task: Dict[str, Any], num_agents: int) -> RoutingDecision:
        """Fallback routing when LLM is not available."""
        available_protocols = list(self.protocols.keys())
        
        if not available_protocols:
            return RoutingDecision(
                selected_protocols=[],
                agent_assignments={},
                reasoning="No protocols available",
                confidence=0.0,
                strategy="none"
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
        
        return RoutingDecision(
            selected_protocols=selected_protocols,
            agent_assignments=agent_assignments,
            reasoning="Fallback round-robin assignment (LLM not available)",
            confidence=0.3,
            strategy="fallback"
        )
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing performance statistics."""
        protocol_stats = {}
        for protocol_name, info in self.protocols.items():
            protocol_stats[protocol_name] = {
                "agent_id": info.agent_id,
                "current_load": info.current_load,
                "avg_response_time": info.avg_response_time,
                "success_rate": info.success_rate,
                "strengths": info.strengths,
                "best_for": info.best_for
            }
        
        return {
            "total_decisions": len(self.task_history),
            "protocol_stats": protocol_stats,
            "available_protocols": list(self.protocols.keys())
        }


# Singleton instance for global use
llm_router = LLMIntelligentRouter()


# Utility functions for easy integration
async def route_task_with_llm(task: Dict[str, Any], num_agents: int = 1, 
                            llm_client=None) -> RoutingDecision:
    """
    Convenience function for LLM-based task routing.
    
    Args:
        task: Task dictionary with question, context, metadata
        num_agents: Number of agents to create
        llm_client: LLM client instance
        
    Returns:
        RoutingDecision with protocol assignments
    """
    if llm_client and not llm_router.llm_client:
        llm_router.set_llm_client(llm_client)
    
    return await llm_router.route_task_with_llm(task, num_agents)


def register_protocol(agent_id: str, protocol_name: str, 
                     strengths: List[str] = None, best_for: List[str] = None):
    """Convenience function for registering protocol agents."""
    llm_router.register_protocol_agent(agent_id, protocol_name, strengths, best_for)


def update_protocol_metrics(agent_id: str, response_time: float, 
                          success: bool, load_change: int = 0):
    """Convenience function for updating protocol metrics."""
    llm_router.update_agent_metrics(agent_id, response_time, success, load_change)


def get_router_stats() -> Dict[str, Any]:
    """Convenience function for getting routing statistics."""
    return llm_router.get_routing_statistics()
