"""
LLM-based Intelligent Router for Meta Protocol

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
        """Initialize protocol information based on actual fail storm recovery performance data."""
        
        # A2A Protocol - Based on actual performance: 178 tasks, 59.6% success, 19.1% answer rate
        self.protocols["a2a"] = ProtocolInfo(
            name="a2a",
            agent_id="A2A-Worker",
            strengths=[
                "High task throughput (178 tasks)",
                "Fast recovery time (6.0s)",
                "Moderate pre-fault latency (7.39s avg, 1.15s median)",
                "Good post-fault performance (6.94s avg)"
            ],
            best_for=[
                "High-volume task processing",
                "Quick recovery scenarios",
                "Moderate complexity questions",
                "Load-balanced workloads"
            ],
            current_load=0,
            avg_response_time=7.39,  # Pre-fault average from data
            success_rate=0.596  # 59.6% from actual data
        )
        
        # ACP Protocol - Based on actual performance: 156 tasks, 59.0% success, 17.9% answer rate
        self.protocols["acp"] = ProtocolInfo(
            name="acp",
            agent_id="ACP-Worker", 
            strengths=[
                "Excellent recovery performance (0.70s avg)",
                "Good fault tolerance (8.0s recovery time)",
                "Stable post-fault operation (7.22s avg)",
                "Most recovery tasks (22)"
            ],
            best_for=[
                "Fault recovery scenarios",
                "Critical system operations",
                "Recovery-phase processing",
                "Resilient task execution"
            ],
            current_load=0,
            avg_response_time=7.83,  # Pre-fault average from data
            success_rate=0.590  # 59.0% from actual data
        )
        
        # Agora Protocol - Based on actual performance: 180 tasks, 60.0% success, 20.0% answer rate
        self.protocols["agora"] = ProtocolInfo(
            name="agora",
            agent_id="Agora-Worker",
            strengths=[
                "Highest task count (180 tasks)",
                "Best overall success rate (60.0%)",
                "Highest answer rate (20.0%)",
                "Good pre-fault performance (7.10s avg, 1.06s median)"
            ],
            best_for=[
                "Maximum task throughput",
                "Best answer discovery",
                "Comprehensive question coverage",
                "High-success scenarios"
            ],
            current_load=0,
            avg_response_time=7.10,  # Pre-fault average from data
            success_rate=0.600  # 60.0% from actual data
        )
        
        # ANP Protocol - Based on actual performance: 164 tasks, 61.0% success, 22.0% answer rate
        self.protocols["anp"] = ProtocolInfo(
            name="anp",
            agent_id="ANP-Worker",
            strengths=[
                "Highest success rate (61.0%)",
                "Best answer rate (22.0%)",
                "Excellent pre-fault performance (6.76s avg, 1.01s median)",
                "Strong security with good performance"
            ],
            best_for=[
                "High-accuracy requirements",
                "Security-sensitive tasks",
                "Quality over quantity scenarios",
                "Encrypted communication needs"
            ],
            current_load=0,
            avg_response_time=6.76,  # Pre-fault average from data - actually fastest!
            success_rate=0.610  # 61.0% from actual data - highest success rate
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
        
        # Create LLM prompt for protocol selection optimized for fail storm recovery
        system_prompt = """You are an intelligent protocol router for a fail storm recovery system. 
Your goal is to select the optimal protocols for ALL agents to maximize answer discovery rate and minimize recovery time during fault scenarios.

ACTUAL PROTOCOL PERFORMANCE DATA (from real fail_storm_recovery tests):
- ANP: 61.0% success rate, 22.0% answer discovery rate, 6.76s avg response, 10.0s recovery time
- Agora: 60.0% success rate, 20.0% answer discovery rate, 7.10s avg response, 6.1s recovery time  
- A2A: 59.6% success rate, 19.1% answer discovery rate, 7.39s avg response, 6.0s recovery time
- ACP: 59.0% success rate, 17.9% answer discovery rate, 7.83s avg response, 8.0s recovery time

FAIL STORM RECOVERY CRITICAL METRICS:
1. ANSWER DISCOVERY RATE (most important) - ANP leads at 22.0%
2. RECOVERY TIME (second most important) - A2A leads at 6.0s
3. Overall success rate - ANP leads at 61.0%
4. Response time - ANP leads at 6.76s

ASSIGNMENT REQUIREMENTS:
- You must assign exactly 8 agents (agent_1 through agent_8)
- PRIORITIZE: Answer discovery rate and fast recovery time
- Consider fault tolerance: distribute across multiple protocols
- Balance performance vs resilience

OPTIMAL STRATEGY:
- Use ANP for highest answer discovery (22.0%) and best response time (6.76s)
- Use A2A for fastest recovery (6.0s) and good throughput
- Use Agora for balanced performance and stability
- AVOID ACP due to lowest answer rate (17.9%) unless specifically needed
- You can assign 0 agents to poor-performing protocols
- Focus on 1-2 best protocols rather than spreading across all protocols

Use tool calling to provide structured protocol selection for all 8 agents."""

        user_prompt = f"""
FAIL STORM RECOVERY TASK:
Question: {task.get('question', '')}
Context: {task.get('context', '')}
Security Required: {task.get('metadata', {}).get('requires_security', False)}
Privacy Critical: {task.get('metadata', {}).get('privacy_critical', False)}

Current Protocol Status:
{json.dumps(protocol_descriptions, indent=2)}

ASSIGNMENT REQUIREMENTS:
- Must assign exactly {num_agents} agents (agent0 through agent{num_agents-1})
- PRIORITIZE: Answer discovery rate (ANP: 22.0% > Agora: 20.0% > A2A: 19.1% > ACP: 17.9%)
- PRIORITIZE: Fast recovery time (A2A: 6.0s < Agora: 6.1s < ACP: 8.0s < ANP: 10.0s)
- Consider fault tolerance: distribute across multiple protocols for resilience
- Balance high answer discovery with fast recovery

RECOMMENDED DISTRIBUTION for {num_agents} agents:
- ANP: 3-4 agents (best answer discovery rate 22.0%)
- A2A: 2-3 agents (fastest recovery 6.0s)
- Agora: 1-2 agents (balanced performance)
- ACP: 0-1 agents (lowest priority due to poor answer rate)

IMPORTANT: You do NOT need to use all protocols. Some protocols can have 0 agents assigned.
Focus on the best performing protocols for the specific task requirements.
Prefer fewer protocols with better performance over using all protocols.

Use the select_protocols tool to make your decision."""

        try:
            # Call LLM for routing decision
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Create dynamic tool definition based on number of agents
            # Use agent0, agent1, agent2... to match fail_storm_recovery naming
            agent_properties = {}
            agent_required = []
            for i in range(num_agents):
                agent_id = f"agent{i}"
                agent_properties[agent_id] = {
                    "type": "string", 
                    "enum": ["anp", "agora", "a2a", "acp"],
                    "description": f"Protocol assignment for {agent_id}"
                }
                agent_required.append(agent_id)

            tools = [{
                "type": "function",
                "function": {
                    "name": "select_protocols",
                    "description": f"Select optimal protocols for {num_agents} agents in fail storm recovery",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selected_protocols": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["anp", "agora", "a2a", "acp"]},
                                "description": "List of selected protocol names",
                                "minItems": 1,
                                "maxItems": 4
                            },
                            "agent_assignments": {
                                "type": "object",
                                "description": f"Mapping of agent IDs to protocol names (must have exactly {num_agents} entries)",
                                "properties": agent_properties,
                                "required": agent_required,
                                "additionalProperties": False
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Detailed explanation focusing on answer discovery rate and recovery time optimization"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Confidence score (0-1)"
                            },
                            "strategy": {
                                "type": "string",
                                "enum": ["answer_discovery_optimized", "recovery_time_optimized", "balanced_performance"],
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
                        
                        # Ensure we have exactly the requested number of agents
                        if len(agent_assignments) != num_agents:
                            raise ValueError(f"LLM must assign exactly {num_agents} agents, got {len(agent_assignments)}")
                        
                        return RoutingDecision(
                            selected_protocols=selected_protocols,
                            agent_assignments=agent_assignments,
                            reasoning=reasoning,
                            confidence=confidence,
                            strategy=strategy
                        )
            
            # If tool calling fails, raise error - no fallback allowed
            raise RuntimeError("LLM tool calling failed - no fallback allowed")
            
        except Exception as e:
            print(f"❌ LLM routing failed: {e}")
            raise RuntimeError(f"LLM routing failed and fallback is disabled: {e}")
    
    
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
