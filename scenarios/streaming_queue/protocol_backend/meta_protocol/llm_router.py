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
        """Initialize protocol information optimized for streaming queue pressure testing."""
        
        # A2A Protocol - Fast and responsive
        self.protocols["a2a"] = ProtocolInfo(
            name="a2a",
            agent_id="A2A-Worker",
            strengths=[
                "Very fast response time",
                "High throughput processing", 
                "Minimal latency",
                "Optimized for speed"
            ],
            best_for=[
                "Quick QA tasks",
                "Simple factual questions",
                "High-speed processing",
                "Time-critical responses"
            ],
            current_load=0,
            avg_response_time=3.42,  # updated from pressure test (seconds)
            success_rate=0.95
        )
        
        # ACP Protocol - Fast and reliable
        self.protocols["acp"] = ProtocolInfo(
            name="acp",
            agent_id="ACP-Worker", 
            strengths=[
                "Fast response time",
                "Reliable processing",
                "Stable performance",
                "Quick turnaround"
            ],
            best_for=[
                "Rapid task completion",
                "Consistent performance",
                "Batch processing",
                "Efficient workflows"
            ],
            current_load=0,
            avg_response_time=4.00,  # updated from pressure test (seconds)
            success_rate=0.98
        )
        
        # Agora Protocol - Stable communication
        self.protocols["agora"] = ProtocolInfo(
            name="agora",
            agent_id="Agora-Worker",
            strengths=[
                "Stable communication",
                "Reliable connections",
                "Consistent performance",
                "Good error handling"
            ],
            best_for=[
                "Complex questions",
                "Multi-step reasoning",
                "Stable long-running tasks",
                "Research-oriented queries"
            ],
            current_load=0,
            avg_response_time=9.00,  # updated from pressure test (seconds)
            success_rate=0.96
        )
        
        # ANP Protocol - Secure but slower
        self.protocols["anp"] = ProtocolInfo(
            name="anp",
            agent_id="ANP-Worker",
            strengths=[
                "Maximum security",
                "Privacy protection",
                "Encrypted communication",
                "Secure processing"
            ],
            best_for=[
                "Security-sensitive tasks",
                "Privacy-critical questions",
                "Confidential data processing",
                "Secure communications"
            ],
            current_load=0,
            avg_response_time=4.78,  # updated from pressure test (seconds)
            success_rate=0.92
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
        
        # Create LLM prompt for protocol selection optimized for pressure testing
        system_prompt = """You are "ProtoRouter", a deterministic protocol selector for streaming queue multi-agent systems.
Your job: For each worker agent, pick exactly ONE protocol from {A2A, ACP, Agora, ANP} that best matches the requirements.
You must justify choices with transparent, capability-level reasoning and produce machine-checkable JSON only.

--------------------------------------------
1) Canonical Feature Model (authoritative; use this only)
--------------------------------------------
A2A (Agent-to-Agent Protocol)
- Transport/Model: HTTP + JSON-RPC + SSE; first-class long-running tasks; task/artifact lifecycle.
- Capability/UX: Multimodal messages and explicit UI capability negotiation.
- Integration: Complements MCP; broad vendor ecosystem.
- Primary orientation: sustained agent-to-agent interaction and lightweight turn-taking.

ACP (Agent Communication Protocol)
- Transport/Model: REST-first over HTTP; MIME-based multimodality; async-first with streaming support.
- Discovery: Agent Manifest; clear single/multi-server topologies.
- Integration: Minimal SDK expectations; straightforward REST exposure.
- Primary orientation: structured, addressable operations with clear progress semantics and repeatable handling at scale.

Agora (Meta-Protocol)
- Positioning: Minimal "meta" wrapper; sessions carry a protocolHash binding to a plain-text protocol doc.
- Discovery: /.well-known returns supported protocol hashes.
- Primary orientation: explicit procedure governance - selecting and following a concrete routine/version.

ANP (Agent Network Protocol)
- Positioning: Network & trust substrate; three layers: identity+E2E, meta-protocol, application protocols.
- Security/Trust: W3C DID-based identities; ECDHE-based end-to-end encryption.
- Primary orientation: relationship assurance and information protection across boundaries.

--------------------------------------------
2) Streaming Queue Scenario Requirements
--------------------------------------------
SCENARIO CHARACTERISTICS:
- High-throughput question-answering system
- Star topology: 1 coordinator + 4 workers
- Batch processing with queue-based task distribution
- Focus: minimizing end-to-end latency while maintaining reliability

SELECTION PRIORITY ORDER:
1. Identity/Confidentiality requirements → ANP (if E2E/DID required)
2. Operation semantics → ACP (REST/structured repeatable ops) vs A2A (sustained interaction)
3. Interaction preferences → streaming, async processing

ASSIGNMENT REQUIREMENTS:
- Total agents: 4 workers (assign exactly 4)
- Match protocols to workload characteristics based on capabilities
- Consider: async processing, load balancing, structured operations
- Provide clear reasoning citing capability matches only
- No numeric performance claims in rationale

Use tool calling to provide structured protocol selection."""

        user_prompt = f"""
PRESSURE TEST TASK:
Question: {task.get('question', '')}
Context: {task.get('context', '')}
Security Required: {task.get('metadata', {}).get('requires_security', False)}
Privacy Critical: {task.get('metadata', {}).get('privacy_critical', False)}

Current Protocol Status:
{json.dumps(protocol_descriptions, indent=2)}

ASSIGNMENT REQUIREMENTS:
- Must assign exactly 4 agents (agent_1, agent_2, agent_3, agent_4)
- Optimize for MAXIMUM SPEED in pressure test environment
- Use A2A and ACP for fastest processing (unless security required)
- Include Agora for stability on complex tasks
- Use ANP only for security/privacy critical tasks

Use the select_protocols tool to make your decision."""

        try:
            # Call LLM for routing decision
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Use tool calling for structured output if available
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
                        
                        # Ensure we have exactly 4 agents for streaming queue
                        if len(agent_assignments) != 4:
                            # Auto-generate 4 agent assignments for pressure testing
                            agent_assignments = {}
                            # Default fast assignment for pressure testing
                            default_protocols = ["a2a", "acp", "agora", "a2a"]  # Favor fast protocols
                            if selected_protocols:
                                # Use LLM selected protocols, repeat if needed
                                for i in range(4):
                                    agent_id = f"agent_{i+1}"
                                    protocol = selected_protocols[i % len(selected_protocols)]
                                    agent_assignments[agent_id] = protocol
                            else:
                                # Use default fast protocols
                                for i in range(4):
                                    agent_id = f"agent_{i+1}"
                                    agent_assignments[agent_id] = default_protocols[i]
                        
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
        return self._fallback_routing_sync({}, num_agents)
    
    def _fallback_routing_sync(self, task: Dict[str, Any], num_agents: int) -> RoutingDecision:
        """Synchronous fallback routing when LLM is not available."""
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
    
    async def _fallback_routing(self, task: Dict[str, Any], num_agents: int) -> RoutingDecision:
        """Async fallback routing when LLM is not available."""
        return self._fallback_routing_sync(task, num_agents)
    
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
