"""
LLM Router for GAIA Meta Protocol.
Provides intelligent protocol selection based on GAIA task characteristics.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class ProtocolInfo:
    """Information about a protocol's capabilities and performance."""
    name: str
    agent_id: str
    strengths: List[str]
    best_for: List[str]
    current_load: int = 0
    avg_response_time: float = 0.0
    success_rate: float = 0.0


@dataclass
class RoutingDecision:
    """LLM routing decision for protocol selection."""
    selected_protocols: List[str]
    agent_assignments: Dict[str, str]  # agent_id -> protocol
    reasoning: str
    confidence: float
    strategy: str


class GAIALLMRouter:
    """
    LLM-based intelligent router specifically designed for GAIA tasks.
    
    Selects optimal protocols based on GAIA task characteristics:
    - Task complexity and type
    - Required tools and capabilities  
    - Domain areas and content type
    - Performance requirements
    """
    
    def __init__(self, llm_client=None):
        self.protocols: Dict[str, ProtocolInfo] = {}
        self.task_history: List[Dict[str, Any]] = []
        self.llm_client = llm_client
        
        # Initialize protocol information based on GAIA requirements
        self._initialize_gaia_protocol_info()
    
    def _initialize_gaia_protocol_info(self):
        """Initialize protocol information optimized for GAIA tasks."""
        
        # A2A Protocol - Fast for computational tasks and data transfer
        self.protocols["a2a"] = ProtocolInfo(
            name="a2a",
            agent_id="A2A-Worker",
            strengths=[
                "High throughput for computational tasks",
                "Fast data transfer between agents",
                "Good for python_execute and file operations",
                "Efficient for sequential workflows"
            ],
            best_for=[
                "computational_task",
                "data_processing",
                "file_manipulation", 
                "python_execute tool",
                "statistical_analysis"
            ],
            current_load=0,
            avg_response_time=7.39,
            success_rate=0.596
        )
        
        # ACP Protocol - Reliable for complex reasoning and synthesis
        self.protocols["acp"] = ProtocolInfo(
            name="acp",
            agent_id="ACP-Worker",
            strengths=[
                "Reliable for complex reasoning tasks",
                "Good for create_chat_completion",
                "Stable for final synthesis",
                "Enterprise-grade reliability"
            ],
            best_for=[
                "research_task",
                "reasoning_task",
                "create_chat_completion tool",
                "final_synthesis",
                "evidence_analysis"
            ],
            current_load=0,
            avg_response_time=7.83,
            success_rate=0.590
        )
        
        # Agora Protocol - Balanced performance for diverse tasks
        self.protocols["agora"] = ProtocolInfo(
            name="agora",
            agent_id="Agora-Worker", 
            strengths=[
                "Balanced performance across task types",
                "Good for browser_use and web search",
                "Reliable for information gathering",
                "Stable for multi-step workflows"
            ],
            best_for=[
                "research_task",
                "information_gathering",
                "browser_use tool",
                "web_search",
                "document_analysis"
            ],
            current_load=0,
            avg_response_time=7.10,
            success_rate=0.600
        )
        
        # ANP Protocol - Secure for sensitive tasks and high accuracy
        self.protocols["anp"] = ProtocolInfo(
            name="anp",
            agent_id="ANP-Worker",
            strengths=[
                "High accuracy for critical tasks",
                "Secure communication",
                "Good for sensitive data handling",
                "Best response quality"
            ],
            best_for=[
                "critical_analysis",
                "sensitive_data",
                "high_accuracy_required",
                "security_critical",
                "academic_research"
            ],
            current_load=0,
            avg_response_time=6.76,
            success_rate=0.610
        )

    def set_llm_client(self, llm_client):
        """Set the LLM client for making routing decisions."""
        self.llm_client = llm_client

    async def route_gaia_task(self, task_analysis: Dict[str, Any], 
                             agents_config: List[Dict[str, Any]]) -> RoutingDecision:
        """
        Route GAIA task using LLM-based intelligent protocol selection.
        
        Args:
            task_analysis: GAIA task analysis including type, complexity, tools, domains
            agents_config: List of agent configurations
            
        Returns:
            RoutingDecision with protocol assignments for each agent
        """
        try:
            if not self.llm_client:
                return self._fallback_routing(task_analysis, agents_config)
            
            # Create LLM prompt for GAIA-specific protocol selection
            system_prompt = self._create_gaia_system_prompt()
            user_prompt = self._create_gaia_user_prompt(task_analysis, agents_config)
            
            # Call LLM for routing decision
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Define tool schema for protocol selection
            tools = [{
                "type": "function",
                "function": {
                    "name": "select_protocols_for_gaia",
                    "description": "Select optimal protocols for GAIA agents based on task analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selected_protocols": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["a2a", "acp", "agora", "anp"]},
                                "description": "List of protocols to use for this task"
                            },
                            "agent_assignments": {
                                "type": "object",
                                "description": "Mapping of agent_id to protocol",
                                "additionalProperties": {"type": "string", "enum": ["a2a", "acp", "agora", "anp"]}
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Explanation for protocol selection decisions"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Confidence level in the routing decision"
                            }
                        },
                        "required": ["selected_protocols", "agent_assignments", "reasoning", "confidence"]
                    }
                }
            }]
            
            # Make LLM call with tool calling
            response = await self.llm_client.ask_tool(
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "select_protocols_for_gaia"}}
            )
            
            # Extract and validate routing decision
            if response and response.get("tool_calls"):
                tool_call = response["tool_calls"][0]
                if tool_call.get("function"):
                    result = json.loads(tool_call["function"]["arguments"])
                    
                    return RoutingDecision(
                        selected_protocols=result.get("selected_protocols", ["a2a"]),
                        agent_assignments=result.get("agent_assignments", {}),
                        reasoning=result.get("reasoning", "LLM routing decision"),
                        confidence=result.get("confidence", 0.5),
                        strategy="llm_based"
                    )
            
            # Fallback if LLM response is invalid
            return self._fallback_routing(task_analysis, agents_config)
            
        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            return self._fallback_routing(task_analysis, agents_config)

    def _create_gaia_system_prompt(self) -> str:
        """Create system prompt optimized for GAIA task routing."""
        return """You are an intelligent protocol router for GAIA (General Agent Interaction Architecture) tasks.
Your goal is to select optimal communication protocols for multi-agent systems working on diverse academic and research tasks.

AVAILABLE PROTOCOLS (based on performance data):
- A2A: Fast data transfer (7.39s avg), good for computational tasks, python_execute tool
- ACP: Reliable reasoning (7.83s avg), good for create_chat_completion, final synthesis  
- Agora: Balanced performance (7.10s avg), good for browser_use, web research
- ANP: High accuracy (6.76s avg), secure, good for academic research, critical analysis

GAIA TASK CHARACTERISTICS:
- Research tasks: Often require web search, document analysis, academic paper retrieval
- Computational tasks: Need python execution, statistical analysis, data processing
- Reasoning tasks: Require complex logical reasoning and synthesis
- Mixed tasks: Combine multiple capabilities and tool types

PROTOCOL SELECTION STRATEGY:
- For browser_use tool: Prefer Agora (optimized for web interactions)
- For python_execute tool: Prefer A2A (fast computational processing)
- For create_chat_completion tool: Prefer ACP (reliable reasoning) or ANP (high accuracy)
- For document analysis: Prefer ANP (high accuracy) or Agora (balanced)
- For final synthesis: Prefer ACP (reliable) or ANP (accurate)

ASSIGNMENT REQUIREMENTS:
- Match protocols to agent tools and roles
- Consider task complexity and domain areas
- Balance performance and reliability
- Ensure appropriate protocol for each agent's specific function

Use tool calling to provide structured protocol selection."""

    def _create_gaia_user_prompt(self, task_analysis: Dict[str, Any], 
                                agents_config: List[Dict[str, Any]]) -> str:
        """Create user prompt with specific GAIA task information."""
        
        # Extract task information
        task_type = task_analysis.get("task_type", "unknown")
        complexity = task_analysis.get("complexity", "medium")
        required_tools = task_analysis.get("required_tools", [])
        domain_areas = task_analysis.get("domain_areas", [])
        
        # Create agent descriptions
        agent_descriptions = []
        for agent_config in agents_config:
            agent_desc = {
                "id": agent_config["id"],
                "name": agent_config["name"],
                "tool": agent_config["tool"],
                "role": agent_config["role"],
                "port": agent_config["port"]
            }
            agent_descriptions.append(agent_desc)
        
        return f"""
GAIA TASK ANALYSIS:
- Task Type: {task_type}
- Complexity: {complexity}
- Required Tools: {required_tools}
- Domain Areas: {domain_areas}

AGENTS TO ASSIGN ({len(agents_config)} agents):
{json.dumps(agent_descriptions, indent=2)}

ASSIGNMENT REQUIREMENTS:
- Must assign exactly {len(agents_config)} agents
- Match protocols to agent tools:
  * browser_use → Agora (optimized for web interactions)
  * python_execute → A2A (fast computational processing)  
  * create_chat_completion → ACP/ANP (reliable reasoning/high accuracy)
  * str_replace_editor → A2A/Agora (file operations)
- Consider agent roles and task complexity
- Optimize for the specific GAIA task characteristics

Use the select_protocols_for_gaia tool to make your decision.
"""

    def _fallback_routing(self, task_analysis: Dict[str, Any], 
                         agents_config: List[Dict[str, Any]]) -> RoutingDecision:
        """Fallback routing when LLM is not available."""
        
        # Simple heuristic-based assignment for GAIA
        agent_assignments = {}
        selected_protocols = set()
        
        for agent_config in agents_config:
            agent_id = str(agent_config["id"])
            tool = agent_config.get("tool", "")
            role = agent_config.get("role", "")
            
            # Protocol assignment based on tool type
            if tool == "browser_use":
                protocol = "agora"  # Best for web interactions
            elif tool == "python_execute":
                protocol = "a2a"    # Fast for computational tasks
            elif tool == "create_chat_completion":
                protocol = "acp"    # Reliable for reasoning
            elif tool == "str_replace_editor":
                protocol = "a2a"    # Good for file operations
            else:
                protocol = "agora"  # Default balanced choice
            
            agent_assignments[agent_id] = protocol
            selected_protocols.add(protocol)
        
        return RoutingDecision(
            selected_protocols=list(selected_protocols),
            agent_assignments=agent_assignments,
            reasoning="Fallback heuristic routing based on tool types",
            confidence=0.3,
            strategy="heuristic_fallback"
        )

    def update_agent_metrics(self, agent_id: str, response_time: float, 
                           success: bool, load_change: int = 0):
        """Update agent performance metrics for future routing decisions."""
        # Find the protocol for this agent
        for protocol_info in self.protocols.values():
            if protocol_info.agent_id == agent_id:
                # Update metrics
                protocol_info.current_load = max(0, protocol_info.current_load + load_change)
                
                # Update performance history
                # This could be expanded to maintain rolling averages
                break

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing performance statistics."""
        return {
            "total_tasks_routed": len(self.task_history),
            "protocols_available": list(self.protocols.keys()),
            "llm_client_available": self.llm_client is not None,
            "protocol_load": {name: info.current_load for name, info in self.protocols.items()}
        }
