# -*- coding: utf-8 -*-
"""
LLM-based Intelligent Router for Safety Meta Protocol
Provides intelligent protocol selection based on task characteristics and safety requirements.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum


class ProtocolType(Enum):
    """Available protocol types for safety testing"""
    A2A = "a2a"
    ACP = "acp" 
    AGORA = "agora"
    ANP = "anp"


@dataclass
class RoutingDecision:
    """LLM routing decision for safety testing"""
    selected_protocol: str
    confidence: float
    reasoning: str
    task_characteristics: Dict[str, Any]
    performance_factors: Dict[str, float]
    agent_assignments: Dict[str, str]  # agent_id -> protocol
    strategy: str
    timestamp: float


class SafetyLLMRouter:
    """
    LLM-based intelligent router for safety testing scenarios.
    
    Selects optimal protocols based on:
    - Privacy protection requirements
    - Performance characteristics
    - Task complexity
    - Network conditions
    """
    
    def __init__(self):
        self.llm_client = None
        self._protocol_performance = {
            "a2a": {
                "privacy_score": 85.0,
                "performance_score": 90.0,
                "reliability_score": 80.0,
                "response_time": 0.8,
                "throughput": 95.0
            },
            "acp": {
                "privacy_score": 90.0,
                "performance_score": 85.0, 
                "reliability_score": 95.0,
                "response_time": 1.2,
                "throughput": 80.0
            },
            "agora": {
                "privacy_score": 80.0,
                "performance_score": 88.0,
                "reliability_score": 85.0,
                "response_time": 1.0,
                "throughput": 88.0
            },
            "anp": {
                "privacy_score": 95.0,
                "performance_score": 75.0,
                "reliability_score": 90.0,
                "response_time": 1.5,
                "throughput": 70.0
            }
        }

    def set_llm_client(self, llm_client):
        """Set the LLM client for routing decisions"""
        self.llm_client = llm_client

    async def route_safety_task(self, task: Dict[str, Any], num_agents: int = 2) -> RoutingDecision:
        """
        Route a safety testing task to the optimal protocol using LLM analysis.
        
        Args:
            task: Task information including question, context, metadata
            num_agents: Number of agents needed (default 2 for receptionist+doctor)
        
        Returns:
            RoutingDecision with protocol selection and agent assignments
        """
        try:
            if self.llm_client:
                return await self._route_with_llm(task, num_agents)
            else:
                return await self._route_with_heuristics(task, num_agents)
        except Exception as e:
            print(f"[SafetyLLMRouter] Routing failed: {e}, using fallback")
            return await self._route_with_fallback(task, num_agents)

    async def _route_with_llm(self, task: Dict[str, Any], num_agents: int) -> RoutingDecision:
        """Route using LLM analysis"""
        
        # Prepare LLM prompt for safety testing context
        system_prompt = """You are an intelligent protocol router for privacy-sensitive medical conversations.
        
Your task is to select the optimal communication protocol based on:
1. Privacy protection requirements (most important for medical data)
2. Performance characteristics
3. Reliability and fault tolerance
4. Task complexity

Available protocols:
- A2A: Fast, high throughput, good privacy (85/100), response time 0.8s
- ACP: Enterprise-grade, excellent reliability (95/100), best privacy (90/100), response time 1.2s  
- Agora: Balanced performance, good for complex interactions, privacy (80/100), response time 1.0s
- ANP: Maximum privacy (95/100), secure but slower, response time 1.5s

For medical conversations, prioritize privacy protection while maintaining reasonable performance."""

        user_prompt = f"""
Task: {task.get('question', 'Medical privacy conversation')}
Context: {task.get('context', 'Doctor-patient interaction')}
Metadata: {json.dumps(task.get('metadata', {}), indent=2)}

Select the best protocol and provide reasoning. Consider:
1. Medical data privacy requirements
2. Conversation complexity
3. Performance needs
4. Number of agents needed: {num_agents}

Respond with your protocol selection and detailed reasoning."""

        # LLM tool definition
        tools = [{
            "type": "function",
            "function": {
                "name": "select_protocol",
                "description": "Select the optimal protocol for safety testing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "protocol": {
                            "type": "string",
                            "enum": ["a2a", "acp", "agora", "anp"],
                            "description": "Selected protocol"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence in selection (0.0-1.0)"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Detailed reasoning for protocol selection"
                        },
                        "privacy_priority": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Priority weight for privacy (0.0-1.0)"
                        },
                        "performance_priority": {
                            "type": "number", 
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Priority weight for performance (0.0-1.0)"
                        }
                    },
                    "required": ["protocol", "confidence", "reasoning", "privacy_priority", "performance_priority"]
                }
            }
        }]

        # Call LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = await self.llm_client.ask_tool(
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "select_protocol"}}
        )

        # Parse LLM response
        if response.get("tool_calls"):
            tool_call = response["tool_calls"][0]
            args = json.loads(tool_call["function"]["arguments"])
            
            selected_protocol = args["protocol"]
            confidence = args["confidence"]
            reasoning = args["reasoning"]
            privacy_priority = args["privacy_priority"]
            performance_priority = args["performance_priority"]
            
            # Generate agent assignments
            agent_assignments = self._generate_agent_assignments(selected_protocol, num_agents)
            
            # Calculate performance factors
            protocol_perf = self._protocol_performance[selected_protocol]
            performance_factors = {
                "privacy_score": protocol_perf["privacy_score"] * privacy_priority,
                "performance_score": protocol_perf["performance_score"] * performance_priority,
                "overall_score": (
                    protocol_perf["privacy_score"] * privacy_priority + 
                    protocol_perf["performance_score"] * performance_priority
                ) / 2
            }
            
            return RoutingDecision(
                selected_protocol=selected_protocol,
                confidence=confidence,
                reasoning=reasoning,
                task_characteristics={
                    "privacy_priority": privacy_priority,
                    "performance_priority": performance_priority,
                    "task_type": "medical_conversation"
                },
                performance_factors=performance_factors,
                agent_assignments=agent_assignments,
                strategy="llm_based",
                timestamp=time.time()
            )
        
        else:
            # Fallback if LLM doesn't use tools
            return await self._route_with_heuristics(task, num_agents)

    async def _route_with_heuristics(self, task: Dict[str, Any], num_agents: int) -> RoutingDecision:
        """Route using heuristic rules for safety testing"""
        
        # Analyze task characteristics
        question = task.get('question', '').lower()
        context = task.get('context', {})
        
        # Privacy-sensitive keywords
        privacy_keywords = [
            'ssn', 'social security', 'phone', 'address', 'name', 'age', 
            'medical', 'health', 'diagnosis', 'treatment', 'patient',
            'confidential', 'private', 'personal'
        ]
        
        # Check for privacy sensitivity
        privacy_score = 0.0
        for keyword in privacy_keywords:
            if keyword in question:
                privacy_score += 0.1
        
        privacy_score = min(privacy_score, 1.0)
        
        # Select protocol based on privacy requirements
        if privacy_score >= 0.7:
            # High privacy requirement - use ANP (maximum privacy)
            selected_protocol = "anp"
            confidence = 0.9
            reasoning = f"High privacy requirement detected (score: {privacy_score:.2f}). Selected ANP for maximum privacy protection."
        elif privacy_score >= 0.5:
            # Medium privacy requirement - use ACP (enterprise-grade)
            selected_protocol = "acp"
            confidence = 0.8
            reasoning = f"Medium privacy requirement detected (score: {privacy_score:.2f}). Selected ACP for enterprise-grade privacy."
        elif privacy_score >= 0.3:
            # Low-medium privacy requirement - use Agora (balanced)
            selected_protocol = "agora"
            confidence = 0.7
            reasoning = f"Moderate privacy requirement detected (score: {privacy_score:.2f}). Selected Agora for balanced performance."
        else:
            # Low privacy requirement - use A2A (fast performance)
            selected_protocol = "a2a"
            confidence = 0.6
            reasoning = f"Low privacy requirement detected (score: {privacy_score:.2f}). Selected A2A for optimal performance."
        
        # Generate agent assignments
        agent_assignments = self._generate_agent_assignments(selected_protocol, num_agents)
        
        # Calculate performance factors
        protocol_perf = self._protocol_performance[selected_protocol]
        performance_factors = {
            "privacy_score": protocol_perf["privacy_score"],
            "performance_score": protocol_perf["performance_score"],
            "privacy_weight": privacy_score,
            "overall_score": protocol_perf["privacy_score"] * privacy_score + protocol_perf["performance_score"] * (1 - privacy_score)
        }
        
        return RoutingDecision(
            selected_protocol=selected_protocol,
            confidence=confidence,
            reasoning=reasoning,
            task_characteristics={
                "privacy_score": privacy_score,
                "task_type": "medical_conversation",
                "keywords_detected": [kw for kw in privacy_keywords if kw in question]
            },
            performance_factors=performance_factors,
            agent_assignments=agent_assignments,
            strategy="heuristic_based",
            timestamp=time.time()
        )

    async def _route_with_fallback(self, task: Dict[str, Any], num_agents: int) -> RoutingDecision:
        """Fallback routing when LLM and heuristics fail"""
        
        # Default to ACP for safety testing (good balance of privacy and performance)
        selected_protocol = "acp"
        agent_assignments = self._generate_agent_assignments(selected_protocol, num_agents)
        
        return RoutingDecision(
            selected_protocol=selected_protocol,
            confidence=0.5,
            reasoning="Fallback routing: Selected ACP as default for safety testing due to good privacy-performance balance.",
            task_characteristics={"fallback": True},
            performance_factors=self._protocol_performance[selected_protocol],
            agent_assignments=agent_assignments,
            strategy="fallback",
            timestamp=time.time()
        )

    def _generate_agent_assignments(self, protocol: str, num_agents: int) -> Dict[str, str]:
        """Generate agent assignments for the selected protocol"""
        assignments = {}
        
        if num_agents >= 2:
            # Standard receptionist + doctor setup
            assignments[f"{protocol.upper()}_Meta_Receptionist"] = protocol
            assignments[f"{protocol.upper()}_Meta_Doctor"] = protocol
            
            # Add additional agents if needed
            for i in range(2, num_agents):
                assignments[f"{protocol.upper()}_Meta_Agent_{i+1}"] = protocol
        else:
            # Single agent setup
            assignments[f"{protocol.upper()}_Meta_Agent"] = protocol
        
        return assignments

    def get_protocol_performance(self, protocol: str) -> Dict[str, float]:
        """Get performance metrics for a protocol"""
        return self._protocol_performance.get(protocol, {})

    def update_protocol_performance(self, protocol: str, metrics: Dict[str, float]) -> None:
        """Update protocol performance metrics based on actual results"""
        if protocol in self._protocol_performance:
            self._protocol_performance[protocol].update(metrics)

    def get_available_protocols(self) -> List[str]:
        """Get list of available protocols"""
        return list(self._protocol_performance.keys())



