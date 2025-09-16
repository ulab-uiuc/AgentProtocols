"""
Intelligent Network Manager - Integrates IntelligentRouter with AgentNetwork

This module extends the existing AgentNetwork with intelligent routing capabilities,
allowing for automatic protocol selection based on task requirements.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional

try:
    from .network import AgentNetwork
    from .base_agent import BaseAgent
    from .intelligent_router import intelligent_router, RoutingDecision, TaskRequirement
except ImportError:
    from src.core.network import AgentNetwork
    from src.core.base_agent import BaseAgent
    from src.core.intelligent_router import intelligent_router, RoutingDecision, TaskRequirement


class IntelligentAgentNetwork(AgentNetwork):
    """
    Enhanced AgentNetwork with intelligent routing capabilities.
    
    This network automatically selects the best protocol agents for each task
    based on task requirements and agent capabilities.
    """
    
    def __init__(self):
        super().__init__()
        self.router = intelligent_router
        self.task_queue = asyncio.Queue()
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.routing_enabled = True
        
        print("🧠 Intelligent Agent Network initialized")
    
    async def register_protocol_agent(self, agent: BaseAgent, protocol_name: str, 
                                    capabilities: List[str] = None,
                                    specialties: List[str] = None):
        """
        Register a protocol agent with its capabilities.
        
        Args:
            agent: BaseAgent instance
            protocol_name: Protocol name (a2a, acp, anp, agora)
            capabilities: List of capability strings
            specialties: List of specialty strings
        """
        # Register with parent network
        await self.register_agent(agent)
        
        # Create protocol profile for intelligent routing
        from .intelligent_router import ProtocolProfile, ProtocolCapability
        
        # Map capability strings to enums
        capability_mapping = {
            "high_throughput": ProtocolCapability.HIGH_THROUGHPUT,
            "low_latency": ProtocolCapability.LOW_LATENCY,
            "secure_communication": ProtocolCapability.SECURE_COMMUNICATION,
            "complex_reasoning": ProtocolCapability.COMPLEX_REASONING,
            "structured_data": ProtocolCapability.STRUCTURED_DATA,
            "real_time": ProtocolCapability.REAL_TIME,
            "fault_tolerant": ProtocolCapability.FAULT_TOLERANT
        }
        
        # Convert capability strings to enums
        agent_capabilities = []
        if capabilities:
            for cap in capabilities:
                if cap in capability_mapping:
                    agent_capabilities.append(capability_mapping[cap])
        
        # Use default capabilities if none provided
        if not agent_capabilities:
            if protocol_name == "a2a":
                agent_capabilities = [ProtocolCapability.HIGH_THROUGHPUT, ProtocolCapability.STRUCTURED_DATA]
            elif protocol_name == "acp":
                agent_capabilities = [ProtocolCapability.SECURE_COMMUNICATION, ProtocolCapability.FAULT_TOLERANT]
            elif protocol_name == "anp":
                agent_capabilities = [ProtocolCapability.SECURE_COMMUNICATION, ProtocolCapability.LOW_LATENCY]
            elif protocol_name == "agora":
                agent_capabilities = [ProtocolCapability.COMPLEX_REASONING, ProtocolCapability.HIGH_THROUGHPUT]
        
        # Create protocol profile
        profile = ProtocolProfile(
            protocol_name=protocol_name,
            agent_id=agent.agent_id,
            capabilities=agent_capabilities,
            avg_response_time=1.0,  # Will be updated based on actual performance
            success_rate=1.0,       # Will be updated based on actual performance
            current_load=0,
            max_concurrent=10,      # Default, can be configured
            specialties=specialties or []
        )
        
        # Register with router
        self.router.register_protocol_agent(agent.agent_id, profile)
        
        print(f"🔧 Registered intelligent protocol agent: {agent.agent_id} ({protocol_name})")
        print(f"   Capabilities: {[cap.value for cap in agent_capabilities]}")
        if specialties:
            print(f"   Specialties: {specialties}")
    
    async def execute_task_intelligently(self, task: Dict[str, Any], 
                                       timeout: float = 30.0) -> Dict[str, Any]:
        """
        Execute a task using intelligent routing to select the best agent(s).
        
        Args:
            task: Task dictionary containing question, context, metadata
            timeout: Maximum execution time
            
        Returns:
            Task execution result with routing information
        """
        if not self.routing_enabled:
            return await self._execute_task_simple(task, timeout)
        
        start_time = time.time()
        task_id = f"task_{int(start_time * 1000)}"
        
        try:
            # Get available agents
            available_agents = list(self._agents.keys())
            if not available_agents:
                return {
                    "task_id": task_id,
                    "success": False,
                    "error": "No agents available",
                    "execution_time": 0.0
                }
            
            # Make routing decision
            routing_decision = await self.router.route_task(task, available_agents)
            
            if not routing_decision.selected_agents:
                return {
                    "task_id": task_id,
                    "success": False,
                    "error": "No suitable agents found",
                    "routing_decision": routing_decision,
                    "execution_time": time.time() - start_time
                }
            
            print(f"🎯 Routing Decision: {routing_decision.reasoning}")
            print(f"   Selected agents: {routing_decision.selected_agents}")
            print(f"   Strategy: {routing_decision.routing_strategy}")
            print(f"   Confidence: {routing_decision.confidence_score:.2%}")
            
            # Track active task
            self.active_tasks[task_id] = {
                "start_time": start_time,
                "selected_agents": routing_decision.selected_agents,
                "routing_decision": routing_decision
            }
            
            # Update agent loads
            for agent_id in routing_decision.selected_agents:
                self.router.update_agent_metrics(agent_id, 0, True, load_change=1)
            
            # Execute task based on routing strategy
            if routing_decision.routing_strategy == "single_agent":
                result = await self._execute_single_agent_task(
                    task, routing_decision.selected_agents[0], timeout
                )
            elif routing_decision.routing_strategy == "multi_agent_collaboration":
                result = await self._execute_multi_agent_task(
                    task, routing_decision.selected_agents, timeout
                )
            else:
                result = await self._execute_task_fallback(
                    task, routing_decision.selected_agents[0], timeout
                )
            
            execution_time = time.time() - start_time
            success = result.get("success", False)
            
            # Update agent performance metrics
            for agent_id in routing_decision.selected_agents:
                self.router.update_agent_metrics(
                    agent_id, execution_time, success, load_change=-1
                )
            
            # Add routing information to result
            result.update({
                "task_id": task_id,
                "routing_decision": {
                    "selected_agents": routing_decision.selected_agents,
                    "strategy": routing_decision.routing_strategy,
                    "confidence": routing_decision.confidence_score,
                    "reasoning": routing_decision.reasoning
                },
                "execution_time": execution_time
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Update metrics for failed task
            if task_id in self.active_tasks:
                for agent_id in self.active_tasks[task_id]["selected_agents"]:
                    self.router.update_agent_metrics(agent_id, execution_time, False, load_change=-1)
            
            return {
                "task_id": task_id,
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
        
        finally:
            # Clean up active task tracking
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def _execute_single_agent_task(self, task: Dict[str, Any], 
                                       agent_id: str, timeout: float) -> Dict[str, Any]:
        """Execute task on a single selected agent."""
        if agent_id not in self._agents:
            return {"success": False, "error": f"Agent {agent_id} not found"}
        
        agent = self._agents[agent_id]
        
        try:
            # Prepare payload
            payload = {
                "text": task.get("question", ""),
                "question": task.get("question", ""),
                "context": task.get("context", ""),
                "metadata": task.get("metadata", {})
            }
            
            # Execute with timeout
            response = await asyncio.wait_for(
                agent.send(agent_id, payload),  # Self-send for processing
                timeout=timeout
            )
            
            return {
                "success": True,
                "response": response,
                "agent_id": agent_id,
                "strategy": "single_agent"
            }
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Task timed out after {timeout}s",
                "agent_id": agent_id
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent_id": agent_id
            }
    
    async def _execute_multi_agent_task(self, task: Dict[str, Any], 
                                      agent_ids: List[str], timeout: float) -> Dict[str, Any]:
        """Execute task using multiple agents in collaboration."""
        if len(agent_ids) < 2:
            return await self._execute_single_agent_task(task, agent_ids[0], timeout)
        
        try:
            # Execute on all selected agents concurrently
            tasks_to_run = []
            for agent_id in agent_ids:
                if agent_id in self._agents:
                    task_coro = self._execute_single_agent_task(task, agent_id, timeout / 2)
                    tasks_to_run.append(task_coro)
            
            if not tasks_to_run:
                return {"success": False, "error": "No valid agents found"}
            
            # Wait for all agents to complete
            results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
            
            # Process results
            successful_results = []
            errors = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append(f"Agent {agent_ids[i]}: {str(result)}")
                elif isinstance(result, dict) and result.get("success"):
                    successful_results.append(result)
                else:
                    errors.append(f"Agent {agent_ids[i]}: {result.get('error', 'Unknown error')}")
            
            if successful_results:
                # Combine successful results
                combined_response = self._combine_agent_responses(successful_results)
                return {
                    "success": True,
                    "response": combined_response,
                    "agent_ids": agent_ids,
                    "strategy": "multi_agent_collaboration",
                    "individual_results": successful_results,
                    "errors": errors if errors else None
                }
            else:
                return {
                    "success": False,
                    "error": "All agents failed",
                    "errors": errors,
                    "agent_ids": agent_ids
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent_ids": agent_ids
            }
    
    def _combine_agent_responses(self, results: List[Dict[str, Any]]) -> str:
        """Combine multiple agent responses into a coherent answer."""
        if not results:
            return "No responses received"
        
        if len(results) == 1:
            return str(results[0].get("response", ""))
        
        # Simple combination strategy - can be enhanced with LLM-based synthesis
        combined_parts = []
        for i, result in enumerate(results):
            agent_id = result.get("agent_id", f"Agent_{i}")
            response = str(result.get("response", ""))
            if response:
                combined_parts.append(f"[{agent_id}]: {response}")
        
        if combined_parts:
            return "\n\n".join(combined_parts)
        else:
            return "No meaningful responses received"
    
    async def _execute_task_simple(self, task: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        """Fallback to simple round-robin execution when routing is disabled."""
        available_agents = list(self._agents.keys())
        if not available_agents:
            return {"success": False, "error": "No agents available"}
        
        # Simple round-robin selection
        agent_id = available_agents[0]  # Or implement round-robin logic
        return await self._execute_single_agent_task(task, agent_id, timeout)
    
    async def _execute_task_fallback(self, task: Dict[str, Any], 
                                   agent_id: str, timeout: float) -> Dict[str, Any]:
        """Fallback execution method."""
        return await self._execute_single_agent_task(task, agent_id, timeout)
    
    def enable_intelligent_routing(self, enabled: bool = True):
        """Enable or disable intelligent routing."""
        self.routing_enabled = enabled
        status = "enabled" if enabled else "disabled"
        print(f"🧠 Intelligent routing {status}")
    
    def get_network_intelligence_stats(self) -> Dict[str, Any]:
        """Get comprehensive network and routing statistics."""
        base_stats = self.snapshot_metrics()
        routing_stats = self.router.get_routing_statistics()
        
        return {
            "network_metrics": base_stats,
            "routing_statistics": routing_stats,
            "active_tasks": len(self.active_tasks),
            "routing_enabled": self.routing_enabled,
            "registered_agents": len(self._agents),
            "protocol_agents": {
                profile.protocol_name: profile.agent_id 
                for profile in self.router.protocol_profiles.values()
            }
        }
    
    async def benchmark_agents(self, test_tasks: List[Dict[str, Any]], 
                             iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark all agents with test tasks to calibrate routing decisions.
        
        Args:
            test_tasks: List of test tasks
            iterations: Number of iterations per task per agent
            
        Returns:
            Benchmark results
        """
        print(f"🏁 Starting agent benchmark with {len(test_tasks)} tasks, {iterations} iterations")
        
        benchmark_results = {}
        
        for agent_id in self._agents.keys():
            benchmark_results[agent_id] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "total_time": 0.0,
                "avg_response_time": 0.0,
                "success_rate": 0.0,
                "task_results": []
            }
        
        # Run benchmark tasks
        for task_idx, task in enumerate(test_tasks):
            for agent_id in self._agents.keys():
                for iteration in range(iterations):
                    try:
                        start_time = time.time()
                        result = await self._execute_single_agent_task(task, agent_id, 30.0)
                        execution_time = time.time() - start_time
                        
                        stats = benchmark_results[agent_id]
                        stats["total_tasks"] += 1
                        stats["total_time"] += execution_time
                        
                        if result.get("success"):
                            stats["successful_tasks"] += 1
                        
                        stats["task_results"].append({
                            "task_idx": task_idx,
                            "iteration": iteration,
                            "success": result.get("success", False),
                            "execution_time": execution_time,
                            "error": result.get("error")
                        })
                        
                    except Exception as e:
                        stats = benchmark_results[agent_id]
                        stats["total_tasks"] += 1
                        stats["task_results"].append({
                            "task_idx": task_idx,
                            "iteration": iteration,
                            "success": False,
                            "execution_time": 0.0,
                            "error": str(e)
                        })
        
        # Calculate final statistics
        for agent_id, stats in benchmark_results.items():
            if stats["total_tasks"] > 0:
                stats["success_rate"] = stats["successful_tasks"] / stats["total_tasks"]
                stats["avg_response_time"] = stats["total_time"] / stats["total_tasks"]
                
                # Update router with benchmark results
                self.router.update_agent_metrics(
                    agent_id, stats["avg_response_time"], 
                    stats["success_rate"] > 0.5, 0
                )
        
        print("✅ Agent benchmark completed")
        return benchmark_results


# Factory function for easy creation
def create_intelligent_network() -> IntelligentAgentNetwork:
    """Create a new IntelligentAgentNetwork instance."""
    return IntelligentAgentNetwork()
