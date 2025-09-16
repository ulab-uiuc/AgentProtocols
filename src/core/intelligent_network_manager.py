"""
Intelligent Network Manager for Agent Network SDK

Provides high-level network management with integrated intelligent routing capabilities.
This is the main entry point for SDK users who want intelligent protocol selection.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

try:
    from .network import AgentNetwork
    from .base_agent import BaseAgent
    from .router_interface import (
        RouterInterface, RouterType, ProtocolCapability, 
        RoutingRequest, RoutingResult, RouterFactory
    )
except ImportError:
    from src.core.network import AgentNetwork
    from src.core.base_agent import BaseAgent
    from src.core.router_interface import (
        RouterInterface, RouterType, ProtocolCapability,
        RoutingRequest, RoutingResult, RouterFactory
    )


class IntelligentNetworkManager:
    """
    High-level network manager with intelligent routing capabilities.
    
    This manager provides:
    - Automatic protocol selection based on task analysis
    - Dynamic agent creation and management
    - Performance monitoring and optimization
    - Easy integration for SDK users
    """
    
    def __init__(self, router_type: RouterType = RouterType.LLM_BASED, 
                 router_config: Dict[str, Any] = None):
        """
        Initialize the intelligent network manager.
        
        Args:
            router_type: Type of router to use
            router_config: Configuration for the router
        """
        self.network = AgentNetwork()
        self.router_config = router_config or {}
        
        # Initialize router
        try:
            self.router = RouterFactory.create_router(router_type, **self.router_config)
            print(f"🧠 Initialized {router_type.value} router")
        except Exception as e:
            print(f"⚠️ Failed to create {router_type.value} router: {e}")
            print("   Falling back to load balanced router")
            self.router = RouterFactory.create_router(RouterType.LOAD_BALANCED)
        
        # Track active agents and their protocols
        self.active_agents: Dict[str, BaseAgent] = {}
        self.agent_protocols: Dict[str, str] = {}
        self.protocol_factories: Dict[str, callable] = {}
        
        # Performance tracking
        self.task_metrics: Dict[str, Dict[str, Any]] = {}
        
        print("🌐 Intelligent Network Manager initialized")
    
    def register_protocol_factory(self, protocol_name: str, factory_func: callable,
                                capability: ProtocolCapability) -> None:
        """
        Register a protocol factory function with its capabilities.
        
        Args:
            protocol_name: Name of the protocol (e.g., "a2a", "acp")
            factory_func: Async function to create agents of this protocol
            capability: ProtocolCapability describing this protocol
        """
        self.protocol_factories[protocol_name] = factory_func
        self.router.register_protocol(capability)
        
        print(f"🔧 Registered protocol factory: {protocol_name}")
        print(f"   Strengths: {', '.join(capability.strengths[:2])}...")
        print(f"   Best for: {', '.join(capability.best_for[:2])}...")
    
    async def execute_task_intelligently(self, task: Dict[str, Any], 
                                       num_agents: int = 4,
                                       constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a task using intelligent routing to select and create optimal agents.
        
        Args:
            task: Task dictionary with question, context, metadata
            num_agents: Number of agents to create for this task
            constraints: Optional constraints (max_latency, requires_security, etc.)
            
        Returns:
            Task execution result with routing information
        """
        start_time = time.time()
        task_id = f"task_{int(start_time * 1000)}"
        
        try:
            # Create routing request
            routing_request = RoutingRequest(
                task=task,
                num_agents=num_agents,
                constraints=constraints or {},
                context={"timestamp": start_time, "task_id": task_id}
            )
            
            # Get routing decision from router
            routing_result = await self.router.route(routing_request)
            
            print(f"🎯 Intelligent Routing Decision for Task {task_id}:")
            print(f"   Selected protocols: {routing_result.selected_protocols}")
            print(f"   Agent assignments: {routing_result.agent_assignments}")
            print(f"   Strategy: {routing_result.strategy}")
            print(f"   Confidence: {routing_result.confidence:.1%}")
            print(f"   Reasoning: {routing_result.reasoning[:100]}...")
            
            # Create agents based on routing decision
            created_agents = await self._create_agents_from_routing(routing_result)
            
            if not created_agents:
                return {
                    "task_id": task_id,
                    "success": False,
                    "error": "Failed to create any agents",
                    "routing_result": routing_result,
                    "execution_time": time.time() - start_time
                }
            
            # Execute task on created agents
            execution_result = await self._execute_on_agents(task, created_agents)
            
            # Update router metrics based on execution
            await self._update_router_metrics(created_agents, execution_result, time.time() - start_time)
            
            # Prepare final result
            result = {
                "task_id": task_id,
                "success": execution_result.get("success", False),
                "response": execution_result.get("response"),
                "routing_decision": {
                    "selected_protocols": routing_result.selected_protocols,
                    "agent_assignments": routing_result.agent_assignments,
                    "strategy": routing_result.strategy,
                    "confidence": routing_result.confidence,
                    "reasoning": routing_result.reasoning
                },
                "execution_time": time.time() - start_time,
                "agents_used": len(created_agents),
                "protocol_distribution": self._analyze_protocol_distribution(created_agents)
            }
            
            # Record task metrics
            self.task_metrics[task_id] = result
            
            return result
            
        except Exception as e:
            return {
                "task_id": task_id,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    async def _create_agents_from_routing(self, routing_result: RoutingResult) -> List[str]:
        """Create agents based on routing decision."""
        created_agents = []
        
        for agent_id, protocol_name in routing_result.agent_assignments.items():
            if protocol_name not in self.protocol_factories:
                print(f"⚠️ No factory registered for protocol: {protocol_name}")
                continue
            
            try:
                # Call the protocol factory function
                factory_func = self.protocol_factories[protocol_name]
                
                # Create agent using factory (assumes factory returns agent with base_agent attribute)
                agent_instance = await factory_func(agent_id, self.router_config)
                
                # Extract BaseAgent if wrapped
                if hasattr(agent_instance, 'base_agent'):
                    base_agent = agent_instance.base_agent
                else:
                    base_agent = agent_instance
                
                # Register with network
                await self.network.register_agent(base_agent)
                
                # Track agent
                self.active_agents[agent_id] = base_agent
                self.agent_protocols[agent_id] = protocol_name
                
                created_agents.append(agent_id)
                
                print(f"✅ Created {protocol_name.upper()} agent: {agent_id}")
                
            except Exception as e:
                print(f"❌ Failed to create {protocol_name} agent {agent_id}: {e}")
                continue
        
        return created_agents
    
    async def _execute_on_agents(self, task: Dict[str, Any], agent_ids: List[str]) -> Dict[str, Any]:
        """Execute task on the created agents."""
        if not agent_ids:
            return {"success": False, "error": "No agents available"}
        
        try:
            # Simple execution strategy: use first agent for now
            # In a real implementation, this could be more sophisticated
            primary_agent_id = agent_ids[0]
            primary_agent = self.active_agents[primary_agent_id]
            
            # Prepare payload
            payload = {
                "text": task.get("question", ""),
                "question": task.get("question", ""),
                "context": task.get("context", ""),
                "metadata": task.get("metadata", {})
            }
            
            # Execute on primary agent
            response = await primary_agent.send(primary_agent_id, payload)
            
            return {
                "success": True,
                "response": response,
                "primary_agent": primary_agent_id,
                "agents_used": agent_ids
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agents_used": agent_ids
            }
    
    async def _update_router_metrics(self, agent_ids: List[str], 
                                   execution_result: Dict[str, Any], 
                                   execution_time: float) -> None:
        """Update router metrics based on task execution."""
        success = execution_result.get("success", False)
        
        for agent_id in agent_ids:
            metrics = {
                "response_time": execution_time / len(agent_ids),  # Distribute time across agents
                "success_rate": 1.0 if success else 0.0,
                "last_execution": time.time()
            }
            self.router.update_metrics(agent_id, metrics)
    
    def _analyze_protocol_distribution(self, agent_ids: List[str]) -> Dict[str, int]:
        """Analyze the distribution of protocols in created agents."""
        distribution = {}
        for agent_id in agent_ids:
            protocol = self.agent_protocols.get(agent_id, "unknown")
            distribution[protocol] = distribution.get(protocol, 0) + 1
        return distribution
    
    async def cleanup(self) -> None:
        """Cleanup all created agents and resources."""
        print("🧹 Cleaning up intelligent network...")
        
        for agent_id, agent in self.active_agents.items():
            try:
                await agent.stop()
                print(f"✅ Stopped agent: {agent_id}")
            except Exception as e:
                print(f"❌ Error stopping agent {agent_id}: {e}")
        
        self.active_agents.clear()
        self.agent_protocols.clear()
        
        print("✅ Intelligent network cleanup completed")
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status."""
        router_stats = self.router.get_statistics()
        network_metrics = self.network.snapshot_metrics()
        
        return {
            "network_metrics": network_metrics,
            "router_statistics": router_stats,
            "active_agents": len(self.active_agents),
            "protocol_distribution": self._analyze_protocol_distribution(list(self.active_agents.keys())),
            "registered_protocols": list(self.protocol_factories.keys()),
            "total_tasks_executed": len(self.task_metrics)
        }


# Convenience function for SDK users
async def create_intelligent_network(router_type: RouterType = RouterType.LLM_BASED,
                                   router_config: Dict[str, Any] = None) -> IntelligentNetworkManager:
    """
    Create an intelligent network manager with the specified router.
    
    Args:
        router_type: Type of router to use
        router_config: Configuration for the router
        
    Returns:
        Configured IntelligentNetworkManager
    """
    return IntelligentNetworkManager(router_type, router_config)
