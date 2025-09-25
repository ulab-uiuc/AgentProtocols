# -*- coding: utf-8 -*-
"""
Meta Protocol Registration Gateway Adapter for Privacy Testing
Adapts the registration gateway for meta protocol with intelligent routing capabilities.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Optional, List
from pathlib import Path

# Import base registration gateway
try:
    from ...core.registration_gateway import RegistrationGateway
except ImportError:
    try:
        from core.registration_gateway import RegistrationGateway
    except ImportError:
        # Create minimal base class if not available
        class RegistrationGateway:
            def __init__(self, config, output):
                self.config = config
                self.output = output
            
            async def register_agent(self, agent_id, agent_config, host="127.0.0.1", port=8080):
                return {"success": True, "agent_id": agent_id, "endpoint": f"http://{host}:{port}"}
            
            async def unregister_agent(self, agent_id):
                return {"success": True, "agent_id": agent_id}
            
            async def get_agent_info(self, agent_id):
                return {"agent_id": agent_id, "status": "active"}
            
            async def health_check_agent(self, agent_id):
                return {"agent_id": agent_id, "healthy": True}
            
            async def cleanup(self):
                pass


class MetaRegistrationAdapter:
    """
    Registration adapter for meta protocol that handles agent registration
    with intelligent protocol selection and routing information.
    """
    
    def __init__(self, config: Dict[str, Any], selected_protocol: str, routing_decision=None, output=None):
        self.config = config
        self.selected_protocol = selected_protocol
        self.routing_decision = routing_decision
        self.output = output
        
        # Initialize base registration gateway
        self.base_gateway = RegistrationGateway(config, output)
        
        # Meta protocol specific state
        self._registered_agents: Dict[str, Dict[str, Any]] = {}
        self._protocol_mapping: Dict[str, str] = {}  # agent_id -> actual protocol
        self._routing_info: Dict[str, Any] = {}
        
        # Registration statistics
        self._registration_stats = {
            "total_registrations": 0,
            "successful_registrations": 0,
            "failed_registrations": 0,
            "protocol_distribution": {}
        }

    async def register_meta_agent(self, agent_id: str, agent_config: Dict[str, Any], 
                                host: str = "127.0.0.1", port: int = 8080) -> Dict[str, Any]:
        """
        Register a meta protocol agent with routing information.
        
        Args:
            agent_id: Unique agent identifier
            agent_config: Agent configuration
            host: Agent host address
            port: Agent port
            
        Returns:
            Registration result with meta protocol information
        """
        self._registration_stats["total_registrations"] += 1
        
        try:
            # Prepare meta protocol agent configuration
            meta_agent_config = {
                **agent_config,
                "meta_protocol": {
                    "selected_protocol": self.selected_protocol,
                    "routing_decision": {
                        "confidence": self.routing_decision.confidence,
                        "reasoning": self.routing_decision.reasoning,
                        "strategy": self.routing_decision.strategy
                    } if self.routing_decision else None,
                    "registration_timestamp": time.time()
                },
                "agent_type": "meta_protocol",
                "underlying_protocol": self.selected_protocol
            }
            
            # Register with base gateway using selected protocol configuration
            base_result = await self.base_gateway.register_agent(
                agent_id=agent_id,
                agent_config=meta_agent_config,
                host=host,
                port=port
            )
            
            if base_result.get("success", False):
                # Store meta protocol specific information
                self._registered_agents[agent_id] = {
                    "agent_id": agent_id,
                    "host": host,
                    "port": port,
                    "selected_protocol": self.selected_protocol,
                    "routing_confidence": self.routing_decision.confidence if self.routing_decision else 0.8,
                    "registration_time": time.time(),
                    "base_result": base_result
                }
                
                self._protocol_mapping[agent_id] = self.selected_protocol
                
                # Update statistics
                self._registration_stats["successful_registrations"] += 1
                if self.selected_protocol not in self._registration_stats["protocol_distribution"]:
                    self._registration_stats["protocol_distribution"][self.selected_protocol] = 0
                self._registration_stats["protocol_distribution"][self.selected_protocol] += 1
                
                if self.output:
                    self.output.success(f"✅ Meta protocol agent registered: {agent_id} ({self.selected_protocol.upper()})")
                
                return {
                    "success": True,
                    "agent_id": agent_id,
                    "selected_protocol": self.selected_protocol,
                    "routing_info": self.routing_decision.__dict__ if self.routing_decision else None,
                    "endpoint": f"http://{host}:{port}",
                    "registration_time": time.time(),
                    "base_registration": base_result
                }
            else:
                self._registration_stats["failed_registrations"] += 1
                error_msg = base_result.get("error", "Unknown registration error")
                
                if self.output:
                    self.output.error(f"❌ Meta protocol agent registration failed: {agent_id}: {error_msg}")
                
                return {
                    "success": False,
                    "agent_id": agent_id,
                    "error": f"Base registration failed: {error_msg}",
                    "selected_protocol": self.selected_protocol
                }
                
        except Exception as e:
            self._registration_stats["failed_registrations"] += 1
            error_msg = f"Meta protocol registration error: {str(e)}"
            
            if self.output:
                self.output.error(f"❌ {error_msg}")
            
            return {
                "success": False,
                "agent_id": agent_id,
                "error": error_msg,
                "selected_protocol": self.selected_protocol
            }

    async def unregister_meta_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Unregister a meta protocol agent.
        
        Args:
            agent_id: Agent identifier to unregister
            
        Returns:
            Unregistration result
        """
        try:
            if agent_id not in self._registered_agents:
                return {
                    "success": False,
                    "agent_id": agent_id,
                    "error": "Agent not registered in meta protocol"
                }
            
            # Unregister from base gateway
            base_result = await self.base_gateway.unregister_agent(agent_id)
            
            # Remove from meta protocol tracking
            agent_info = self._registered_agents.pop(agent_id, {})
            self._protocol_mapping.pop(agent_id, None)
            
            # Update statistics
            protocol = agent_info.get("selected_protocol", "unknown")
            if protocol in self._registration_stats["protocol_distribution"]:
                self._registration_stats["protocol_distribution"][protocol] -= 1
                if self._registration_stats["protocol_distribution"][protocol] <= 0:
                    del self._registration_stats["protocol_distribution"][protocol]
            
            if self.output:
                self.output.info(f"🗑️  Meta protocol agent unregistered: {agent_id}")
            
            return {
                "success": True,
                "agent_id": agent_id,
                "selected_protocol": protocol,
                "base_unregistration": base_result
            }
            
        except Exception as e:
            error_msg = f"Meta protocol unregistration error: {str(e)}"
            
            if self.output:
                self.output.error(f"❌ {error_msg}")
            
            return {
                "success": False,
                "agent_id": agent_id,
                "error": error_msg
            }

    async def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered meta protocol agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent information or None if not found
        """
        if agent_id not in self._registered_agents:
            return None
        
        agent_info = self._registered_agents[agent_id]
        base_info = await self.base_gateway.get_agent_info(agent_id)
        
        return {
            **agent_info,
            "base_info": base_info,
            "uptime_seconds": time.time() - agent_info.get("registration_time", 0),
            "protocol_mapping": self._protocol_mapping.get(agent_id, "unknown")
        }

    async def list_registered_agents(self) -> List[Dict[str, Any]]:
        """
        List all registered meta protocol agents.
        
        Returns:
            List of agent information dictionaries
        """
        agents_list = []
        
        for agent_id in self._registered_agents.keys():
            agent_info = await self.get_agent_info(agent_id)
            if agent_info:
                agents_list.append(agent_info)
        
        return agents_list

    async def health_check_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Perform health check on a registered meta protocol agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Health check result
        """
        try:
            if agent_id not in self._registered_agents:
                return {
                    "agent_id": agent_id,
                    "healthy": False,
                    "error": "Agent not registered in meta protocol"
                }
            
            agent_info = self._registered_agents[agent_id]
            
            # Perform base health check
            base_health = await self.base_gateway.health_check_agent(agent_id)
            
            # Additional meta protocol health checks
            meta_health = {
                "protocol_selected": agent_info.get("selected_protocol"),
                "routing_confidence": agent_info.get("routing_confidence", 0.0),
                "uptime_seconds": time.time() - agent_info.get("registration_time", 0),
                "endpoint": f"http://{agent_info.get('host', 'unknown')}:{agent_info.get('port', 0)}"
            }
            
            overall_healthy = base_health.get("healthy", False)
            
            return {
                "agent_id": agent_id,
                "healthy": overall_healthy,
                "base_health": base_health,
                "meta_health": meta_health,
                "check_timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "agent_id": agent_id,
                "healthy": False,
                "error": f"Health check failed: {str(e)}",
                "check_timestamp": time.time()
            }

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform health check on all registered meta protocol agents.
        
        Returns:
            Dictionary mapping agent_id to health check results
        """
        health_results = {}
        
        for agent_id in self._registered_agents.keys():
            health_results[agent_id] = await self.health_check_agent(agent_id)
        
        return health_results

    def get_registration_stats(self) -> Dict[str, Any]:
        """
        Get meta protocol registration statistics.
        
        Returns:
            Registration statistics dictionary
        """
        return {
            **self._registration_stats,
            "currently_registered": len(self._registered_agents),
            "selected_protocol": self.selected_protocol,
            "routing_confidence": self.routing_decision.confidence if self.routing_decision else 0.8,
            "routing_strategy": self.routing_decision.strategy if self.routing_decision else "heuristic"
        }

    def get_protocol_distribution(self) -> Dict[str, int]:
        """
        Get distribution of protocols among registered agents.
        
        Returns:
            Dictionary mapping protocol names to agent counts
        """
        return dict(self._registration_stats["protocol_distribution"])

    async def update_routing_info(self, new_routing_decision) -> None:
        """
        Update routing information for all registered agents.
        
        Args:
            new_routing_decision: New routing decision object
        """
        self.routing_decision = new_routing_decision
        
        # Update routing info for all registered agents
        for agent_id, agent_info in self._registered_agents.items():
            agent_info["routing_confidence"] = new_routing_decision.confidence
            agent_info["routing_updated"] = time.time()
        
        if self.output:
            self.output.info(f"🔄 Updated routing info for {len(self._registered_agents)} meta protocol agents")

    async def cleanup(self) -> None:
        """Cleanup meta protocol registration adapter."""
        try:
            # Unregister all agents
            agent_ids = list(self._registered_agents.keys())
            for agent_id in agent_ids:
                await self.unregister_meta_agent(agent_id)
            
            # Cleanup base gateway
            await self.base_gateway.cleanup()
            
            self._registered_agents.clear()
            self._protocol_mapping.clear()
            self._routing_info.clear()
            
            if self.output:
                self.output.info("🧹 Meta protocol registration adapter cleanup completed")
                
        except Exception as e:
            if self.output:
                self.output.warning(f"⚠️  Cleanup warning: {e}")
