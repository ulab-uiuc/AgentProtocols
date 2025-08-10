"""
Protocol Factory for managing different protocol implementations.
"""

import sys
import os
from typing import Dict, Any, List, Tuple, Optional

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocols.dummy.network import DummyNetwork, create_dummy_agent


class ProtocolFactory:
    """Factory for creating protocol-specific networks and agents."""
    
    def __init__(self):
        self.protocols = {
            'dummy': {
                'network_class': DummyNetwork,
                'create_agent_func': create_dummy_agent,
                'description': 'Dummy protocol for testing'
            }
        }
        self.default_protocol = 'dummy'
    
    def get_available_protocols(self) -> List[str]:
        """Get list of available protocols."""
        return list(self.protocols.keys())
    
    def create_network(self, protocol: str = None) -> Any:
        """Create a network instance for the specified protocol."""
        protocol = protocol or self.default_protocol
        
        if protocol not in self.protocols:
            raise ValueError(f"Unknown protocol: {protocol}. Available: {list(self.protocols.keys())}")
        
        network_class = self.protocols[protocol]['network_class']
        return network_class()
    
    def create_agent(self, agent_config: Dict[str, Any], task_id: str, protocol: str = None) -> Any:
        """Create an agent instance for the specified protocol."""
        protocol = protocol or self.default_protocol
        
        if protocol not in self.protocols:
            raise ValueError(f"Unknown protocol: {protocol}. Available: {list(self.protocols.keys())}")
        
        create_func = self.protocols[protocol]['create_agent_func']
        return create_func(agent_config, task_id)
    
    def create_multi_agent_system(self, agents_config: List[Dict[str, Any]], 
                                  task_id: str = "test", protocol: str = None) -> Tuple[Any, List[Any]]:
        """
        Create a complete multi-agent system with network and agents.
        
        Args:
            agents_config: List of agent configurations
            task_id: Task identifier
            protocol: Protocol to use
            
        Returns:
            Tuple of (network, agents_list)
        """
        protocol = protocol or self.default_protocol
        
        # Create network
        network = self.create_network(protocol)
        network.config = {"task_id": task_id}
        
        # Create and register agents
        agents = []
        for agent_config in agents_config:
            agent = self.create_agent(agent_config, task_id, protocol)
            network.register_agent(agent)
            agents.append(agent)
        
        return network, agents
    
    def get_protocol_info(self, protocol: str) -> Dict[str, Any]:
        """Get information about a specific protocol."""
        if protocol not in self.protocols:
            return {}
        return self.protocols[protocol]


# Global instance
protocol_factory = ProtocolFactory()
