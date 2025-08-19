"""
SimpleJSON protocol network implementation for fail-storm recovery scenarios.

Implements the FailStormNetworkBase using SimpleJSON protocol for communication.
"""

import asyncio
import time
from typing import Dict, List, Any, Tuple, Optional

from core.network_base import FailStormNetworkBase
from core.comm import AbstractCommAdapter
from protocol_backends.simple_json.adapter import SimpleJSONAdapter


class SimpleJSONNetwork(FailStormNetworkBase):
    """
    SimpleJSON protocol implementation of FailStormNetworkBase.
    
    Uses SimpleJSON adapters for communication while inheriting all
    fault detection and recovery logic from the base class.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 heartbeat_interval: float = 5.0,
                 heartbeat_timeout: float = 15.0,
                 debug_mode: bool = False):
        """
        Initialize SimpleJSON network coordinator.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing network settings
        heartbeat_interval : float
            Interval between heartbeat messages (seconds)
        heartbeat_timeout : float
            Time to wait before considering a node failed (seconds)
        debug_mode : bool
            Enable verbose debug output
        """
        # Create SimpleJSON adapter for the coordinator
        coordinator_id = config.get("coordinator_id", "coordinator")
        coordinator_card = config.get("coordinator_card", {
            "protocol": "simple_json",
            "agent_id": coordinator_id,
            "capabilities": ["network_coordination", "fault_detection"]
        })
        
        comm_adapter = SimpleJSONAdapter(coordinator_id, coordinator_card)
        
        # Initialize base class
        super().__init__(
            comm_adapter=comm_adapter,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            debug_mode=debug_mode
        )
        
        # SimpleJSON specific configuration
        self.config = config
        self.coordinator_id = coordinator_id
        
        # Agent adapters for local communication
        self._agent_adapters: Dict[str, SimpleJSONAdapter] = {}
        
        # Message processing state
        self._message_buffer: List[Tuple[str, Dict[str, Any]]] = []
        
    async def deliver(self, dst: str, msg: Dict[str, Any]) -> None:
        """
        Deliver message to specific agent via SimpleJSON protocol.
        
        Args:
            dst: Destination agent ID
            msg: Message payload
        """
        if dst in self._agent_adapters:
            # Send directly to agent's adapter
            await self._agent_adapters[dst].get_inbox().put({
                "sender": self.coordinator_id,
                "target": dst,
                "message": msg,
                "timestamp": time.time()
            })
        else:
            # Use broadcast if agent not found
            await self.comm_adapter.broadcast(msg)
    
    async def poll(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Poll for incoming messages from SimpleJSON protocol.
        
        Returns:
            List of (sender_id, message) tuples
        """
        messages = []
        
        # Check coordinator's inbox
        try:
            while True:
                sender, message = self.comm_adapter.recv_nowait()
                if sender and message:
                    messages.append((sender, message))
                else:
                    break
        except Exception as e:
            if self.debug_mode:
                print(f"[SimpleJSONNetwork] Error polling coordinator messages: {e}")
        
        # Check all agent adapters for messages
        for agent_id, adapter in self._agent_adapters.items():
            try:
                while True:
                    sender, message = adapter.recv_nowait()
                    if sender and message:
                        messages.append((sender, message))
                    else:
                        break
            except Exception as e:
                if self.debug_mode:
                    print(f"[SimpleJSONNetwork] Error polling agent {agent_id} messages: {e}")
        
        return messages
    
    async def register_agent(self, agent_id: str, **kwargs) -> None:
        """Register a new agent with SimpleJSON adapter."""
        # Create SimpleJSON adapter for the agent
        agent_card = kwargs.get("agent_card", {
            "protocol": "simple_json",
            "agent_id": agent_id,
            "capabilities": kwargs.get("capabilities", [])
        })
        
        agent_adapter = SimpleJSONAdapter(agent_id, agent_card)
        
        # Register with coordinator adapter
        self.comm_adapter.register_agent(agent_id, agent_adapter.get_inbox())
        
        # Register coordinator with agent adapter
        agent_adapter.register_agent(self.coordinator_id, self.comm_adapter.get_inbox())
        
        # Store agent adapter
        self._agent_adapters[agent_id] = agent_adapter
        
        # Connect agent adapter
        await agent_adapter.connect()
        
        # Register with base class
        await super().register_agent(agent_id, **kwargs)
        
        # Register with other agents for peer-to-peer communication
        for other_id, other_adapter in self._agent_adapters.items():
            if other_id != agent_id:
                agent_adapter.register_agent(other_id, other_adapter.get_inbox())
                other_adapter.register_agent(agent_id, agent_adapter.get_inbox())
        
        print(f"[SimpleJSONNetwork] Agent {agent_id} registered with SimpleJSON adapter")
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent and clean up SimpleJSON adapter."""
        # Disconnect agent adapter
        if agent_id in self._agent_adapters:
            await self._agent_adapters[agent_id].disconnect()
            
            # Unregister from other agents
            for other_id, other_adapter in self._agent_adapters.items():
                if other_id != agent_id:
                    other_adapter.unregister_agent(agent_id)
            
            # Remove agent adapter
            self._agent_adapters.pop(agent_id, None)
        
        # Unregister from coordinator
        self.comm_adapter.unregister_agent(agent_id)
        
        # Unregister from base class
        await super().unregister_agent(agent_id)
        
        print(f"[SimpleJSONNetwork] Agent {agent_id} unregistered")
    
    async def start(self) -> None:
        """Start the SimpleJSON network coordinator."""
        await super().start()
        
        # Connect all agent adapters
        for agent_id, adapter in self._agent_adapters.items():
            await adapter.connect()
        
        print(f"[SimpleJSONNetwork] SimpleJSON network coordinator started")
    
    async def stop(self) -> None:
        """Stop the SimpleJSON network coordinator."""
        # Disconnect all agent adapters
        for agent_id, adapter in self._agent_adapters.items():
            await adapter.disconnect()
        
        await super().stop()
        
        print(f"[SimpleJSONNetwork] SimpleJSON network coordinator stopped")
    
    def get_agent_adapter(self, agent_id: str) -> Optional[SimpleJSONAdapter]:
        """Get the SimpleJSON adapter for a specific agent."""
        return self._agent_adapters.get(agent_id)
    
    def get_all_agent_adapters(self) -> Dict[str, SimpleJSONAdapter]:
        """Get all registered agent adapters."""
        return self._agent_adapters.copy()
    
    async def _handle_protocol_message(self, sender_id: str, message: Dict[str, Any]) -> None:
        """Handle SimpleJSON-specific messages."""
        msg_type = message.get("type")
        
        if msg_type == "agent_status":
            # Handle agent status updates
            await self._handle_agent_status(sender_id, message)
        elif msg_type == "topology_request":
            # Handle topology information requests
            await self._handle_topology_request(sender_id, message)
        else:
            # Call parent implementation for other messages
            await super()._handle_protocol_message(sender_id, message)
    
    async def _handle_agent_status(self, sender_id: str, message: Dict[str, Any]) -> None:
        """Handle agent status update messages."""
        if sender_id in self.agents:
            # Update agent state
            status = message.get("status", {})
            self.agents[sender_id].metadata.update(status)
            self.agents[sender_id].last_update = time.time()
            
            if self.debug_mode:
                print(f"[SimpleJSONNetwork] Agent {sender_id} status update: {status}")
    
    async def _handle_topology_request(self, sender_id: str, message: Dict[str, Any]) -> None:
        """Handle topology information requests."""
        topology_info = {
            "type": "topology_response",
            "active_agents": list(self.active_agents),
            "failed_agents": list(self.failed_agents),
            "total_agents": len(self.agents),
            "topology_changes": len(self._topology_changes),
            "timestamp": time.time()
        }
        
        await self.deliver(sender_id, topology_info) 