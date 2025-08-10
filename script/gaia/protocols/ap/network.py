from __future__ import annotations
import asyncio
import json
import time
from typing import Dict, Any, List, Tuple
from script.gaia.core.network import MeshNetwork
from .agent import APAgent


class APNetwork(MeshNetwork):
    """
    Agent Protocol implementation for GAIA multi-agent framework.
    
    Implements network coordination methods using Agent Protocol HTTP API while inheriting
    all coordination and task management logic from MeshNetwork.
    """
    
    def __init__(self, adapter, **kwargs):
        """
        Initialize Agent Protocol network coordinator.
        
        Args:
            adapter: Protocol adapter instance
            **kwargs: Additional configuration
        """
        super().__init__(adapter)
        self.client_config = kwargs
        
        # Initialize required attributes from MeshNetwork
        self.active_agents = set()
        self.agents = []
        self.config = {}
        
        # Agent Protocol specific setup
        self._recv_queue: asyncio.Queue[Tuple[int, Dict[str, Any]]] = asyncio.Queue()
        self._setup_message_handler()
        
        # Network settings for Agent Protocol
        self.controller_channel = kwargs.get("controller_channel", "ap-controller")
        self.broadcast_channel = kwargs.get("broadcast_channel", "ap-broadcast")
        
    def _setup_message_handler(self) -> None:
        """Setup Agent Protocol message handler for coordinator messages."""
        # For Agent Protocol, messages come via HTTP API calls
        # This method prepares the message handling infrastructure
        pass
    
    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """
        Deliver message to specific agent via Agent Protocol HTTP API.
        
        Args:
            dst: Destination agent ID
            msg: Message payload
        """
        try:
            # Find target agent
            target_agent = None
            for agent in self.agents:
                if agent.id == dst:
                    target_agent = agent
                    break
                    
            if target_agent is None:
                print(f"AP Network: Target agent {dst} not found")
                return

            # Create Agent Protocol task/step based on message type
            msg_type = msg.get("type", "task_result")
            
            if msg_type == "doc_init":
                # Create a new task with document
                ap_payload = {
                    "type": "create_task",
                    "input": "".join(msg.get("chunks", [])),
                    "additional_input": {
                        "source": "doc_init",
                        "total_length": msg.get("total_length", 0),
                        "timestamp": int(time.time())
                    }
                }
            elif msg_type == "task_result":
                # Execute step with task result
                ap_payload = {
                    "type": "execute_step",
                    "task_id": getattr(self, 'current_task_id', None),
                    "input": msg.get("result", ""),
                    "additional_input": {
                        "source": msg.get("source", "unknown"),
                        "agent_id": msg.get("agent_id"),
                        "priority": msg.get("priority", 1),
                        "timestamp": int(time.time())
                    }
                }
            else:
                # Default message handling
                ap_payload = {
                    "type": "create_task",
                    "input": json.dumps(msg, ensure_ascii=False),
                    "additional_input": {
                        "original_type": msg_type,
                        "timestamp": int(time.time())
                    }
                }
            
            # Use the agent's send_msg method
            await target_agent.send_msg(dst, ap_payload)
            
        except Exception as e:
            print(f"AP Network: Failed to deliver message to agent {dst}: {e}")
    
    async def poll(self) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Poll for incoming messages from agents via Agent Protocol.
        
        Returns:
            List of (sender_id, message) tuples
        """
        messages = []
        
        try:
            # For Agent Protocol, poll all agents for their current state
            for agent in self.agents:
                try:
                    # Use the agent's recv_msg method to get messages
                    msg = await agent.recv_msg(timeout=0.0)  # Non-blocking
                    if msg:
                        messages.append((agent.id, msg))
                except Exception as e:
                    print(f"AP Network: Error polling agent {agent.id}: {e}")
                    
            # Also check internal message queue
            while not self._recv_queue.empty():
                sender_id, message = self._recv_queue.get_nowait()
                messages.append((sender_id, message))
                
        except Exception as e:
            print(f"AP Network: Error polling messages: {e}")
        
        return messages
    
    # async def broadcast(self, msg: Dict[str, Any]) -> None:
    #     """
    #     Broadcast message to all active agents via Agent Protocol.
        
    #     Args:
    #         msg: Message to broadcast
    #     """
    #     try:
    #         # Add broadcast metadata
    #         message = {
    #             **msg,
    #             "sender_id": -1,  # Network coordinator
    #             "broadcast": True,
    #             "timestamp": int(time.time())
    #         }
            
    #         # Send to each active agent using deliver method
    #         for agent in self.agents:
    #             if agent.id in self.active_agents:
    #                 await self.deliver(agent.id, message)
                
    #     except Exception as e:
    #         print(f"AP Network: Failed to broadcast message: {e}")
    
    async def connect(self) -> bool:
        """
        Establish Agent Protocol network connections.
        
        Returns:
            True if connection successful
        """
        try:
            # For Agent Protocol, connections are established when agents start their HTTP servers
            # This method can be used to verify all agents are ready
            print("AP Network: Checking agent connectivity...")
            
            connected_count = 0
            for agent in self.agents:
                # Check if agent HTTP server is running (simplified check)
                if hasattr(agent, 'running') and agent.running:
                    connected_count += 1
            
            if connected_count == len(self.agents):
                print(f"AP Network: All {connected_count} agents connected successfully")
                return True
            else:
                print(f"AP Network: Only {connected_count}/{len(self.agents)} agents connected")
                return False
                
        except Exception as e:
            print(f"AP Network: Connection check failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Agent Protocol network."""
        try:
            # For Agent Protocol, stop all agent HTTP servers
            print("AP Network: Disconnecting agents...")
            
            for agent in self.agents:
                if hasattr(agent, 'stop'):
                    await agent.stop()
                    
            print("AP Network: All agents disconnected")
            
        except Exception as e:
            print(f"AP Network: Disconnect error: {e}")
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current Agent Protocol network status and statistics."""
        return {
            "protocol": "Agent Protocol",
            "connected": len([a for a in self.agents if hasattr(a, 'running') and a.running]),
            "total_agents": len(self.agents),
            "controller_channel": self.controller_channel,
            "broadcast_channel": self.broadcast_channel,
            "active_agents": len(self.active_agents),
            "message_queue_size": self._recv_queue.qsize(),
            "current_tick": getattr(self, 'current_tick', 0)
        }
    
    # ==================== Agent Management ====================
    
    def register_agent(self, agent: APAgent) -> None:
        """Register a new Agent Protocol agent in the system."""
        self.active_agents.add(agent.id)
        self.agents.append(agent)
        
        print(f"AP Network: Registered agent {agent.id} ({agent.name}) with tool {agent.tool_name}")
    
    def unregister_agent(self, agent_id: int) -> None:
        """Remove agent from active tracking."""
        self.active_agents.discard(agent_id)
        
        # Remove from agents list
        self.agents = [agent for agent in self.agents if agent.id != agent_id]
        
        print(f"AP Network: Unregistered agent {agent_id}")
    
    def get_agent(self, agent_id: int) -> APAgent:
        """Get agent by ID."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None
    
    # ==================== Dynamic Agent Management ====================
    async def load_and_create_agents(self, config_path: str):
        """Load configuration and dynamically create Agent Protocol agents."""
        print(f"📋 Loading Agent Protocol configuration from {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        print(f"🤖 Creating {len(self.config['agents'])} Agent Protocol agents...")
        
        # Create APAgent instances according to configuration
        for agent_config in self.config["agents"]:
            agent = APAgent(
                node_id=agent_config["id"],
                name=agent_config["name"],
                tool=agent_config["tool"],
                adapter=self.adapter,
                port=agent_config["port"],
                config=agent_config,
                task_id=self.config.get("task_id"),
                network=self  # Pass network reference to agent
            )
            
            # Register the agent
            self.register_agent(agent)
            
            # Start agent server
            try:
                await agent.start()
                print(f"✅ Agent {agent.id} ({agent.name}) started successfully on port {agent.port}")
            except Exception as e:
                print(f"❌ Failed to start agent {agent.id}: {e}")
        
        print(f"🌐 Agent Protocol network initialized with {len(self.agents)} agents") 

    async def execute_workflow_with_task(self, initial_task: str) -> str:
        """
        Execute the workflow with the given initial task using Agent Protocol.
        
        Args:
            initial_task: The initial task to start the workflow
            
        Returns:
            Final result from workflow execution
        """
        try:
            print(f"🎯 Starting Agent Protocol workflow execution...")
            print(f"📋 Initial task: {initial_task[:100]}...")
            
            # Call the base class execute_workflow method
            result = await self.execute_workflow(self.config, initial_task)
            
            print(f"✅ Agent Protocol workflow completed")
            return result
            
        except Exception as e:
            print(f"❌ Agent Protocol workflow execution failed: {e}")
            return f"Workflow execution failed: {e}" 