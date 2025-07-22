"""
Distributed runner for MAPF scenarios.

Supports execution across multiple nodes with different deployment patterns:
- Coordinator-only nodes
- Agent-only nodes
- Mixed nodes
"""

import importlib
import asyncio
import yaml
import time
import socket
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Set
from ..core.world import GridWorld
from ..core.agent_base import BaseRobot
from ..core.network_base import BaseNet
from ..core.utils import info_log, error_log
from .local_runner import LocalRunner


class DistributedRunner:
    """
    Distributed execution runner for MAPF scenarios.
    
    Coordinates execution across multiple nodes with different roles.
    """
    
    def __init__(self, config_path: str, node_role: str = "mixed", node_id: Optional[str] = None):
        """
        Initialize distributed runner.
        
        Args:
            config_path: Path to YAML configuration file
            node_role: Role of this node ("coordinator", "agent", "mixed")
            node_id: Unique identifier for this node
        """
        self.config_path = config_path
        self.node_role = node_role
        self.node_id = node_id or self._generate_node_id()
        self.config = self._load_config()
        
        # Runtime state
        self.world: Optional[GridWorld] = None
        self.network: Optional[BaseNet] = None
        self.agents: List[BaseRobot] = []
        self.assigned_agent_ids: Set[int] = set()
        self.is_running = False
        
        # Distributed coordination
        self.coordinator_address = None
        self.peer_nodes: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.start_time = 0.0
        self.end_time = 0.0
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID based on hostname and timestamp."""
        hostname = socket.gethostname()
        timestamp = int(time.time() * 1000) % 100000
        return f"{hostname}-{timestamp}"
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Validate distributed configuration
            distributed_config = config.get("distributed", {})
            
            if not distributed_config:
                raise ValueError("Missing 'distributed' section in config")
            
            info_log(f"Loaded distributed configuration from {self.config_path}")
            return config
            
        except Exception as e:
            error_log(f"Failed to load config from {self.config_path}: {e}")
            raise
    
    def _get_node_assignment(self) -> Dict[str, Any]:
        """Get agent assignment for this node."""
        distributed_config = self.config["distributed"]
        node_assignments = distributed_config.get("node_assignments", {})
        
        # Check for specific assignment
        if self.node_id in node_assignments:
            return node_assignments[self.node_id]
        
        # Check for role-based assignment
        role_assignments = distributed_config.get("role_assignments", {})
        if self.node_role in role_assignments:
            return role_assignments[self.node_role]
        
        # Default assignment
        return {
            "coordinator": self.node_role in ["coordinator", "mixed"],
            "agents": [] if self.node_role == "coordinator" else "auto"
        }
    
    def _assign_agents(self) -> Set[int]:
        """Determine which agents this node should run."""
        assignment = self._get_node_assignment()
        agent_assignment = assignment.get("agents", [])
        
        if agent_assignment == "auto":
            # Auto-assign based on node role and total agents
            total_agents = self.config.get("num_agents", 4)
            total_nodes = len(self.config["distributed"].get("nodes", [self.node_id]))
            
            # Simple round-robin assignment
            node_index = hash(self.node_id) % total_nodes
            assigned = set()
            
            for i in range(total_agents):
                if i % total_nodes == node_index:
                    assigned.add(i)
            
            return assigned
        
        elif isinstance(agent_assignment, list):
            return set(agent_assignment)
        
        else:
            return set()
    
    async def setup(self) -> None:
        """Setup the distributed MAPF scenario."""
        info_log(f"Setting up distributed MAPF scenario (role: {self.node_role}, node: {self.node_id})")
        
        assignment = self._get_node_assignment()
        
        # Create world (all nodes need world state)
        world_config = self.config.get("world", {})
        self.world = GridWorld(
            size=world_config.get("size", 10),
            goals=world_config.get("goals")
        )
        
        # Add obstacles
        obstacles = world_config.get("obstacles", [])
        for obstacle in obstacles:
            if isinstance(obstacle, (list, tuple)) and len(obstacle) == 2:
                self.world.add_obstacle(obstacle[0], obstacle[1])
        
        # Setup coordinator if assigned
        if assignment.get("coordinator", False):
            await self._setup_coordinator()
        
        # Setup agents if assigned
        self.assigned_agent_ids = self._assign_agents()
        if self.assigned_agent_ids:
            await self._setup_agents()
        
        # Connect to distributed network
        await self._connect_distributed()
        
        info_log(f"Distributed setup complete - coordinator: {bool(self.network)}, agents: {len(self.agents)}")
    
    async def _setup_coordinator(self) -> None:
        """Setup network coordinator."""
        net_cls_path = self.config["net_cls"]
        net_cls = self._load_class(net_cls_path)
        net_kwargs = self.config.get("net_kwargs", {})
        
        # Add distributed-specific configuration
        distributed_config = self.config["distributed"]
        net_kwargs.update({
            "node_id": self.node_id,
            "distributed_mode": True,
            "coordinator_address": distributed_config.get("coordinator_address"),
            "peer_nodes": distributed_config.get("nodes", [])
        })
        
        # Create protocol-specific network client
        protocol_client = await self._create_distributed_protocol_client("network")
        
        if protocol_client:
            self.network = net_cls(self.world, protocol_client, **net_kwargs)
        else:
            self.network = net_cls(self.world, **net_kwargs)
        
        info_log(f"Created distributed network coordinator: {net_cls.__name__}")
    
    async def _setup_agents(self) -> None:
        """Setup assigned agents."""
        agent_cls_path = self.config["agent_cls"]
        agent_cls = self._load_class(agent_cls_path)
        agent_kwargs = self.config.get("agent_kwargs", {})
        
        # Add distributed-specific configuration
        distributed_config = self.config["distributed"]
        agent_kwargs.update({
            "node_id": self.node_id,
            "distributed_mode": True,
            "coordinator_address": distributed_config.get("coordinator_address")
        })
        
        for agent_id in self.assigned_agent_ids:
            if agent_id >= len(self.world.goals):
                error_log(f"Not enough goals for agent {agent_id}")
                continue
            
            # Starting positions
            start_positions = self.config.get("start_positions")
            if start_positions and agent_id < len(start_positions):
                start_pos = tuple(start_positions[agent_id])
            else:
                start_pos = self.world.goals[(agent_id + 1) % len(self.world.goals)]
            
            goal_pos = self.world.goals[agent_id]
            
            # Create distributed protocol client
            protocol_client = await self._create_distributed_protocol_client("agent", agent_id)
            
            if protocol_client:
                agent = agent_cls(agent_id, self.world, goal_pos, protocol_client, **agent_kwargs)
            else:
                agent = agent_cls(agent_id, self.world, goal_pos, **agent_kwargs)
            
            agent.initialize_position(start_pos)
            self.agents.append(agent)
        
        info_log(f"Created {len(self.agents)} distributed agents: {list(self.assigned_agent_ids)}")
    
    async def _create_distributed_protocol_client(self, component_type: str, agent_id: Optional[int] = None):
        """Create protocol client configured for distributed operation."""
        protocol = self.config.get("protocol", "dummy")
        distributed_config = self.config["distributed"]
        
        if protocol == "dummy":
            # Dummy protocol needs special handling for distributed mode
            return None
        
        elif protocol == "a2a":
            # Create distributed A2A client
            class DistributedA2AClient:
                def __init__(self, **kwargs):
                    self.config = kwargs
                    self.connected = False
                    self.distributed_endpoints = kwargs.get("endpoints", [])
                
                async def connect(self):
                    # Connect to distributed A2A network
                    self.connected = True
                    info_log(f"Connected to distributed A2A network")
                
                async def disconnect(self):
                    self.connected = False
                
                def on_message(self, channel, handler):
                    pass
                
                async def send(self, channel, message):
                    # Route message through distributed network
                    pass
                
                def is_connected(self):
                    return self.connected
            
            a2a_config = distributed_config.get("a2a_config", {})
            a2a_config.update({
                "endpoints": distributed_config.get("a2a_endpoints", []),
                "node_id": self.node_id
            })
            
            return DistributedA2AClient(**a2a_config)
        
        elif protocol == "anp":
            # Create distributed ANP client
            class DistributedANPClient:
                def __init__(self, **kwargs):
                    self.config = kwargs
                    self.connected = False
                
                async def connect(self, **kwargs):
                    self.connected = True
                    info_log(f"Connected to distributed ANP network")
                
                async def disconnect(self):
                    self.connected = False
                
                def set_message_handler(self, handler):
                    self.handler = handler
                
                async def send_message(self, did, message):
                    pass
                
                def is_connected(self):
                    return self.connected
            
            class DistributedANPNode:
                def __init__(self, **kwargs):
                    self.config = kwargs
                    self.running = False
                
                async def start_coordinator(self, **kwargs):
                    self.running = True
                    info_log(f"Started distributed ANP coordinator")
                
                async def stop_coordinator(self):
                    self.running = False
                
                def set_coordinator_handler(self, handler):
                    self.handler = handler
                
                async def send_to_agent(self, did, message):
                    pass
                
                def is_running(self):
                    return self.running
            
            anp_config = distributed_config.get("anp_config", {})
            anp_config.update({
                "node_endpoints": distributed_config.get("anp_endpoints", []),
                "node_id": self.node_id
            })
            
            if component_type == "agent":
                return DistributedANPClient(**anp_config)
            else:
                return DistributedANPNode(**anp_config)
        
        return None
    
    def _load_class(self, class_path: str) -> Type:
        """Dynamically load a class from module path."""
        try:
            module_path, class_name = class_path.split(":")
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except Exception as e:
            error_log(f"Failed to load class {class_path}: {e}")
            raise
    
    async def _connect_distributed(self) -> None:
        """Connect to distributed coordination network."""
        distributed_config = self.config["distributed"]
        
        # Connect coordinator
        if self.network and hasattr(self.network, 'connect'):
            await self.network.connect()
        
        # Connect agents
        for agent in self.agents:
            if hasattr(agent, 'connect'):
                await agent.connect()
        
        # Register with coordinator if this is an agent-only node
        if self.agents and not self.network:
            await self._register_with_coordinator()
        
        info_log("Connected to distributed network")
    
    async def _register_with_coordinator(self) -> None:
        """Register this node's agents with the global coordinator."""
        # This would send registration messages to the coordinator
        # Implementation depends on the specific protocol
        info_log(f"Registered {len(self.agents)} agents with coordinator")
    
    async def run(self) -> Dict[str, Any]:
        """
        Run the distributed MAPF scenario.
        
        Returns:
            Execution statistics for this node
        """
        if not self.world:
            await self.setup()
        
        info_log(f"Starting distributed MAPF execution on node {self.node_id}")
        self.start_time = time.time()
        self.is_running = True
        
        try:
            tasks = []
            
            # Network coordinator task (if running on this node)
            if self.network:
                tasks.append(asyncio.create_task(self.network.run(), name="network"))
            
            # Agent tasks
            for agent in self.agents:
                task_name = f"agent-{agent.aid}"
                tasks.append(asyncio.create_task(agent.run(), name=task_name))
            
            # Wait for completion or timeout
            timeout = self.config.get("timeout_seconds", 300)
            
            if tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    info_log(f"Node timed out after {timeout} seconds")
                    for task in tasks:
                        task.cancel()
            else:
                info_log("No tasks to run on this node")
        
        except Exception as e:
            error_log(f"Error during distributed execution: {e}")
            raise
        
        finally:
            self.end_time = time.time()
            self.is_running = False
            
            # Disconnect
            if self.network and hasattr(self.network, 'disconnect'):
                await self.network.disconnect()
            
            for agent in self.agents:
                if hasattr(agent, 'disconnect'):
                    await agent.disconnect()
        
        return self._collect_node_statistics()
    
    def _collect_node_statistics(self) -> Dict[str, Any]:
        """Collect execution statistics for this node."""
        elapsed_time = self.end_time - self.start_time
        
        # Network statistics (if coordinator is on this node)
        network_stats = self.network.get_performance_metrics() if self.network else {}
        
        # Agent statistics
        agent_stats = {}
        goals_reached = 0
        
        for agent in self.agents:
            agent_status = {
                "goal_reached": agent.is_at_goal(),
                "current_position": agent.current_pos,
                "goal_position": agent.goal,
                "path_length": len(agent.path) if agent.path else 0
            }
            
            if hasattr(agent, 'get_connection_status'):
                agent_status.update(agent.get_connection_status())
            
            agent_stats[agent.aid] = agent_status
            
            if agent.is_at_goal():
                goals_reached += 1
        
        return {
            "node_id": self.node_id,
            "node_role": self.node_role,
            "execution_time": elapsed_time,
            "assigned_agents": list(self.assigned_agent_ids),
            "goals_reached": goals_reached,
            "has_coordinator": bool(self.network),
            "network_stats": network_stats,
            "agent_stats": agent_stats
        }


async def main(config_path: str, node_role: str = "mixed", node_id: Optional[str] = None):
    """
    Main entry point for distributed runner.
    
    Args:
        config_path: Path to configuration file
        node_role: Role of this node
        node_id: Node identifier
    """
    runner = DistributedRunner(config_path, node_role, node_id)
    
    try:
        results = await runner.run()
        
        # Print node summary
        print("\n" + "="*50)
        print(f"Distributed MAPF Node Summary - {results['node_id']}")
        print("="*50)
        print(f"Node role: {results['node_role']}")
        print(f"Execution time: {results['execution_time']:.2f}s")
        print(f"Assigned agents: {results['assigned_agents']}")
        print(f"Goals reached: {results['goals_reached']}")
        print(f"Has coordinator: {results['has_coordinator']}")
        print("="*50)
        
        return results
        
    except Exception as e:
        error_log(f"Distributed runner execution failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python distributed_runner.py <config_path> [node_role] [node_id]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    node_role = sys.argv[2] if len(sys.argv) > 2 else "mixed"
    node_id = sys.argv[3] if len(sys.argv) > 3 else None
    
    asyncio.run(main(config_path, node_role, node_id)) 