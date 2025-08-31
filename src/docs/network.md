# AgentNetwork - Network Architecture

### Overview

The `AgentNetwork` class serves as the central coordinator and topology manager for the multi-agent system. It maintains the registry of all agents, manages their interconnections, handles message routing, and provides comprehensive monitoring capabilities.

### Core Responsibilities

1. **Agent Lifecycle Management**: Register, unregister, and track agent instances
2. **Topology Management**: Define and maintain network topologies (star, mesh, custom)
3. **Message Routing**: Route messages between agents based on topology rules
4. **Health Monitoring**: Continuous health checks and failure detection
5. **Metrics Collection**: Performance monitoring and observability
6. **Failure Recovery**: Automatic detection and recovery from agent failures

### Class Structure

```python
class AgentNetwork:
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}           # Agent registry
        self._graph: Dict[str, Set[str]] = defaultdict(set)  # Topology graph
        self._metrics: Dict[str, Any] = {}                # Runtime metrics
        self._lock = asyncio.Lock()                       # Thread safety
```

### Key Methods

#### Agent Management

```python
async def register_agent(self, agent: BaseAgent) -> None:
    """Register a new agent instance (thread-safe)."""
    
async def unregister_agent(self, agent_id: str) -> None:
    """Remove an agent and all its connections."""
```

#### Connection Management

```python
async def connect_agents(self, src_id: str, dst_id: str) -> None:
    """Create a directed edge src → dst (idempotent)."""
    
async def disconnect_agents(self, src_id: str, dst_id: str) -> None:
    """Remove a directed edge src → dst."""
```

#### Topology Setup

```python
def setup_star_topology(self, center_id: str) -> None:
    """Setup star topology with center_id as hub."""
    
def setup_mesh_topology(self) -> None:
    """Setup full mesh topology (all-to-all connections)."""
```

#### Message Routing

```python
async def route_message(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
    """Forward message if edge exists."""
    
async def broadcast_message(self, src_id: str, payload: Dict[str, Any], exclude: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Broadcast message to all connected agents."""
```

#### Monitoring

```python
async def health_check(self) -> Dict[str, bool]:
    """Check health of all registered agents."""
    
def snapshot_metrics(self) -> Dict[str, Any]:
    """Return current metrics dict."""
```

### Topology Types

#### Star Topology
- **Description**: One central hub connected to all other agents
- **Use Case**: Centralized coordination, hub-and-spoke patterns
- **Characteristics**: 
  - Low complexity
  - Single point of failure
  - Efficient for broadcast operations

```
    A ---- B
    |      |
    D ---- C (center)
    |      |
    F ---- E
```

#### Mesh Topology
- **Description**: Every agent connected to every other agent
- **Use Case**: High availability, peer-to-peer communication
- **Characteristics**:
  - High redundancy
  - Complex routing
  - No single point of failure

```
A ---- B
|\    /|
| \  / |
|  \/  |
|  /\  |
| /  \ |
|/    \|
D ---- C
```

### Connection Process

1. **Agent Registration**: Agents register with the network
2. **Adapter Creation**: Protocol-specific adapters are created for each connection
3. **Topology Application**: Network topology is applied to create connections
4. **Health Monitoring**: Continuous monitoring begins

### Error Handling

- **Connection Failures**: Automatic retry with exponential backoff
- **Agent Failures**: Immediate detection and isolation
- **Network Partitions**: Graceful degradation and recovery
- **Protocol Errors**: Error propagation and logging

### Performance Considerations

- **Concurrent Operations**: Thread-safe operations using asyncio locks
- **Connection Pooling**: Reuse of HTTP connections for efficiency
- **Batched Operations**: Group operations for better throughput
- **Memory Management**: Automatic cleanup of disconnected agents

## Example Usage

```python
import asyncio
from agent_network import AgentNetwork, BaseAgent

async def example():
    # Create network
    network = AgentNetwork()
    
    # Create and register agents
    agent1 = await BaseAgent.create_a2a("agent-1", port=8001, executor=executor1)
    agent2 = await BaseAgent.create_a2a("agent-2", port=8002, executor=executor2)
    
    await network.register_agent(agent1)
    await network.register_agent(agent2)
    
    # Setup star topology
    network.setup_star_topology("agent-1")
    
    # Send message
    response = await network.route_message(
        "agent-1", "agent-2", 
        {"message": "Hello"}
    )
    
    # Health check
    health = await network.health_check()
    print(f"Health status: {health}")
    
    # Get metrics
    metrics = network.snapshot_metrics()
    print(f"Network metrics: {metrics}")

asyncio.run(example())
```

## Performance Metrics

- **Latency**: Message routing latency
- **Throughput**: Messages per second
- **Availability**: Agent uptime percentage
- **Resource Usage**: Memory and CPU utilization 