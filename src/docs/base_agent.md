# BaseAgent - Dual-Role Agent Implementation

### Overview

The `BaseAgent` class is a sophisticated dual-role agent implementation that functions as both a server (receiving messages) and a multi-client (sending messages to different targets using various protocols). It provides a unified interface for agent communication while supporting multiple protocol adapters and server implementations.

### Key Features

- **Dual-Role Architecture**: Acts as both HTTP server and client
- **Protocol Agnostic**: Supports multiple communication protocols through adapters
- **A2A SDK Integration**: Native support for A2A (Agent-to-Agent) protocol
- **Concurrent Operations**: Asynchronous message handling and processing
- **Health Monitoring**: Built-in health check endpoints
- **Automatic Discovery**: Agent card discovery and capability exchange
- **Resource Management**: Proper lifecycle management and cleanup

### Class Architecture

```python
class BaseAgent:
    def __init__(self, agent_id: str, host: str = "0.0.0.0", port: Optional[int] = None, 
                 httpx_client: Optional[httpx.AsyncClient] = None, 
                 server_adapter: Optional[BaseServerAdapter] = None):
        self.agent_id = agent_id
        self._host = host
        self._port = port or self._find_free_port()
        self._httpx_client = httpx_client or httpx.AsyncClient(timeout=30.0)
        self._server_adapter = server_adapter or A2AServerAdapter()
        
        # Multi-adapter support: dst_id -> adapter
        self._outbound: Dict[str, BaseProtocolAdapter] = {}
        
        # Server components
        self._server_task: Optional[asyncio.Task] = None
        self._server_instance: Optional[uvicorn.Server] = None
        self._self_agent_card: Optional[Dict[str, Any]] = None
```

### Factory Methods

#### A2A Agent Creation

```python
@classmethod
async def create_a2a(
    cls,
    agent_id: str,
    host: str = "0.0.0.0", 
    port: Optional[int] = None,
    executor: Optional[Any] = None,
    httpx_client: Optional[httpx.AsyncClient] = None,
    server_adapter: Optional[BaseServerAdapter] = None
) -> "BaseAgent":
    """Create BaseAgent with A2A server capability."""
```

**Features:**
- Validates executor implements SDK native interface
- Starts HTTP server with A2A endpoints
- Fetches and caches agent card
- Returns fully initialized agent instance

#### Legacy Methods (Deprecated)

```python
@classmethod
async def from_a2a(cls, agent_id: str, base_url: str, 
                   httpx_client: Optional[httpx.AsyncClient] = None) -> "BaseAgent":
    """DEPRECATED: v0.x compatibility method (client-only mode)."""
```

### Core Operations

#### Message Sending

```python
async def send(self, dst_id: str, payload: Dict[str, Any]) -> Any:
    """Send message to destination agent using appropriate outbound adapter."""
```

**Process:**
1. Validate agent is initialized
2. Find appropriate outbound adapter for destination
3. Enrich payload with source information
4. Record metrics and timing
5. Delegate to protocol adapter
6. Handle errors and record failures

#### Health Monitoring

```python
async def health_check(self) -> bool:
    """Check if the agent's server is healthy and responsive."""
```

**Capabilities:**
- Server responsiveness check
- HTTP endpoint validation
- Timeout handling
- Status code verification

#### Agent Discovery

```python
def get_card(self) -> Dict[str, Any]:
    """Get this agent's card (from local server)."""
```

**Agent Card Contents:**
- Agent ID and metadata
- Server address and capabilities
- Protocol support information
- Connection statistics

### Server Architecture

#### A2A Server Endpoints

- `/.well-known/agent.json` - Agent card discovery
- `/health` - Health check endpoint  
- `/message` - Message handling endpoint

#### Executor Interface

```python
# Required executor interface
async def execute(context: RequestContext, event_queue: EventQueue) -> None:
    """Process incoming requests and enqueue response events."""
```

**Requirements:**
- Must implement async execute method
- Accept RequestContext and EventQueue parameters
- Handle message processing and response generation
- Support both JSON and streaming responses

### Connection Management

#### Outbound Adapters

```python
def add_outbound_adapter(self, dst_id: str, adapter: BaseProtocolAdapter) -> None:
    """Add an outbound adapter for connecting to a destination agent."""

def remove_outbound_adapter(self, dst_id: str) -> None:
    """Remove an outbound adapter."""
    
def get_outbound_adapters(self) -> Dict[str, BaseProtocolAdapter]:
    """Get all outbound adapters (for debugging/monitoring)."""
```

#### Protocol Support

- **A2A (Agent-to-Agent)**: Primary protocol support
- **IoA (Internet of Agents)**: Planned future support
- **Custom Protocols**: Extensible through adapter pattern

### Lifecycle Management

#### Startup Process

1. **Port Allocation**: Find available port if not specified
2. **Server Creation**: Initialize server adapter with executor
3. **Server Start**: Launch uvicorn server in background task
4. **Health Wait**: Wait for server to become ready
5. **Card Fetch**: Retrieve agent card from server
6. **Initialization**: Mark agent as initialized

#### Shutdown Process

```python
async def stop(self) -> None:
    """Gracefully stop the agent server and clean up all resources."""
```

**Steps:**
1. Signal server shutdown
2. Wait for graceful shutdown (with timeout)
3. Force cancel if needed
4. Clean up all adapters
5. Clear tracking data
6. Mark as uninitialized

### Error Handling

#### Connection Errors
- Automatic retry with exponential backoff
- Circuit breaker pattern for failing endpoints
- Graceful degradation for partial failures

#### Server Errors
- Health check failure detection
- Automatic server restart capabilities
- Error logging and metrics

#### Protocol Errors
- Protocol-specific error handling
- Error propagation to callers
- Standardized error formats

### Performance Optimization

#### Connection Pooling
- Shared httpx client across adapters
- Persistent connections for efficiency
- Connection limit management

#### Concurrent Processing
- Asynchronous message handling
- Non-blocking operations
- Parallel adapter management

#### Resource Management
- Automatic cleanup of unused connections
- Memory-efficient data structures
- Garbage collection integration

## Example Implementation

```python
import asyncio
from agent_network import BaseAgent
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

class SimpleExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Get user input
        user_input = context.get_user_input()
        
        # Process message
        response = f"Echo: {user_input}"
        
        # Send response
        await event_queue.enqueue_event(new_agent_text_message(response))
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(new_agent_text_message("Operation cancelled"))

async def example():
    # Create agent with executor
    agent = await BaseAgent.create_a2a(
        agent_id="example-agent",
        host="localhost",
        port=8080,
        executor=SimpleExecutor()
    )
    
    # Agent is now running and ready to receive messages
    print(f"Agent running at: {agent.get_listening_address()}")
    print(f"Agent card: {agent.get_card()}")
    
    # Keep running
    try:
        await asyncio.sleep(3600)  # Run for 1 hour
    finally:
        await agent.stop()  # Clean shutdown

asyncio.run(example())
```

## Best Practices

1. **Executor Design**: Keep executors lightweight and focused
2. **Error Handling**: Always implement proper error handling in executors
3. **Resource Cleanup**: Use try/finally blocks for proper cleanup
4. **Health Monitoring**: Regularly check agent health in production
5. **Protocol Selection**: Choose appropriate protocols based on use case requirements 