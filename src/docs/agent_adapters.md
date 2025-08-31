# Agent Adapters - Protocol-Specific Client Adapters

### Overview

Agent Adapters provide protocol-specific client implementations that enable BaseAgent instances to communicate with different types of agents using various protocols. They abstract the complexity of different communication protocols behind a unified interface, allowing seamless integration and protocol switching.

### Architecture

The adapter system follows the **Adapter Pattern**, providing a consistent interface for different protocol implementations:

```
BaseProtocolAdapter (Abstract Interface)
    ├── A2AAdapter (Agent-to-Agent Protocol)
    ├── IoAAdapter (Internet of Agents - Future)
    └── CustomAdapter (User-defined protocols)
```

### Base Protocol Adapter

The `BaseProtocolAdapter` class defines the common interface that all protocol adapters must implement:

```python
class BaseProtocolAdapter:
    """Base class for all protocol adapters."""
    
    def __init__(self, base_url: str, auth_headers: Dict[str, str] = None):
        self.base_url = base_url
        self.auth_headers = auth_headers or {}
    
    async def initialize(self) -> None:
        """Initialize the adapter (fetch capabilities, setup connections, etc.)."""
        raise NotImplementedError
    
    async def send_message(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        """Send message to destination agent."""
        raise NotImplementedError
    
    async def receive_message(self) -> Dict[str, Any]:
        """Receive incoming messages (for protocols that support polling)."""
        raise NotImplementedError
    
    async def health_check(self) -> bool:
        """Check if the destination agent is healthy."""
        raise NotImplementedError
    
    async def cleanup(self) -> None:
        """Clean up adapter resources."""
        raise NotImplementedError
```

### A2A Adapter Implementation

The `A2AAdapter` is the primary implementation for the Agent-to-Agent protocol:

#### Core Features

- **HTTP-based Communication**: Uses HTTP/HTTPS for agent communication
- **JSON Message Format**: Standardized JSON payload structure
- **Streaming Support**: Both one-shot and streaming message delivery
- **Agent Discovery**: Automatic agent card fetching and capability discovery
- **Authentication**: Support for custom authentication headers
- **Connection Pooling**: Efficient connection reuse through shared httpx clients

#### Class Structure

```python
class A2AAdapter(BaseProtocolAdapter):
    def __init__(
        self, 
        httpx_client: httpx.AsyncClient, 
        base_url: str,
        auth_headers: Optional[Dict[str, str]] = None,
        agent_card_path: str = "/.well-known/agent.json"
    ):
        super().__init__(base_url=base_url, auth_headers=auth_headers or {})
        self.httpx_client = httpx_client
        self.base_url = base_url.rstrip('/')
        self.auth_headers = auth_headers or {}
        self.agent_card_path = agent_card_path
        self.agent_card: Dict[str, Any] = {}
        self._inbox_not_available = False
```

#### Message Format

The A2A adapter uses the official A2A message format:

```python
# Outgoing message structure
request_data = {
    "id": request_id,
    "params": {
        "message": payload.get("message", payload),
        "context": payload.get("context", {}),
        "routing": {
            "destination": dst_id,
            "source": payload.get("source", "unknown")
        }
    }
}
```

#### Key Methods

##### Message Sending

```python
async def send_message(self, dst_id: str, payload: Dict[str, Any]) -> Any:
    """Send message via A2A protocol using official A2A format."""
    
    # Construct A2A official message format
    request_id = str(uuid4())
    request_data = {
        "id": request_id,
        "params": {
            "message": payload.get("message", payload),
            "context": payload.get("context", {}),
            "routing": {
                "destination": dst_id,
                "source": payload.get("source", "unknown")
            }
        }
    }
    
    try:
        headers = {"Content-Type": "application/json"}
        headers.update(self.auth_headers)
        
        response = await self.httpx_client.post(
            f"{self.base_url}/message",
            content=json.dumps(request_data, separators=(',', ':')),
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise RuntimeError(f"A2A send failed: {e}") from e
```

##### Streaming Messages

```python
async def send_message_streaming(self, dst_id: str, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
    """Send message via A2A protocol and return streaming response."""
    
    async with self.httpx_client.stream(
        "POST",
        f"{self.base_url}/message",
        content=json.dumps(request_data, separators=(',', ':')),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        timeout=httpx.Timeout(30.0, read=None)
    ) as response:
        response.raise_for_status()
        
        async for line in response.aiter_lines():
            if line.strip():
                try:
                    clean_line = line.lstrip("data:").strip()
                    if clean_line:
                        event_data = json.loads(clean_line)
                        yield event_data
                except json.JSONDecodeError:
                    continue
```

##### Agent Discovery

```python
async def initialize(self) -> None:
    """Initialize by fetching the agent card from /.well-known/agent.json."""
    
    try:
        headers = {}
        headers.update(self.auth_headers)
        
        response = await self.httpx_client.get(
            f"{self.base_url}{self.agent_card_path}",
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        self.agent_card = response.json()
    except Exception as e:
        raise ConnectionError(f"Failed to initialize A2A adapter: {e}") from e
```

##### Health Monitoring

```python
async def health_check(self) -> bool:
    """Check if the A2A agent is responsive."""
    
    try:
        headers = {}
        headers.update(self.auth_headers)
        
        response = await self.httpx_client.get(
            f"{self.base_url}/health",
            headers=headers,
            timeout=5.0
        )
        return response.status_code == 200
    except Exception:
        return False
```

### Adapter Lifecycle

#### Initialization Process

1. **Adapter Creation**: Instantiate adapter with target agent URL
2. **Connection Setup**: Configure HTTP client and authentication
3. **Agent Discovery**: Fetch agent card and capabilities
4. **Capability Validation**: Verify protocol compatibility
5. **Ready State**: Adapter ready for message exchange

#### Runtime Operations

- **Message Routing**: Route messages based on destination agent ID
- **Error Handling**: Handle network failures and timeouts
- **Health Monitoring**: Regular health checks for destination agents
- **Metrics Collection**: Track performance and error metrics

#### Cleanup Process

- **Connection Termination**: Close active connections
- **Resource Release**: Free allocated resources
- **Cache Clearing**: Clear cached agent information

### Integration with BaseAgent

BaseAgent uses adapters through the outbound adapter registry:

```python
class BaseAgent:
    def __init__(self, ...):
        # Multi-adapter support: dst_id -> adapter
        self._outbound: Dict[str, BaseProtocolAdapter] = {}
    
    def add_outbound_adapter(self, dst_id: str, adapter: BaseProtocolAdapter) -> None:
        """Add an outbound adapter for connecting to a destination agent."""
        self._outbound[dst_id] = adapter
        self.outgoing_edges.add(dst_id)
    
    async def send(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        """Send message to destination agent using appropriate outbound adapter."""
        
        if dst_id not in self._outbound:
            raise RuntimeError(f"No outbound adapter found for destination {dst_id}")
        
        adapter = self._outbound[dst_id]
        
        # Add source information to payload
        enriched_payload = payload.copy()
        enriched_payload.setdefault("source", self.agent_id)
        
        # Delegate to protocol adapter
        response = await adapter.send_message(dst_id, enriched_payload)
        return response
```

### Error Handling

#### Connection Errors

```python
try:
    response = await adapter.send_message(dst_id, payload)
except httpx.TimeoutException as e:
    raise TimeoutError(f"A2A message timeout to {dst_id}") from e
except httpx.HTTPStatusError as e:
    raise ConnectionError(f"A2A HTTP error {e.response.status_code}") from e
except Exception as e:
    raise RuntimeError(f"A2A send failed: {e}") from e
```

#### Protocol Errors

- **Invalid Message Format**: Validation of A2A message structure
- **Authentication Failures**: Handling of auth-related errors
- **Protocol Mismatches**: Detection of incompatible protocol versions

### Performance Optimization

#### Connection Pooling

- **Shared HTTP Clients**: Single httpx client per adapter instance
- **Connection Limits**: Configurable connection pool size
- **Keep-Alive**: Persistent connections for better performance

#### Message Optimization

- **JSON Compression**: Compact JSON serialization
- **Batch Operations**: Group multiple messages when possible
- **Async Operations**: Non-blocking message sending

## Usage Examples

### Creating and Using an A2A Adapter

```python
import asyncio
import httpx
from agent_network.agent_adapters import A2AAdapter

async def example():
    # Create shared HTTP client
    client = httpx.AsyncClient(timeout=30.0)
    
    # Create A2A adapter
    adapter = A2AAdapter(
        httpx_client=client,
        base_url="http://target-agent:8080",
        auth_headers={"Authorization": "Bearer token123"}
    )
    
    # Initialize adapter
    await adapter.initialize()
    
    # Send message
    response = await adapter.send_message(
        "target-agent-id",
        {"message": "Hello from source agent"}
    )
    
    print(f"Response: {response}")
    
    # Check health
    is_healthy = await adapter.health_check()
    print(f"Agent healthy: {is_healthy}")
    
    # Cleanup
    await adapter.cleanup()
    await client.aclose()

asyncio.run(example())
```

### Custom Adapter Implementation

```python
from agent_network.agent_adapters import BaseProtocolAdapter

class CustomProtocolAdapter(BaseProtocolAdapter):
    """Custom protocol adapter implementation."""
    
    async def initialize(self) -> None:
        """Initialize custom protocol connection."""
        # Custom initialization logic
        pass
    
    async def send_message(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        """Send message using custom protocol."""
        # Custom message sending logic
        return {"status": "success", "data": "custom response"}
    
    async def receive_message(self) -> Dict[str, Any]:
        """Receive message using custom protocol."""
        # Custom message receiving logic
        return {"message": "custom incoming message"}
    
    async def health_check(self) -> bool:
        """Check agent health using custom protocol."""
        # Custom health check logic
        return True
    
    async def cleanup(self) -> None:
        """Clean up custom protocol resources."""
        # Custom cleanup logic
        pass
```

## Best Practices

1. **Connection Reuse**: Always reuse HTTP clients across adapters
2. **Error Handling**: Implement comprehensive error handling with retries
3. **Timeout Configuration**: Set appropriate timeouts for different operation types
4. **Resource Cleanup**: Always clean up adapters during shutdown
5. **Health Monitoring**: Regularly check adapter and destination health 