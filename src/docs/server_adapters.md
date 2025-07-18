# Server Adapters - Protocol-Specific Server Implementations

### Overview

Server Adapters provide protocol-specific server implementations that enable BaseAgent instances to receive and process incoming messages using various protocols. They handle the server-side communication, request parsing, executor integration, and response formatting for different protocol standards.

### Architecture

The server adapter system follows a pluggable architecture pattern:

```
BaseServerAdapter (Abstract Interface)
    ├── A2AServerAdapter (Agent-to-Agent Protocol Server)
    ├── DummyServerAdapter (Testing/Development Server)
    └── CustomServerAdapter (User-defined protocols)
```

### Base Server Adapter

The `BaseServerAdapter` class defines the interface for all server adapter implementations:

```python
class BaseServerAdapter:
    """Base class for protocol-specific server adapters."""
    
    protocol_name = "BaseProtocol"
    
    def build(
        self,
        host: str,
        port: int,
        agent_id: str,
        executor: Any,
        **kwargs
    ) -> Tuple[uvicorn.Server, Dict[str, Any]]:
        """
        Build and configure the server instance.
        
        Returns:
            Tuple of (uvicorn.Server, agent_card)
        """
        raise NotImplementedError("Subclasses must implement build method")
```

### A2A Server Adapter

The `A2AServerAdapter` is the primary implementation for the Agent-to-Agent protocol server:

#### Core Features

- **A2A Protocol Compliance**: Full compliance with A2A specification
- **SDK Native Integration**: Direct integration with A2A SDK executor interface
- **Dual Response Modes**: Support for both JSON and SSE streaming responses
- **Agent Card Generation**: Automatic agent card creation and serving
- **Health Monitoring**: Built-in health check endpoints
- **Request Validation**: Comprehensive request validation and error handling

#### Class Structure

```python
class A2AServerAdapter(BaseServerAdapter):
    """Server adapter for A2A (Agent-to-Agent) protocol with SDK native interface."""
    
    protocol_name = "A2A"
    
    def build(
        self,
        host: str,
        port: int,
        agent_id: str,
        executor: Any,
        **kwargs
    ) -> Tuple[uvicorn.Server, Dict[str, Any]]:
        """Build A2A server using SDK native executor interface."""
```

#### Agent Card Generation

The adapter automatically generates A2A-compliant agent cards:

```python
agent_card = {
    "name": f"Agent {agent_id}",
    "url": f"http://{host}:{port}/",
    "protocolVersion": "1.0.0",
    "skills": [
        {
            "id": "agent_execution",
            "name": "Agent Execution",
            "description": "Execute agent tasks using A2A SDK native interface"
        }
    ],
    "capabilities": {
        "streaming": True,
        "supportsAuthenticatedExtendedCard": False,
        "nativeSDK": True
    }
}
```

#### Server Endpoints

The A2A server exposes the following endpoints:

- **`/.well-known/agent.json`** - Agent card discovery endpoint
- **`/health`** - Health check endpoint
- **`/message`** - Message processing endpoint

#### Request Processing Flow

1. **Request Reception**: HTTP POST request received on `/message`
2. **Request Parsing**: Extract A2A message structure from JSON body
3. **Context Creation**: Create SDK RequestContext from request data
4. **Executor Invocation**: Call executor.execute(context, event_queue)
5. **Response Collection**: Collect events from event queue
6. **Response Formatting**: Format response based on Accept header
7. **Response Delivery**: Send JSON or SSE response to client

### A2A Starlette Application

The core server implementation uses Starlette framework:

```python
class A2AStarletteApplication:
    """A2A server implementation using SDK native executor interface."""
    
    def __init__(self, agent_card: Dict[str, Any], executor: Any):
        self.agent_card = agent_card
        self.executor = executor
        
    def build(self) -> Starlette:
        """Build the Starlette application."""
        routes = [
            Route("/.well-known/agent.json", self.get_agent_card, methods=["GET"]),
            Route("/health", self.health_check, methods=["GET"]),
            Route("/message", self.handle_message, methods=["POST"]),
        ]
        
        return Starlette(routes=routes)
```

#### Message Handling

```python
async def handle_message(self, request: Request) -> JSONResponse | StreamingResponse:
    """Handle incoming messages using SDK native executor interface."""
    
    try:
        # Parse request body
        body = await request.json()
        
        # Create SDK RequestContext
        if 'params' in body and 'message' in body['params']:
            message_data = body['params']['message']
            message = Message(**message_data)
            params = MessageSendParams(message=message)
            ctx = RequestContext(params)
        else:
            # Fallback for simple messages
            text = body.get('text', str(body))
            message = new_agent_text_message(text, role=Role.user)
            params = MessageSendParams(message=message)
            ctx = RequestContext(params)
        
        # Create EventQueue
        queue = EventQueue()
        
        # Call SDK native executor
        await self.executor.execute(ctx, queue)
        
        # Handle response based on Accept header
        accept_header = request.headers.get("accept", "")
        if "text/event-stream" in accept_header:
            return StreamingResponse(
                self._sse_generator(queue),
                media_type="text/event-stream"
            )
        else:
            # Collect events and return JSON
            events = []
            try:
                while True:
                    event = await queue.dequeue_event(no_wait=True)
                    events.append(self._event_to_dict(event))
            except asyncio.QueueEmpty:
                pass
            
            return JSONResponse({"events": events})
            
    except Exception as e:
        return JSONResponse(
            {"error": f"Message handling failed: {e}"},
            status_code=500
        )
```

#### SSE Streaming Support

```python
async def _sse_generator(self, queue: EventQueue):
    """Generate SSE events from EventQueue."""
    
    try:
        while True:
            try:
                event = await queue.dequeue_event(no_wait=True)
                event_data = self._event_to_dict(event)
                yield f"data: {json.dumps(event_data)}\n\n"
            except asyncio.QueueEmpty:
                try:
                    event = await queue.dequeue_event(no_wait=False)
                    event_data = self._event_to_dict(event)
                    yield f"data: {json.dumps(event_data)}\n\n"
                except asyncio.QueueEmpty:
                    break
    except Exception as e:
        error_data = {"error": str(e), "type": "stream_error"}
        yield f"data: {json.dumps(error_data)}\n\n"
```

### Dummy Server Adapter

For testing and development purposes, a dummy server adapter is provided:

```python
class DummyServerAdapter(BaseServerAdapter):
    """Dummy server adapter for testing purposes."""
    
    protocol_name = "Dummy"
    
    def build(
        self,
        host: str,
        port: int,
        agent_id: str,
        executor: Any,
        **kwargs
    ) -> Tuple[uvicorn.Server, Dict[str, Any]]:
        """Build a simple dummy server for testing."""
        
        # Create simple dummy application
        app = self._create_dummy_app(agent_id, executor)
        
        # Generate dummy agent card
        agent_card = {
            "name": f"Dummy Agent {agent_id}",
            "url": f"http://{host}:{port}/",
            "protocol": "dummy",
            "capabilities": ["echo", "ping"]
        }
        
        # Configure uvicorn server
        config = uvicorn.Config(app, host=host, port=port, log_level="error")
        server = uvicorn.Server(config)
        
        return server, agent_card
```

### Server Lifecycle Management

#### Startup Process

1. **Adapter Selection**: Choose appropriate server adapter
2. **Executor Validation**: Validate executor interface compatibility
3. **Server Building**: Build server instance and agent card
4. **Background Task**: Start server in asyncio background task
5. **Health Polling**: Wait for server readiness
6. **Card Fetching**: Retrieve and cache agent card

#### Runtime Operations

- **Request Processing**: Handle incoming protocol requests
- **Executor Integration**: Bridge requests to executor interface
- **Response Handling**: Format and deliver responses
- **Health Monitoring**: Respond to health check requests
- **Error Handling**: Handle and report processing errors

#### Shutdown Process

- **Graceful Shutdown**: Signal server to stop accepting requests
- **Request Completion**: Wait for active requests to complete
- **Resource Cleanup**: Clean up server resources
- **Task Cancellation**: Cancel server background task

### Integration with BaseAgent

Server adapters are seamlessly integrated into BaseAgent:

```python
class BaseAgent:
    async def _start_server(self, executor: Any) -> None:
        """Start the internal server using pluggable adapter."""
        
        # Use server adapter to build server and agent card
        self._server_instance, self._self_agent_card = self._server_adapter.build(
            host=self._host,
            port=self._port,
            agent_id=self.agent_id,
            executor=executor
        )
        
        # Start server in background task
        self._server_task = asyncio.create_task(self._server_instance.serve())
        
        # Wait for server to be ready
        await self._wait_for_server_ready()
```

### Error Handling

#### Request Validation Errors

```python
try:
    body = await request.json()
    # Validate A2A message structure
    if not self._validate_a2a_message(body):
        return JSONResponse(
            {"error": "Invalid A2A message format"},
            status_code=400
        )
except json.JSONDecodeError:
    return JSONResponse(
        {"error": "Invalid JSON in request body"},
        status_code=400
    )
```

#### Executor Errors

```python
try:
    await self.executor.execute(ctx, queue)
except Exception as e:
    logger.error(f"Executor error: {e}")
    return JSONResponse(
        {"error": f"Execution failed: {str(e)}"},
        status_code=500
    )
```

#### Protocol Errors

- **Unsupported Methods**: Return 405 Method Not Allowed
- **Missing Headers**: Return 400 Bad Request with details
- **Protocol Violations**: Return 400 Bad Request with violation details

### Performance Optimization

#### Request Processing

- **Async Handlers**: Non-blocking request processing
- **Connection Pooling**: Efficient connection management
- **Response Streaming**: Memory-efficient streaming responses
- **Request Batching**: Batch processing when applicable

#### Resource Management

- **Memory Efficiency**: Minimal memory footprint per request
- **CPU Optimization**: Efficient JSON parsing and serialization
- **Connection Limits**: Configurable connection limits

## Usage Examples

### Creating a Custom Server Adapter

```python
from agent_network.server_adapters import BaseServerAdapter
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

class CustomServerAdapter(BaseServerAdapter):
    """Custom protocol server adapter."""
    
    protocol_name = "Custom"
    
    def build(self, host: str, port: int, agent_id: str, executor: Any, **kwargs):
        """Build custom protocol server."""
        
        # Create custom Starlette app
        app = self._create_custom_app(agent_id, executor)
        
        # Generate custom agent card
        agent_card = {
            "name": f"Custom Agent {agent_id}",
            "url": f"http://{host}:{port}/",
            "protocol": "custom-v1",
            "capabilities": ["custom_processing", "custom_streaming"]
        }
        
        # Configure server
        config = uvicorn.Config(app, host=host, port=port, log_level="error")
        server = uvicorn.Server(config)
        
        return server, agent_card
    
    def _create_custom_app(self, agent_id: str, executor: Any) -> Starlette:
        """Create custom Starlette application."""
        
        async def custom_handler(request):
            """Handle custom protocol requests."""
            try:
                body = await request.json()
                # Custom processing logic
                result = await self._process_custom_message(body, executor)
                return JSONResponse({"result": result})
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
        
        routes = [
            Route("/custom", custom_handler, methods=["POST"]),
            Route("/health", lambda r: JSONResponse({"status": "ok"}), methods=["GET"])
        ]
        
        return Starlette(routes=routes)
    
    async def _process_custom_message(self, message: dict, executor: Any):
        """Process message using custom protocol."""
        # Implement custom message processing
        return {"processed": True, "message": message}
```

### Using Custom Server Adapter

```python
import asyncio
from agent_network import BaseAgent

async def example():
    # Create agent with custom server adapter
    custom_adapter = CustomServerAdapter()
    
    agent = await BaseAgent.create_a2a(
        agent_id="custom-agent",
        host="localhost",
        port=8080,
        executor=your_custom_executor,
        server_adapter=custom_adapter
    )
    
    print(f"Custom agent running at: {agent.get_listening_address()}")
    
    # Agent is now serving custom protocol
    await asyncio.sleep(3600)  # Keep running
    
    await agent.stop()

asyncio.run(example())
```

## Configuration Examples

### A2A Server Configuration

```python
# Uvicorn configuration for A2A server
config = uvicorn.Config(
    app,
    host=host,
    port=port,
    log_level="error",  # Minimize logging
    access_log=False,   # Disable access logs
    loop="uvloop",      # Use uvloop for better performance
    http="httptools"    # Use httptools for HTTP parsing
)
```

### Server Adapter Registry

```python
# Register custom server adapters
SERVER_ADAPTERS = {
    "a2a": A2AServerAdapter,
    "dummy": DummyServerAdapter,
    "custom": CustomServerAdapter
}

def get_server_adapter(protocol: str) -> BaseServerAdapter:
    """Get server adapter by protocol name."""
    if protocol not in SERVER_ADAPTERS:
        raise ValueError(f"Unknown protocol: {protocol}")
    return SERVER_ADAPTERS[protocol]()
```

## Best Practices

1. **Executor Validation**: Always validate executor interface before server start
2. **Error Handling**: Implement comprehensive error handling for all endpoints
3. **Health Endpoints**: Always provide health check endpoints for monitoring
4. **Resource Limits**: Set appropriate connection and request limits
5. **Security**: Implement proper authentication and input validation 