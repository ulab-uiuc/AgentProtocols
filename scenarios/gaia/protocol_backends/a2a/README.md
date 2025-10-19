# A2A Protocol Backend for GAIA Framework

This directory contains the A2A (Agent-to-Agent) protocol backend implementation for the GAIA framework.

## Files

- `__init__.py` - Module exports
- `agent.py` - A2AAgent and A2AExecutor implementation
- `network.py` - A2ANetwork implementation
- `README.md` - This documentation

## Features

- ✅ Full A2A SDK integration using native A2A protocol
- ✅ HTTP-based agent communication with Starlette/Uvicorn
- ✅ GAIA Core LLM integration (OpenAI-compatible)
- ✅ Real-time message processing with EventQueue
- ✅ Health check endpoints for monitoring
- ✅ Proper async/await patterns throughout
- ✅ No mock implementations or fallbacks

## Architecture

### A2AAgent
- Inherits from `MeshAgent` following GAIA framework conventions
- Manages HTTP server lifecycle using Starlette + Uvicorn
- Integrates with A2A SDK's `AgentExecutor` pattern
- Provides `/message` and `/health` endpoints

### A2AExecutor
- Implements A2A SDK's `AgentExecutor` interface
- Bridges A2A protocol messages to GAIA Core LLM
- Handles request context and event queue management
- Supports text message extraction and response formatting

### A2ANetwork
- Inherits from `MeshNetwork` following GAIA framework conventions
- Manages multiple A2A agents and their communication
- Provides agent registration and discovery
- Implements message delivery via HTTP backend

## Usage

```python
from protocol_backends.a2a import A2ANetwork

config = {
    'agents': [
        {
            'id': 1,
            'name': 'TestAgent',
            'tool': 'test',
            'port': 9000,
            'model_name': 'gpt-4o',
            'openai_api_key': 'your-key-here'
        }
    ],
    'agent_prompts': {
        '1': {'system_prompt': 'You are a helpful assistant.'}
    },
    'workflow': {'task_id': 'test_task'}
}

# Create and start network
network = A2ANetwork(config)
await network.start()

# Send message to agent
await network.deliver(1, {"content": "Hello, agent!"})

# Stop network
await network.stop()
```

## Testing

Run the included tests to verify functionality:

```bash
# Basic functionality test
python test_a2a_simple.py

# Standalone server test  
python test_a2a_standalone.py
```

## Requirements

- A2A SDK (`pip install a2a`)
- GAIA Core LLM (available in `src/utils/core.py`)
- Starlette/Uvicorn for HTTP server
- OpenAI API key for LLM calls
