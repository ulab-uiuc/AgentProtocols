# QA Worker Agent

A question-answering agent built with the A2A SDK, using the `agent_network/utils/core.py` for LLM execution.

## Architecture

This QA worker follows the exact same pattern as `protocols/a2a-samples/samples/python/agents/helloworld`:

```
qa_worker/
├── __init__.py         # Package initialization  
├── __main__.py         # Main server entry point (A2A server setup)
├── agent_executor.py   # AgentExecutor implementation using Core LLM
├── test_client.py      # Test client for the QA agent
└── README.md          # This file
```

## Core Integration

The agent uses `agent_network/utils/core.py` for unified LLM execution:

- **OpenAI API**: If `OPENAI_API_KEY` environment variable is set
- **Local vLLM**: If running on localhost:8000  
- **Mock responses**: Fallback for testing when LLM is unavailable

## Quick Start

### 1. Run the QA Worker Agent

```bash
# Navigate to qa_worker directory
cd agent_network/streaming_queue/qa_worker

# Run the agent (starts A2A server on port 8000)
python -m qa_worker
```

### 2. Test the Agent

```bash
# Run test client (make sure agent is running first)
python test_client.py
```

## Agent Features

### Skills

- **question_answering**: Basic QA using Core LLM
  - Examples: "What is AI?", "How does ML work?", "Explain quantum computing"

- **advanced_qa**: Extended QA with detailed analysis (authenticated users)
  - Examples: "Detailed analysis of neural networks", "Mathematical foundations of ML"

### Agent Card

- **Name**: QA Worker Agent
- **URL**: http://localhost:8000/
- **Input/Output**: Text only
- **Streaming**: Supported
- **Extended Card**: Supported with authentication

## Integration with BaseAgent Network

This A2A agent can be integrated with the BaseAgent network framework:

```python
from base_agent import BaseAgent
import httpx

# Create BaseAgent wrapper for the A2A QA worker
client = httpx.AsyncClient()
agent = await BaseAgent.from_a2a(
    agent_id="QAWorker",
    url="http://localhost:8000",
    client=client
)

# Now the agent can be used in the BaseAgent network
```

## Configuration

The agent automatically detects and configures the best available LLM:

1. **OpenAI**: Uses `gpt-3.5-turbo` if API key is available
2. **Local**: Uses `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` on localhost:8000
3. **Mock**: Provides test responses when no LLM is available

## A2A SDK Compliance

This implementation is fully compliant with A2A SDK standards:

- ✅ Standard `AgentExecutor` pattern
- ✅ Proper `RequestContext` and `EventQueue` usage  
- ✅ Agent card with skills definition
- ✅ Support for public and extended agent cards
- ✅ Streaming message support
- ✅ Standard A2A server setup with `A2AStarletteApplication` 