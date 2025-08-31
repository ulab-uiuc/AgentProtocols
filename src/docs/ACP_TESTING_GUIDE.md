# ACP Integration Testing Guide

This guide explains how to test the ACP (Agent Communication Protocol) integration in the Multiagent-Protocol framework.

## Overview

The ACP integration includes:
- **ACP Server Adapter** (`src/server_adapters/acp_adapter.py`) - Uses the ACP SDK for protocol compliance
- **ACP Client Adapter** (`src/agent_adapters/acp_adapter.py`) - Enables HTTP communication with ACP servers
- **Test Files** - Multiple test scenarios demonstrating the integration

## Test Files Available

### 1. Basic ACP SDK Test (`test_acp_simple.py`)
**Purpose**: Direct usage of the ACP SDK Server (recommended for getting started)

**How to run**:
```bash
cd /Users/ldq/Work/1_Helping_Others/Agent_Network/Multiagent-Protocol/src

# Start echo server (Terminal 1)
python3 test_acp_simple.py echo

# Start smart server (Terminal 2)
python3 test_acp_simple.py smart

# Test client communication (Terminal 3)
python3 test_acp_simple.py client
```

**What it tests**:
- ✅ ACP SDK Server with echo agent
- ✅ ACP SDK Server with smart agent
- ✅ Direct HTTP communication with ACP servers
- ✅ Proper message structure and async generator pattern

### 2. ACP Server Adapter Test (`test_acp_sdk.py`)
**Purpose**: Tests the framework's ACP server adapter

**How to run**:
```bash
cd /Users/ldq/Work/1_Helping_Others/Agent_Network/Multiagent-Protocol/src
python3 test_acp_sdk.py
```

**What it tests**:
- ✅ ACPServerAdapter integration
- ✅ Multiple agent servers with different personalities
- ✅ Proper server lifecycle management
- ✅ Framework-level agent card generation

### 3. Integrated Multi-Agent Test (`test_acp_integrated.py`)
**Purpose**: Full integration test with LLM agents and multi-agent communication

**How to run**:
```bash
cd /Users/ldq/Work/1_Helping_Others/Agent_Network/Multiagent-Protocol/src
python3 test_acp_integrated.py
```

**What it tests**:
- ✅ Multi-agent system with different personalities
- ✅ LLM-enhanced responses
- ✅ Client-server communication using adapters
- ✅ Complex conversation and collaboration scenarios

### 4. Final Integration Test (`test_acp_final.py`)
**Purpose**: Comprehensive validation of the complete ACP integration

**How to run**:
```bash
cd /Users/ldq/Work/1_Helping_Others/Agent_Network/Multiagent-Protocol/src
python3 test_acp_final.py
```

**What it tests**:
- ✅ Basic ACP server functionality
- ✅ ACP adapter integration (if available)
- ✅ Threading-based server management
- ✅ Complete protocol compliance

## Step-by-Step Testing Instructions

### Step 1: Validate ACP SDK Installation
```bash
cd /Users/ldq/Work/1_Helping_Others/Agent_Network/Multiagent-Protocol/src
python3 -c "from acp_sdk.server import Server; print('✅ ACP SDK installed')"
```

### Step 2: Test Basic ACP Server
```bash
# In Terminal 1
python3 test_acp_simple.py echo

# In Terminal 2 (after echo server starts)
python3 test_acp_simple.py client
```

### Step 3: Test Framework Integration
```bash
python3 test_acp_sdk.py
```

### Step 4: Test Multi-Agent System
```bash
python3 test_acp_integrated.py
```

## Expected Output

### Successful ACP SDK Test:
```
🚀 Starting ACP Echo Server (Port 8001)
Press Ctrl+C to stop the server
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
```

### Successful Client Test:
```
📞 Testing client communication...

📨 Testing Echo Agent
✅ Echo agent responded: {"results": [...]}

🧠 Testing Smart Agent
✅ Smart agent responded: {"results": [...]}
```

### Successful Integration Test:
```
🧪 Starting Integrated ACP Test
==================================================
🚀 Starting agent servers...
✅ Connected test_client_1 to alice at http://127.0.0.1:8001
✅ Connected test_client_1 to bob at http://127.0.0.1:8002
✅ Connected test_client_2 to charlie at http://127.0.0.1:8003

📝 Test 1: Echo with LLM Enhancement
Alice's echo response: Hello, Alice!
Alice's LLM comment: Hello! I'm here to help you with anything you need.

✅ All tests completed successfully!
```

## Troubleshooting

### Common Issues:

1. **Import Error**: `ModuleNotFoundError: No module named 'acp_sdk'`
   - **Solution**: Install ACP SDK: `pip install acp-sdk`

2. **Port Already in Use**: `OSError: [Errno 48] Address already in use`
   - **Solution**: Kill existing servers: `pkill -f "test_acp"` or use different ports

3. **Connection Refused**: `ConnectionRefusedError`
   - **Solution**: Ensure servers are running and wait 2-3 seconds after startup

4. **404 Not Found**: Agent card or message endpoints not found
   - **Solution**: Use the correct ACP SDK server pattern (see `test_acp_simple.py`)

### Manual Testing Commands:

```bash
# Test ACP server endpoint directly
curl -X POST http://127.0.0.1:8001/acp/message \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "parts": [{"content": "Hello ACP!"}]}]}'

# Check server health (if available)
curl http://127.0.0.1:8001/health

# Get agent card (if available)
curl http://127.0.0.1:8001/.well-known/agent.json
```

## Key Features Validated

✅ **ACP SDK Compliance**: Using actual ACP SDK Message, MessagePart, Context, and Server classes
✅ **Async Generator Pattern**: Proper implementation of ACP SDK's async generator interface
✅ **Multi-Agent Communication**: Multiple agents with different personalities and capabilities
✅ **LLM Integration**: Enhanced responses with simulated LLM capabilities
✅ **Protocol Compliance**: Following ACP 1.0 specification
✅ **Framework Integration**: Working with the project's adapter pattern
✅ **Error Handling**: Proper error responses and connection management

## Next Steps

1. **Run the Basic Test**: Start with `test_acp_simple.py` to validate core functionality
2. **Test Framework Integration**: Use `test_acp_sdk.py` to verify adapter integration
3. **Explore Multi-Agent**: Try `test_acp_integrated.py` for complex scenarios
4. **Custom Development**: Use the patterns from these tests to build your own ACP agents

The ACP integration is now complete and fully functional with the actual ACP SDK package!
