# 🤖 Multiagent-Protocol

[Website](https://example.com) | [Paper (placeholder)](#) | [Licenses](LICENSE)

A comprehensive multi-agent communication framework supporting multiple protocols for distributed AI systems. This framework enables seamless interaction between AI agents across different communication paradigms with built-in security, monitoring, and scalability features.

## 🌟 Features

- **🔗 Multi-Protocol Support**: ANP, A2A, ACP, Agora, and custom protocols
- **🏗️ Modular Architecture**: Protocol-agnostic design with pluggable backends
- **🔒 Security-First**: DID authentication, E2E encryption, privacy protection
- **📊 Real-time Monitoring**: Performance metrics, health checks, and observability
- **🌐 Distributed Systems**: Support for complex multi-agent workflows
- **🧪 Testing Frameworks**: Comprehensive testing suites for each protocol
- **⚡ High Performance**: Async/await patterns, concurrent execution

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Supported Scenarios](#-supported-scenarios)
- [Protocol Guide](#-protocol-guide)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Development](#-development)
- [Contributing](#-contributing)
- [中文文档 (简体中文)](README_zh_CN.md)

## 🚀 Quick Start

### Prerequisites

```bash
# Required environment
Python 3.11+
OpenAI API Key (for LLM-based agents)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/MultiagentBench/Multiagent-Protocol.git
cd Multiagent-Protocol

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY='sk-your-openai-api-key-here'
```

## 🎯 Supported Scenarios

### 1. 🌍 GAIA (General AI Agent) Framework
**Purpose**: Task execution and coordination across distributed AI agents

**Supported Protocols**:
- **ANP (Agent Network Protocol)**: DID-based authentication with E2E encryption
- **A2A (Agent-to-Agent)**: Direct peer communication with message routing
- **ACP (Agent Communication Protocol)**: Session-based conversation management
- **Agora**: Tool-based agent orchestration with LangChain integration

**Usage**:
```bash
# Set API key
export OPENAI_API_KEY='sk-your-key-here'

# Run with different protocols
python -m script.gaia.runners.run_anp        # ANP Protocol
python -m script.gaia.runners.run_a2a        # A2A Protocol
python -m script.gaia.runners.run_acp        # ACP Protocol
python -m script.gaia.runners.run_agora      # Agora Protocol
python -m script.gaia.runners.run_dummy      # Dummy Protocol

# Meta-protocol coordination
python -m script.gaia.runners.run_meta_protocol
```

### 2. 📡 Streaming Queue
**Purpose**: High-throughput message processing with coordinator-worker patterns

**Supported Protocols**: ANP, A2A, ACP, Agora, Meta-Protocol

**Usage**:
```bash
export OPENAI_API_KEY='sk-your-key-here'

# Stream processing with different protocols
python -m script.streaming_queue.runner.run_anp     # ANP Streaming
python -m script.streaming_queue.runner.run_a2a     # A2A Streaming
python -m script.streaming_queue.runner.run_acp     # ACP Streaming
python -m script.streaming_queue.runner.run_agora   # Agora Streaming

# Meta-network coordination
python -m script.streaming_queue.runner.run_meta_network
```

### 3. 🛡️ Safety Tech
**Purpose**: Privacy-preserving agent communication and security testing

**Supported Protocols**: ANP, A2A, ACP, Agora, Meta

**Usage**:
```bash
export OPENAI_API_KEY='sk-your-key-here'

# Unified security testing
python -m script.safety_tech.runners.run_unified_security_test_anp
python -m script.safety_tech.runners.run_unified_security_test_a2a
python -m script.safety_tech.runners.run_unified_security_test_acp
python -m script.safety_tech.runners.run_unified_security_test_agora

# S2 Meta-protocol security analysis
python -m script.safety_tech.runners.run_s2_meta
```

### 4. 🔄 Fail Storm Recovery
**Purpose**: Fault-tolerant systems with automatic recovery mechanisms

**Supported Protocols**: ANP, A2A, ACP, Agora, Simple JSON, Meta-Protocol

**Usage**:
```bash
export OPENAI_API_KEY='sk-your-key-here'

# Fault tolerance testing
python -m script.fail_storm_recovery.runners.run_anp
python -m script.fail_storm_recovery.runners.run_a2a
python -m script.fail_storm_recovery.runners.run_acp
python -m script.fail_storm_recovery.runners.run_agora
python -m script.fail_storm_recovery.runners.run_simple_json

# Meta-protocol coordination
python -m script.fail_storm_recovery.runners.run_meta
python -m script.fail_storm_recovery.runners.run_meta_network
```

## 🔧 Protocol Guide

### ANP (Agent Network Protocol)
- **Features**: DID authentication, E2E encryption, WebSocket communication
- **Use Cases**: Secure agent networks, identity-verified communications
- **Dependencies**: `agentconnect_src/` (AgentConnect SDK)

### A2A (Agent-to-Agent Protocol)
- **Features**: Direct peer communication, JSON-RPC messaging, event streaming
- **Use Cases**: High-performance agent coordination, real-time messaging
- **Dependencies**: `a2a-sdk`, `a2a-server`

### ACP (Agent Communication Protocol)
- **Features**: Session management, conversation threads, message history
- **Use Cases**: Conversational agents, multi-turn interactions
- **Dependencies**: `acp-sdk`

### Agora Protocol
- **Features**: Tool orchestration, LangChain integration, function calling
- **Use Cases**: Tool-enabled agents, LLM-powered workflows
- **Dependencies**: `agora-protocol`, `langchain`

### Meta-Protocol
- **Features**: Protocol abstraction, adaptive routing, multi-protocol support
- **Use Cases**: Protocol-agnostic applications, seamless migration

## ⚙️ Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY='sk-your-openai-api-key'

# Optional
export ANTHROPIC_API_KEY='your-anthropic-key'
export OPENAI_BASE_URL='https://api.openai.com/v1'  # Custom endpoint
export LOG_LEVEL='INFO'                              # DEBUG, INFO, WARNING, ERROR
```

### Configuration Files

Each scenario uses YAML configuration files located in `script/{scenario}/config/`:

```yaml
# Example: script/gaia/config/anp.yaml
model:
  type: "openai"
  name: "gpt-4o"
  temperature: 0.0
  api_key: "${OPENAI_API_KEY}"

network:
  host: "127.0.0.1"
  port_range:
    start: 9000
    end: 9010

agents:
  - id: 1
    name: "Agent1"
    tool: "create_chat_completion"
    max_tokens: 500

workflow:
  type: "sequential"
  max_steps: 5
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Multiagent-Protocol                      │
├─────────────────────────────────────────────────────────────┤
│  Scenarios                                                  │
│  ┌─────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │  GAIA   │ │  Streaming  │ │ Safety Tech │ │ Fail Storm  │ │
│  │         │ │    Queue    │ │             │ │  Recovery   │ │
│  └─────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Protocol Backends                                          │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────────────┐             │
│  │ ANP │ │ A2A │ │ ACP │ │Agora│ │Meta-Protocol│             │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────────────┘             │
├─────────────────────────────────────────────────────────────┤
│  Core Infrastructure                                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │   Network   │ │   Agents    │ │ Monitoring  │            │
│  │   Layer     │ │   Layer     │ │   Layer     │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# Run tests for specific scenario
pytest script/gaia/tests/
pytest script/safety_tech/tests/

# Run with coverage
pytest --cov=src --cov-report=html
```

### Adding New Protocols

1. Create protocol backend in `script/{scenario}/protocol_backends/{protocol_name}/`
2. Implement required interfaces: `agent.py`, `network.py`, `comm.py`
3. Add configuration in `script/{scenario}/config/{protocol_name}.yaml`
4. Create runner in `script/{scenario}/runners/run_{protocol_name}.py`

### Code Structure

```
script/
├── {scenario}/                    # Scenario implementation
│   ├── config/                   # Configuration files
│   ├── protocol_backends/        # Protocol implementations
│   │   ├── {protocol}/
│   │   │   ├── agent.py         # Agent implementation
│   │   │   ├── network.py       # Network coordinator
│   │   │   └── comm.py          # Communication backend
│   ├── runners/                  # Entry point scripts
│   └── tools/                    # Scenario-specific tools
├── common/                       # Shared utilities
└── requirements.txt              # Dependencies
```

## 📊 Monitoring & Observability

The framework includes comprehensive monitoring capabilities:

- **Performance Metrics**: Message throughput, latency, success rates
- **Health Monitoring**: Agent status, network connectivity, resource usage
- **Security Auditing**: Authentication events, encryption status, privacy compliance
- **Custom Dashboards**: Protocol-specific visualizations and alerts

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure all tests pass: `pytest`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Wiki](https://github.com/your-org/Multiagent-Protocol/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-org/Multiagent-Protocol/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/Multiagent-Protocol/discussions)

## 🚀 Getting Started Examples

### Example 1: Simple GAIA Task Execution

```bash
# 1. Set up environment
export OPENAI_API_KEY='sk-your-key'

# 2. Run GAIA with ANP protocol
python -m script.gaia.runners.run_anp

# 3. Monitor output for agent interactions
# Expected: Multiple agents collaborating on tasks with DID authentication
```

### Example 2: High-Throughput Streaming

```bash
# 1. Set up environment
export OPENAI_API_KEY='sk-your-key'

# 2. Run streaming queue with A2A
python -m script.streaming_queue.runner.run_a2a

# 3. Observe coordinator-worker message processing
# Expected: High-frequency message exchange with load balancing
```

### Example 3: Security Testing

```bash
# 1. Set up environment
export OPENAI_API_KEY='sk-your-key'

# 2. Run privacy-aware security tests
python -m script.safety_tech.runners.run_unified_security_test_anp

# 3. Review privacy protection mechanisms
# Expected: Encrypted communication with privacy compliance reports
```

---

**Built with ❤️ by the Multi-Agent Systems Community**