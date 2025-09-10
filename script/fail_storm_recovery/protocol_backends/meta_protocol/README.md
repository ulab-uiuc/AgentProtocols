# Fail-Storm Meta-Protocol Integration

This directory contains the meta-protocol integration for fail-storm recovery testing. It provides a unified BaseAgent interface for all protocol backends (ACP, ANP, Agora, A2A) to enable cross-protocol communication and fault tolerance testing.

## Architecture

The meta-protocol integration follows the same pattern as `streaming_queue` but is adapted for fail-storm recovery scenarios:

```
protocol_backends/meta_protocol/
├── __init__.py                 # Main exports
├── meta_coordinator.py         # Central coordinator for all protocols
├── a2a_meta_agent.py          # A2A protocol wrapper
├── anp_meta_agent.py          # ANP protocol wrapper  
├── acp_meta_agent.py          # ACP protocol wrapper
├── agora_meta_agent.py        # Agora protocol wrapper
└── README.md                  # This file
```

## Components

### Meta Coordinator (`FailStormMetaCoordinator`)

Central coordinator that manages all protocol agents through BaseAgent interfaces:
- Dynamic load balancing across protocols
- Unified message routing and response handling  
- Protocol-agnostic shard QA task dispatching
- Cross-protocol performance comparison
- Fault tolerance and recovery metrics

### Protocol Meta Agents

Each protocol has a meta agent wrapper that:
- Wraps the original protocol agent with BaseAgent interface
- Converts between protocol-specific and unified message formats
- Integrates with ShardWorkerExecutor for fail-storm tasks
- Provides health monitoring and metrics collection

## Usage

### Basic Setup

```python
from protocol_backends.meta_protocol import (
    FailStormMetaCoordinator,
    create_a2a_meta_worker,
    create_anp_meta_worker,
    create_acp_meta_worker,
    create_agora_meta_worker
)

# Create coordinator
coordinator = FailStormMetaCoordinator()

# Add agents (2 per protocol for fail-storm testing)
config = {"core": {"type": "openai", "name": "gpt-4o", ...}}

await coordinator.add_protocol_agent("a2a", "A2A-Agent-1", config, port=9000)
await coordinator.add_protocol_agent("a2a", "A2A-Agent-2", config, port=9001)
await coordinator.add_protocol_agent("anp", "ANP-Agent-1", config, port=9002)
await coordinator.add_protocol_agent("anp", "ANP-Agent-2", config, port=9003)
# ... etc for acp and agora

# Setup network topology
await coordinator.install_outbound_adapters()
```

### Running Tests

Use the provided runner script:

```bash
# Run meta network test
cd script/fail_storm_recovery
python runners/run_meta_network.py --config config_meta.yaml

# Or run integration test first
python test_meta_integration.py
```

### Configuration

See `config_meta.yaml` for example configuration:

```yaml
protocols:
  acp:
    core:
      type: "openai"
      name: "gpt-4o"
      openai_api_key: "${OPENAI_API_KEY}"
      # ...
  anp:
    # ... similar config for each protocol
  agora:
    # ...
  a2a:
    # ...

base_port: 9000  # Starting port for agents
```

## Network Topology

The meta-protocol network creates:
- **8 agents total**: 2 agents per protocol (ACP, ANP, Agora, A2A)
- **Cross-protocol communication**: All agents can communicate through BaseAgent adapters
- **Fault tolerance**: Network remains functional even if some agents fail
- **Load balancing**: Tasks can be distributed across available agents

## Metrics and Monitoring

The coordinator collects comprehensive metrics:

```python
metrics = await coordinator.get_failstorm_metrics()
```

Metrics include:
- Protocol-specific performance stats
- Failure and recovery statistics  
- Network health status
- Cross-protocol communication metrics

## Fail-Storm Testing

The integration supports fail-storm recovery testing:

1. **Fault Injection**: Simulate agent failures
2. **Recovery Testing**: Test network resilience
3. **Performance Monitoring**: Track recovery times
4. **Cross-Protocol Fallback**: Route tasks to healthy agents

## Dependencies

- `src/core/base_agent.py` - BaseAgent interface
- `src/core/network.py` - Network management
- `src/agent_adapters/` - Protocol-specific adapters
- Protocol-specific SDKs (ACP, ANP, Agora, A2A)
- ShardWorkerExecutor for task execution

## Files Generated

- `results/meta_network_metrics.json` - Detailed performance metrics
- Logs from individual protocol agents
- Network topology and health reports
