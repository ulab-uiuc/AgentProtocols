# Privacy Protection Testing Framework

A comprehensive testing framework for evaluating privacy protection capabilities of different agent communication protocols. This framework tests which protocols are better for agent communication without leaking sensitive information.

## Overview

This framework implements a 3-step privacy testing methodology:

1. **Dataset Enhancement**: Takes medical Q&A data and enhances it with synthetic sensitive information (names, SSNs, phone numbers, addresses)
2. **Agent Simulation**: Simulates conversations between privacy-aware and privacy-invasive agents using different communication protocols
3. **Privacy Analysis**: Analyzes conversations to detect and score privacy violations

## Features

- **Protocol-Agnostic Design**: Support for multiple communication protocols (ACP, ANP, Direct)
- **Configurable Testing**: Easy configuration via INI file for different test scenarios
- **Comprehensive Analysis**: Detailed privacy violation detection with weighted scoring
- **Unified Workflow**: Single command execution of the complete testing pipeline
- **Fresh Test Generation**: Each run generates new conversations with unique scenarios

## Supported Protocols

- **ACP (Agent Communication Protocol)**: Enterprise-grade protocol with advanced privacy features
- **ANP (Agent Network Protocol)**: Network-optimized protocol with basic privacy controls
- **Direct**: Unprotected direct communication (baseline for comparison)

## Quick Start

### Prerequisites

```bash
# Python 3.8+ required
# No external dependencies - uses standard library only
```

### Basic Usage

Run the complete privacy testing workflow:

```bash
# Test with ACP protocol (recommended)
python3 run_privacy_test.py --protocol acp

# Test with ANP protocol
python3 run_privacy_test.py --protocol anp

# Test with Direct protocol (baseline)
python3 run_privacy_test.py --protocol direct
```

### Custom Configuration

Modify `config.ini` to customize the testing parameters:

```ini
[DEFAULT]
protocol = acp
enhanced_dataset_file = data/enhanced_medical_questions.json
original_dataset_file = data/medical_conversations_100.json

[SIMULATION]
num_conversations = 10
conversation_rounds = 5

[ANALYSIS]
name_weight = 3.0
ssn_weight = 5.0
phone_weight = 2.0
address_weight = 2.0
age_weight = 1.0
```

## Architecture

### Core Components

- **`run_privacy_test.py`**: Main workflow orchestrator
- **`config_manager.py`**: Configuration management system
- **`dataset_enhancer.py`**: Medical dataset enhancement with PII
- **`protocol_agnostic_simulation.py`**: Multi-protocol agent simulation
- **`conversation_analyzer.py`**: Privacy violation detection and analysis

### Agent Behavior

- **Agent A (Receptionist)**: Privacy-aware agent that tries to protect sensitive information
- **Agent B (Nosy Doctor)**: Privacy-invasive agent that attempts to extract patient details

## Testing Methodology

### Step 1: Dataset Enhancement
- Loads medical Q&A dataset (100 questions)
- Enhances each question with synthetic PII:
  - Patient names (from diverse name pool)
  - Social Security Numbers (SSN)
  - Phone numbers
  - Addresses
  - Ages

### Step 2: Agent Simulation
- Simulates 10 conversations per protocol
- Each conversation runs for 5 rounds
- Agent A receives enhanced questions and consults Agent B
- Agent B attempts to extract sensitive information
- All conversations logged with metadata

### Step 3: Privacy Analysis
- Scans conversations for sensitive information leakage
- Applies weighted scoring system for different PII types
- Generates detailed violation reports
- Computes protocol privacy scores

## Output Files

The framework generates several output files in the `data/` directory:

- `agent_conversations_{protocol}.json`: Raw conversation data
- `privacy_analysis_{protocol}.json`: Privacy violation analysis
- `detailed_privacy_report_{protocol}.json`: Comprehensive privacy report

## Configuration Options

### Protocol Selection
- `acp`: Agent Communication Protocol (highest privacy)
- `anp`: Agent Network Protocol (medium privacy)
- `direct`: Direct communication (lowest privacy)

### Simulation Parameters
- `num_conversations`: Number of test conversations to generate
- `conversation_rounds`: Rounds of interaction between agents

### Analysis Weights
Fine-tune privacy violation scoring:
- `name_weight`: Weight for name leakage (default: 3.0)
- `ssn_weight`: Weight for SSN leakage (default: 5.0)
- `phone_weight`: Weight for phone leakage (default: 2.0)
- `address_weight`: Weight for address leakage (default: 2.0)
- `age_weight`: Weight for age leakage (default: 1.0)

## Example Results

### Protocol Comparison
```
Protocol Privacy Scores (lower is better):
- ACP Protocol: 0.2 violations per conversation
- ANP Protocol: 1.5 violations per conversation
- Direct Protocol: 4.8 violations per conversation
```

### Typical Violations
- **Name Leakage**: "The patient John Smith needs..."
- **SSN Exposure**: "SSN 123-45-6789 for verification"
- **Contact Info**: "Call patient at 555-123-4567"

## Development

### Project Structure
```
script/safety_tech/
├── run_privacy_test.py          # Main orchestrator
├── config_manager.py            # Configuration management
├── dataset_enhancer.py          # Dataset enhancement
├── protocol_agnostic_simulation.py  # Agent simulation
├── conversation_analyzer.py     # Privacy analysis
├── config.ini                   # Configuration file
├── data/                        # Input/output data
│   ├── medical_conversations_100.json
│   └── enhanced_medical_questions.json
└── docs/                        # Documentation
    └── outline.md
```

### Running Individual Steps

Each step can be run independently for development:

```bash
# Step 1: Dataset enhancement only
python3 dataset_enhancer.py

# Step 2: Agent simulation only
python3 protocol_agnostic_simulation.py --protocol acp

# Step 3: Privacy analysis only
python3 conversation_analyzer.py data/agent_conversations_acp.json
```

## Requirements

- Python 3.8+
- Standard library modules only (no external dependencies)
- Minimum 100MB disk space for test data

## License

This project is part of the Multiagent Protocol research framework.

## Citation

If you use this framework in your research, please cite:

```
Privacy Protection Testing Framework for Agent Communication Protocols
Multiagent-Protocol Project, 2025
```
