# GAIA Model Comparison Experiments

This directory contains scripts and configurations for running comparative experiments across different LLMs and protocols in the GAIA benchmark.

## Overview

The experiments compare three state-of-the-art language models:
- **GPT-4o** (OpenAI)
- **Gemini 2.5 Flash** (Google)
- **Claude 3.5 Sonnet** (Anthropic)

Across four agent communication protocols:
- **ACP** (Agent Communication Protocol)
- **A2A** (Agent-to-Agent)
- **Agora** (Multi-agent coordination)
- **ANP** (Agent Network Protocol)

## Directory Structure

```
model_comparison/
├── configs/                    # Model-specific configurations
│   ├── gpt4o.yaml
│   ├── gemini2.5flash.yaml
│   └── claude3.5.yaml
├── data/                       # Sampled dataset
│   └── sampled_metadata.jsonl # 10% sample from each difficulty level
├── results/                    # Experiment results (auto-created)
│   ├── gpt-4o/
│   │   ├── acp/
│   │   ├── a2a/
│   │   ├── agora/
│   │   └── anp/
│   ├── gemini-2.5-flash/
│   └── claude-3.5-sonnet/
├── llm_patches.py              # LLM API compatibility layer
├── run_protocol.py             # Single protocol runner
├── run_all_experiments.sh      # Run all experiments
├── sample_data.py              # Data sampling script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `openai` - For GPT-4o
- `anthropic` - For Claude 3.5
- `google-generativeai` - For Gemini 2.5 Flash

### 2. Set API Keys

Set environment variables for the models you want to test:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

### 3. Sample Data

Generate the 10% sample from each difficulty level:

```bash
python sample_data.py
```

This creates `data/sampled_metadata.jsonl` with a balanced sample.

## Usage

### Run Single Experiment

Run a specific protocol with a specific model:

```bash
# ACP with GPT-4o
python run_protocol.py --protocol acp --model gpt4o

# A2A with Claude 3.5
python run_protocol.py --protocol a2a --model claude3.5

# Agora with Gemini 2.5 Flash
python run_protocol.py --protocol agora --model gemini2.5flash

# ANP with GPT-4o
python run_protocol.py --protocol anp --model gpt4o
```

### Run All Experiments

Run all combinations of models and protocols:

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

This will run 12 experiments (3 models × 4 protocols).

## Results

Results are saved in the following structure:

```
workspaces/
└── {model_name}/           # e.g., gpt-4o
    └── {protocol}/         # e.g., acp
        └── results_{timestamp}.json
```

Each result file contains:
- Task-level results with predictions and ground truth
- Execution metrics (time, token usage)
- Quality scores from LLM judge
- Agent assignments and tool usage statistics

## LLM Compatibility Layer

The `llm_patches.py` module provides adapters for different LLM APIs:

- **AnthropicLLMAdapter**: Converts OpenAI-style calls to Anthropic Claude API
- **GoogleLLMAdapter**: Converts OpenAI-style calls to Google Gemini API
- **get_llm_instance()**: Factory function that auto-selects the right adapter

### Usage Example

```python
from llm_patches import get_llm_instance

# Automatically detects API type from config
llm = get_llm_instance(config_path="configs/claude3.5.yaml")

# Use with OpenAI-style interface
response = await llm.ask(messages=[
    {"role": "user", "content": "Hello!"}
])
```

## Configuration

Each model has its own YAML configuration:

### GPT-4o (`configs/gpt4o.yaml`)
```yaml
model:
  name: "gpt-4o"
  base_url: "https://api.openai.com/v1"
  api_key: ""  # Uses OPENAI_API_KEY env var
```

### Claude 3.5 (`configs/claude3.5.yaml`)
```yaml
model:
  name: "claude-3-5-sonnet-20241022"
  base_url: "https://api.anthropic.com/v1"
  api_key: ""  # Uses ANTHROPIC_API_KEY env var
  api_type: "anthropic"
```

### Gemini 2.5 Flash (`configs/gemini2.5flash.yaml`)
```yaml
model:
  name: "gemini-2.0-flash-exp"
  base_url: "https://generativelanguage.googleapis.com/v1beta"
  api_key: ""  # Uses GOOGLE_API_KEY env var
  api_type: "google"
```

## Analysis

After running experiments, you can analyze results:

```python
import json
from pathlib import Path

# Load results
results_dir = Path("../../workspaces/gpt-4o/acp")
result_files = list(results_dir.glob("results_*.json"))

for result_file in result_files:
    with open(result_file) as f:
        data = json.load(f)
    
    metadata = data["metadata"]
    print(f"Model: GPT-4o, Protocol: ACP")
    print(f"Success rate: {metadata['success_rate']:.1f}%")
    print(f"Avg quality: {metadata['avg_quality_score']:.2f}")
    print(f"Total time: {metadata['total_execution_time']:.1f}s")
```

## Troubleshooting

### API Key Issues

If you see "API key not configured" errors:
1. Check that environment variables are set: `echo $OPENAI_API_KEY`
2. Ensure the config files use empty strings for api_key (to use env vars)

### Import Errors

If you see module import errors:
```bash
# Install missing dependencies
pip install anthropic google-generativeai

# Or install all at once
pip install -r requirements.txt
```

### Rate Limits

If you hit rate limits:
1. Add delays between experiments in `run_all_experiments.sh`
2. Run protocols sequentially rather than all at once
3. Check your API quota limits

## Notes

- Each task runs with a 300-second timeout
- Results include detailed metrics: execution time, token usage, quality scores
- The LLM judge evaluates answer quality on a 1-5 scale
- Workspace directories are automatically created for each task
- Task files are copied to workspace directories for isolation
- Results are stored in `/workspaces/{model_name}/{protocol}/` structure

## Citation

If you use this framework in your research, please cite the GAIA benchmark and the AgentProtocols framework.

