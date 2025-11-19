#!/bin/bash
# Run all three model comparison experiments using command-line arguments
# Usage: ./run_comparison.sh [protocol]
# Example: ./run_comparison.sh acp

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Protocol to test (default: acp)
PROTOCOL="${1:-acp}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      GAIA Model Comparison Experiment Runner              ║${NC}"
echo -e "${BLUE}║  GPT-4o vs Gemini 2.0 Flash vs Claude 3.5 Sonnet          ║${NC}"
echo -e "${BLUE}║  Protocol: ${PROTOCOL}                                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

# Step 1: Sample data
echo -e "\n${GREEN}[1/4] Sampling 10% of data from each difficulty level...${NC}"
cd "$SCRIPT_DIR"
python sample_data.py

# Check if sampled data exists
if [ ! -f "data/sampled_metadata.jsonl" ]; then
    echo -e "${RED}Error: Sampled data not found!${NC}"
    exit 1
fi

# Count sampled tasks
SAMPLE_COUNT=$(wc -l < data/sampled_metadata.jsonl)
echo -e "${GREEN}✓ Sampled ${SAMPLE_COUNT} tasks${NC}"

# Create results directories for each model
mkdir -p results/gpt4o results/gemini results/claude

# Step 2: Run GPT-4o
echo -e "\n${GREEN}[2/4] Running GPT-4o experiment with ${PROTOCOL}...${NC}"
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Warning: OPENAI_API_KEY not set. Skipping GPT-4o.${NC}"
else
    cd "$BASE_DIR"
    python -m scenarios.gaia.runners.run_${PROTOCOL} \
        --config scenarios/gaia/model_comparison/configs/config_gpt4o.yaml \
        --api-type openai \
        --model gpt-4o \
        || echo -e "${RED}GPT-4o run failed${NC}"
    echo -e "${GREEN}✓ GPT-4o experiment completed${NC}"
fi

# Step 3: Run Gemini
echo -e "\n${GREEN}[3/4] Running Gemini 2.0 Flash experiment with ${PROTOCOL}...${NC}"
if [ -z "$GOOGLE_API_KEY" ]; then
    echo -e "${RED}Warning: GOOGLE_API_KEY not set. Skipping Gemini.${NC}"
else
    cd "$BASE_DIR"
    python -m scenarios.gaia.runners.run_${PROTOCOL} \
        --config scenarios/gaia/model_comparison/configs/config_gemini.yaml \
        --api-type google \
        --model gemini-2.0-flash-exp \
        || echo -e "${RED}Gemini run failed${NC}"
    echo -e "${GREEN}✓ Gemini experiment completed${NC}"
fi

# Step 4: Run Claude
echo -e "\n${GREEN}[4/4] Running Claude 3.5 Sonnet experiment with ${PROTOCOL}...${NC}"
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${RED}Warning: ANTHROPIC_API_KEY not set. Skipping Claude.${NC}"
else
    cd "$BASE_DIR"
    python -m scenarios.gaia.runners.run_${PROTOCOL} \
        --config scenarios/gaia/model_comparison/configs/config_claude.yaml \
        --api-type anthropic \
        --model claude-3-5-sonnet-20241022 \
        || echo -e "${RED}Claude run failed${NC}"
    echo -e "${GREEN}✓ Claude experiment completed${NC}"
fi

# Summary
echo -e "\n${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              Experiment Runs Completed                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo -e "\nProtocol: ${YELLOW}${PROTOCOL}${NC}"
echo -e "Results saved in: ${SCRIPT_DIR}/results/"
echo -e "  - results/gpt4o/${PROTOCOL}_results.json"
echo -e "  - results/gemini/${PROTOCOL}_results.json"
echo -e "  - results/claude/${PROTOCOL}_results.json"
echo -e "\nTo analyze results, run:"
echo -e "  cd ${SCRIPT_DIR}"
echo -e "  python analyze_results.py --protocol ${PROTOCOL}"
