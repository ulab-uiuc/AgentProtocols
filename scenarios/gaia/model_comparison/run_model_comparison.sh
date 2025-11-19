#!/bin/bash
# Run model comparison experiments using the new dual-config system
# Usage: ./run_model_comparison.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$( cd "$SCRIPT_DIR/../../.." && pwd )"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      GAIA Model Comparison Experiment Runner (v2)         ║${NC}"
echo -e "${BLUE}║  GPT-4o vs Gemini 2.5 Flash vs Claude 3.5 Sonnet          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

# Define models and protocols
MODELS=("gpt4o" "gemini2.5flash" "claude3.5")
MODEL_NAMES=("GPT-4o" "Gemini 2.5 Flash" "Claude 3.5 Sonnet")
PROTOCOLS=("acp" "a2a" "agora" "anp")

# API key environment variables
API_KEYS=("OPENAI_API_KEY" "GOOGLE_API_KEY" "ANTHROPIC_API_KEY")

# Step 1: Sample data
echo -e "\n${GREEN}[1/4] Sampling 10% of data from each difficulty level...${NC}"
cd "$SCRIPT_DIR"
python sample_data.py

# Check if sampled data exists
if [ ! -f "data/sampled_metadata.jsonl" ]; then
    echo -e "${RED}Error: Sampled data not found!${NC}"
    exit 1
fi

SAMPLE_COUNT=$(wc -l < data/sampled_metadata.jsonl)
echo -e "${GREEN}✓ Sampled ${SAMPLE_COUNT} tasks${NC}"

# Step 2-4: Run experiments for each model and protocol
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"
    API_KEY_VAR="${API_KEYS[$i]}"
    
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Running ${MODEL_NAME} experiments${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    
    # Check API key
    if [ -z "${!API_KEY_VAR}" ]; then
        echo -e "${RED}Warning: ${API_KEY_VAR} not set. Skipping ${MODEL_NAME}.${NC}"
        echo "export ${API_KEY_VAR}='your-key-here'"
        continue
    fi
    
    # Run each protocol
    for PROTOCOL in "${PROTOCOLS[@]}"; do
        echo -e "\n${GREEN}Running ${PROTOCOL^^} with ${MODEL_NAME}...${NC}"
        
        cd "$BASE_DIR"
        python -m scenarios.gaia.runners.run_${PROTOCOL} \
            --protocol-config ${PROTOCOL}.yaml \
            --general-config scenarios/gaia/model_comparison/configs/${MODEL}.yaml \
            || echo -e "${RED}${MODEL_NAME} + ${PROTOCOL^^} run failed${NC}"
        
        echo -e "${GREEN}✓ ${MODEL_NAME} + ${PROTOCOL^^} completed${NC}"
    done
done

# Summary
echo -e "\n${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              All Experiment Runs Completed                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo -e "\nResults organized in: /root/AgentProtocols/workspaces/"
echo -e "  Structure: /workspaces/{protocol}/{uuid}/"
echo -e "\nTo analyze results, run:"
echo -e "  cd ${SCRIPT_DIR}"
echo -e "  python analyze_results.py"
