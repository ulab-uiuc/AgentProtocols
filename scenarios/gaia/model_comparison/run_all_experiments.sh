#!/bin/bash
# Run all model comparison experiments
# This script runs each protocol (ACP, A2A, Agora, ANP) with each model (GPT-4o, Gemini 2.5 Flash, Claude 3.5)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Models to test
MODELS=("gpt4o" "gemini2.5flash" "claude3.5")

# Protocols to test
PROTOCOLS=("acp" "a2a" "agora" "anp")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  GAIA Model Comparison Experiments"
echo "=========================================="
echo ""
echo "Models: ${MODELS[@]}"
echo "Protocols: ${PROTOCOLS[@]}"
echo ""
echo "Total experiments: $((${#MODELS[@]} * ${#PROTOCOLS[@]}))"
echo "=========================================="
echo ""

# Check if API keys are set
check_api_keys() {
    local missing_keys=0
    
    if [ -z "$OPENAI_API_KEY" ]; then
        echo -e "${YELLOW}Warning: OPENAI_API_KEY not set${NC}"
        missing_keys=1
    fi
    
    if [ -z "$GOOGLE_API_KEY" ]; then
        echo -e "${YELLOW}Warning: GOOGLE_API_KEY not set${NC}"
        missing_keys=1
    fi
    
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo -e "${YELLOW}Warning: ANTHROPIC_API_KEY not set${NC}"
        missing_keys=1
    fi
    
    if [ $missing_keys -eq 1 ]; then
        echo ""
        echo -e "${YELLOW}Some API keys are missing. Please set them:${NC}"
        echo "  export OPENAI_API_KEY='your-key'"
        echo "  export GOOGLE_API_KEY='your-key'"
        echo "  export ANTHROPIC_API_KEY='your-key'"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

check_api_keys

# Counter for progress
total_experiments=$((${#MODELS[@]} * ${#PROTOCOLS[@]}))
current_experiment=0
failed_experiments=0

# Log file
log_file="experiments_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $log_file"
echo ""

# Run experiments
for model in "${MODELS[@]}"; do
    for protocol in "${PROTOCOLS[@]}"; do
        current_experiment=$((current_experiment + 1))
        
        echo "=========================================="
        echo -e "${GREEN}[$current_experiment/$total_experiments] Running: $model + $protocol${NC}"
        echo "=========================================="
        
        # Run the experiment
        if python run_protocol.py --protocol "$protocol" --model "$model" 2>&1 | tee -a "$log_file"; then
            echo -e "${GREEN}✅ Success: $model + $protocol${NC}"
        else
            echo -e "${RED}❌ Failed: $model + $protocol${NC}"
            failed_experiments=$((failed_experiments + 1))
        fi
        
        echo ""
        echo "Progress: $current_experiment/$total_experiments"
        echo ""
        
        # Brief pause between experiments
        sleep 2
    done
done

echo "=========================================="
echo "  All Experiments Complete"
echo "=========================================="
echo "Total: $total_experiments"
echo -e "${GREEN}Successful: $((total_experiments - failed_experiments))${NC}"
if [ $failed_experiments -gt 0 ]; then
    echo -e "${RED}Failed: $failed_experiments${NC}"
fi
echo "Log file: $log_file"
echo "=========================================="

exit $failed_experiments
