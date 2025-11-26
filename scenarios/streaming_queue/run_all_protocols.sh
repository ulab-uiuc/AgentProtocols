#!/usr/bin/env bash

# Simplified version: run protocols sequentially, each using its own configuration file
set -e

# Change to repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH-}"

# Protocols to run
# PROTOCOLS=(a2a acp agora anp meta)
PROTOCOLS=(anp agora meta)
if [[ $# -gt 0 ]]; then
    PROTOCOLS=("$@")
fi

echo "Starting protocols: ${PROTOCOLS[*]}"
echo "Working directory: ${REPO_ROOT}"
echo

# Cleanup function: clear status line on exit
cleanup() {
    printf "\r\033[K"  # Clear status line
}
trap cleanup EXIT

# Run each protocol
for i in "${!PROTOCOLS[@]}"; do
    protocol="${PROTOCOLS[$i]}"
    num=$((i + 1))
    total=${#PROTOCOLS[@]}
    
    # Show current status on the bottom
    printf "\r\033[K🔄 Running [$num/$total]: %s protocol..." "$protocol" >&2
    
    case "$protocol" in
        a2a)   python3 -m script.streaming_queue.runner.run_a2a ;;
        acp)   python3 -m script.streaming_queue.runner.run_acp ;;
        agora) python3 -m script.streaming_queue.runner.run_anp ;;
        anp)   python3 -m script.streaming_queue.runner.run_agora ;;
        meta)  python3 -m script.streaming_queue.runner.run_meta_network ;;
    *)     printf "\r\033[K❌ Error: unknown protocol %s\n" "$protocol"; exit 1 ;;
    esac
    
    # Clear status line and show completion message
    printf "\r\033[K✅ [$num/$total] %s protocol completed\n" "$protocol"
done

echo "🎉 All protocols finished!"
