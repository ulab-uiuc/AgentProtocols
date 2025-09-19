#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash script/run_all_failstorm.sh <OPENAI_API_KEY>
# Or set OPENAI_API_KEY in env before running.

if [[ -n "${1:-}" ]]; then
  export OPENAI_API_KEY="$1"
fi
export OPENAI_API_KEY="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"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set.\nUsage: bash script/run_all_failstorm.sh <OPENAI_API_KEY>" >&2
  exit 1
fi

# Move to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Pick python
PYTHON="python3"
command -v python3 >/dev/null 2>&1 || PYTHON="python"

run() {
  local label="$1"; shift
  echo "\n==================== ${label} ====================\n"
  "$PYTHON" -m "$@"
}

run "ACP"   script.fail_storm_recovery.runners.run_acp
run "A2A"   script.fail_storm_recovery.runners.run_a2a
run "ANP"   script.fail_storm_recovery.runners.run_anp
run "AGORA" script.fail_storm_recovery.runners.run_agora
run "META"  script.fail_storm_recovery.runners.run_meta_network

echo "\nAll protocols finished.\n"
