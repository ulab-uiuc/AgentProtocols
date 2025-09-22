#!/bin/bash
cd ~/Multiagent-Protocol

# This script runs all GAIA protocol runners serially from the project root.

echo "🚀 Starting all GAIA protocol runners..."

# --- Runners that don't require an API key ---

echo "---"
echo "1. Running A2A Protocol Runner..."
python -m script.gaia.runners.run_a2a

echo "---"
echo "2. Running ACP Protocol Runner..."
python -m script.gaia.runners.run_acp

echo "---"
echo "3. Running ANP Protocol Runner..."
python -m script.gaia.runners.run_anp


# --- Runners that require an API key ---
export OPENAI_API_KEY="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"
if [ -z "$OPENAI_API_KEY" ]; then
  echo "⚠️ WARNING: OPENAI_API_KEY is not set. Skipping runners that require it (agora, meta)."
else
  echo "---"
  echo "5. Running Agora Protocol Runner..."
  python -m script.gaia.runners.run_agora

  # echo "---"
  # echo "6. Running Meta Protocol Runner..."
  # python -m script.gaia.runners.run_meta_protocol
fi

echo "---"
echo "✅ All protocol runners have completed."
