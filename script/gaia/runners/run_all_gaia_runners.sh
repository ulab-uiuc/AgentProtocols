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

if [ -z "$OPENAI_API_KEY" ]; then
  echo "⚠️ WARNING: OPENAI_API_KEY is not set. Skipping runners that require it (agora, meta)."
else
  echo "---"
  echo "5. Running Agora Protocol Runner..."
  python -m script.gaia.runners.run_agora

  echo "---"
  echo "6. Running Meta Protocol Runner..."
  python -m script.gaia.runners.run_meta_protocol
fi

echo "---"
echo "✅ All protocol runners have completed."
