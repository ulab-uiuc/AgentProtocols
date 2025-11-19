"""
Generic protocol runner for model comparison experiments.

This script runs a specific protocol (ACP, A2A, Agora, ANP) with a specified model configuration,
storing results in /workspaces/{model_name}/{protocol}/{task_id}/ directories.

Usage:
    python run_protocol.py --protocol acp --model gpt4o
    python run_protocol.py --protocol a2a --model gemini2.5flash
    python run_protocol.py --protocol agora --model claude3.5
    python run_protocol.py --protocol anp --model gpt4o
"""
import argparse
import asyncio
import sys
from pathlib import Path
import os
import uuid
from datetime import datetime

# Add paths for imports
HERE = Path(__file__).resolve().parent
GAIA_ROOT = HERE.parent
SCENARIOS_ROOT = GAIA_ROOT.parent
sys.path.insert(0, str(GAIA_ROOT))
sys.path.insert(0, str(SCENARIOS_ROOT))

# Import protocol runners
from runners.run_acp import ACPRunner
from runners.run_a2a import A2ARunner
from runners.run_agora import AgoraRunner
from runners.run_anp import ANPRunner


# Protocol to runner class mapping
PROTOCOL_RUNNERS = {
    'acp': ACPRunner,
    'a2a': A2ARunner,
    'agora': AgoraRunner,
    'anp': ANPRunner,
}

# Model name mapping
MODEL_NAMES = {
    'gpt4o': 'gpt-4o',
    'gemini2.5flash': 'gemini-2.5-flash',
    'claude3.5': 'claude-3.5-sonnet',
}


def setup_environment(model: str, protocol: str) -> tuple[str, str]:
    """
    Set up environment variables and paths for the experiment.
    
    Args:
        model: Model identifier (gpt4o, gemini2.5flash, claude3.5)
        protocol: Protocol name (acp, a2a, agora, anp)
    
    Returns:
        Tuple of (config_path, run_id)
    """
    # Generate unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up config path
    config_path = HERE / "configs" / f"{model}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Create workspace directory structure
    model_name = MODEL_NAMES.get(model, model)
    workspace_base = GAIA_ROOT / "workspaces" / model_name / protocol
    workspace_base.mkdir(parents=True, exist_ok=True)
    
    # Set environment variable for workspace base
    os.environ['GAIA_MODEL_COMPARISON_WORKSPACE'] = str(workspace_base)
    os.environ['GAIA_MODEL_NAME'] = model_name
    os.environ['GAIA_PROTOCOL_NAME'] = protocol
    os.environ['GAIA_RUN_ID'] = run_id
    
    print(f"🔧 Model: {model_name}")
    print(f"🔧 Protocol: {protocol}")
    print(f"🔧 Run ID: {run_id}")
    print(f"🔧 Config: {config_path}")
    print(f"🔧 Workspace: {workspace_base}")
    
    return str(config_path), run_id


async def run_experiment(protocol: str, model: str):
    """
    Run the protocol with the specified model configuration.
    
    Args:
        protocol: Protocol name (acp, a2a, agora, anp)
        model: Model identifier (gpt4o, gemini2.5flash, claude3.5)
    """
    if protocol not in PROTOCOL_RUNNERS:
        raise ValueError(f"Unknown protocol: {protocol}. Choose from: {list(PROTOCOL_RUNNERS.keys())}")
    
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model}. Choose from: {list(MODEL_NAMES.keys())}")
    
    # Set up environment and get config path
    config_path, run_id = setup_environment(model, protocol)
    
    # Get the appropriate runner class
    RunnerClass = PROTOCOL_RUNNERS[protocol]
    
    # Create runner instance with model-specific config
    runner = RunnerClass(config_path=config_path)
    
    # Update output file path to include model and run ID
    model_name = MODEL_NAMES[model]
    output_file = GAIA_ROOT / "workspaces" / model_name / protocol / f"results_{run_id}.json"
    runner.output_file = str(output_file)
    
    print(f"📊 Results will be saved to: {output_file}")
    print("=" * 80)
    
    try:
        # Run the experiment
        await runner.run()
        print(f"\n✅ Experiment completed successfully!")
        print(f"📊 Results: {output_file}")
    except KeyboardInterrupt:
        print(f"\n🛑 Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run protocol comparison experiments with different models"
    )
    parser.add_argument(
        '--protocol',
        type=str,
        required=True,
        choices=['acp', 'a2a', 'agora', 'anp'],
        help='Protocol to run (acp, a2a, agora, anp)'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['gpt4o', 'gemini2.5flash', 'claude3.5'],
        help='Model to use (gpt4o, gemini2.5flash, claude3.5)'
    )
    
    args = parser.parse_args()
    
    # Run the experiment
    asyncio.run(run_experiment(args.protocol, args.model))


if __name__ == "__main__":
    main()
