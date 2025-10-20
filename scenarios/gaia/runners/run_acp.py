"""
ACP Runner for GAIA Multi-Agent System

This runner is responsible for:
1. Reading GAIA tasks and protocol configuration
2. Creating and initializing the ACP network (agent creation and service startup are completed inside ACPNetwork)
3. Calling the common RunnerBase workflow
4. Unified logging redirection (provided by RunnerBase)

Usage:
    python -m script.gaia.runners.run_acp [config_path]

If config_path is not provided, it will try to use script/gaia/config/acp.yaml
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, Any
import sys

# PathSetup
HERE = Path(__file__).resolve().parent
GAIA_ROOT = HERE.parent
sys.path.insert(0, str(GAIA_ROOT))

# Protocol network import (agent creation and service startup are handled internally by the network)
from protocol_backends.acp.network import ACPNetwork

# Base runner (provides common orchestration and logging redirection)
from .runner_base import RunnerBase


class ACPRunner(RunnerBase):
    """ACP protocol runner implementing the create_network hook."""
    def __init__(self, config_path: str = "acp.yaml") -> None:
        super().__init__(config_path, protocol_name="acp")

    def create_network(self, general_config: Dict[str, Any]) -> ACPNetwork:
        """Create and return an ACP network instance."""
        try:
            print("ℹ️  Initializing NetworkBase and ACP agents...")
            network = ACPNetwork(config=general_config)
            print("🌐 ACP network created (agent initialization and ACP server startup handled internally)")
            return network
        except Exception as e:
            print(f"❌ Failed to create ACP network: {e}")
            raise


async def main():
    """Entry point for ACP runner."""
    runner = ACPRunner()
    try:
        await runner.run()
    except KeyboardInterrupt:
        print("\n🛑 ACP run interrupted by user")
    except Exception as e:
        print(f"❌ ACP runner failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
