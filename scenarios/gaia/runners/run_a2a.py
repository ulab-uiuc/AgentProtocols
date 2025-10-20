"""
A2A Runner for GAIA Multi-Agent System

This runner is responsible for:
1. Reading GAIA task and protocol configuration
2. Creating and initializing the A2A network (agent creation is completed inside the network)
3. Calling the common RunnerBase workflow
4. Unified logging redirection (provided by RunnerBase)

Usage:
    python -m script.gaia.runners.run_a2a [config_path] [optional_log_file_name]

If config_path is not provided, it will try to use script/gaia/config/a2a.yaml
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# PathSetup
HERE = Path(__file__).resolve().parent
GAIA_ROOT = HERE.parent
sys.path.insert(0, str(GAIA_ROOT))

# Protocol network import (agent creation is handled internally by the network, not created in the runner)
from protocol_backends.a2a.network import A2ANetwork

# Base runner
from runners.runner_base import RunnerBase


class A2ARunner(RunnerBase):
    """A2A protocol runner implementing the create_network hook."""
    def __init__(self, config_path: str = "a2a.yaml") -> None:
        super().__init__(config_path, protocol_name="a2a")

    # # Only print key info (other logic lives inside the network)
    # print("🔧 A2A runner initialized")
    # if self.config:
    #     print(f"📦 A2A config keys: {list(self.config.keys())}")

    def create_network(self, general_config: Dict[str, Any]) -> A2ANetwork:
        """
        Create and return an A2A network instance.
        Note: Agent instantiation/registration has been moved inside A2ANetwork and is not handled here.
        """
        try:
            # Merge runner config (which has model info) into general_config
            if hasattr(self, 'config') and 'model' in self.config:
                general_config['model'] = self.config['model']
                print(f"🔧 Added model config to general_config: {self.config['model']['name']}")
            
            network = A2ANetwork(config=general_config)
            print("🌐 A2A network created (agent initialization handled internally)")
            return network
        except Exception as e:
            print(f"❌ Failed to create A2A network: {e}")
            raise


async def main():
    """Entry point for A2A runner."""
    runner = A2ARunner()

    try:        
        runner = A2ARunner()
        await runner.run()
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"❌ A2A runner failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())