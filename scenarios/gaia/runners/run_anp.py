"""
ANP Runner for GAIA Multi-Agent System

This runner is responsible for:
1. Reading GAIA tasks and protocol configuration
2. Creating and initializing the ANP network (agent creation and service startup are completed inside ANPNetwork)
3. Calling the common RunnerBase workflow
4. Unified logging redirection (provided by RunnerBase)

Usage:
    python -m script.gaia.runners.run_anp [config_path]

If config_path is not provided, it will try to use script/gaia/config/anp.yaml
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
from protocol_backends.anp.network import ANPNetwork

# Base runner (provides common orchestration and logging redirection)
from .runner_base import RunnerBase


class ANPRunner(RunnerBase):
    """ANP protocol runner implementing the create_network hook."""
    def __init__(self, config_path: str = "anp.yaml") -> None:
        super().__init__(config_path, protocol_name="anp")

    def create_network(self, general_config: Dict[str, Any]) -> ANPNetwork:
        """Create and return an ANP network instance."""
        try:
            print("ℹ️  Initializing NetworkBase and ANP agents...")
            
            # Merge runner config (which has model info) into general_config
            if hasattr(self, 'config') and 'model' in self.config:
                general_config['model'] = self.config['model']
                print(f"🔧 Added model config to general_config: {self.config['model']['name']}")
            
            # Merge network configuration from runner config into planned config
            network_config = self.config.get("network", {})
            if network_config:
                general_config['network'] = network_config
                print(f"🔗 Added network config: timeout={network_config.get('timeout_seconds', 'default')}")
            
            network = ANPNetwork(config=general_config)
            print("🌐 ANP network created (agent initialization and ANP server startup handled internally)")
            return network
        except Exception as e:
            print(f"❌ Failed to create ANP network: {e}")
            raise


async def main():
    """Entry point for ANP runner."""
    runner = ANPRunner()
    try:
        await runner.run()
    except KeyboardInterrupt:
        print("\n🛑 ANP run interrupted by user")
    except Exception as e:
        print(f"❌ ANP runner failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
