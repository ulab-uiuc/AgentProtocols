"""
Agora Runner for GAIA Multi-Agent System

This Runner is responsible for:
1. Loading GAIA task and protocol configuration
2. Creating and initializing the Agora network (agent creation is handled internally by the network)
3. Invoking the generic RunnerBase to execute the workflow
4. Unified log redirection (provided by RunnerBase)

Usage:
    python -m script.gaia.runners.run_agora [config_path] [optional_log_file_name]

If config_path is not provided, it will default to script/gaia/config/agora.yaml
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

# Protocol network import (agent instantiation is handled inside the network, not here)
from protocol_backends.agora.network import AgoraNetwork

# Base Runner
from runners.runner_base import RunnerBase


class AgoraRunner(RunnerBase):
    """Agora protocol Runner, implements the create_network hook."""
    def __init__(self, protocol_config_path: str = "agora.yaml", general_config_path: Optional[str] = None) -> None:
        super().__init__(protocol_config_path=protocol_config_path,
                         general_config_path=general_config_path,
                         protocol_name="agora")

    # # Print only key information (other logic is inside the network)
    # print("🔧 Agora Runner initialized")
    # if self.config:
    #     print(f"📦 Agora protocol config keys: {list(self.config.keys())}")

    def create_network(self, general_config: Dict[str, Any]) -> AgoraNetwork:
        """
        Create and return an Agora network instance.
        Note: Agent instantiation/registration has been moved inside AgoraNetwork and is not handled here.
        """
        try:
            print("ℹ️  Initializing NetworkBase and Agora Agents...")
            network = AgoraNetwork(config=general_config)
            print("🌐 Agora Network created (agents initialized internally)")
            return network
        except Exception as e:
            print(f"❌ Failed to create Agora Network: {e}")
            raise


async def main():
    """Agora Runner entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GAIA benchmark with Agora protocol")
    parser.add_argument("--protocol-config", type=str, default="agora.yaml",
                        help="Path to protocol config file (default: agora.yaml)")
    parser.add_argument("--general-config", type=str, default=None,
                        help="Path to general config file (default: scenarios/gaia/config/general.yaml)")
    
    args = parser.parse_args()
    
    runner = AgoraRunner(protocol_config_path=args.protocol_config,
                         general_config_path=args.general_config)
    
    try:        
        await runner.run()
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())