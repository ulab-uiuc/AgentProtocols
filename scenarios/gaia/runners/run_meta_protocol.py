"""
Meta Protocol Runner for GAIA Framework.
Implements intelligent protocol selection similar to fail_storm_recovery.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# PathSetup
HERE = Path(__file__).resolve().parent
GAIA_ROOT = HERE.parent
sys.path.insert(0, str(GAIA_ROOT))

# Protocol network import
from protocol_backends.meta_protocol.network import MetaProtocolNetwork

# Base runner
from runners.runner_base import RunnerBase


class MetaProtocolRunner(RunnerBase):
    """Meta Protocol runner implementing the create_network hook."""
    
    def __init__(self, protocol_config_path: str = "meta_protocol.yaml", general_config_path: str = None) -> None:
        super().__init__(protocol_config_path, general_config_path, protocol_name="meta_protocol")

    def create_network(self, general_config: Dict[str, Any]) -> MetaProtocolNetwork:
        """
        Create and return a Meta Protocol network instance.
        Integrates intelligent protocol selection and cross-protocol communication.
        """
        try:
            print("ℹ️  Initializing Meta Protocol Network with intelligent routing...")
            
            # Extract task_id from general_config
            task_id = general_config.get("task_id", "meta_protocol_task")
            
            # Create meta protocol network
            network = MetaProtocolNetwork(general_config, task_id)
            
            # Register agents from config (LLM routing will happen during network.start())
            network.register_agents_from_config()
            
            print(f"✅ Meta Protocol network created for task: {task_id}")
            print(f"📊 Available protocols: {network._available_protocols}")
            
            return network
            
        except Exception as e:
            print(f"❌ Failed to create Meta Protocol network: {e}")
            raise


# Main execution
async def main():
    """Main execution for Meta Protocol runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GAIA Meta Protocol Runner")
    parser.add_argument(
        "--protocol-config",
        type=str,
        default="meta_protocol.yaml",
        help="Path to protocol-specific configuration file"
    )
    parser.add_argument(
        "--general-config",
        type=str,
        default=None,
        help="Path to general configuration file (model, runtime, etc.)"
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Starting GAIA Meta Protocol Runner")
    print(f"📋 Protocol Config: {args.protocol_config}")
    print(f"📋 General Config: {args.general_config or 'default'}")
    print(f"📊 Meta Protocol: Intelligent routing enabled")
    
    # Create and run
    runner = MetaProtocolRunner(
        protocol_config_path=args.protocol_config,
        general_config_path=args.general_config
    )
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())