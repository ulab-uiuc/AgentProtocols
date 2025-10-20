#!/usr/bin/env python3
"""
Agora Protocol Fail-Storm Recovery Test Runner

This script runs the fail-storm recovery test using Agora protocol.
All configuration is loaded from protocol_backends/agora/config.yaml.
No command line arguments needed.

Usage:
    python run_agora.py
"""

import asyncio
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))  # fail_storm_recovery directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from protocol_backends.agora.runner import AgoraRunner


async def main():
    """Main entry point for Agora fail-storm testing."""
    try:
        print("🎵 Starting Agora Protocol Fail-Storm Recovery Test")
        print("=" * 60)

        # Note: base_runner's parsing logic will look for plain filenames in fail_storm_recovery/configs/ first
        # Passing multi-level relative paths will cause re-concatenation and result in not finding the file. Just pass the filename.
        # Create Agora runner with default config (will use configs/config_agora.yaml)
        runner = AgoraRunner()

        print(f"📋 Configuration loaded from: configs/config_agora.yaml")
        print(f"🔗 Protocol: Agora")
        print(f"👥 Agents: {runner.config['scenario']['agent_count']}")
        print(f"⏱️  Runtime: {runner.config['scenario']['total_runtime']}s")
        print(f"💥 Fault time: {runner.config['scenario']['fault_injection_time']}s")
        print("=" * 60)

        # Run the scenario
        results = await runner.run_scenario()

        print("\n🎵 Agora Fail-Storm test completed successfully!")

        # Get actual result paths from runner
        result_paths = runner.get_results_paths()
        print(f"📊 Results saved to: {result_paths['results_file']}")
        print(f"📈 Detailed metrics: {result_paths['detailed_results_file']}")

        return results

    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Agora test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the Agora fail-storm test
    asyncio.run(main())
