#!/usr/bin/env python3
"""
ACP Protocol Fail-Storm Recovery Test Runner

This script runs the fail-storm recovery test using ACP protocol.
All configuration is loaded from protocol_backends/acp/config.yaml.
No command line arguments needed.

Usage:
    python run_acp.py
"""

import asyncio
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))  # fail_storm_recovery directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from protocol_backends.acp.runner import ACPRunner


async def main():
    """Main entry point for ACP fail-storm testing."""
    try:
        print("Starting ACP Protocol Fail-Storm Recovery Test")
        print("=" * 60)
        
        # Create ACP runner with protocol-specific config
        runner = ACPRunner()
        
        print(f"📋 Configuration loaded from: configs/config_acp.yaml")
        print(f"🔗 Protocol: ACP")
        print(f"👥 Agents: {runner.config['scenario']['agent_count']}")
        print(f"⏱️  Runtime: {runner.config['scenario']['total_runtime']}s")
        print(f"💥 Fault time: {runner.config['scenario']['fault_injection_time']}s")
        print("=" * 60)
        
        # Run the scenario
        results = await runner.run_scenario()
        
        print("\nACP Fail-Storm test completed successfully!")
        
        # Get actual result paths from runner
        result_paths = runner.get_results_paths()
        print(f"Results saved to: {result_paths['results_file']}")
        print(f"Detailed metrics: {result_paths['detailed_results_file']}")
        
        return results
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return None
    except Exception as e:
        print(f"\nACP test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the ACP fail-storm test
    asyncio.run(main())

