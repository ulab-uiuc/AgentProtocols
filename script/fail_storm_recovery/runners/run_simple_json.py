#!/usr/bin/env python3
"""
Simple JSON Protocol Fail-Storm Recovery Test Runner

This script runs the fail-storm recovery test using Simple JSON protocol.
All configuration is loaded from configs/config_simple_json.yaml.
No command line arguments needed.

Usage:
    python run_simple_json.py
"""

import asyncio
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))  # fail_storm_recovery directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from protocol_backends.simple_json.runner import SimpleJsonRunner


async def main():
    """Main entry point for Simple JSON fail-storm testing."""
    try:
        print("🚀 Starting Simple JSON Protocol Fail-Storm Recovery Test")
        print("=" * 60)
        
        # Create Simple JSON runner with default config (will use configs/config_simple_json.yaml)
        runner = SimpleJsonRunner()
        
        print(f"📋 Configuration loaded from: configs/config_simple_json.yaml")
        print(f"🔗 Protocol: Simple JSON")
        print(f"👥 Agents: {runner.config['scenario']['agent_count']}")
        print(f"⏱️  Runtime: {runner.config['scenario']['total_runtime']}s")
        print(f"💥 Fault time: {runner.config['scenario']['fault_injection_time']}s")
        print("=" * 60)
        
        # Run the scenario
        results = await runner.run_scenario()
        
        print("\n🎉 Simple JSON Fail-Storm test completed successfully!")
        
        # Get actual result paths from runner
        result_paths = runner.get_results_paths()
        print(f"📊 Results saved to: {result_paths['results_file']}")
        print(f"📈 Detailed metrics: {result_paths['detailed_results_file']}")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Simple JSON test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the Simple JSON fail-storm test
    asyncio.run(main())
