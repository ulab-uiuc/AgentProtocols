#!/usr/bin/env python3
"""
A2A Protocol Fail-Storm Recovery Test Runner

This script runs the fail-storm recovery test using A2A protocol.
All configuration is loaded from protocol_backends/a2a/config.yaml.
No command line arguments needed.

Usage:
    python run_a2a.py
"""

import asyncio
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from protocol_backends.a2a.runner import A2ARunner


async def main():
    """Main entry point for A2A fail-storm testing."""
    try:
        print("🚀 Starting A2A Protocol Fail-Storm Recovery Test")
        print("=" * 60)
        
        # Create A2A runner with protocol-specific config
        runner = A2ARunner("protocol_backends/a2a/config.yaml")
        
        print(f"📋 Configuration loaded from: protocol_backends/a2a/config.yaml")
        print(f"🔗 Protocol: A2A")
        print(f"👥 Agents: {runner.config['scenario']['agent_count']}")
        print(f"⏱️  Runtime: {runner.config['scenario']['total_runtime']}s")
        print(f"💥 Fault time: {runner.config['scenario']['fault_injection_time']}s")
        print("=" * 60)
        
        # Run the scenario
        results = await runner.run_scenario()
        
        print("\n🎉 A2A Fail-Storm test completed successfully!")
        print("📊 Results saved to: results/failstorm_metrics.json")
        print("📈 Detailed metrics: results/detailed_failstorm_metrics.json")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ A2A test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the A2A fail-storm test
    asyncio.run(main())

