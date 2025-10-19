#!/usr/bin/env python3
"""
Meta Protocol Fail-Storm Recovery Test Runner

This script runs the fail-storm recovery test using Meta Protocol with LLM-based intelligent routing.
All configuration is loaded from configs/config_meta.yaml.
No command line arguments needed.

Usage:
    python run_meta.py
"""

import asyncio
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))  # fail_storm_recovery directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from protocol_backends.meta_protocol.runner import MetaProtocolRunner


async def main():
    """Main entry point for Meta Protocol fail-storm testing."""
    try:
        print("🚀 Starting Meta Protocol Fail-Storm Recovery Test")
        print("🧠 Using LLM-based Intelligent Routing")
        print("📊 Based on actual fail_storm_recovery performance data:")
        print("   • ANP: 61.0% success, 22.0% answer rate (Best accuracy)")
        print("   • Agora: 60.0% success, 20.0% answer rate (Best throughput)")  
        print("   • A2A: 59.6% success, 19.1% answer rate (High volume)")
        print("   • ACP: 59.0% success, 17.9% answer rate (Best recovery)")
        print("=" * 70)
        
        # Create Meta Protocol runner with default config (will use configs/config_meta.yaml)
        runner = MetaProtocolRunner()
        
        print(f"📋 Configuration loaded from: configs/config_meta.yaml")
        print(f"🔗 Protocol: Meta (Multi-Protocol with LLM Routing)")
        print(f"👥 Agents: {runner.config['scenario']['agent_count']}")
        print(f"⏱️  Runtime: {runner.config['scenario']['total_runtime']}s")
        print(f"💥 Fault time: {runner.config['scenario']['fault_injection_time']}s")
        print("=" * 70)
        
        # Run the scenario
        results = await runner.run_scenario()
        
        print("\n🎉 Meta Protocol Fail-Storm test completed successfully!")
        
        # Get actual result paths from runner
        result_paths = runner.get_results_paths()
        print(f"📊 Results saved to: {result_paths['results_file']}")
        print(f"📈 Detailed metrics: {result_paths['detailed_results_file']}")
        print(f"📋 Meta metrics: {result_paths['metrics_file']}")
        
        # Show protocol distribution
        if hasattr(runner, 'protocol_types') and runner.protocol_types:
            print(f"\n🎯 Final Protocol Distribution:")
            protocol_counts = {}
            for agent_id, protocol in runner.protocol_types.items():
                protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
            
            for protocol, count in sorted(protocol_counts.items()):
                print(f"   {protocol.upper()}: {count} agents")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️  Meta Protocol test interrupted by user")
        return None
    except Exception as e:
        print(f"❌ Meta Protocol test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the Meta Protocol fail-storm test
    results = asyncio.run(main())