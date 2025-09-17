#!/usr/bin/env python3
"""
Meta-Protocol Fail-Storm Recovery Test Runner

This script runs the fail-storm recovery test using Meta-Protocol with LLM-based
intelligent routing. All configuration is loaded from configs/config_meta.yaml.
No command line arguments needed.

The Meta-Protocol runner uses LLM to intelligently select the best protocols
for each agent based on task requirements and fault scenarios.

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
    """Main entry point for Meta-Protocol fail-storm testing."""
    try:
        print("Starting Meta-Protocol Fail-Storm Recovery Test")
        print("=" * 60)
        
        # Create Meta-Protocol runner with intelligent routing
        runner = MetaProtocolRunner()
        
        print(f"📋 Configuration loaded from: {runner.get_config_path()}")
        print(f"🧠 Protocol: Meta-Protocol with LLM Intelligent Routing")
        print(f"👥 Agents: {runner.config['scenario']['agent_count']}")
        print(f"⏱️  Runtime: {runner.config['scenario']['total_runtime']}s")
        print(f"💥 Fault mode: {runner.config['scenario'].get('cyclic_faults', False) and 'Cyclic' or 'Single'}")
        print(f"🤖 LLM Model: {runner.config.get('llm', {}).get('model', 'gpt-4o')}")
        print("=" * 60)
        
        # Run the scenario
        results = await runner.run_scenario()
        
        print("\nMeta-Protocol Fail-Storm test completed successfully!")
        
        # Get actual result paths from runner
        result_paths = runner.get_results_paths()
        print(f"Results saved to: {result_paths['results_file']}")
        print(f"Detailed metrics: {result_paths['detailed_results_file']}")
        
        # Display protocol distribution if available
        meta_info = results.get('meta_protocol_specific', {})
        if meta_info:
            protocol_assignments = meta_info.get('protocol_assignments', {})
            if protocol_assignments:
                print("\n🧠 LLM Protocol Assignments:")
                protocol_counts = {}
                for agent_id, protocol in protocol_assignments.items():
                    protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
                    print(f"  {agent_id}: {protocol.upper()}")
                
                print("\n📊 Protocol Distribution:")
                for protocol, count in protocol_counts.items():
                    print(f"  {protocol.upper()}: {count} agents")
        
        # Display LLM routing statistics
        routing_analysis = results.get('llm_routing_analysis', {})
        if routing_analysis:
            avg_confidence = routing_analysis.get('average_confidence', 0)
            total_decisions = routing_analysis.get('total_decisions', 0)
            print(f"\n🎯 LLM Routing Performance:")
            print(f"  Total decisions: {total_decisions}")
            print(f"  Average confidence: {avg_confidence:.2%}")
        
        return results
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return None
    except Exception as e:
        print(f"\nMeta-Protocol test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the Meta-Protocol fail-storm test
    asyncio.run(main())