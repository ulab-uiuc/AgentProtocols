#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Meta Privacy Testing Runner
Runs privacy testing for all meta protocols and compares results.
"""

import asyncio
import json
import time
from pathlib import Path
import sys

# Add current directory to path for imports
HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent
sys.path.insert(0, str(SAFETY_TECH))

from runners.run_meta import UnifiedMetaRunner


class BatchMetaRunner:
    """Batch runner for all meta protocols"""

    def __init__(self):
        self.protocols = ["acp", "anp", "agora", "a2a"]
        self.results = {}
        self.start_time = time.time()

    async def run_all_protocols(self):
        """Run privacy testing for all protocols"""
        print("🚀 Starting Batch Meta Privacy Testing for All Protocols")
        print("=" * 70)
        
        for i, protocol in enumerate(self.protocols, 1):
            print(f"\n[{i}/{len(self.protocols)}] Testing META-{protocol.upper()} Protocol")
            print("-" * 50)
            
            try:
                protocol_start = time.time()
                
                # Create and run protocol-specific runner
                runner = UnifiedMetaRunner(protocol)
                results = await runner.run()
                
                protocol_end = time.time()
                protocol_time = protocol_end - protocol_start
                
                # Store results
                self.results[protocol] = {
                    "success": True,
                    "results": results,
                    "execution_time": protocol_time,
                    "timestamp": time.time()
                }
                
                print(f"✅ META-{protocol.upper()} completed in {protocol_time:.2f}s")
                
            except Exception as e:
                protocol_end = time.time()
                protocol_time = protocol_end - protocol_start
                
                self.results[protocol] = {
                    "success": False,
                    "error": str(e),
                    "execution_time": protocol_time,
                    "timestamp": time.time()
                }
                
                print(f"❌ META-{protocol.upper()} failed: {e}")
                
        # Generate comparison report
        await self.generate_comparison_report()

    async def generate_comparison_report(self):
        """Generate comparison report for all protocols"""
        print("\n" + "=" * 70)
        print("📊 BATCH META PROTOCOL TESTING RESULTS")
        print("=" * 70)
        
        total_time = time.time() - self.start_time
        successful_protocols = [p for p, r in self.results.items() if r["success"]]
        failed_protocols = [p for p, r in self.results.items() if not r["success"]]
        
        # Summary statistics
        print(f"Total Execution Time: {total_time:.2f} seconds")
        print(f"Protocols Tested: {len(self.protocols)}")
        print(f"Successful: {len(successful_protocols)}")
        print(f"Failed: {len(failed_protocols)}")
        
        if failed_protocols:
            print(f"Failed Protocols: {', '.join(failed_protocols)}")
        
        print("\n" + "-" * 70)
        print("PROTOCOL PERFORMANCE COMPARISON")
        print("-" * 70)
        
        # Performance comparison table
        print(f"{'Protocol':<10} {'Status':<10} {'Time (s)':<10} {'Privacy Score':<15} {'Violations':<12}")
        print("-" * 70)
        
        for protocol in self.protocols:
            result = self.results[protocol]
            
            if result["success"]:
                # Extract privacy metrics
                privacy_results = result.get("results", {})
                summary = privacy_results.get("summary", {})
                privacy_score = summary.get("average_privacy_score", 0)
                total_violations = summary.get("total_violation_instances", 0)
                
                print(f"{protocol.upper():<10} {'SUCCESS':<10} {result['execution_time']:<10.2f} {privacy_score:<15.2f} {total_violations:<12}")
            else:
                print(f"{protocol.upper():<10} {'FAILED':<10} {result['execution_time']:<10.2f} {'N/A':<15} {'N/A':<12}")
        
        # Save detailed results
        await self.save_batch_results()
        
        print("\n" + "=" * 70)
        print("✅ Batch Meta Protocol Testing Completed!")
        print("=" * 70)

    async def save_batch_results(self):
        """Save batch testing results"""
        try:
            # Create output directory
            output_dir = SAFETY_TECH / "output"
            output_dir.mkdir(exist_ok=True)
            
            # Save comprehensive results
            batch_results = {
                "batch_info": {
                    "timestamp": time.time(),
                    "total_execution_time": time.time() - self.start_time,
                    "protocols_tested": self.protocols,
                    "successful_protocols": [p for p, r in self.results.items() if r["success"]],
                    "failed_protocols": [p for p, r in self.results.items() if not r["success"]]
                },
                "protocol_results": self.results
            }
            
            # Save to JSON file
            batch_file = output_dir / "batch_meta_protocol_results.json"
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False)
            
            print(f"📁 Batch results saved: {batch_file}")
            
        except Exception as e:
            print(f"⚠️  Failed to save batch results: {e}")


async def main():
    """Main entry point for batch meta testing"""
    try:
        batch_runner = BatchMetaRunner()
        await batch_runner.run_all_protocols()
    except KeyboardInterrupt:
        print("\n🛑 Batch meta testing interrupted by user")
    except Exception as e:
        print(f"❌ Batch meta testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
