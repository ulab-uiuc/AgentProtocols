#!/usr/bin/env python3
"""
Protocol Performance Comparison Script

This script runs all available protocols and compares their fail-storm recovery performance.
"""

import asyncio
import json
import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any

def run_protocol_test(protocol: str) -> Dict[str, Any]:
    """Run fail-storm test for a specific protocol."""
    print(f"\n🚀 Testing {protocol.upper()} protocol...")
    
    # Map protocol to runner script
    runner_map = {
        'anp': 'run_anp.py',
        'a2a': 'run_a2a.py', 
        'agora': 'run_agora.py'
    }
    
    if protocol not in runner_map:
        print(f"❌ Unknown protocol: {protocol}")
        return None
    
    runner_script = f"runners\\{runner_map[protocol]}"
    
    try:
        # Run the test
        result = subprocess.run(
            [sys.executable, runner_script],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {protocol.upper()} test completed successfully")
            
            # Load results
            results_file = Path(__file__).parent / "results" / "failstorm_metrics.json"
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                
                # Add protocol identifier
                metrics['protocol_name'] = protocol
                return metrics
            else:
                print(f"⚠️  Results file not found for {protocol}")
                return None
        else:
            print(f"❌ {protocol.upper()} test failed:")
            print(f"STDOUT: {result.stdout[-500:]}")  # Last 500 chars
            print(f"STDERR: {result.stderr[-500:]}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {protocol.upper()} test timed out")
        return None
    except Exception as e:
        print(f"❌ Error running {protocol} test: {e}")
        return None

def extract_key_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key performance metrics from test results."""
    if not metrics:
        return {}
    
    protocol = metrics.get('protocol_name', 'unknown')
    
    # Basic timing
    timing = metrics.get('timing', {})
    failstorm = metrics.get('failstorm_metrics', {})
    qa_metrics = metrics.get('qa_metrics', {})
    
    return {
        'protocol': protocol,
        'total_runtime': timing.get('total_runtime', 0),
        'recovery_time_ms': failstorm.get('recovery_ms', 0),
        'steady_state_time_ms': failstorm.get('steady_state_ms', 0),
        'success_rate': qa_metrics.get('success_rate', 0),
        'answer_found_rate': qa_metrics.get('answer_found_rate', 0),
        'total_qa_tasks': qa_metrics.get('total_qa_tasks', 0),
        'avg_task_duration': qa_metrics.get('average_task_duration', 0),
        'pre_fault_answer_rate': qa_metrics.get('pre_fault_answer_rate', 0),
        'post_fault_answer_rate': qa_metrics.get('post_fault_answer_rate', 0),
        'answer_sources': qa_metrics.get('answer_sources', {}),
        'agents_killed': metrics.get('agent_summary', {}).get('temporarily_killed_count', 0),
        'agents_reconnected': metrics.get('agent_summary', {}).get('reconnected_count', 0)
    }

def compare_protocols(results: List[Dict[str, Any]]):
    """Compare protocol performance metrics."""
    print("\n📊 Protocol Performance Comparison")
    print("=" * 80)
    
    # Extract key metrics for each protocol
    comparison_data = []
    for result in results:
        if result:
            metrics = extract_key_metrics(result)
            comparison_data.append(metrics)
    
    if not comparison_data:
        print("❌ No valid test results to compare")
        return
    
    # Print comparison table
    headers = ['Protocol', 'Recovery(ms)', 'Steady(ms)', 'Success%', 'Answer%', 'Tasks', 'Avg Duration(s)']
    print(f"{'Protocol':<12} {'Recovery(ms)':<12} {'Steady(ms)':<12} {'Success%':<10} {'Answer%':<10} {'Tasks':<8} {'Duration(s)':<12}")
    print("-" * 80)
    
    for metrics in comparison_data:
        protocol = metrics['protocol'].upper()
        recovery_ms = f"{metrics['recovery_time_ms']/1000:.1f}s" if metrics['recovery_time_ms'] else "N/A"
        steady_ms = f"{metrics['steady_state_time_ms']/1000:.1f}s" if metrics['steady_state_time_ms'] else "N/A"
        success_rate = f"{metrics['success_rate']*100:.1f}%" if metrics['success_rate'] else "N/A"
        answer_rate = f"{metrics['answer_found_rate']*100:.1f}%" if metrics['answer_found_rate'] else "N/A"
        total_tasks = str(metrics['total_qa_tasks'])
        avg_duration = f"{metrics['avg_task_duration']:.2f}s" if metrics['avg_task_duration'] else "N/A"
        
        print(f"{protocol:<12} {recovery_ms:<12} {steady_ms:<12} {success_rate:<10} {answer_rate:<10} {total_tasks:<8} {avg_duration:<12}")
    
    # Detailed comparison
    print(f"\n📈 Detailed Analysis:")
    
    for metrics in comparison_data:
        protocol = metrics['protocol'].upper()
        print(f"\n🔗 {protocol} Protocol:")
        print(f"  ⏱️  Recovery time: {metrics['recovery_time_ms']/1000:.1f}s")
        print(f"  🎯 Steady state: {metrics['steady_state_time_ms']/1000:.1f}s")
        print(f"  ✅ Success rate: {metrics['success_rate']*100:.1f}%")
        print(f"  💡 Answer found rate: {metrics['answer_found_rate']*100:.1f}%")
        print(f"  📊 Total tasks: {metrics['total_qa_tasks']}")
        print(f"  ⚡ Avg task duration: {metrics['avg_task_duration']:.2f}s")
        
        # Performance change analysis
        pre_fault = metrics['pre_fault_answer_rate']
        post_fault = metrics['post_fault_answer_rate']
        if pre_fault and post_fault:
            change = ((post_fault - pre_fault) / pre_fault) * 100
            if change > 0:
                print(f"  📈 Performance change: +{change:.1f}% (improved after fault)")
            elif change < 0:
                print(f"  📉 Performance change: {change:.1f}% (degraded after fault)")
            else:
                print(f"  ➡️  Performance change: 0% (no change)")
        
        # Answer sources
        sources = metrics['answer_sources']
        if sources:
            total_answers = sum(sources.values())
            if total_answers > 0:
                local_pct = (sources.get('local', 0) / total_answers) * 100
                neighbor_pct = (sources.get('neighbor', 0) / total_answers) * 100
                print(f"  🏠 Local answers: {local_pct:.1f}%")
                print(f"  🤝 Neighbor answers: {neighbor_pct:.1f}%")

def main():
    print("🔥 Fail-Storm Protocol Comparison Test")
    print("=" * 50)
    print("Testing ANP, A2A, and Agora protocols...")
    
    # Test protocols in sequence
    protocols = ['anp', 'a2a', 'agora']
    results = []
    
    for protocol in protocols:
        result = run_protocol_test(protocol)
        if result:
            # Save individual results
            results_file = Path(__file__).parent / "results" / f"failstorm_metrics_{protocol}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"💾 {protocol.upper()} results saved to: {results_file}")
        
        results.append(result)
        
        # Brief pause between tests
        time.sleep(2)
    
    # Compare results
    compare_protocols(results)
    
    # Save comparison summary
    comparison_summary = {
        'test_timestamp': time.time(),
        'protocols_tested': len([r for r in results if r]),
        'results': [extract_key_metrics(r) for r in results if r]
    }
    
    summary_file = Path(__file__).parent / "results" / "protocol_comparison.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Comparison summary saved to: {summary_file}")
    print(f"🎉 Protocol comparison completed!")

if __name__ == "__main__":
    main()
