"""
Export all LLM justifications and ground truth comparisons to a readable format.
"""

import json
import argparse
from typing import Dict, Any


def export_justifications(results_file: str, output_file: str):
    """Export all justifications to a readable text file"""
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    detailed_results = data.get("detailed_results", [])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ProtoRouter Benchmark - Complete Justifications Export\n")
        f.write("=" * 80 + "\n\n")
        
        for result in detailed_results:
            scenario_id = result["scenario_id"]
            difficulty = result["difficulty"]
            scenario_correct = result["scenario_correct"]
            
            f.write(f"Scenario: {scenario_id} ({difficulty})\n")
            f.write(f"Overall Result: {'✅ CORRECT' if scenario_correct else '❌ INCORRECT'}\n")
            f.write("-" * 60 + "\n")
            
            # Process each module
            for module_id, module_result in result["module_results"].items():
                gt_protocol = module_result["ground_truth"]
                llm_protocol = module_result["llm_selection"]
                is_correct = module_result["correct"]
                llm_justification = module_result.get("llm_justification", "No justification provided")
                
                # Get ground truth justification
                gt_justifications = result.get("ground_truth_justifications", {})
                gt_justification = gt_justifications.get(module_id, "No justification provided")
                
                f.write(f"\nModule {module_id}:\n")
                f.write(f"  Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}\n")
                f.write(f"  LLM Choice: {llm_protocol}\n")
                f.write(f"  Ground Truth: {gt_protocol}\n")
                
                if module_result.get("a2a_acp_confusion", False):
                    f.write(f"  ⚠️  A2A/ACP Confusion Detected\n")
                
                f.write(f"\n  LLM Justification:\n")
                f.write(f"    {llm_justification}\n")
                
                f.write(f"\n  Ground Truth Justification:\n")
                f.write(f"    {gt_justification}\n")
                
                f.write("\n")
            
            f.write("=" * 80 + "\n\n")
    
    print(f"Justifications exported to: {output_file}")


def export_summary_by_protocol(results_file: str, output_file: str):
    """Export summary of LLM reasoning patterns by protocol"""
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    detailed_results = data.get("detailed_results", [])
    
    # Group by protocol choices
    protocol_analysis = {
        "A2A": {"correct": [], "incorrect": []},
        "ACP": {"correct": [], "incorrect": []},
        "Agora": {"correct": [], "incorrect": []},
        "ANP": {"correct": [], "incorrect": []}
    }
    
    for result in detailed_results:
        for module_result in result["module_results"].values():
            llm_protocol = module_result["llm_selection"]
            is_correct = module_result["correct"]
            justification = module_result.get("llm_justification", "")
            
            if llm_protocol in protocol_analysis:
                category = "correct" if is_correct else "incorrect"
                protocol_analysis[llm_protocol][category].append({
                    "scenario": result["scenario_id"],
                    "justification": justification,
                    "ground_truth": module_result["ground_truth"]
                })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ProtoRouter Benchmark - Protocol Choice Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        for protocol in ["A2A", "ACP", "Agora", "ANP"]:
            correct_count = len(protocol_analysis[protocol]["correct"])
            incorrect_count = len(protocol_analysis[protocol]["incorrect"])
            total = correct_count + incorrect_count
            
            if total == 0:
                continue
                
            accuracy = correct_count / total * 100
            
            f.write(f"Protocol: {protocol}\n")
            f.write(f"Total selections: {total} | Correct: {correct_count} | Incorrect: {incorrect_count} | Accuracy: {accuracy:.1f}%\n")
            f.write("-" * 60 + "\n")
            
            # Show reasoning patterns for correct choices
            if correct_count > 0:
                f.write(f"\n✅ Correct {protocol} Selections ({correct_count}):\n")
                for i, item in enumerate(protocol_analysis[protocol]["correct"][:3], 1):  # Show first 3
                    f.write(f"  {i}. {item['scenario']} (GT: {item['ground_truth']})\n")
                    f.write(f"     Reasoning: {item['justification'][:300]}...\n\n")
            
            # Show reasoning patterns for incorrect choices
            if incorrect_count > 0:
                f.write(f"❌ Incorrect {protocol} Selections ({incorrect_count}):\n")
                for i, item in enumerate(protocol_analysis[protocol]["incorrect"][:3], 1):  # Show first 3
                    f.write(f"  {i}. {item['scenario']} (GT: {item['ground_truth']})\n")
                    f.write(f"     Reasoning: {item['justification'][:300]}...\n\n")
            
            f.write("=" * 80 + "\n\n")
    
    print(f"Protocol analysis exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Export LLM justifications from benchmark results')
    parser.add_argument('--results', default='results/benchmark_results.json', help='Path to benchmark results JSON file')
    parser.add_argument('--output', default='results/justifications_export.txt', help='Output file for justifications')
    parser.add_argument('--summary', default='results/protocol_analysis.txt', help='Output file for protocol analysis')
    
    args = parser.parse_args()
    
    print("📄 Exporting complete justifications...")
    export_justifications(args.results, args.output)
    
    print("📊 Exporting protocol analysis...")
    export_summary_by_protocol(args.results, args.summary)
    
    print("✅ Export completed!")


if __name__ == "__main__":
    main()
