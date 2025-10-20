"""
Evaluation System - Measure ProtoRouter protocol selection accuracy
"""

import json
from typing import Dict, List, Any, Tuple
from collections import defaultdict


class RouterBenchmarkEvaluator:
    """Router benchmark evaluator that tracks various accuracy metrics"""
    
    def __init__(self):
        self.results = []
        self.difficulty_levels = ["L1", "L2", "L3", "L4", "L5"]
    
    def evaluate_scenario(self, scenario_id: str, llm_response: Dict[str, Any], 
                         ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the results of a single scenario"""
        
        # Parse difficulty level
        difficulty = scenario_id.split("-")[0] if "-" in scenario_id else "Unknown"
        
        # Get protocol selections from LLM
        llm_selections = {}
        if llm_response and "module_selections" in llm_response:
            for selection in llm_response["module_selections"]:
                module_id = str(selection.get("module_id", 0))
                protocol = selection.get("selected_protocol", "")
                llm_selections[module_id] = protocol
        
        # Get ground truth
        gt_selections = {}
        for module_id, gt_data in ground_truth.items():
            gt_selections[module_id] = gt_data.get("module_protocol", "")
        
        # Calculate metrics
        module_results = {}
        correct_modules = 0
        total_modules = len(gt_selections)
        a2a_acp_confusion = 0  # A2A/ACP confusion cases
        
        for module_id in gt_selections:
            gt_protocol = gt_selections[module_id]
            llm_protocol = llm_selections.get(module_id, "")
            
            is_correct = gt_protocol == llm_protocol
            if is_correct:
                correct_modules += 1
            
            # Check for A2A/ACP confusion
            if ((gt_protocol == "A2A" and llm_protocol == "ACP") or 
                (gt_protocol == "ACP" and llm_protocol == "A2A")):
                a2a_acp_confusion += 1
            
            # Get LLM justification if available
            llm_justification = ""
            if llm_response and "module_selections" in llm_response:
                for selection in llm_response["module_selections"]:
                    if str(selection.get("module_id", 0)) == module_id:
                        llm_justification = selection.get("justification", "")
                        break
            
            module_results[module_id] = {
                "ground_truth": gt_protocol,
                "llm_selection": llm_protocol,
                "llm_justification": llm_justification,
                "correct": is_correct,
                "a2a_acp_confusion": ((gt_protocol == "A2A" and llm_protocol == "ACP") or 
                                     (gt_protocol == "ACP" and llm_protocol == "A2A"))
            }
        
        # For L2 and above, require all answers to be correct
        scenario_correct = correct_modules == total_modules
        if difficulty == "L1":
            # L1 allows partial correctness
            scenario_correct = correct_modules > 0
        
        individual_accuracy = correct_modules / total_modules if total_modules > 0 else 0
        
        # Add ground truth justifications
        gt_justifications = {}
        for module_id, gt_data in ground_truth.items():
            gt_justifications[module_id] = gt_data.get("justification", "")
        
        result = {
            "scenario_id": scenario_id,
            "difficulty": difficulty,
            "total_modules": total_modules,
            "correct_modules": correct_modules,
            "scenario_correct": scenario_correct,
            "individual_accuracy": individual_accuracy,
            "a2a_acp_confusion_count": a2a_acp_confusion,
            "module_results": module_results,
            "ground_truth_justifications": gt_justifications,
            "llm_raw_response": llm_response  # Save complete LLM response
        }
        
        return result
    
    def calculate_overall_statistics(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall statistics"""
        
        # Group by difficulty
        by_difficulty = defaultdict(list)
        for result in evaluation_results:
            difficulty = result["difficulty"]
            by_difficulty[difficulty].append(result)
        
        # Overall statistics
        total_scenarios = len(evaluation_results)
        correct_scenarios = sum(1 for r in evaluation_results if r["scenario_correct"])
        overall_scenario_accuracy = correct_scenarios / total_scenarios if total_scenarios > 0 else 0
        
        # Individual module statistics
        total_modules = sum(r["total_modules"] for r in evaluation_results)
        correct_modules = sum(r["correct_modules"] for r in evaluation_results)
        individual_module_accuracy = correct_modules / total_modules if total_modules > 0 else 0
        
        # A2A/ACP confusion statistics
        total_a2a_acp_confusion = sum(r["a2a_acp_confusion_count"] for r in evaluation_results)
        
        # Statistics by difficulty
        difficulty_stats = {}
        for difficulty in self.difficulty_levels:
            if difficulty in by_difficulty:
                scenarios = by_difficulty[difficulty]
                total_diff_scenarios = len(scenarios)
                correct_diff_scenarios = sum(1 for s in scenarios if s["scenario_correct"])
                diff_scenario_accuracy = correct_diff_scenarios / total_diff_scenarios if total_diff_scenarios > 0 else 0
                
                total_diff_modules = sum(s["total_modules"] for s in scenarios)
                correct_diff_modules = sum(s["correct_modules"] for s in scenarios)
                diff_module_accuracy = correct_diff_modules / total_diff_modules if total_diff_modules > 0 else 0
                
                difficulty_stats[difficulty] = {
                    "total_scenarios": total_diff_scenarios,
                    "correct_scenarios": correct_diff_scenarios,
                    "scenario_accuracy": diff_scenario_accuracy,
                    "total_modules": total_diff_modules,
                    "correct_modules": correct_diff_modules,
                    "module_accuracy": diff_module_accuracy
                }
        
        # Protocol confusion matrix
        confusion_matrix = self._calculate_confusion_matrix(evaluation_results)
        
        return {
            "overall_statistics": {
                "total_scenarios": total_scenarios,
                "correct_scenarios": correct_scenarios,
                "overall_scenario_accuracy": overall_scenario_accuracy,
                "total_modules": total_modules,
                "correct_modules": correct_modules,
                "individual_module_accuracy": individual_module_accuracy,
                "a2a_acp_confusion_count": total_a2a_acp_confusion
            },
            "difficulty_statistics": difficulty_stats,
            "confusion_matrix": confusion_matrix,
            "detailed_results": evaluation_results
        }
    
    def _calculate_confusion_matrix(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Calculate confusion matrix for protocol selection"""
        protocols = ["A2A", "ACP", "Agora", "ANP"]
        matrix = {gt: {pred: 0 for pred in protocols} for gt in protocols}
        
        for result in evaluation_results:
            for module_result in result["module_results"].values():
                gt = module_result["ground_truth"]
                pred = module_result["llm_selection"]
                if gt in protocols and pred in protocols:
                    matrix[gt][pred] += 1
        
        return matrix
    
    def generate_report(self, statistics: Dict[str, Any]) -> str:
        """Generate detailed evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("ProtoRouter Benchmark Evaluation Report")
        report.append("=" * 60)
        
        # Overall statistics
        overall = statistics["overall_statistics"]
        report.append(f"\n📊 Overall Statistics:")
        report.append(f"  Total scenarios: {overall['total_scenarios']}")
        report.append(f"  Correct scenarios: {overall['correct_scenarios']}")
        report.append(f"  Overall scenario accuracy: {overall['overall_scenario_accuracy']:.2%}")
        report.append(f"  Total modules: {overall['total_modules']}")
        report.append(f"  Correct modules: {overall['correct_modules']}")
        report.append(f"  Individual module accuracy: {overall['individual_module_accuracy']:.2%}")
        report.append(f"  A2A/ACP confusion count: {overall['a2a_acp_confusion_count']}")
        
        # Statistics by difficulty
        report.append(f"\n📈 Statistics by Difficulty:")
        for difficulty, stats in statistics["difficulty_statistics"].items():
            report.append(f"  {difficulty}:")
            report.append(f"    Scenario accuracy: {stats['scenario_accuracy']:.2%} ({stats['correct_scenarios']}/{stats['total_scenarios']})")
            report.append(f"    Module accuracy: {stats['module_accuracy']:.2%} ({stats['correct_modules']}/{stats['total_modules']})")
        
        # Confusion matrix
        report.append(f"\n🔄 Protocol Confusion Matrix:")
        matrix = statistics["confusion_matrix"]
        protocols = ["A2A", "ACP", "Agora", "ANP"]
        
        # Header
        header = "Actual\\Pred".ljust(12)
        for p in protocols:
            header += f"{p:>8}"
        report.append(f"  {header}")
        
        # Matrix content
        for gt in protocols:
            row = f"{gt}".ljust(12)
            for pred in protocols:
                row += f"{matrix[gt][pred]:>8}"
            report.append(f"  {row}")
        
        # Detailed error analysis
        report.append(f"\n❌ Error Analysis:")
        error_count = defaultdict(int)
        error_examples = defaultdict(list)
        
        for result in statistics["detailed_results"]:
            if not result["scenario_correct"]:
                for module_id, module_result in result["module_results"].items():
                    if not module_result["correct"]:
                        gt = module_result["ground_truth"]
                        pred = module_result["llm_selection"]
                        error_type = f"{gt} → {pred}"
                        error_count[error_type] += 1
                        
                        # Save example with justifications
                        if len(error_examples[error_type]) < 2:  # Keep max 2 examples per error type
                            example = {
                                "scenario_id": result["scenario_id"],
                                "module_id": module_id,
                                "llm_justification": module_result.get("llm_justification", ""),
                                "gt_justification": result.get("ground_truth_justifications", {}).get(module_id, "")
                            }
                            error_examples[error_type].append(example)
        
        for error_type, count in sorted(error_count.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {error_type}: {count} times")
            
            # Add examples with justifications
            for example in error_examples[error_type]:
                report.append(f"    Example - {example['scenario_id']} Module {example['module_id']}:")
                if example['llm_justification']:
                    report.append(f"      LLM: {example['llm_justification'][:200]}...")
                if example['gt_justification']:
                    report.append(f"      GT:  {example['gt_justification'][:200]}...")
                report.append("")
        
        return "\n".join(report)
    
    def save_results(self, statistics: Dict[str, Any], output_file: str):
        """Save results to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_file}")
