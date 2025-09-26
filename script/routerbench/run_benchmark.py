"""
ProtoRouter Benchmark 主运行脚本
"""

import json
import os
import sys
import yaml
from typing import Dict, Any
import argparse

# Add src to path for importing utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from proto_router import ProtoRouterBenchmark
from evaluator import RouterBenchmarkEvaluator


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    if not os.path.exists(config_path):
        # 创建默认配置
        default_config = {
            "model": {
                "type": "openai",  # or "local"
                "name": "gpt-4o",
                "temperature": 0.1,
                "openai_api_key": "your_api_key_here",
                "openai_base_url": "https://api.openai.com/v1"
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"已创建默认配置文件: {config_path}")
        print("请编辑配置文件并设置正确的API密钥")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_dataset(data_path: str) -> list:
    """加载数据集"""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='运行ProtoRouter Benchmark')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--data', default='data/data.json', help='数据集路径')
    parser.add_argument('--output', default='results', help='输出目录')
    parser.add_argument('--limit', type=int, help='限制处理的场景数量（用于测试）')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 加载配置和数据
    print("🔧 Loading configuration...")
    config = load_config(args.config)
    
    print("📚 Loading dataset...")
    dataset = load_dataset(args.data)
    
    if args.limit:
        dataset = dataset[:args.limit]
        print(f"⚠️  Limited to {args.limit} scenarios")
    
    print(f"📊 Dataset contains {len(dataset)} scenarios")
    
    # Initialize system
    print("🚀 Initializing ProtoRouter...")
    router = ProtoRouterBenchmark(config)
    evaluator = RouterBenchmarkEvaluator()
    
    # Process all scenarios
    print("\n🔄 Processing scenarios...")
    evaluation_results = []
    
    for i, scenario in enumerate(dataset, 1):
        scenario_id = scenario.get("id", f"scenario_{i}")
        print(f"  [{i}/{len(dataset)}] Processing scenario: {scenario_id}")
        
        try:
            # LLM processes scenario
            llm_result = router.process_scenario(scenario)
            
            if llm_result["success"]:
                # Evaluate results
                ground_truth = scenario.get("ground_truth", {})
                eval_result = evaluator.evaluate_scenario(
                    scenario_id, 
                    llm_result["llm_response"], 
                    ground_truth
                )
                evaluation_results.append(eval_result)
                
                # Show detailed results for each module
                correct = eval_result["correct_modules"]
                total = eval_result["total_modules"]
                print(f"    📊 Result: {correct}/{total} modules correct")
                
                # Show each module's selection vs ground truth
                for module_id, module_result in eval_result["module_results"].items():
                    llm_choice = module_result["llm_selection"]
                    gt_choice = module_result["ground_truth"]
                    is_correct = module_result["correct"]
                    status = "✅" if is_correct else "❌"
                    print(f"      Module {module_id}: {status} LLM={llm_choice}, GT={gt_choice}")
                    
                    # Show A2A/ACP confusion specifically
                    if module_result.get("a2a_acp_confusion", False):
                        print(f"        ⚠️  A2A/ACP confusion detected")
            else:
                print(f"    ❌ Processing failed: {llm_result['error']}")
                
        except Exception as e:
            print(f"    💥 Exception: {str(e)}")
    
    # Calculate statistics
    print("\n📊 Calculating statistics...")
    statistics = evaluator.calculate_overall_statistics(evaluation_results)
    
    # Generate report
    print("\n📋 Generating report...")
    report = evaluator.generate_report(statistics)
    
    # Save results
    results_file = os.path.join(args.output, "benchmark_results.json")
    report_file = os.path.join(args.output, "benchmark_report.txt")
    
    evaluator.save_results(statistics, results_file)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 Report saved to: {report_file}")
    
    # Show brief results
    print("\n" + "="*60)
    print("🎯 Brief Results:")
    overall = statistics["overall_statistics"]
    print(f"  Overall scenario accuracy: {overall['overall_scenario_accuracy']:.2%}")
    print(f"  Individual module accuracy: {overall['individual_module_accuracy']:.2%}")
    print(f"  A2A/ACP confusion count: {overall['a2a_acp_confusion_count']}")
    
    print("\nBy difficulty level:")
    for difficulty, stats in statistics["difficulty_statistics"].items():
        print(f"  {difficulty}: scenario {stats['scenario_accuracy']:.1%}, module {stats['module_accuracy']:.1%}")
    
    print("="*60)


if __name__ == "__main__":
    main()
