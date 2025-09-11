#!/usr/bin/env python3
"""
Summarize all protocol test results
"""

import json
from pathlib import Path

def summarize_results():
    results_dir = Path("script/fail_storm_recovery/results")
    
    # 检查所有结果文件
    result_files = [
        ('ANP', 'detailed_failstorm_metrics_20250910_222552_anp.json'),
        ('A2A', 'detailed_failstorm_metrics_20250910_225847_a2a.json'),
        ('Agora', 'detailed_failstorm_metrics_20250910_233802_agora.json'),
        ('Meta', 'meta_failstorm_metrics_20250910_220202_meta.json')
    ]

    print('🎉 所有协议循环故障测试结果汇总:')
    print('=' * 60)

    for protocol, filename in result_files:
        try:
            filepath = results_dir / filename
            if not filepath.exists():
                print(f'{protocol}协议: 文件不存在')
                print()
                continue
                
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if 'qa_summary' in data:
                qa = data['qa_summary']
                print(f'{protocol}协议:')
                print(f'  总任务数: {qa.get("total_tasks", 0):,}')
                print(f'  找到答案: {qa.get("found_answers", 0):,}')
                print(f'  答案率: {qa.get("answer_found_rate", 0):.1%}')
                print(f'  测试组数: {qa.get("groups_tested", 0)}')
                print(f'  组范围: {qa.get("group_range", "N/A")}')
            else:
                # 手动统计
                tasks = data.get('detailed_data', {}).get('task_executions', [])
                qa_tasks = [t for t in tasks if 'qa' in t.get('task_type', '')]
                found = [t for t in qa_tasks if t.get('answer_found')]
                groups = set(t.get('group_id', 0) for t in qa_tasks if t.get('group_id') is not None)
                
                print(f'{protocol}协议:')
                print(f'  总任务数: {len(qa_tasks):,}')
                print(f'  找到答案: {len(found):,}')
                print(f'  答案率: {len(found)/len(qa_tasks):.1%}' if qa_tasks else '0%')
                print(f'  测试组数: {len(groups)}')
                print(f'  组范围: {min(groups)}-{max(groups)}' if groups else 'N/A')
            print()
        except Exception as e:
            print(f'{protocol}协议: 错误 - {e}')
            print()

if __name__ == "__main__":
    summarize_results()
