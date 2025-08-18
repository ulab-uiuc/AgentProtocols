#!/usr/bin/env python3
"""
基于现有QA系统的拓扑结构评测
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Dict, List
import statistics

# 正确导入streaming_queue_new模块
sys.path.insert(0, str(Path(__file__).parent))
from agent_network.script.streaming_queue.old_scripts.streaming_queue_new import RealAgentNetworkDemo, ColoredOutput

class TopologyBenchmark:
    """拓扑结构评测类"""
    def __init__(self, config_path="config.yaml"):
        self.output = ColoredOutput()
        self.results = {}
        self.test_questions = self.load_test_questions()
    
    def load_test_questions(self):
        """加载测试问题"""
        # 检查数据文件是否存在
        questions_file = Path(__file__).parent / "data" / "top1000_simplified.jsonl"
        
        if not questions_file.exists():
            # 如果数据文件不存在，使用默认测试问题
            self.output.warning(f"数据文件不存在: {questions_file}，使用默认测试问题")
            return [
                {'id': i, 'question': q} for i, q in enumerate([
                    "What is artificial intelligence?",
                    "Explain machine learning basics",
                    "What are neural networks?",
                    "How does deep learning work?",
                    "What is natural language processing?",
                    "Explain computer vision",
                    "What is reinforcement learning?",
                    "How do transformers work?",
                    "What is supervised learning?",
                    "Explain unsupervised learning"
                ])
            ]
        
        questions = []
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        questions.append({
                            'id': item.get('id', len(questions)),
                            'question': item.get('q', item.get('question', ''))
                        })
                        if len(questions) >= 20:  # 使用20个问题进行测试
                            break
        except Exception as e:
            self.output.error(f"读取问题文件失败: {e}")
            return []
        
        return questions
    
    async def test_topology_performance(self, topology_name: str, demo: RealAgentNetworkDemo):
        """测试指定拓扑的性能"""
        self.output.info(f"测试 {topology_name} 拓扑性能...")
        
        # 基础性能测试
        basic_performance = await self.test_basic_qa_performance(demo)
        
        # 并发性能测试
        concurrent_performance = await self.test_concurrent_performance(demo)
        
        # 网络分析（使用基础网络信息）
        network_analysis = self.get_network_analysis(demo)
        
        # 连接测试
        connectivity_test = await self.test_connectivity(demo)
        
        return {
            'basic_performance': basic_performance,
            'concurrent_performance': concurrent_performance,
            'network_analysis': network_analysis,
            'connectivity_test': connectivity_test
        }
    
    def get_network_analysis(self, demo: RealAgentNetworkDemo):
        """获取网络分析信息"""
        try:
            topology_info = demo.network.get_topology()
            
            # 计算基础网络指标
            total_nodes = len(topology_info)
            total_edges = sum(len(edges) for edges in topology_info.values())
            
            # 简单的连通性分析
            max_connections = max(len(edges) for edges in topology_info.values()) if topology_info else 0
            min_connections = min(len(edges) for edges in topology_info.values()) if topology_info else 0
            avg_connections = total_edges / total_nodes if total_nodes > 0 else 0
            
            return {
                'total_nodes': total_nodes,
                'total_edges': total_edges,
                'max_connections_per_node': max_connections,
                'min_connections_per_node': min_connections,
                'avg_connections_per_node': avg_connections,
                'network_density': total_edges / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0
            }
        except Exception as e:
            self.output.warning(f"网络分析失败: {e}")
            return {'error': str(e)}
    
    async def test_basic_qa_performance(self, demo: RealAgentNetworkDemo):
        """测试基础QA性能"""
        test_questions = self.test_questions[:10]  # 使用10个问题
        
        if not test_questions:
            return {'error': 'No test questions available'}
        
        start_time = time.time()
        
        # 构造问题格式，适配dispatch_questions_dynamically方法
        formatted_questions = []
        for q in test_questions:
            formatted_questions.append({
                'type': 'general',
                'message': q['question']
            })
        
        try:
            # 使用现有的dispatch方法
            results = await demo.dispatch_questions_dynamically(formatted_questions)
            
            end_time = time.time()
            
            successful_results = [r for r in results if r and 'result' in r]
            
            return {
                'total_time_ms': (end_time - start_time) * 1000,
                'questions_processed': len(test_questions),
                'successful_responses': len(successful_results),
                'success_rate': len(successful_results) / len(test_questions) if test_questions else 0,
                'avg_time_per_question': (end_time - start_time) * 1000 / len(test_questions) if test_questions else 0,
                'throughput_qps': len(test_questions) / (end_time - start_time) if (end_time - start_time) > 0 else 0
            }
        except Exception as e:
            self.output.error(f"基础性能测试失败: {e}")
            return {'error': str(e)}
    
    async def test_concurrent_performance(self, demo: RealAgentNetworkDemo):
        """测试并发性能"""
        concurrent_questions = self.test_questions[:5]
        
        if not concurrent_questions:
            return {'error': 'No test questions available'}
        
        start_time = time.time()
        
        try:
            # 并发发送消息
            tasks = []
            for question in concurrent_questions:
                task = demo.send_message_to_coordinator(question['question'], 'general')
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            
            successful_results = [r for r in results if not isinstance(r, Exception) and r is not None and 'result' in r]
            
            return {
                'concurrent_requests': len(concurrent_questions),
                'successful_responses': len(successful_results),
                'total_time_ms': (end_time - start_time) * 1000,
                'concurrency_efficiency': len(successful_results) / len(concurrent_questions) if concurrent_questions else 0,
                'avg_concurrent_latency': (end_time - start_time) * 1000 / len(concurrent_questions) if concurrent_questions else 0
            }
        except Exception as e:
            self.output.error(f"并发性能测试失败: {e}")
            return {'error': str(e)}
    
    async def test_connectivity(self, demo: RealAgentNetworkDemo):
        """测试网络连通性"""
        try:
            # 执行健康检查
            health_status = await demo.network.health_check()
            
            healthy_count = sum(1 for status in health_status.values() if status)
            total_count = len(health_status)
            
            return {
                'total_agents': total_count,
                'healthy_agents': healthy_count,
                'connectivity_rate': healthy_count / total_count if total_count > 0 else 0,
                'health_details': health_status
            }
        except Exception as e:
            self.output.error(f"连通性测试失败: {e}")
            return {'error': str(e)}
    
    async def run_topology_comparison(self):
        """运行拓扑对比测试"""
        topologies_to_test = [
            ('star', 'Star Topology'),
            ('ring', 'Ring Topology'),
            ('mesh', 'Mesh Topology'),
            ('hierarchical', 'Hierarchical Topology')
        ]

        self.output.info("开始拓扑结构对比评测...")
        
        for topology_name, description in topologies_to_test:
            self.output.info(f"\n=== 测试 {description} ===")
            # 创建demo实例
            demo = RealAgentNetworkDemo()
            
            try:
                # 设置agents
                worker_ids = await demo.setup_agents()
                self.output.success(f"成功创建 {len(worker_ids)} 个worker agents")
                
                # 应用指定拓扑
                if topology_name == "star":
                    demo.network.setup_star_topology("Coordinator-1")
                    self.output.success("应用星型拓扑")
                elif topology_name == "mesh":
                    demo.network.setup_mesh_topology()
                    self.output.success("应用网状拓扑")
                elif topology_name == "ring":
                    if hasattr(demo.network, 'setup_ring_topology'):
                        demo.network.setup_ring_topology()
                    else:
                        raise Exception("Ring topology not implemented")
                elif topology_name == "hierarchical":
                    if len(worker_ids) >= 4:
                        sub_coordinators = worker_ids[:2]
                        if hasattr(demo.network, 'setup_hierarchical_topology'):
                            demo.network.setup_hierarchical_topology("Coordinator-1", sub_coordinators)
                        else:
                            raise Exception("Hierarchical topology not implemented")
                    else:
                        raise Exception("需要至少4个worker进行分层拓扑测试")

                # 等待连接建立
                await asyncio.sleep(2)
                
                # 运行健康检查
                await demo.run_health_check()
                
                # 运行性能测试
                performance_results = await self.test_topology_performance(topology_name, demo)
                
                # 保存结果
                self.results[topology_name] = performance_results
                
                self.output.success(f"{description} 测试完成")
                
                # 显示即时结果
                self.display_immediate_results(topology_name, performance_results)
                
            except Exception as e:
                self.output.error(f"{description} 测试失败: {e}")
                import traceback
                traceback.print_exc()
                
            finally:
                await self.simple_cleanup(demo)
                await asyncio.sleep(2)
                
        # 生成对比报告
        self.generate_comparison_report()
    
    async def simple_cleanup(self, demo):
        """简单清理 - 忽略所有错误"""
        if not demo:
            return
        
        self.output.info("开始清理资源...")
        
        try:
            # 尝试正常清理，但不等待
            cleanup_task = asyncio.create_task(demo.cleanup())
            try:
                await asyncio.wait_for(cleanup_task, timeout=3.0)
                self.output.success("正常清理完成")
            except asyncio.TimeoutError:
                self.output.info("清理超时，执行强制清理")
                cleanup_task.cancel()
                await self.force_cleanup(demo)
        except Exception as e:
            self.output.info(f"清理时出现预期错误: {type(e).__name__}")
            await self.force_cleanup(demo)
    
    async def force_cleanup(self, demo):
        """强制清理资源"""
        try:
            # 强制关闭HTTP客户端
            if hasattr(demo, 'httpx_client') and demo.httpx_client:
                if not demo.httpx_client.is_closed:
                    await demo.httpx_client.aclose()
            
            # 取消所有任务
            agents = []
            if hasattr(demo, 'coordinator') and demo.coordinator:
                agents.append(demo.coordinator)
            if hasattr(demo, 'workers') and demo.workers:
                agents.extend(demo.workers)
            
            for agent in agents:
                try:
                    if hasattr(agent, '_server_task') and agent._server_task:
                        agent._server_task.cancel()
                except Exception:
                    pass
            
            self.output.success("强制清理完成")
            
        except Exception as e:
            self.output.info(f"强制清理也出现错误: {e} (可忽略)")

    def display_immediate_results(self, topology_name: str, results: Dict):
        """显示即时测试结果"""
        self.output.info(f"\n--- {topology_name.upper()} 拓扑测试结果 ---")
        
        basic_perf = results.get('basic_performance', {})
        if 'error' not in basic_perf:
            self.output.progress(f"成功率: {basic_perf.get('success_rate', 0):.2%}")
            self.output.progress(f"吞吐量: {basic_perf.get('throughput_qps', 0):.2f} QPS")
            self.output.progress(f"平均延迟: {basic_perf.get('avg_time_per_question', 0):.2f} ms")
        
        concurrent_perf = results.get('concurrent_performance', {})
        if 'error' not in concurrent_perf:
            self.output.progress(f"并发效率: {concurrent_perf.get('concurrency_efficiency', 0):.2%}")
        
        network_analysis = results.get('network_analysis', {})
        if 'error' not in network_analysis:
            self.output.progress(f"网络节点数: {network_analysis.get('total_nodes', 0)}")
            self.output.progress(f"网络连接数: {network_analysis.get('total_edges', 0)}")
            self.output.progress(f"网络密度: {network_analysis.get('network_density', 0):.3f}")
    
    def generate_comparison_report(self):
        """生成拓扑对比报告"""
        self.output.info("\n=== 拓扑对比报告 ===")
        
        if not self.results:
            self.output.error("没有测试结果")
            return
        
        # 打印摘要表格
        print(f"\n{'拓扑类型':<12} {'成功率':<10} {'吞吐量(QPS)':<12} {'平均延迟(ms)':<14} {'网络密度':<10}")
        print("-" * 70)
        
        for topology_name, results in self.results.items():
            basic_perf = results.get('basic_performance', {})
            network_analysis = results.get('network_analysis', {})
            
            if 'error' in basic_perf:
                print(f"{topology_name:<12} {'ERROR':<10} {'ERROR':<12} {'ERROR':<14} {'ERROR':<10}")
                continue
            
            success_rate = basic_perf.get('success_rate', 0)
            throughput = basic_perf.get('throughput_qps', 0)
            avg_latency = basic_perf.get('avg_time_per_question', 0)
            density = network_analysis.get('network_density', 0)
            
            print(f"{topology_name:<12} {success_rate:<10.2%} {throughput:<12.2f} {avg_latency:<14.2f} {density:<10.3f}")
        
        # 保存详细报告
        report_file = Path(__file__).parent / 'topology_comparison_report.json'
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': time.time(),
                    'test_results': self.results,
                    'summary': self.generate_summary()
                }, f, indent=2, ensure_ascii=False)
            
            self.output.success(f"详细报告已保存到: {report_file}")
        except Exception as e:
            self.output.error(f"保存报告失败: {e}")
    
    def generate_summary(self):
        """生成测试摘要"""
        if not self.results:
            return {}
        
        # 找出各项指标的最佳表现者
        best_performers = {}
        
        metrics = [
            ('success_rate', ['basic_performance', 'success_rate'], 'max'),
            ('throughput', ['basic_performance', 'throughput_qps'], 'max'),
            ('latency', ['basic_performance', 'avg_time_per_question'], 'min'),
            ('network_density', ['network_analysis', 'network_density'], 'max')
        ]
        
        for metric_name, metric_path, optimization in metrics:
            values = {}
            for topology, results in self.results.items():
                try:
                    value = results
                    for key in metric_path:
                        value = value[key]
                    values[topology] = value
                except (KeyError, TypeError):
                    continue
            
            if values:
                if optimization == 'max':
                    best_topology = max(values, key=values.get)
                else:
                    best_topology = min(values, key=values.get)
                
                best_performers[metric_name] = {
                    'topology': best_topology,
                    'value': values[best_topology],
                    'all_values': values
                }
        
        return best_performers

# 主函数
async def main():
    """主函数"""
    try:
        benchmark = TopologyBenchmark()
        await benchmark.run_topology_comparison()
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())