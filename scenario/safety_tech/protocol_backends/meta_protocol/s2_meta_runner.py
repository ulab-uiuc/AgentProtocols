# -*- coding: utf-8 -*-
"""
S2 Meta Protocol Runner for Safety_Tech

Main entry point for S2 security testing with intelligent protocol routing.
Integrates LLM-based protocol selection with comprehensive S2 security probes.
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Dict, Any, Optional
from pathlib import Path

# Add paths
HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent.parent
PROJECT_ROOT = SAFETY_TECH.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SAFETY_TECH) not in sys.path:
    sys.path.insert(0, str(SAFETY_TECH))

# Import S2 Meta components
from .s2_meta_coordinator import S2MetaCoordinator
from .s2_llm_router import S2LLMRouter

# Import safety_tech base components
from runners.runner_base import RunnerBase


class S2MetaProtocolRunner(RunnerBase):
    """
    S2 Meta Protocol Runner
    
    Main orchestrator for S2 security testing using intelligent protocol routing.
    Combines LLM-based protocol selection with comprehensive security probes.
    """
    
    def __init__(self, config_path: str = "config_meta_s2.yaml"):
        super().__init__(config_path)
        
        self.config_path = config_path  # Store config_path for coordinator
        self.coordinator: Optional[S2MetaCoordinator] = None
        self.router: Optional[S2LLMRouter] = None
        self.test_results: Dict[str, Any] = {}
        
        # S2 Meta testing parameters
        self.test_focus = self.config.get("general", {}).get("test_focus", "comprehensive")
        self.enable_cross_protocol = self.config.get("general", {}).get("enable_cross_protocol", True)
        self.enable_mitm = self.config.get("general", {}).get("enable_mitm", False)
        
    async def run(self) -> Dict[str, Any]:
        """Run S2 Meta Protocol security test."""
        
        try:
            self.output.info("🚀 启动S2 Meta协议保密性测试")
            self.output.info(f"   测试重点: {self.test_focus}")
            self.output.info(f"   跨协议通信: {'启用' if self.enable_cross_protocol else '禁用'}")
            self.output.info(f"   MITM测试: {'启用' if self.enable_mitm else '禁用'}")
            
            # Initialize S2 Meta Coordinator
            await self.initialize_coordinator()
            
            # Create meta network infrastructure  
            await self.coordinator.create_network()
            
            # Setup dual doctors with intelligent protocol routing
            await self.coordinator.setup_s2_doctors(self.test_focus)
            
            # Run health checks
            await self.coordinator.run_health_checks()
            
            # Execute S2 security tests
            self.test_results = await self.coordinator.run_s2_security_test()
            
            # Display results with detailed S2 analysis
            # 尝试获取详细的S2分析结果
            s2_detailed_results = getattr(self.coordinator, '_last_s2_detailed_results', None)
            self.coordinator.display_results(self.test_results, s2_detailed_results)
            
            # Generate summary report
            await self.generate_summary_report()
            
            self.output.success("✅ S2 Meta协议测试完成!")
            
            return self.test_results
            
        except Exception as e:
            self.output.error(f"S2 Meta协议测试失败: {e}")
            raise
        finally:
            # Cleanup resources
            if self.coordinator:
                await self.coordinator.cleanup()
    
    async def initialize_coordinator(self):
        """Initialize S2 Meta Coordinator."""
        
        try:
            self.coordinator = S2MetaCoordinator(self.config_path)
            
            # Set test focus and parameters
            self.coordinator.config["general"]["test_focus"] = self.test_focus
            self.coordinator.config["general"]["enable_cross_protocol"] = self.enable_cross_protocol
            self.coordinator.config["general"]["enable_mitm"] = self.enable_mitm
            
            self.output.success("📋 S2 Meta协调器已初始化")
            
            # Display S2 security profiles
            security_summary = self.coordinator.s2_router.get_s2_security_summary()
            self.output.info("🔒 S2安全档案:")
            
            for rank_info in security_summary["ranking_by_s2_score"]:
                protocol = rank_info["protocol"]
                score = rank_info["s2_score"]
                self.output.info(f"   {rank_info['rank']}. {protocol.upper()}: {score:.1f}分")
            
        except Exception as e:
            self.output.error(f"S2协调器初始化失败: {e}")
            raise
    
    async def generate_summary_report(self):
        """Generate comprehensive S2 Meta testing summary report."""
        
        try:
            # Get routing statistics
            routing_stats = self.coordinator.s2_router.get_routing_statistics()
            security_summary = self.coordinator.s2_router.get_s2_security_summary()
            
            # Prepare summary data
            summary_data = {
                "test_configuration": {
                    "test_focus": self.test_focus,
                    "cross_protocol_enabled": self.enable_cross_protocol,
                    "mitm_enabled": self.enable_mitm,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "protocol_routing": {
                    "routing_decision": self.coordinator.routing_decision.__dict__ if self.coordinator.routing_decision else {},
                    "routing_statistics": routing_stats,
                    "llm_routing_used": routing_stats.get("llm_available", False)
                },
                "s2_security_profiles": security_summary,
                "test_results": self.test_results,
                "conversation_statistics": self.coordinator.conversation_stats
            }
            
            # Save comprehensive summary
            summary_file = self._get_output_path("s2_meta_protocol_summary.json")
            
            import json
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            # Generate human-readable report
            readable_report = self._generate_readable_summary(summary_data)
            report_file = self._get_output_path("s2_meta_protocol_summary_report.txt")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(readable_report)
            
            self.output.success(f"📊 S2总结报告已生成:")
            self.output.info(f"   JSON数据: {summary_file}")
            self.output.info(f"   可读报告: {report_file}")
            
        except Exception as e:
            self.output.warning(f"生成S2总结报告失败: {e}")
    
    def _generate_readable_summary(self, summary_data: Dict[str, Any]) -> str:
        """Generate human-readable summary report."""
        
        test_config = summary_data["test_configuration"]
        routing = summary_data["protocol_routing"]
        profiles = summary_data["s2_security_profiles"]
        results = summary_data["test_results"]
        conv_stats = summary_data["conversation_statistics"]
        
        report = f"""
=== S2 Meta协议保密性测试总结报告 ===

测试配置:
- 测试时间: {test_config.get('timestamp', 'unknown')}
- 测试重点: {test_config.get('test_focus', 'unknown')}  
- 跨协议通信: {'启用' if test_config.get('cross_protocol_enabled', False) else '禁用'}
- MITM测试: {'启用' if test_config.get('mitm_enabled', False) else '禁用'}

协议路由决策:
- 路由方式: {'LLM智能路由' if routing.get('llm_routing_used', False) else '规则路由'}"""
        
        routing_decision = routing.get("routing_decision", {})
        if routing_decision:
            report += f"""
- Doctor_A协议: {routing_decision.get('doctor_a_protocol', 'unknown').upper()}
- Doctor_B协议: {routing_decision.get('doctor_b_protocol', 'unknown').upper()}  
- 路由策略: {routing_decision.get('routing_strategy', 'unknown')}
- 置信度: {routing_decision.get('confidence', 0.0):.1%}"""
        
        report += f"""

S2协议安全排名:"""
        
        for rank_info in profiles.get("ranking_by_s2_score", []):
            protocol = rank_info["protocol"]
            score = rank_info["s2_score"]  
            rank = rank_info["rank"]
            report += f"""
{rank}. {protocol.upper()}: {score:.1f}/100"""
        
        report += f"""

测试执行统计:
- 总对话数: {conv_stats.get('total_conversations', 0)}
- 总探针数: {conv_stats.get('total_security_probes', 0)}
- 跨协议消息: {conv_stats.get('cross_protocol_messages', 0)}
- 探针注入成功率: {conv_stats.get('probe_injection_success_rate', 0.0):.1%}
- 检测到的S2违规: {conv_stats.get('s2_violations_detected', 0)}

S2保密性评分:"""
        
        s2_scores = results.get("s2_scores", {})
        for protocol, score in s2_scores.items():
            grade = "优秀" if score >= 90 else "良好" if score >= 80 else "中等" if score >= 60 else "较差"
            report += f"""
- {protocol.upper()}: {score:.1f}/100 ({grade})"""
        
        violations = results.get("security_violations", {})
        total_violations = sum(violations.values()) if violations else 0
        
        report += f"""

S2安全违规统计:
- 总违规数: {total_violations}"""
        
        if violations:
            for vtype, count in violations.items():
                if count > 0:
                    report += f"""
  - {vtype}: {count}"""
        
        cross_protocol_comparison = results.get("cross_protocol_security_comparison", {})
        if cross_protocol_comparison:
            report += f"""

跨协议安全对比:
- 对比协议: {cross_protocol_comparison.get('protocols_compared', [])}
- 安全差异: {cross_protocol_comparison.get('security_differential', 0):.1f}分
- 安全性更强: {cross_protocol_comparison.get('stronger_protocol', 'unknown').upper()}"""
        
        # Add routing effectiveness analysis
        routing_stats = routing.get("routing_statistics", {})
        if routing_stats and routing_stats.get("total_decisions", 0) > 0:
            report += f"""

路由决策分析:
- 总决策数: {routing_stats.get('total_decisions', 0)}
- 跨协议比率: {routing_stats.get('cross_protocol_ratio', 0.0):.1%}
- LLM可用性: {'是' if routing_stats.get('llm_available', False) else '否'}"""
        
        report += f"""

=== 测试结论 ===

"""
        
        # Generate conclusions based on results
        if s2_scores:
            max_score = max(s2_scores.values())
            if max_score >= 90:
                conclusion = "优秀 - 协议具备强大的S2保密性防护能力"
            elif max_score >= 80:
                conclusion = "良好 - 协议提供了可靠的S2保密性保护"  
            elif max_score >= 60:
                conclusion = "中等 - 协议具备基础的S2保密性能力，仍有改进空间"
            else:
                conclusion = "较差 - 协议的S2保密性防护能力需要显著改进"
            
            report += f"整体S2保密性评估: {conclusion}\n"
        
        if cross_protocol_comparison:
            protocols = cross_protocol_comparison.get('protocols_compared', [])
            stronger = cross_protocol_comparison.get('stronger_protocol', '')
            if len(protocols) == 2 and stronger:
                report += f"跨协议对比结论: {stronger.upper()}协议在S2保密性方面表现更优\n"
        
        if total_violations == 0:
            report += "安全防护结论: ✅ 未检测到S2安全违规，协议防护机制有效\n"
        else:
            report += f"安全防护结论: ⚠️ 检测到{total_violations}个S2违规，需要加强防护措施\n"
        
        report += f"""
Meta协议路由效果: {'✅ LLM智能路由有效提升了协议选择准确性' if routing.get('llm_routing_used', False) else '📋 规则路由提供了基础的协议选择能力'}

=== 报告结束 ===
"""
        
        return report
    
    def display_results(self, results: Dict[str, Any]):
        """Display S2 Meta Protocol final results."""
        
        self.output.info("📊 S2 Meta协议测试最终结果")
        self.output.progress("=" * 70)
        
        # Display routing decision
        if self.coordinator and self.coordinator.routing_decision:
            decision = self.coordinator.routing_decision
            self.output.progress(f"协议路由: {decision.doctor_a_protocol.upper()} ↔ {decision.doctor_b_protocol.upper()}")
            self.output.progress(f"路由策略: {decision.routing_strategy}")
            self.output.progress(f"置信度: {decision.confidence:.1%}")
            self.output.progress("")
        
        # Display S2 scores
        s2_scores = results.get("s2_scores", {})
        if s2_scores:
            self.output.progress("S2保密性评分:")
            for protocol, score in s2_scores.items():
                grade = "🟢" if score >= 90 else "🟡" if score >= 80 else "🟠" if score >= 60 else "🔴"
                self.output.progress(f"  {grade} {protocol.upper()}: {score:.1f}/100")
            self.output.progress("")
        
        # Display security violations summary
        violations = results.get("security_violations", {})
        total_violations = sum(violations.values()) if violations else 0
        
        if total_violations == 0:
            self.output.success("🛡️ S2安全评估: 未检测到安全违规")
        else:
            self.output.warning(f"⚠️ S2安全评估: 检测到{total_violations}个违规")
            for vtype, count in violations.items():
                if count > 0:
                    self.output.progress(f"     {vtype}: {count}")
        
        self.output.progress("")
        
        # Display conversation statistics
        conv_stats = self.coordinator.conversation_stats if self.coordinator else {}
        self.output.progress(f"测试统计: {conv_stats.get('total_conversations', 0)}对话, {conv_stats.get('total_security_probes', 0)}探针")
        
        if conv_stats.get('cross_protocol_messages', 0) > 0:
            self.output.progress(f"跨协议消息: {conv_stats.get('cross_protocol_messages', 0)}")
        
        self.output.progress("=" * 70)


async def main():
    """Main entry point for S2 Meta Protocol testing."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='S2 Meta Protocol Security Testing\n\nNOTE: E2E encryption testing requires sudo privileges for network packet capture.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', default='config_meta_s2.yaml', 
                       help='Configuration file path')
    parser.add_argument('--test-focus', choices=['comprehensive', 'tls_focused', 'e2e_focused', 'session_focused', 'anp_pure', 'agora_pure', 'acp_pure', 'a2a_pure'],
                       default='comprehensive', help='S2 test focus area (e2e_focused requires sudo, *_pure for single protocol tests)')
    parser.add_argument('--enable-cross-protocol', action='store_true', default=True,
                       help='Enable cross-protocol communication testing')
    parser.add_argument('--enable-mitm', action='store_true', default=False,
                       help='Enable MITM testing (requires sudo privileges)')
    
    args = parser.parse_args()
    
    try:
        runner = S2MetaProtocolRunner(args.config)
        
        # Override config with CLI arguments  
        runner.test_focus = args.test_focus
        runner.enable_cross_protocol = args.enable_cross_protocol
        runner.enable_mitm = args.enable_mitm
        
        # Run S2 Meta Protocol test
        results = await runner.run()
        
        print(f"\n🎉 S2 Meta协议测试完成! 结果已保存到 output/ 目录")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n⏹️ S2测试被用户中断")
        return None
    except Exception as e:
        print(f"\n❌ S2测试失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
