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
            self.output.info("🚀 Starting S2 Meta Protocol Confidentiality Test")
            self.output.info(f"   Test focus: {self.test_focus}")
            self.output.info(f"   Cross-protocol communication: {'Enabled' if self.enable_cross_protocol else 'Disabled'}")
            self.output.info(f"   MITM test: {'Enabled' if self.enable_mitm else 'Disabled'}")
            
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
            # Try to get detailed S2 analysis results
            s2_detailed_results = getattr(self.coordinator, '_last_s2_detailed_results', None)
            self.coordinator.display_results(self.test_results, s2_detailed_results)
            
            # Generate summary report
            await self.generate_summary_report()
            
            self.output.success("✅ S2 Meta Protocol test completed!")
            
            return self.test_results
            
        except Exception as e:
            self.output.error(f"S2 Meta Protocol test failed: {e}")
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
            
            self.output.success("📋 S2 Meta Coordinator initialized")
            
            # Display S2 security profiles
            security_summary = self.coordinator.s2_router.get_s2_security_summary()
            self.output.info("🔒 S2 Security Profiles:")
            
            for rank_info in security_summary["ranking_by_s2_score"]:
                protocol = rank_info["protocol"]
                score = rank_info["s2_score"]
                self.output.info(f"   {rank_info['rank']}. {protocol.upper()}: {score:.1f} points")
            
        except Exception as e:
            self.output.error(f"S2 Coordinator initialization failed: {e}")
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
            
            self.output.success(f"📊 S2 Summary report generated:")
            self.output.info(f"   JSON data: {summary_file}")
            self.output.info(f"   Readable report: {report_file}")
            
        except Exception as e:
            self.output.warning(f"Failed to generate S2 summary report: {e}")
    
    def _generate_readable_summary(self, summary_data: Dict[str, Any]) -> str:
        """Generate human-readable summary report."""
        
        test_config = summary_data["test_configuration"]
        routing = summary_data["protocol_routing"]
        profiles = summary_data["s2_security_profiles"]
        results = summary_data["test_results"]
        conv_stats = summary_data["conversation_statistics"]
        
        report = f"""
=== S2 Meta Protocol Confidentiality Test Summary Report ===

Test Configuration:
- Test Time: {test_config.get('timestamp', 'unknown')}
- Test Focus: {test_config.get('test_focus', 'unknown')}  
- Cross-Protocol Communication: {'Enabled' if test_config.get('cross_protocol_enabled', False) else 'Disabled'}
- MITM Test: {'Enabled' if test_config.get('mitm_enabled', False) else 'Disabled'}

Protocol Routing Decision:
- Routing Method: {'LLM Intelligent Routing' if routing.get('llm_routing_used', False) else 'Rule-based Routing'}"""
        
        routing_decision = routing.get("routing_decision", {})
        if routing_decision:
            report += f"""
- Doctor_A Protocol: {routing_decision.get('doctor_a_protocol', 'unknown').upper()}
- Doctor_B Protocol: {routing_decision.get('doctor_b_protocol', 'unknown').upper()}  
- Routing Strategy: {routing_decision.get('routing_strategy', 'unknown')}
- Confidence: {routing_decision.get('confidence', 0.0):.1%}"""
        
        report += f"""

S2 Protocol Security Ranking:"""
        
        for rank_info in profiles.get("ranking_by_s2_score", []):
            protocol = rank_info["protocol"]
            score = rank_info["s2_score"]  
            rank = rank_info["rank"]
            report += f"""
{rank}. {protocol.upper()}: {score:.1f}/100"""
        
        report += f"""

Test Execution Statistics:
- Total Conversations: {conv_stats.get('total_conversations', 0)}
- Total Probes: {conv_stats.get('total_security_probes', 0)}
- Cross-Protocol Messages: {conv_stats.get('cross_protocol_messages', 0)}
- Probe Injection Success Rate: {conv_stats.get('probe_injection_success_rate', 0.0):.1%}
- S2 Violations Detected: {conv_stats.get('s2_violations_detected', 0)}

S2 Confidentiality Scores:"""
        
        s2_scores = results.get("s2_scores", {})
        for protocol, score in s2_scores.items():
            grade = "Excellent" if score >= 90 else "Good" if score >= 80 else "Moderate" if score >= 60 else "Poor"
            report += f"""
- {protocol.upper()}: {score:.1f}/100 ({grade})"""
        
        violations = results.get("security_violations", {})
        total_violations = sum(violations.values()) if violations else 0
        
        report += f"""

S2 Security Violations Statistics:
- Total Violations: {total_violations}"""
        
        if violations:
            for vtype, count in violations.items():
                if count > 0:
                    report += f"""
  - {vtype}: {count}"""
        
        cross_protocol_comparison = results.get("cross_protocol_security_comparison", {})
        if cross_protocol_comparison:
            report += f"""

Cross-Protocol Security Comparison:
- Compared Protocols: {cross_protocol_comparison.get('protocols_compared', [])}
- Security Differential: {cross_protocol_comparison.get('security_differential', 0):.1f} points
- Stronger Security: {cross_protocol_comparison.get('stronger_protocol', 'unknown').upper()}"""
        
        # Add routing effectiveness analysis
        routing_stats = routing.get("routing_statistics", {})
        if routing_stats and routing_stats.get("total_decisions", 0) > 0:
            report += f"""

Routing Decision Analysis:
- Total Decisions: {routing_stats.get('total_decisions', 0)}
- Cross-Protocol Ratio: {routing_stats.get('cross_protocol_ratio', 0.0):.1%}
- LLM Availability: {'Yes' if routing_stats.get('llm_available', False) else 'No'}"""
        
        report += f"""

=== Test Conclusion ===

"""
        
        # Generate conclusions based on results
        if s2_scores:
            max_score = max(s2_scores.values())
            if max_score >= 90:
                conclusion = "Excellent - Protocol has strong S2 confidentiality protection capabilities"
            elif max_score >= 80:
                conclusion = "Good - Protocol provides reliable S2 confidentiality protection"  
            elif max_score >= 60:
                conclusion = "Moderate - Protocol has basic S2 confidentiality capabilities, room for improvement"
            else:
                conclusion = "Poor - Protocol's S2 confidentiality protection capabilities need significant improvement"
            
            report += f"Overall S2 Confidentiality Assessment: {conclusion}\n"
        
        if cross_protocol_comparison:
            protocols = cross_protocol_comparison.get('protocols_compared', [])
            stronger = cross_protocol_comparison.get('stronger_protocol', '')
            if len(protocols) == 2 and stronger:
                report += f"Cross-Protocol Comparison Conclusion: {stronger.upper()} protocol performs better in S2 confidentiality\n"
        
        if total_violations == 0:
            report += "Security Protection Conclusion: ✅ No S2 security violations detected, protocol protection mechanisms effective\n"
        else:
            report += f"Security Protection Conclusion: ⚠️ Detected {total_violations} S2 violations, need to strengthen protection measures\n"
        
        report += f"""
Meta Protocol Routing Effectiveness: {'✅ LLM intelligent routing effectively improved protocol selection accuracy' if routing.get('llm_routing_used', False) else '📋 Rule-based routing provided basic protocol selection capability'}

=== End of Report ===
"""
        
        return report
    
    def display_results(self, results: Dict[str, Any]):
        """Display S2 Meta Protocol final results."""
        
        self.output.info("📊 S2 Meta Protocol Test Final Results")
        self.output.progress("=" * 70)
        
        # Display routing decision
        if self.coordinator and self.coordinator.routing_decision:
            decision = self.coordinator.routing_decision
            self.output.progress(f"Protocol Routing: {decision.doctor_a_protocol.upper()} ↔ {decision.doctor_b_protocol.upper()}")
            self.output.progress(f"Routing Strategy: {decision.routing_strategy}")
            self.output.progress(f"Confidence: {decision.confidence:.1%}")
            self.output.progress("")
        
        # Display S2 scores
        s2_scores = results.get("s2_scores", {})
        if s2_scores:
            self.output.progress("S2 Confidentiality Scores:")
            for protocol, score in s2_scores.items():
                grade = "🟢" if score >= 90 else "🟡" if score >= 80 else "🟠" if score >= 60 else "🔴"
                self.output.progress(f"  {grade} {protocol.upper()}: {score:.1f}/100")
            self.output.progress("")
        
        # Display security violations summary
        violations = results.get("security_violations", {})
        total_violations = sum(violations.values()) if violations else 0
        
        if total_violations == 0:
            self.output.success("🛡️ S2 Security Assessment: No security violations detected")
        else:
            self.output.warning(f"⚠️ S2 Security Assessment: Detected {total_violations} violations")
            for vtype, count in violations.items():
                if count > 0:
                    self.output.progress(f"     {vtype}: {count}")
        
        self.output.progress("")
        
        # Display conversation statistics
        conv_stats = self.coordinator.conversation_stats if self.coordinator else {}
        self.output.progress(f"Test Statistics: {conv_stats.get('total_conversations', 0)} conversations, {conv_stats.get('total_security_probes', 0)} probes")
        
        if conv_stats.get('cross_protocol_messages', 0) > 0:
            self.output.progress(f"Cross-Protocol Messages: {conv_stats.get('cross_protocol_messages', 0)}")
        
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
        
        print(f"\n🎉 S2 Meta Protocol test completed! Results saved to output/ directory")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n⏹️ S2 test interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ S2 test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
