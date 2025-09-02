#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANP Privacy Protection Testing Runner
Enhanced version using the improved ANP privacy analyzer.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import base classes
from core.privacy_analyzer_base import PrivacyAnalyzerBase
from runners.runner_base import RunnerBase
from core.network_base import NetworkBase

# Import ANP components
from protocol_backends.anp import (
    ANPCommBackend, ANPPrivacyAnalyzer, ANPPrivacySimulator
)

# Try to import AgentConnect for integrated testing
try:
    from agent_connect.python.simple_node import SimpleNode, SimpleNodeSession
    import uvicorn
    ANP_SERVER_AVAILABLE = True
    print("[ANP Integrated] AgentConnect available for SimpleNode creation")
except ImportError:
    ANP_SERVER_AVAILABLE = False
    print("[ANP Integrated] AgentConnect not available - cannot run integrated nodes")


class ANPIntegratedRunner(RunnerBase):
    """ANP runner with integrated AgentConnect SimpleNodes for complete end-to-end testing"""

    def __init__(self, config_path: str = "config_anp.yaml"):
        # Pass bare filename; RunnerBase resolves to configs dir
        super().__init__(config_path)
        
        # Set data directory
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.network = None
        self._backend = None
        self._simulator = None
        self._analyzer = None

    async def create_network(self) -> NetworkBase:
        """Create ANP network with AgentConnect SimpleNodes"""
        try:
            if ANP_SERVER_AVAILABLE:
                # SimpleNodes will be created during agent setup
                self.output.success("ANP AgentConnect infrastructure ready")
            else:
                self.output.warning("AgentConnect not available - running in mock mode")
            
            # Initialize ANP communication backend
            self._backend = ANPCommBackend(self.config, self.output)
            network = NetworkBase(comm_backend=self._backend)
            
            self.output.success("ANP network infrastructure created")
            return network
        except Exception as e:
            self.output.error(f"Failed to create ANP network: {e}")
            raise

    async def setup_agents(self) -> Dict[str, Any]:
        """Setup ANP privacy testing agents with AgentConnect SimpleNodes"""
        try:
            agents = {}

            # Create ANP privacy simulator
            self._simulator = ANPPrivacySimulator(self.config, self.output)

            # Register ANP agents with SimpleNode endpoints
            receptionist_id = "ANP_Receptionist"
            doctor_id = "ANP_Doctor"

            # Register agents (SimpleNodes will be created in the backend)
            await self.network.register_agent(receptionist_id, "http://127.0.0.1:9001")
            await self.network.register_agent(doctor_id, "http://127.0.0.1:9002")

            # Wire the simulator with the network
            self._simulator.set_network(self.network)

            agents[receptionist_id] = self._simulator.receptionist
            agents[doctor_id] = self._simulator.doctor

            self.output.success(f"ANP agents configured: {list(agents.keys())}")
            return agents

        except Exception as e:
            self.output.error(f"Failed to setup ANP agents: {e}")
            raise

    async def run_privacy_test(self) -> Dict[str, Any]:
        """Run ANP privacy protection tests"""
        try:
            # Load enhanced dataset
            enhanced_questions = self.load_enhanced_dataset()
            
            # Get test parameters
            general_config = self.config.get("general", {})
            num_conversations = general_config.get("num_conversations", 5)
            max_rounds = general_config.get("max_rounds", 3)
            
            self.output.info(f"Loaded {len(enhanced_questions)} enhanced questions")
            self.output.info(f"Running {num_conversations} ANP conversations with {max_rounds} rounds each")
            
            # Run conversations using the simulator
            conversation_data = await self._simulator.run_privacy_conversations(
                enhanced_questions[:num_conversations], max_rounds
            )
            
            # Save conversation data
            conv_file = self.data_dir / "agent_conversations_anp_integrated.json"
            with open(conv_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            self.output.success(f"Conversation data saved: {conv_file}")
            
            # Analyze privacy using enhanced analyzer
            self._analyzer = ANPPrivacyAnalyzer(self.config, self.output)
            analysis_results = self._analyzer.analyze_conversations_from_data(conversation_data)
            
            # Save analysis results
            analysis_file = self.data_dir / "privacy_analysis_anp_integrated.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
            
            self.output.success(f"Analysis results saved: {analysis_file}")
            
            # Generate detailed privacy report
            report_file = self.data_dir / "detailed_privacy_report_anp_integrated.txt"
            self.generate_privacy_report(analysis_results, report_file)
            self.output.success(f"Privacy report saved: {report_file}")
            
            return analysis_results
            
        except Exception as e:
            self.output.error(f"Failed to run ANP privacy test: {e}")
            raise

    def generate_privacy_report(self, analysis_results: Dict[str, Any], report_file: Path):
        """Generate detailed privacy report for ANP"""
        try:
            summary = analysis_results.get("analysis_summary", {})
            scores = analysis_results.get("privacy_scores", {})
            anp_metrics = analysis_results.get("anp_metrics", {})
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ANP PRIVACY PROTECTION ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Conversations: {summary.get('total_conversations', 0)}\n")
                f.write(f"Analysis Focus: {summary.get('analysis_focus', 'agent_responses_only')}\n")
                f.write("\n")
                
                f.write("OVERALL PRIVACY ASSESSMENT\n")
                f.write("-" * 40 + "\n")
                f.write(f"Final Privacy Score: {scores.get('final_score', 0.0):.2f}/100\n")
                f.write(f"Base Score (before ANP bonuses): {scores.get('average_base_score', 0.0):.2f}/100\n")
                f.write(f"ANP Protocol Bonus: +{scores.get('average_anp_bonus', 0.0):.2f}\n")
                f.write(f"Overall Privacy Grade: {scores.get('grade', 'Critical')}\n")
                f.write("\n")
                
                f.write("ANP PRIVACY FEATURES\n")
                f.write("-" * 40 + "\n")
                anp_effectiveness = scores.get('anp_effectiveness', {})
                f.write(f"DID Authentication: {anp_effectiveness.get('did_authentication_rate', 1.0)*100:.1f}% coverage\n")
                f.write(f"Privacy Protection Rate: {anp_effectiveness.get('privacy_protection_rate', 0.0)*100:.1f}%\n")
                f.write(f"Data Minimization Rate: {anp_effectiveness.get('data_minimization_rate', 0.0)*100:.1f}%\n")
                f.write(f"Agent Responses Analyzed: {anp_metrics.get('agent_responses_analyzed', 0)}\n")
                f.write(f"Patient Inputs Ignored: {anp_metrics.get('patient_inputs_ignored', 0)}\n")
                f.write("\n")
                
                f.write("VIOLATION SUMMARY (Agent Responses Only)\n")
                f.write("-" * 40 + "\n")
                violations = summary.get('violations_in_agent_responses', {})
                f.write(f"SSN Violations: {violations.get('SSN', 0)}\n")
                f.write(f"Phone Violations: {violations.get('PHONE', 0)}\n")
                f.write(f"Address Violations: {violations.get('ADDRESS', 0)}\n")
                f.write(f"Age Violations: {violations.get('AGE', 0)}\n")
                f.write(f"Name Violations: {violations.get('NAME', 0)}\n")
                f.write(f"Total Agent Violations: {summary.get('total_agent_violations', 0)}\n")
                f.write("\n")
                
                f.write("PRIVACY PROTECTION EFFECTIVENESS\n")
                f.write("-" * 40 + "\n")
                if summary.get('total_agent_violations', 0) == 0:
                    f.write("✅ Excellent: No privacy violations detected in agent responses\n")
                    f.write("✅ ANP privacy protection mechanisms working effectively\n")
                else:
                    f.write(f"⚠️  {summary.get('total_agent_violations', 0)} privacy violations found in agent responses\n")
                    f.write("🔧 ANP privacy filters may need adjustment\n")
                f.write("\n")
                
                f.write("=" * 80 + "\n")
                
        except Exception as e:
            self.output.error(f"Error generating privacy report: {e}")
            raise

    def display_results_summary(self, analysis_results: Dict[str, Any]):
        """Display enhanced results summary for ANP"""
        try:
            summary = analysis_results.get("analysis_summary", {})
            scores = analysis_results.get("privacy_scores", {})
            anp_metrics = analysis_results.get("anp_metrics", {})
            
            self.output.info("📊 ANP Privacy Testing Results Summary")
            self.output.info("=" * 50)
            self.output.info(f"Analysis Focus: {summary.get('analysis_focus', 'agent_responses_only')}")
            self.output.info(f"Total Conversations: {summary.get('total_conversations', 0)}")
            self.output.info(f"Agent Responses Analyzed: {anp_metrics.get('agent_responses_analyzed', 0)}")
            self.output.info(f"Patient Inputs Ignored: {anp_metrics.get('patient_inputs_ignored', 0)}")
            self.output.info(f"Final Privacy Score: {scores.get('final_score', 0.0):.2f}/100")
            self.output.info(f"Privacy Grade: {scores.get('grade', 'Critical')}")
            
            # ANP Features
            anp_effectiveness = scores.get('anp_effectiveness', {})
            self.output.info(f"ANP Privacy Protection Rate: {anp_effectiveness.get('privacy_protection_rate', 0.0)*100:.1f}%")
            self.output.info(f"ANP Data Minimization Rate: {anp_effectiveness.get('data_minimization_rate', 0.0)*100:.1f}%")
            self.output.info(f"ANP Protocol Bonus: +{scores.get('average_anp_bonus', 0.0):.2f}")
            
            # Violations (only from agent responses)
            violations = summary.get('violations_in_agent_responses', {})
            total_violations = summary.get('total_agent_violations', 0)
            
            if total_violations > 0:
                self.output.warning("⚠️  Privacy violations detected in agent responses:")
                for violation_type, count in violations.items():
                    if count > 0:
                        self.output.warning(f"     {violation_type}: {count}")
            else:
                self.output.success("✅ No privacy violations detected in agent responses!")
                self.output.success("✅ ANP privacy protection working effectively!")
            
        except Exception as e:
            self.output.error(f"Error displaying results: {e}")


async def main():
    """Main entry point for ANP privacy testing"""
    try:
        runner = ANPIntegratedRunner()
        await runner.run()
    except Exception as e:
        print(f"❌ ANP integrated testing failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
