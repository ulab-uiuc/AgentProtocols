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
            
            # Save conversation data with smart path
            conv_output_file = self._get_output_path("agent_conversations_anp_integrated.json")
            self.save_conversation_data(conversation_data, conv_output_file)
            
            # Analyze privacy using enhanced analyzer
            self._analyzer = ANPPrivacyAnalyzer(self.config, self.output)
            analysis_results = self._analyzer.analyze_conversations_from_data(conversation_data)
            
            # Save analysis results
            analysis_output_file = self._get_output_path("privacy_analysis_anp_integrated.json")
            self._analyzer.save_analysis_results(analysis_results, analysis_output_file)
            
            # Generate report
            report = self._analyzer.generate_privacy_report(analysis_results)
            report_output_file = self._get_output_path("detailed_privacy_report_anp_integrated.txt")
            with open(report_output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.output.success(f"Conversation data saved: {conv_output_file}")
            self.output.success(f"Analysis results saved: {analysis_output_file}")
            self.output.success(f"Privacy report saved: {report_output_file}")
            
            return analysis_results
            
        except Exception as e:
            self.output.error(f"Failed to run ANP privacy test: {e}")
            raise

    def save_conversation_data(self, conversation_data: Dict[str, Any], output_file: str) -> None:
        """Save ANP conversation data"""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.output.error(f"Failed to save conversation data: {e}")
            raise

    def display_results(self, results: Dict[str, Any]) -> None:
        """Display ANP-specific results summary."""
        try:
            self.output.info("📊 Privacy Testing Results Summary")
            self.output.progress("=" * 50)
            
            # ANP analyzer returns "analysis_summary", not "summary"
            summary = results.get("analysis_summary", {})
            scores = results.get("privacy_scores", {})
            anp_metrics = results.get("anp_metrics", {})
            
            # Basic info
            total_convs = results.get('total_conversations', summary.get('total_conversations', 0))
            self.output.progress(f"Total Conversations: {total_convs}")
            self.output.progress(f"Average Privacy Score: {scores.get('final_score', 0.0):.2f}/100")
            self.output.progress(f"Privacy Grade: {scores.get('grade', 'Unknown')}")
            
            # ANP-specific info
            self.output.progress(f"Agent Responses Analyzed: {anp_metrics.get('agent_responses_analyzed', 0)}")
            self.output.progress(f"Patient Inputs Ignored: {anp_metrics.get('patient_inputs_ignored', 0)}")
            
            # Violations (only from agent responses)
            violations = summary.get('violations_in_agent_responses', {})
            total_violations = summary.get('total_agent_violations', 0)
            self.output.progress(f"Total Privacy Violations: {total_violations}")
            
            if total_violations > 0:
                self.output.warning("Privacy violations detected:")
                for vtype, count in violations.items():
                    if count > 0:
                        self.output.progress(f"  {vtype.upper()}: {count}")
            else:
                self.output.success("✅ No privacy violations detected in agent responses!")
                
        except Exception as e:
            self.output.error(f"Error displaying results: {e}")
            # Fall back to base class method
            super().display_results(results)


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
