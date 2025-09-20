# -*- coding: utf-8 -*-
"""
ACP Privacy Testing Runner
Implements privacy testing using ACP (Agent Communication Protocol).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add current directory to path for imports
HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent
sys.path.insert(0, str(SAFETY_TECH))

from runners.runner_base import RunnerBase

# Import ACP components
try:
    from protocol_backends.acp import (
        ACPCommBackend,
        ACPReceptionistExecutor,
        ACPNosyDoctorExecutor,
        ACPPrivacySimulator,
        ACPPrivacyAnalyzer
    )
    from core.network_base import NetworkBase
    from core.privacy_analyzer_base import PrivacyAnalyzerBase
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed")
    sys.exit(1)


class ExtendedACPPrivacyAnalyzer(PrivacyAnalyzerBase):
    """Extended ACP analyzer that can handle direct conversation data."""
    
    def extract_conversation_data(self, conversation_file: str) -> List[Dict[str, Any]]:
        """Extract conversation data from ACP conversation file format."""
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                acp_data = json.load(f)
            
            conversations = acp_data.get("conversations", [])
            standardized_conversations = []
            
            for conv in conversations:
                standardized_conv = self._standardize_conversation_data(conv)
                if standardized_conv:
                    standardized_conversations.append(standardized_conv)
            
            return standardized_conversations
            
        except Exception as e:
            self._log(f"Error extracting conversation data: {e}")
            raise
    
    def analyze_conversations_from_data(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conversations from direct data structure."""
        try:
            conversations = conversation_data.get("conversations", [])
            
            standardized_conversations = []
            for conv in conversations:
                standardized_conv = self._standardize_conversation_data(conv)
                if standardized_conv:
                    standardized_conversations.append(standardized_conv)
            
            # Use the standardized data for analysis
            return self._analyze_standardized_conversations(standardized_conversations)
            
        except Exception as e:
            self._log(f"Error analyzing conversation data: {e}")
            raise

    def _standardize_conversation_data(self, conv_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Standardize conversation data from simulator."""
        try:
            messages = []
            
            # Convert simulator message format to standard format
            for msg in conv_data.get("messages", []):
                messages.append({
                    "sender": msg.get("sender", "unknown"),
                    "content": msg.get("message", ""),
                    "timestamp": str(msg.get("timestamp", ""))
                })
            
            return {
                "conversation_id": conv_data.get("conversation_id", "unknown"),
                "messages": messages,
                "metadata": {
                    "protocol": "acp",
                    "original_question": conv_data.get("original_question", "")
                }
            }
            
        except Exception as e:
            self._log(f"Error standardizing conversation: {e}")
            return None

    def _analyze_standardized_conversations(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze standardized conversations."""
        
        results = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_conversations": len(conversations),
                "protocol": "acp",
                "analyzer_config": {
                    "weights": {
                        "ssn": self.ssn_weight,
                        "phone": self.phone_weight,
                        "address": self.address_weight,
                        "age": self.age_weight,
                        "name": self.name_weight
                    }
                }
            },
            "conversation_analyses": [],
            "summary": {}
        }
        
        total_violations = {"ssn": 0, "phone": 0, "address": 0, "age": 0, "name": 0}
        total_privacy_score = 0
        
        for conversation in conversations:
            # Combine all messages in conversation
            full_text = " ".join([msg["content"] for msg in conversation["messages"]])
            
            analysis = self.analyze_conversation_text(full_text)
            analysis["conversation_id"] = conversation["conversation_id"]
            analysis["message_count"] = len(conversation["messages"])
            
            results["conversation_analyses"].append(analysis)
            
            # Accumulate for summary
            for vtype in total_violations:
                total_violations[vtype] += len(analysis["violations"][vtype])
            total_privacy_score += analysis["privacy_score"]
        
        # Calculate summary statistics
        avg_privacy_score = total_privacy_score / len(conversations) if conversations else 0
        
        results["summary"] = {
            "total_violations": total_violations,
            "total_violation_instances": sum(total_violations.values()),
            "average_privacy_score": avg_privacy_score,
            "overall_privacy_grade": self._get_privacy_grade(avg_privacy_score),
            "conversations_by_grade": self._count_conversations_by_grade(results["conversation_analyses"])
        }
        
        return results


class ACPRunner(RunnerBase):
    """ACP-specific privacy testing runner."""

    def __init__(self, config_path: str = "config_acp.yaml"):
        # Pass bare filename; RunnerBase resolves to configs dir
        super().__init__(config_path)
        self._backend: Optional[ACPCommBackend] = None
        self._simulator: Optional[ACPPrivacySimulator] = None
        self._analyzer: Optional[ExtendedACPPrivacyAnalyzer] = None

    async def create_network(self) -> NetworkBase:
        """Create ACP network with communication backend."""
        try:
            self._backend = ACPCommBackend()
            network = NetworkBase(comm_backend=self._backend)
            self.output.success("ACP network infrastructure created")
            return network
        except Exception as e:
            self.output.error(f"Failed to create ACP network: {e}")
            raise

    async def setup_agents(self) -> Dict[str, Any]:
        """Setup ACP agents for privacy testing."""
        try:
            agents = {}
            
            # Create privacy simulator
            self._simulator = ACPPrivacySimulator(self.config, self.output)
            
            # For ACP, we use direct agent simulation (most reliable)
            receptionist_id = "ACP_Receptionist"
            doctor_id = "ACP_Doctor"
            
            # Register agents in network (using mock endpoints for direct simulation)
            await self.network.register_agent(receptionist_id, f"acp://localhost:8001")
            await self.network.register_agent(doctor_id, f"acp://localhost:8002")
            
            # Set network for simulator
            self._simulator.set_network(self.network)
            
            agents[receptionist_id] = self._simulator.receptionist
            agents[doctor_id] = self._simulator.doctor
            
            self.output.success(f"ACP agents configured: {list(agents.keys())}")
            return agents
            
        except Exception as e:
            self.output.error(f"Failed to setup ACP agents: {e}")
            raise

    async def run_privacy_test(self) -> Dict[str, Any]:
        """Run ACP privacy protection test."""
        try:
            # Load enhanced dataset
            enhanced_questions = self.load_enhanced_dataset()
            
            # Get test parameters
            general_config = self.config.get("general", {})
            num_conversations = min(
                general_config.get("num_conversations", 5),
                len(enhanced_questions)
            )
            max_rounds = general_config.get("max_rounds", 3)
            
            self.output.info(f"Running {num_conversations} conversations with {max_rounds} rounds each")
            
            # Select questions for testing
            test_questions = enhanced_questions[:num_conversations]
            
            # Run privacy test batch using simulator
            conversation_results = await self._simulator.run_privacy_test_batch(
                test_questions, 
                max_rounds
            )
            
            # Save conversation data with smart path
            conv_output_file = self._get_output_path("agent_conversations_acp.json")
            self.save_conversation_data(conversation_results, conv_output_file)
            
            # Analyze privacy violations with extended analyzer
            self._analyzer = ExtendedACPPrivacyAnalyzer(self.config, self.output)
            analysis_results = self._analyzer.analyze_conversations_from_data(conversation_results)
            
            # Save analysis results
            analysis_output_file = self._get_output_path("privacy_analysis_acp.json")
            self._analyzer.save_analysis_results(analysis_results, analysis_output_file)
            
            # Generate report
            report = self._analyzer.generate_privacy_report(analysis_results)
            report_output_file = self._get_output_path("detailed_privacy_report_acp.txt")
            with open(report_output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.output.success(f"Conversation data saved: {conv_output_file}")
            self.output.success(f"Analysis results saved: {analysis_output_file}")
            self.output.success(f"Privacy report saved: {report_output_file}")
            
            return analysis_results
            
        except Exception as e:
            self.output.error(f"Privacy test failed: {e}")
            raise

    def save_conversation_data(self, conversation_data: Dict[str, Any], output_file: str) -> None:
        """Save ACP conversation data"""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.output.error(f"Failed to save conversation data: {e}")
            raise

    def display_results(self, results: Dict[str, Any]) -> None:
        """Display ACP-specific results summary."""
        try:
            self.output.info("📊 Privacy Testing Results Summary")
            self.output.progress("=" * 50)
            
            # ACP analyzer returns "summary" in results
            summary = results.get("summary", {})
            metadata = results.get("analysis_metadata", {})
            
            # Basic info
            total_convs = metadata.get('total_conversations', 0)
            self.output.progress(f"Total Conversations: {total_convs}")
            self.output.progress(f"Average Privacy Score: {summary.get('average_privacy_score', 0):.2f}/100")
            self.output.progress(f"Privacy Grade: {summary.get('overall_privacy_grade', 'Unknown')}")
            
            # Violations
            violations = summary.get("total_violations", {})
            total_violations = summary.get("total_violation_instances", 0)
            self.output.progress(f"Total Privacy Violations: {total_violations}")
            
            if total_violations > 0:
                self.output.warning("Privacy violations detected:")
                for vtype, count in violations.items():
                    if count > 0:
                        self.output.progress(f"  {vtype.upper()}: {count}")
            else:
                self.output.success("✅ No privacy violations detected!")
                
        except Exception as e:
            self.output.error(f"Error displaying results: {e}")
            # Fall back to base class method
            super().display_results(results)


async def main():
    """Main entry point for ACP privacy testing."""
    try:        
        runner = ACPRunner()
        await runner.run()
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())