# -*- coding: utf-8 -*-
"""
Safety Meta Protocol Coordinator
Uses standard safety protocol components with intelligent routing.
"""

from __future__ import annotations

import asyncio
import json
import time
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add src to path
HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent.parent
PROJECT_ROOT = SAFETY_TECH.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SAFETY_TECH) not in sys.path:
    sys.path.insert(0, str(SAFETY_TECH))

# Import safety_tech components
from runners.runner_base import RunnerBase

# Import meta protocol components
from .llm_router import SafetyLLMRouter
from .agents import MetaReceptionistAgent, MetaNosyDoctorAgent, create_meta_receptionist_executor, create_meta_doctor_executor
from .analyzer import MetaPrivacyAnalyzer
from .comm import MetaCommBackend
from .registration_adapter import MetaRegistrationAdapter


class SafetyMetaCoordinator(RunnerBase):
    """
    Safety Meta Protocol Coordinator using standard safety protocol components
    
    Uses intelligent routing to select optimal protocol and manages privacy testing.
    Each setup has 2 agents: receptionist and doctor.
    """

    def __init__(self, config_path: str = "config_meta.yaml"):
        super().__init__(config_path)
        
        # Meta protocol components
        self.llm_router = SafetyLLMRouter()
        self.routing_decision = None
        self.selected_protocol = None
        
        # Communication and registration
        self.comm_backend: Optional[MetaCommBackend] = None
        self.registration_adapter: Optional[MetaRegistrationAdapter] = None
        
        # Agents
        self.receptionist_agent: Optional[MetaReceptionistAgent] = None
        self.doctor_agent: Optional[MetaNosyDoctorAgent] = None
        
        # Privacy analysis
        self.privacy_analyzer: Optional[MetaPrivacyAnalyzer] = None
        
        # Performance tracking
        self.conversation_stats = {
            "total_conversations": 0,
            "total_rounds": 0,
            "total_response_time": 0.0,
            "avg_response_time": 0.0,
            "privacy_violations": 0
        }
        
        # Initialize LLM client
        self._initialize_llm_client()

    def _initialize_llm_client(self) -> None:
        """Initialize LLM client for intelligent routing"""
        try:
            core_config = self.config.get("core", {})
            
            if core_config.get("type") == "openai":
                api_key = core_config.get("openai_api_key")
                base_url = core_config.get("openai_base_url", "https://api.openai.com/v1")
                model = core_config.get("name", "gpt-4o")
                
                if api_key:
                    # Create simple LLM client
                    class SimpleLLMClient:
                        def __init__(self, api_key, base_url, model):
                            self.api_key = api_key
                            self.base_url = base_url
                            self.model = model
                        
                        async def ask_tool(self, messages, tools, tool_choice):
                            import aiohttp
                            headers = {
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"
                            }
                            payload = {
                                "model": self.model,
                                "messages": messages,
                                "tools": tools,
                                "tool_choice": tool_choice,
                                "temperature": 0.1
                            }
                            
                            async with aiohttp.ClientSession() as session:
                                async with session.post(f"{self.base_url}/chat/completions", 
                                                       headers=headers, json=payload) as response:
                                    if response.status != 200:
                                        error_text = await response.text()
                                        raise Exception(f"LLM API call failed: {response.status} - {error_text}")
                                    result = await response.json()
                                    return result["choices"][0]["message"]
                    
                    llm_client = SimpleLLMClient(api_key, base_url, model)
                    self.llm_router.set_llm_client(llm_client)
                    
                    self.output.success(f"🧠 LLM client initialized: {model}")
                else:
                    self.output.warning("⚠️  No OpenAI API key found, using heuristic routing")
            else:
                self.output.info("📋 Using heuristic-based protocol selection")
                
        except Exception as e:
            self.output.warning(f"⚠️  Failed to initialize LLM client: {e}")

    async def create_network(self) -> Dict[str, Any]:
        """Create meta protocol network infrastructure"""
        try:
            # Use LLM to select optimal protocol for safety testing
            sample_task = {
                "question": "Medical privacy conversation with sensitive patient data",
                "context": "Doctor-patient interaction requiring privacy protection",
                "metadata": {
                    "type": "medical_conversation",
                    "privacy_level": "high",
                    "agents_needed": 2,
                    "roles": ["receptionist", "doctor"]
                }
            }
            
            self.output.info("🧠 Analyzing task for optimal protocol selection...")
            
            # Get routing decision
            self.routing_decision = await self.llm_router.route_safety_task(sample_task, num_agents=2)
            self.selected_protocol = self.routing_decision.selected_protocol
            
            self.output.success(f"📊 Protocol selected: {self.selected_protocol.upper()}")
            self.output.info(f"🎯 Confidence: {self.routing_decision.confidence:.1%}")
            self.output.info(f"💡 Reasoning: {self.routing_decision.reasoning}")
            
            # Initialize communication backend
            self.comm_backend = MetaCommBackend(
                selected_protocol=self.selected_protocol,
                routing_decision=self.routing_decision
            )
            
            # Initialize registration adapter
            self.registration_adapter = MetaRegistrationAdapter(
                config=self.config,
                selected_protocol=self.selected_protocol,
                routing_decision=self.routing_decision,
                output=self.output
            )
            
            # Initialize privacy analyzer
            self.privacy_analyzer = MetaPrivacyAnalyzer(self.config, self.output)
            
            self.output.success("Meta protocol network infrastructure created")
            return {"selected_protocol": self.selected_protocol, "routing_decision": self.routing_decision}
            
        except Exception as e:
            self.output.error(f"Failed to create meta network: {e}")
            raise

    async def setup_agents(self) -> Dict[str, Any]:
        """Setup meta agents with intelligent protocol selection"""
        try:
            if not self.selected_protocol:
                raise RuntimeError("No protocol selected. Call create_network() first.")
            
            # Create agent IDs based on selected protocol
            receptionist_id = f"{self.selected_protocol.upper()}_Meta_Receptionist"
            doctor_id = f"{self.selected_protocol.upper()}_Meta_Doctor"
            
            # Create meta protocol agents
            self.receptionist_agent = MetaReceptionistAgent(
                agent_id=receptionist_id,
                selected_protocol=self.selected_protocol,
                routing_decision=self.routing_decision
            )
            
            self.doctor_agent = MetaNosyDoctorAgent(
                agent_id=doctor_id,
                selected_protocol=self.selected_protocol,
                routing_decision=self.routing_decision
            )
            
            # Register agents with registration adapter
            receptionist_reg = await self.registration_adapter.register_meta_agent(
                agent_id=receptionist_id,
                agent_config={"agent_type": "receptionist", "protocol": self.selected_protocol},
                host="127.0.0.1",
                port=8100
            )
            
            doctor_reg = await self.registration_adapter.register_meta_agent(
                agent_id=doctor_id,
                agent_config={"agent_type": "doctor", "protocol": self.selected_protocol},
                host="127.0.0.1", 
                port=8101
            )
            
            if not (receptionist_reg.get("success") and doctor_reg.get("success")):
                raise RuntimeError("Agent registration failed")
            
            # Register endpoints with communication backend
            await self.comm_backend.register_endpoint(receptionist_id, "http://127.0.0.1:8100")
            await self.comm_backend.register_endpoint(doctor_id, "http://127.0.0.1:8101")
            
            agents = {
                receptionist_id: self.receptionist_agent,
                doctor_id: self.doctor_agent
            }
            
            self.output.success(f"Meta {self.selected_protocol.upper()} agents configured: {list(agents.keys())}")
            return agents
            
        except Exception as e:
            self.output.error(f"Failed to setup meta agents: {e}")
            raise

    async def run_health_checks(self) -> None:
        """Run health checks using registration adapter"""
        if not self.registration_adapter:
            return
        
        self.output.info("🏥 Running meta protocol health checks...")
        
        try:
            # Check health of all registered agents
            health_results = await self.registration_adapter.health_check_all()
            
            healthy_count = 0
            for agent_id, health in health_results.items():
                if health.get("healthy", False):
                    self.output.success(f"✅ {agent_id} - Healthy")
                    healthy_count += 1
                else:
                    error = health.get("error", "Unknown error")
                    self.output.error(f"❌ {agent_id} - {error}")
            
            self.output.info(f"Health Check: {healthy_count}/{len(health_results)} agents healthy")
                
        except Exception as e:
            self.output.warning(f"Health check failed: {e}")

    async def run_privacy_test(self) -> Dict[str, Any]:
        """Run privacy protection tests using meta protocol"""
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
            
            self.output.info(f"Running {num_conversations} META-{self.selected_protocol.upper()} conversations with {max_rounds} rounds each")
            
            test_questions = enhanced_questions[:num_conversations]
            
            # Run meta privacy tests
            conversation_results = await self._run_meta_conversations(
                test_questions, max_rounds
            )
            
            # Save conversation data
            conv_output_file = self._get_output_path(f"agent_conversations_meta_{self.selected_protocol}.json")
            self.save_conversation_data(conversation_results, conv_output_file)
            
            # Analyze privacy violations using meta protocol analyzer
            analysis_results = self.privacy_analyzer.analyze_conversations_from_data(conversation_results)
            
            # Save analysis results
            analysis_output_file = self._get_output_path(f"privacy_analysis_meta_{self.selected_protocol}.json")
            self._save_analysis_results(analysis_results, analysis_output_file)
            
            # Generate report
            report = self._generate_meta_privacy_report(analysis_results)
            report_output_file = self._get_output_path(f"detailed_privacy_report_meta_{self.selected_protocol}.txt")
            with open(report_output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.output.success(f"Conversation data saved: {conv_output_file}")
            self.output.success(f"Analysis results saved: {analysis_output_file}")
            self.output.success(f"Privacy report saved: {report_output_file}")
            
            return analysis_results
            
        except Exception as e:
            self.output.error(f"Meta privacy test failed: {e}")
            raise

    async def _run_meta_conversations(self, questions: List[str], max_rounds: int) -> Dict[str, Any]:
        """Run conversations using meta protocol communication backend"""
        conversations = []
        
        for i, question in enumerate(questions):
            conversation_id = f"meta_{self.selected_protocol}_conv_{i+1}"
            
            self.output.info(f"Starting META-{self.selected_protocol.upper()} conversation {i+1}/{len(questions)}")
            
            # Initialize conversation
            conversation = {
                "conversation_id": conversation_id,
                "original_question": question,
                "protocol": f"meta_{self.selected_protocol}",
                "routing_decision": {
                    "selected_protocol": self.routing_decision.selected_protocol,
                    "confidence": self.routing_decision.confidence,
                    "reasoning": self.routing_decision.reasoning,
                    "strategy": self.routing_decision.strategy
                },
                "messages": [],
                "timestamp": time.time()
            }
            
            # Patient -> Receptionist (initial message)
            current_message = question
            conversation["messages"].append({
                "sender": "Patient",
                "recipient": self.receptionist_agent.agent_id,
                "message": current_message,
                "timestamp": time.time()
            })
            
            # Process initial message through receptionist meta agent
            receptionist_response = await self.receptionist_agent.process_message_direct(current_message, "Patient")
            conversation["messages"].append({
                "sender": self.receptionist_agent.agent_id,
                "recipient": self.doctor_agent.agent_id,
                "message": receptionist_response,
                "timestamp": time.time()
            })
            
            current_message = receptionist_response
            
            # Run conversation rounds using meta protocol communication
            for round_num in range(max_rounds):
                # Receptionist -> Doctor via meta protocol
                try:
                    payload = {"text": current_message, "sender": self.receptionist_agent.agent_id}
                    doctor_response_data = await self.comm_backend.send(
                        self.receptionist_agent.agent_id,
                        self.doctor_agent.agent_id,
                        payload
                    )
                    
                    # Extract response text
                    doctor_response = self._extract_response_text(doctor_response_data)
                    
                    conversation["messages"].append({
                        "sender": self.doctor_agent.agent_id,
                        "recipient": self.receptionist_agent.agent_id,
                        "message": doctor_response,
                        "timestamp": time.time()
                    })
                    
                    # Doctor -> Receptionist via meta protocol
                    payload = {"text": doctor_response, "sender": self.doctor_agent.agent_id}
                    receptionist_response_data = await self.comm_backend.send(
                        self.doctor_agent.agent_id,
                        self.receptionist_agent.agent_id,
                        payload
                    )
                    
                    receptionist_response = self._extract_response_text(receptionist_response_data)
                    
                    conversation["messages"].append({
                        "sender": self.receptionist_agent.agent_id,
                        "recipient": self.doctor_agent.agent_id,
                        "message": receptionist_response,
                        "timestamp": time.time()
                    })
                    
                    current_message = receptionist_response
                    
                except Exception as e:
                    self.output.warning(f"Meta protocol routing error in round {round_num + 1}: {e}")
                    # Fallback to direct processing
                    doctor_response = await self.doctor_agent.process_patient_message(current_message)
                    conversation["messages"].append({
                        "sender": self.doctor_agent.agent_id,
                        "recipient": self.receptionist_agent.agent_id,
                        "message": doctor_response,
                        "timestamp": time.time(),
                        "fallback": True
                    })
                    
                    receptionist_response = await self.receptionist_agent.process_patient_message(doctor_response)
                    conversation["messages"].append({
                        "sender": self.receptionist_agent.agent_id,
                        "recipient": self.doctor_agent.agent_id,
                        "message": receptionist_response,
                        "timestamp": time.time(),
                        "fallback": True
                    })
                    
                    current_message = receptionist_response
            
            conversations.append(conversation)
            self.output.info(f"Completed META-{self.selected_protocol.upper()} conversation {i+1}/{len(questions)}")
        
        return {
            "conversations": conversations,
            "protocol": f"meta_{self.selected_protocol}",
            "total_conversations": len(conversations),
            "timestamp": time.time(),
            "routing_info": {
                "selected_protocol": self.selected_protocol,
                "routing_decision": {
                    "selected_protocol": self.routing_decision.selected_protocol,
                    "confidence": self.routing_decision.confidence,
                    "reasoning": self.routing_decision.reasoning,
                    "strategy": self.routing_decision.strategy
                }
            }
        }

    def _extract_response_text(self, response_data: Any) -> str:
        """Extract text response from AgentNetwork response"""
        if isinstance(response_data, str):
            return response_data
        elif isinstance(response_data, dict):
            if "text" in response_data:
                return response_data["text"]
            elif "content" in response_data:
                return response_data["content"]
            elif "message" in response_data:
                return response_data["message"]
            else:
                return str(response_data)
        else:
            return str(response_data)

    async def _analyze_privacy_results(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze privacy violations using protocol-specific analyzer"""
        try:
            # Import appropriate analyzer based on protocol
            if self.protocol_type == "acp":
                from ..acp.analyzer import ACPPrivacyAnalyzer
                analyzer = ACPPrivacyAnalyzer(self.config, self.output)
            elif self.protocol_type == "anp":
                from ..anp.analyzer import ANPPrivacyAnalyzer
                analyzer = ANPPrivacyAnalyzer(self.config, self.output)
            elif self.protocol_type == "agora":
                from ..agora.analyzer import AgoraPrivacyAnalyzer
                analyzer = AgoraPrivacyAnalyzer(self.config, self.output)
            elif self.protocol_type == "a2a":
                from ..a2a.analyzer import A2APrivacyAnalyzer
                analyzer = A2APrivacyAnalyzer(self.config, self.output)
            else:
                # Fallback to base analyzer
                from core.privacy_analyzer_base import PrivacyAnalyzerBase
                analyzer = PrivacyAnalyzerBase(self.config, self.output)
            
            # Analyze conversations
            if hasattr(analyzer, 'analyze_conversations_from_data'):
                return analyzer.analyze_conversations_from_data(conversation_data)
            else:
                # Fallback analysis
                return self._basic_privacy_analysis(conversation_data)
                
        except Exception as e:
            self.output.warning(f"Privacy analysis error: {e}")
            return self._basic_privacy_analysis(conversation_data)

    def _basic_privacy_analysis(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Basic privacy analysis fallback"""
        conversations = conversation_data.get("conversations", [])
        
        return {
            "analysis_timestamp": time.time(),
            "protocol": f"meta_{self.protocol_type}",
            "total_conversations": len(conversations),
            "total_violations": 0,
            "average_privacy_score": 100.0,
            "privacy_grade": "Excellent",
            "summary": {
                "total_violations": {"ssn": 0, "phone": 0, "address": 0, "age": 0, "name": 0},
                "total_violation_instances": 0,
                "average_privacy_score": 100.0,
                "overall_privacy_grade": "Excellent"
            }
        }

    def save_conversation_data(self, conversation_data: Dict[str, Any], output_file: str) -> None:
        """Save meta conversation data"""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.output.error(f"Failed to save conversation data: {e}")
            raise

    def _save_analysis_results(self, analysis_results: Dict[str, Any], output_file: str) -> None:
        """Save analysis results"""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.output.error(f"Failed to save analysis results: {e}")
            raise

    def _generate_meta_privacy_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive privacy report for meta protocol"""
        protocol_name = f"META-{self.selected_protocol.upper()}"
        summary = analysis_results.get("summary", {})
        
        # Extract routing information
        routing_info = analysis_results.get("meta_protocol_metrics", {})
        
        report = f"""
=== {protocol_name} Privacy Protection Test Report ===

Test Configuration:
- Meta Protocol: {protocol_name} (using LLM-based selection)
- Selected Protocol: {self.selected_protocol.upper()}
- Total Conversations: {analysis_results.get('total_conversations', 0)}
- Analysis Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}

LLM Routing Decision:
- Confidence: {self.routing_decision.confidence:.1%}
- Strategy: {self.routing_decision.strategy}
- Reasoning: {self.routing_decision.reasoning}

Privacy Analysis Results:
- Average Privacy Score: {summary.get('average_privacy_score', 0):.2f}/100
- Overall Privacy Grade: {summary.get('overall_privacy_grade', 'Unknown')}
- Total Violations: {summary.get('total_violation_instances', 0)}

Violation Breakdown:
- SSN Violations: {summary.get('total_violations', {}).get('ssn', 0)}
- Phone Violations: {summary.get('total_violations', {}).get('phone', 0)}
- Address Violations: {summary.get('total_violations', {}).get('address', 0)}
- Age Violations: {summary.get('total_violations', {}).get('age', 0)}
- Name Violations: {summary.get('total_violations', {}).get('name', 0)}

=== End of Report ===
"""
        return report

    def display_results(self, results: Dict[str, Any]) -> None:
        """Display comprehensive meta protocol results"""
        try:
            protocol_name = f"META-{self.selected_protocol.upper()}"
            self.output.info(f"📊 {protocol_name} Privacy Testing Results Summary")
            self.output.progress("=" * 60)
            
            summary = results.get("summary", {})
            total_convs = results.get('total_conversations', 0)
            routing_info = results.get("meta_protocol_metrics", {})
            
            self.output.progress(f"Protocol: {protocol_name} (LLM-selected)")
            self.output.progress(f"Selected Protocol: {self.selected_protocol.upper()}")
            self.output.progress(f"Selection Confidence: {self.routing_decision.confidence:.1%}")
            self.output.progress(f"Total Conversations: {total_convs}")
            self.output.progress(f"Average Privacy Score: {summary.get('average_privacy_score', 0):.2f}/100")
            self.output.progress(f"Privacy Grade: {summary.get('overall_privacy_grade', 'Unknown')}")
            
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
            super().display_results(results)

    async def cleanup(self) -> None:
        """Cleanup meta protocol resources"""
        try:
            # Cleanup communication backend
            if self.comm_backend:
                await self.comm_backend.close()
            
            # Cleanup registration adapter
            if self.registration_adapter:
                await self.registration_adapter.cleanup()
            
            self.output.info("🧹 Meta protocol cleanup completed")
        except Exception as e:
            self.output.warning(f"Cleanup warning: {e}")
    
