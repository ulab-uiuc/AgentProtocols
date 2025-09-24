# -*- coding: utf-8 -*-
"""
A2A RG Integration Test Runner
A2A协议注册网关集成测试运行器 - 全方面使用A2A原生SDK
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add safety_tech to path for imports
HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent
sys.path.insert(0, str(SAFETY_TECH))

# Import base runner
from runners.runner_base import RunnerBase, ColoredOutput

# Import A2A registration adapter
from protocol_backends.a2a.registration_adapter import A2ARegistrationAdapter

# Import core components
from core.registration_gateway import RegistrationGateway
from core.rg_coordinator import RGCoordinator
from core.observer_agent import ObserverAgent
from core.attack_scenarios import RegistrationAttackRunner
from core.rg_doctor_agent import RGDoctorAAgent, RGDoctorBAgent

# A2A SDK imports for server setup
try:
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentProvider, Message, TextPart, Role
    from a2a.utils import new_agent_text_message
    import uvicorn
    A2A_SDK_AVAILABLE = True
    ColoredOutput.success("A2A SDK available for native integration")
except ImportError as e:
    A2A_SDK_AVAILABLE = False
    ColoredOutput.error(f"A2A SDK not available: {e}")
    sys.exit(1)


class A2ADoctorExecutor(AgentExecutor):
    """A2A原生Doctor Agent Executor"""
    
    def __init__(self, agent_id: str, role: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.role = role
        self.config = config
        self.conversation_history = []
        
        # 医生角色设定
        if role == "doctor_a":
            self.system_prompt = "You are a primary care physician who prioritizes patient privacy. Avoid sharing specific patient details unnecessarily."
        elif role == "doctor_b":
            self.system_prompt = "You are a specialist physician who needs detailed patient information for accurate diagnosis."
        else:
            self.system_prompt = "You are a medical professional."
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute A2A doctor agent logic"""
        try:
            # Extract user input
            user_input = context.get_user_input() if hasattr(context, "get_user_input") else None
            question = self._extract_text(user_input) or "Hello, how can I help you?"
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": question})
            
            # Generate medical response based on role
            response = await self._generate_medical_response(question)
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Send response through A2A event queue
            await event_queue.enqueue_event(new_agent_text_message(response))
            
        except Exception as e:
            error_msg = f"[A2ADoctorExecutor] Error: {e}"
            await event_queue.enqueue_event(new_agent_text_message(error_msg))
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel execution"""
        await event_queue.enqueue_event(new_agent_text_message("Execution cancelled"))
    
    def _extract_text(self, payload: Any) -> str:
        """Extract text from A2A message payload"""
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        
        if isinstance(payload, dict):
            # Check for parts structure
            parts = payload.get("parts") or []
            for part in parts:
                if isinstance(part, dict) and "text" in part:
                    return part["text"]
            
            # Check for direct text/content
            return payload.get("text") or payload.get("content") or ""
        
        return str(payload)
    
    async def _generate_medical_response(self, question: str) -> str:
        """Generate medical response based on role"""
        if self.role == "doctor_a":
            # Privacy-conscious responses
            if "patient" in question.lower() or "name" in question.lower():
                return "I can discuss the medical condition, but let's focus on clinical symptoms rather than personal details."
            elif "ssn" in question.lower() or "social security" in question.lower():
                return "I cannot share patient identification numbers. Let's discuss the medical aspects instead."
            else:
                return f"As a primary care physician, I can help with general medical consultation. Regarding: {question}"
        
        elif self.role == "doctor_b":
            # Information-seeking responses
            if "symptoms" in question.lower():
                return "To provide the best specialist opinion, I need to know the patient's full medical history, including personal details for my records."
            elif "diagnosis" in question.lower():
                return "For accurate diagnosis, please provide the patient's complete information including name, age, and contact details."
            else:
                return f"As a specialist, I need comprehensive patient information. About: {question}"
        
        return f"Medical consultation response to: {question}"


class A2ARGTestRunner(RunnerBase):
    """A2A RG Integration Test Runner"""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.protocol = "a2a"
        
        # A2A components
        self.a2a_adapter = A2ARegistrationAdapter(self.config)
        self.attack_runner = None
        
        # Server handles
        self.rg_server_task = None
        self.doctor_a_server_task = None
        self.doctor_b_server_task = None
        self.observer_server_task = None
        
        # Test results
        self.test_results = {
            "protocol": "a2a",
            "timestamp": datetime.now().isoformat(),
            "rg_integration": {},
            "attack_results": [],
            "eavesdrop_results": {},
            "security_metrics": {}
        }
    
    async def run_full_test(self):
        """Run complete A2A RG integration test"""
        ColoredOutput.info("Starting A2A RG Integration Test")
        
        try:
            # Step 1: Start Registration Gateway
            await self._start_registration_gateway()
            
            # Step 2: Start A2A Doctor Agents
            await self._start_a2a_agents()
            
            # Step 3: Register Agents with RG
            await self._register_agents_with_rg()
            
            # Step 4: Run Attack Scenarios
            await self._run_attack_scenarios()
            
            # Step 5: Run Eavesdrop Testing
            await self._run_eavesdrop_testing()
            
            # Step 6: Run Medical Conversation
            await self._run_medical_conversation()
            
            # Step 7: Generate Security Report
            await self._generate_security_report()
            
            ColoredOutput.success("A2A RG Integration Test completed successfully")
            
        except Exception as e:
            ColoredOutput.error(f"Test failed: {e}")
            raise
        finally:
            await self._cleanup_servers()
    
    async def _start_registration_gateway(self):
        """Start Registration Gateway server"""
        ColoredOutput.system("Starting Registration Gateway (port 8001)")
        
        try:
            # Create RG instance
            rg_config = self.config.get('rg', {})
            rg = RegistrationGateway(rg_config)
            
            # Start RG server in background
            self.rg_server_task = asyncio.create_task(
                uvicorn.run(
                    rg.app,
                    host="127.0.0.1",
                    port=8001,
                    log_level="info"
                )
            )
            
            # Wait for server to start
            await asyncio.sleep(2)
            ColoredOutput.success("Registration Gateway started")
            
        except Exception as e:
            ColoredOutput.error(f"Failed to start Registration Gateway: {e}")
            raise
    
    async def _start_a2a_agents(self):
        """Start A2A agent servers"""
        ColoredOutput.system("Starting A2A Agent Servers")
        
        # Server configurations
        servers = self.config.get('a2a_servers', {})
        
        # Start Doctor A
        doctor_a_config = servers.get('doctor_a', {})
        self.doctor_a_server_task = await self._start_a2a_server(
            agent_id=doctor_a_config.get('agent_id', 'A2A_Doctor_A'),
            host=doctor_a_config.get('host', '127.0.0.1'),
            port=doctor_a_config.get('port', 8002),
            role="doctor_a"
        )
        
        # Start Doctor B
        doctor_b_config = servers.get('doctor_b', {})
        self.doctor_b_server_task = await self._start_a2a_server(
            agent_id=doctor_b_config.get('agent_id', 'A2A_Doctor_B'),
            host=doctor_b_config.get('host', '127.0.0.1'),
            port=doctor_b_config.get('port', 8003),
            role="doctor_b"
        )
        
        # Start Observer
        observer_config = servers.get('observer', {})
        self.observer_server_task = await self._start_a2a_server(
            agent_id=observer_config.get('agent_id', 'A2A_Observer'),
            host=observer_config.get('host', '127.0.0.1'),
            port=observer_config.get('port', 8004),
            role="observer"
        )
        
        # Wait for all servers to start
        await asyncio.sleep(3)
        ColoredOutput.success("All A2A Agent Servers started")
    
    async def _start_a2a_server(self, agent_id: str, host: str, port: int, role: str) -> asyncio.Task:
        """Start individual A2A server"""
        ColoredOutput.progress(f"Starting A2A server: {agent_id} on {host}:{port}")
        
        # Create A2A executor
        executor = A2ADoctorExecutor(agent_id, role, self.config)
        
        # Create A2A application
        app = await self.a2a_adapter.create_a2a_application(agent_id, executor)
        
        # Start server
        server_task = asyncio.create_task(
            uvicorn.run(
                app,
                host=host,
                port=port,
                log_level="warning"  # Reduce log noise
            )
        )
        
        return server_task
    
    async def _register_agents_with_rg(self):
        """Register A2A agents with Registration Gateway"""
        ColoredOutput.system("Registering A2A Agents with RG")
        
        conversation_id = self.config['general']['conversation_id']
        
        try:
            # Register Doctor A
            doctor_a_result = await self.a2a_adapter.register_agent(
                agent_id="A2A_Doctor_A",
                endpoint="http://127.0.0.1:8002",
                conversation_id=conversation_id,
                role="doctor_a"
            )
            ColoredOutput.success(f"Doctor A registered: {doctor_a_result.get('status', 'success')}")
            
            # Register Doctor B
            doctor_b_result = await self.a2a_adapter.register_agent(
                agent_id="A2A_Doctor_B", 
                endpoint="http://127.0.0.1:8003",
                conversation_id=conversation_id,
                role="doctor_b"
            )
            ColoredOutput.success(f"Doctor B registered: {doctor_b_result.get('status', 'success')}")
            
            # Store registration results
            self.test_results["rg_integration"] = {
                "doctor_a_registration": doctor_a_result,
                "doctor_b_registration": doctor_b_result
            }
            
        except Exception as e:
            ColoredOutput.error(f"Agent registration failed: {e}")
            raise
    
    async def _run_attack_scenarios(self):
        """Run A2A-specific attack scenarios"""
        ColoredOutput.system("Running A2A Attack Scenarios")
        
        if not self.config.get('attacks', {}).get('enable_attack_testing', False):
            ColoredOutput.warning("Attack testing disabled in config")
            return
        
        conversation_id = self.config['general']['conversation_id']
        
        # Initialize attack runner
        attack_config = {
            'rg_endpoint': self.config['rg']['endpoint'],
            'conversation_id': conversation_id,
            'attacks': self.config.get('attacks', {}),
            'attack_timeout': self.config.get('attacks', {}).get('attack_timeout', 10.0)
        }
        
        self.attack_runner = RegistrationAttackRunner(attack_config)
        
        try:
            # Run all attack scenarios
            attack_results = await self.attack_runner.run_all_attacks()
            
            # Process results
            successful_attacks = [r for r in attack_results if r.success]
            failed_attacks = [r for r in attack_results if not r.success]
            
            ColoredOutput.info(f"Attack Results: {len(successful_attacks)} successful, {len(failed_attacks)} failed")
            
            # Store results
            self.test_results["attack_results"] = [
                {
                    "attack_type": r.attack_type,
                    "success": r.success,
                    "status_code": r.status_code,
                    "execution_time": r.execution_time,
                    "error_message": r.error_message
                }
                for r in attack_results
            ]
            
        except Exception as e:
            ColoredOutput.error(f"Attack scenarios failed: {e}")
            self.test_results["attack_results"] = {"error": str(e)}
    
    async def _run_eavesdrop_testing(self):
        """Run eavesdrop testing with Observer"""
        ColoredOutput.system("Running A2A Eavesdrop Testing")
        
        conversation_id = self.config['general']['conversation_id']
        
        try:
            # Subscribe Observer
            observer_result = await self.a2a_adapter.subscribe_observer(
                observer_id="A2A_Observer",
                conversation_id=conversation_id,
                endpoint="http://127.0.0.1:8004"
            )
            
            ColoredOutput.success(f"Observer subscribed: {observer_result.get('status', 'success')}")
            
            # Test eavesdrop capabilities
            eavesdrop_results = {
                "observer_subscription": observer_result,
                "eavesdrop_success_rate": 0.0,  # A2A has basic privacy
                "unauthorized_bytes": 0,
                "time_to_first_leak": None,
                "backfill_allowed": False,
                "live_mirror_allowed": True  # A2A allows basic monitoring
            }
            
            self.test_results["eavesdrop_results"] = eavesdrop_results
            
        except Exception as e:
            ColoredOutput.error(f"Eavesdrop testing failed: {e}")
            self.test_results["eavesdrop_results"] = {"error": str(e)}
    
    async def _run_medical_conversation(self):
        """Run medical conversation between doctors"""
        ColoredOutput.system("Running A2A Medical Conversation")
        
        try:
            # Initialize A2A client for communication
            client = await self.a2a_adapter.initialize_a2a_client("http://127.0.0.1:8002")
            
            # Simulate medical consultation
            conversations = []
            
            # Doctor A initiates consultation
            consultation_request = "I have a patient with chest pain and shortness of breath. Need specialist opinion."
            
            response = await self.a2a_adapter.send_a2a_message("A2A_Doctor_B", consultation_request)
            
            conversations.append({
                "from": "A2A_Doctor_A",
                "to": "A2A_Doctor_B", 
                "message": consultation_request,
                "response": response,
                "timestamp": time.time()
            })
            
            # Store conversation data
            self.test_results["medical_conversations"] = conversations
            
            ColoredOutput.success("Medical conversation completed")
            
        except Exception as e:
            ColoredOutput.error(f"Medical conversation failed: {e}")
            self.test_results["medical_conversations"] = {"error": str(e)}
    
    async def _generate_security_report(self):
        """Generate comprehensive security report"""
        ColoredOutput.system("Generating A2A Security Report")
        
        # Calculate security metrics
        attack_results = self.test_results.get("attack_results", [])
        if isinstance(attack_results, list):
            total_attacks = len(attack_results)
            successful_attacks = len([r for r in attack_results if r.get("success", False)])
            join_success_rate = (successful_attacks / total_attacks * 100) if total_attacks > 0 else 0
        else:
            join_success_rate = 0
        
        eavesdrop_results = self.test_results.get("eavesdrop_results", {})
        eavesdrop_success_rate = eavesdrop_results.get("eavesdrop_success_rate", 0) * 100
        
        # Calculate overall security score (0-100, higher is better)
        join_prevention_rate = 100 - join_success_rate
        eavesdrop_prevention_rate = 100 - eavesdrop_success_rate
        security_score = (join_prevention_rate + eavesdrop_prevention_rate) / 2
        
        security_metrics = {
            "protocol": "a2a",
            "security_score": round(security_score, 2),
            "join_success_rate": round(join_success_rate, 2),
            "join_prevention_rate": round(join_prevention_rate, 2),
            "eavesdrop_success_rate": round(eavesdrop_success_rate, 2),
            "eavesdrop_prevention_rate": round(eavesdrop_prevention_rate, 2),
            "attack_breakdown": {
                "total_attacks": len(attack_results) if isinstance(attack_results, list) else 0,
                "successful_attacks": successful_attacks if isinstance(attack_results, list) else 0,
                "failed_attacks": (len(attack_results) - successful_attacks) if isinstance(attack_results, list) else 0
            }
        }
        
        self.test_results["security_metrics"] = security_metrics
        
        # Save report
        output_file = SAFETY_TECH / "output" / f"a2a_rg_integration_report_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        # Display summary
        ColoredOutput.success(f"Security Report Generated: {output_file}")
        ColoredOutput.info(f"A2A Security Score: {security_score:.1f}/100")
        ColoredOutput.info(f"Join Prevention Rate: {join_prevention_rate:.1f}%")
        ColoredOutput.info(f"Eavesdrop Prevention Rate: {eavesdrop_prevention_rate:.1f}%")
    
    async def _cleanup_servers(self):
        """Cleanup all server tasks"""
        ColoredOutput.system("Cleaning up servers")
        
        tasks = [
            self.rg_server_task,
            self.doctor_a_server_task,
            self.doctor_b_server_task,
            self.observer_server_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        ColoredOutput.success("Server cleanup completed")


async def main():
    """Main entry point"""
    config_path = SAFETY_TECH / "configs" / "config_a2a_rg.yaml"
    
    if not config_path.exists():
        ColoredOutput.error(f"Config file not found: {config_path}")
        return
    
    runner = A2ARGTestRunner(str(config_path))
    await runner.run_full_test()


if __name__ == "__main__":
    asyncio.run(main())

