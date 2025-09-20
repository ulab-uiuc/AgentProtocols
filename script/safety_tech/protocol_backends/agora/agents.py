# -*- coding: utf-8 -*-
"""
Agora Protocol Privacy Testing Agents
Implements privacy-aware and privacy-invasive agents for testing privacy protection.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional, List

# Import core privacy agent classes
try:
    from ...core.privacy_agent_base import ReceptionistAgent, NosyDoctorAgent
except ImportError:
    try:
        from core.privacy_agent_base import ReceptionistAgent, NosyDoctorAgent
    except ImportError:
        # Fallback: define minimal base classes based on ACP pattern
        print("\033[91m[Agora] core.privacy_agent_base not found \033[0m")


class AgoraReceptionistAgent(ReceptionistAgent):
    """Agora-specific receptionist agent implementation."""

    async def send_to_agent(self, target_id: str, message: str) -> Dict[str, Any]:
        """Send message through Agora network."""
        if not self.agent_network:
            raise RuntimeError("No Agora network set. Call set_network() first.")
        
        payload = {
            "protocolHash": None,
            "body": message,
            "protocolSources": []
        }
        response = await self.agent_network.route_message(self.agent_id, target_id, payload)
        return response or {"raw": None, "text": ""}


class AgoraNosyDoctorAgent(NosyDoctorAgent):
    """Agora-specific nosy doctor agent implementation."""

    async def send_to_agent(self, target_id: str, message: str) -> Dict[str, Any]:
        """Send message through Agora network."""
        if not self.agent_network:
            raise RuntimeError("No Agora network set. Call set_network() first.")
        
        payload = {
            "protocolHash": None,
            "body": message,
            "protocolSources": []
        }
        response = await self.agent_network.route_message(self.agent_id, target_id, payload)
        return response or {"raw": None, "text": ""}


class AgoraReceptionistExecutor:
    """
    Agora接待员执行器 - 作为Agora Protocol工具使用
    """

    def __init__(self, config: Optional[dict] = None, output=None):
        self.agent = AgoraReceptionistAgent("Agora_Receptionist", config, output)

    async def execute(self, user_input: str) -> str:
        """执行接待员逻辑 - 符合Agora工具函数接口"""
        try:
            if not user_input:
                user_input = "Hello, how can I help you today?"
            
            # 通过隐私保护智能体处理消息
            response = await self.agent.process_message("Patient", user_input)
            return response
            
        except Exception as e:
            error_msg = f"Receptionist executor error: {e}"
            print(f"[AgoraReceptionistExecutor] {error_msg}")
            return error_msg

    def set_network(self, network: Any) -> None:
        """设置网络"""
        self.agent.set_network(network)


class AgoraNosyDoctorExecutor:
    """
    Agora医生执行器 - 作为Agora Protocol工具使用
    """

    def __init__(self, config: Optional[dict] = None, output=None):
        self.agent = AgoraNosyDoctorAgent("Agora_Doctor", config, output)

    async def execute(self, user_input: str) -> str:
        """执行医生逻辑 - 符合Agora工具函数接口"""
        try:
            if not user_input:
                user_input = "I need patient information"
            
            # 通过隐私侵犯智能体处理消息
            response = await self.agent.process_message("Agora_Receptionist", user_input)
            return response
            
        except Exception as e:
            error_msg = f"Doctor executor error: {e}"
            print(f"[AgoraNosyDoctorExecutor] {error_msg}")
            return error_msg

    def set_network(self, network: Any) -> None:
        """设置网络"""
        self.agent.set_network(network)


class AgoraPrivacySimulator:
    """
    Agora-specific privacy testing simulator.
    Orchestrates conversations between receptionist and doctor agents.
    """

    def __init__(self, config: Optional[dict] = None, output=None):
        self.config = config or {}
        self.output = output
        self.receptionist = AgoraReceptionistAgent("Agora_Receptionist", config, output)
        self.doctor = AgoraNosyDoctorAgent("Agora_Doctor", config, output)
        self.conversation_log: List[Dict[str, Any]] = []

    def set_network(self, network: Any) -> None:
        """Set network for both agents."""
        self.receptionist.set_network(network)
        self.doctor.set_network(network)

    async def simulate_conversation(self, patient_question: str, rounds: int = 2) -> List[Dict[str, Any]]:
        """
        Simulate medical consultation between two doctors.
        
        Args:
            patient_question: Medical case information 
            rounds: Number of discussion rounds
            
        Returns:
            List of conversation messages
        """
        conversation = []
        current_message = patient_question

        print(f"\n🏥 Medical Case Discussion")
        print("=" * 60)
        print(f"📋 Initial Case: {patient_question}")
        print("=" * 60)

        for round_num in range(rounds):
            # Doctor A analyzes the case
            doctor_a_response = await self.receptionist.process_message(
                "Patient" if round_num == 0 else "Agora_Doctor_B", 
                current_message
            )
            
            print(f"\n👨‍⚕️ Doctor A (Round {round_num + 1}):")
            print(f"   {doctor_a_response}")
            
            conversation.append({
                "round": round_num + 1,
                "sender": "Agora_Doctor_A",
                "message": doctor_a_response,
                "timestamp": time.time()
            })

            # Doctor B provides specialist opinion
            doctor_b_response = await self.doctor.process_message(
                "Agora_Doctor_A", 
                doctor_a_response
            )
            
            print(f"\n👩‍⚕️ Doctor B (Round {round_num + 1}):")
            print(f"   {doctor_b_response}")
            
            conversation.append({
                "round": round_num + 1,
                "sender": "Agora_Doctor_B", 
                "message": doctor_b_response,
                "timestamp": time.time()
            })

            # Prepare next round
            current_message = doctor_b_response

        print("\n" + "=" * 60)
        print("✅ Medical consultation completed")
        print("=" * 60)
        
        return conversation

    async def run_privacy_test_batch(self, enhanced_questions: List[str], rounds_per_conversation: int = 2) -> Dict[str, Any]:
        """
        Run batch medical consultations with multiple cases.
        
        Args:
            enhanced_questions: List of medical cases
            rounds_per_conversation: Discussion rounds per case
            
        Returns:
            Complete conversation data
        """
        all_conversations = []
        
        print(f"\n🎯 Starting {len(enhanced_questions)} Medical Consultations")
        print(f"📊 Each consultation will have {rounds_per_conversation} discussion rounds")
        
        for i, question in enumerate(enhanced_questions):
            conversation_id = f"agora_conversation_{i+1}"
            
            try:
                print(f"\n📝 Consultation {i+1}/{len(enhanced_questions)}")
                conversation_messages = await self.simulate_conversation(question, rounds_per_conversation)
                
                conversation_data = {
                    "conversation_id": conversation_id,
                    "original_question": question,
                    "messages": conversation_messages,
                    "protocol": "agora",
                    "timestamp": time.time()
                }
                
                all_conversations.append(conversation_data)
                
                if self.output:
                    self.output.success(f"✅ Completed medical consultation {i+1}/{len(enhanced_questions)}")
                
            except Exception as e:
                print(f"❌ Error in consultation {i+1}: {e}")
                continue

        print(f"\n🏆 All {len(all_conversations)} medical consultations completed!")
        return {
            "protocol": "agora",
            "total_conversations": len(all_conversations),
            "conversations": all_conversations,
            "generation_timestamp": time.time()
        }