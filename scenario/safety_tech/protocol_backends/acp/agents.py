# -*- coding: utf-8 -*-
"""
ACP Protocol Agent Adapters for Privacy Testing
Bridges between ACP protocol and privacy testing agent logic using real acp_sdk types.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional, List, AsyncGenerator

# acp_sdk real types
from acp_sdk.server import Context
from acp_sdk.models import Message, MessagePart

# Import core agent classes
try:
    from ...core.privacy_agent_base import ReceptionistAgent, NosyDoctorAgent
except ImportError:
    from core.privacy_agent_base import ReceptionistAgent, NosyDoctorAgent


class ACPReceptionistAgent(ReceptionistAgent):
    """ACP-specific receptionist agent implementation."""

    async def send_to_agent(self, target_id: str, message: str) -> Dict[str, Any]:
        """Send message through ACP network."""
        if not self.agent_network:
            raise RuntimeError("No ACP network set. Call set_network() first.")
        
        payload = {"text": message}
        response = await self.agent_network.route_message(self.agent_id, target_id, payload)
        return response or {"raw": None, "text": ""}


class ACPNosyDoctorAgent(NosyDoctorAgent):
    """ACP-specific nosy doctor agent implementation."""

    async def send_to_agent(self, target_id: str, message: str) -> Dict[str, Any]:
        """Send message through ACP network."""
        if not self.agent_network:
            raise RuntimeError("No ACP network set. Call set_network() first.")
        
        payload = {"text": message}
        response = await self.agent_network.route_message(self.agent_id, target_id, payload)
        return response or {"raw": None, "text": ""}


class ACPReceptionistExecutor:
    """
    ACP Executor wrapper for receptionist agent.
    Handles ACP protocol integration for privacy-aware agent.
    """

    def __init__(self, config: Optional[dict] = None, output=None):
        self.agent = ACPReceptionistAgent("ACP_Receptionist", config, output)

    async def execute(self, messages: List[Message], context: Context) -> AsyncGenerator[Message, None]:
        """Execute receptionist agent logic through ACP interface (streaming)."""
        try:
            # Extract user input from messages
            user_input = ""
            if messages:
                for part in getattr(messages[0], 'parts', []) or []:
                    if isinstance(getattr(part, 'content', None), str):
                        user_input += part.content
            if not user_input:
                user_input = "status"

            response = await self.agent.process_message("ACP_Client", user_input)
            yield Message(parts=[MessagePart(content=response, content_type="text/plain")])

        except Exception as e:
            error_msg = f"Receptionist agent error: {e}"
            print(f"[ACPReceptionistExecutor] {error_msg}")
            yield Message(parts=[MessagePart(content=error_msg, content_type="text/plain")])

    async def cancel(self, messages: List[Message], context: Context) -> AsyncGenerator[Message, None]:
        """Handle ACP cancellation (streaming)."""
        yield Message(parts=[MessagePart(content="Operation cancelled", content_type="text/plain")])

    def set_network(self, network: Any) -> None:
        """Set network for agent communication."""
        self.agent.set_network(network)


class ACPNosyDoctorExecutor:
    """
    ACP Executor wrapper for nosy doctor agent.
    Handles ACP protocol integration for privacy-invasive agent.
    """

    def __init__(self, config: Optional[dict] = None, output=None):
        self.agent = ACPNosyDoctorAgent("ACP_Doctor", config, output)

    async def execute(self, messages: List[Message], context: Context) -> AsyncGenerator[Message, None]:
        """Execute nosy doctor agent logic through ACP interface (streaming)."""
        try:
            user_input = ""
            if messages:
                for part in getattr(messages[0], 'parts', []) or []:
                    if isinstance(getattr(part, 'content', None), str):
                        user_input += part.content
            if not user_input:
                user_input = "Please provide patient information"

            response = await self.agent.process_message("ACP_Client", user_input)
            yield Message(parts=[MessagePart(content=response, content_type="text/plain")])

        except Exception as e:
            error_msg = f"Doctor agent error: {e}"
            print(f"[ACPNosyDoctorExecutor] {error_msg}")
            yield Message(parts=[MessagePart(content=error_msg, content_type="text/plain")])

    async def cancel(self, messages: List[Message], context: Context) -> AsyncGenerator[Message, None]:
        """Handle ACP cancellation (streaming)."""
        yield Message(parts=[MessagePart(content="Operation cancelled", content_type="text/plain")])

    def set_network(self, network: Any) -> None:
        """Set network for agent communication."""
        self.agent.set_network(network)


class ACPPrivacySimulator:
    """
    ACP-specific privacy testing simulator.
    Orchestrates conversations between receptionist and doctor agents.
    """

    def __init__(self, config: Optional[dict] = None, output=None):
        self.config = config or {}
        self.output = output
        self.receptionist = ACPReceptionistAgent("ACP_Receptionist", config, output)
        self.doctor = ACPNosyDoctorAgent("ACP_Doctor", config, output)
        self.conversation_log: List[Dict[str, Any]] = []

    def set_network(self, network: Any) -> None:
        """Set network for both agents."""
        self.receptionist.set_network(network)
        self.doctor.set_network(network)

    async def simulate_conversation(self, patient_question: str, rounds: int = 3) -> List[Dict[str, Any]]:
        """
        Simulate privacy testing conversation between receptionist and doctor.
        
        Args:
            patient_question: Enhanced patient question with sensitive information
            rounds: Number of conversation rounds
            
        Returns:
            List of conversation messages
        """
        conversation = []
        current_message = patient_question

        for round_num in range(rounds):
            # Receptionist processes patient question/doctor request
            receptionist_response = await self.receptionist.process_message(
                "Patient" if round_num == 0 else "ACP_Doctor", 
                current_message
            )
            
            conversation.append({
                "round": round_num + 1,
                "sender": "ACP_Receptionist",
                "message": receptionist_response,
                "timestamp": time.time()
            })

            # Doctor responds with invasive query
            doctor_response = await self.doctor.process_message(
                "ACP_Receptionist", 
                receptionist_response
            )
            
            conversation.append({
                "round": round_num + 1,
                "sender": "ACP_Doctor", 
                "message": doctor_response,
                "timestamp": time.time()
            })

            # Prepare next round
            current_message = doctor_response

        return conversation

    async def run_privacy_test_batch(self, enhanced_questions: List[str], rounds_per_conversation: int = 3) -> Dict[str, Any]:
        """
        Run batch privacy testing with multiple enhanced questions.
        
        Args:
            enhanced_questions: List of patient questions with sensitive info
            rounds_per_conversation: Conversation rounds per question
            
        Returns:
            Complete conversation data for privacy analysis
        """
        import time
        
        all_conversations = []
        
        for i, question in enumerate(enhanced_questions):
            conversation_id = f"acp_conversation_{i+1}"
            
            try:
                conversation_messages = await self.simulate_conversation(question, rounds_per_conversation)
                
                conversation_data = {
                    "conversation_id": conversation_id,
                    "original_question": question,
                    "messages": conversation_messages,
                    "protocol": "acp",
                    "timestamp": time.time()
                }
                
                all_conversations.append(conversation_data)
                
                if self.output:
                    self.output.info(f"Completed ACP conversation {i+1}/{len(enhanced_questions)}")
                
            except Exception as e:
                print(f"[ACPPrivacySimulator] Error in conversation {i+1}: {e}")
                continue

        return {
            "protocol": "acp",
            "total_conversations": len(all_conversations),
            "conversations": all_conversations,
            "generation_timestamp": time.time()
        }
