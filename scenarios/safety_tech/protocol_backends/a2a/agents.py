# -*- coding: utf-8 -*-
"""
A2A Privacy Testing Agent Adapters
Implements A2A-compatible agents for privacy protection testing scenarios.
"""

from __future__ import annotations
import asyncio
import json
import time
import uuid
from typing import Any, Dict, Optional, List

# Import core agent classes
try:
    from ...core.privacy_agent_base import ReceptionistAgent, NosyDoctorAgent
except ImportError:
    from core.privacy_agent_base import ReceptionistAgent, NosyDoctorAgent

# A2A SDK imports for agent execution
try:
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.utils import new_agent_text_message
    print("[A2A Privacy] Successfully imported A2A SDK agent execution")
except ImportError as e:
    raise ImportError(f"A2A SDK required but not available: {e}")


# Safe wrapper for EventQueue.enqueue_event
async def safe_enqueue_event(event_queue, event):
    """Safely enqueue event, handling both sync and async EventQueue implementations."""
    try:
        if hasattr(event_queue, 'enqueue_event'):
            result = event_queue.enqueue_event(event)
            if hasattr(result, '__await__'):
                return await result
            return result
        else:
            # Fallback for mock event queue
            return event
    except Exception as e:
        print(f"[A2A Privacy] Error enqueuing event: {e}")
        return None


class A2AReceptionistAgent(ReceptionistAgent):
    """A2A-compatible privacy-aware receptionist agent"""

    def __init__(self, agent_id: str, config: Optional[dict] = None, output=None):
        super().__init__(agent_id, config, output)
        self.a2a_context = {
            "agent_type": "receptionist",
            "privacy_role": "protector",
            "capabilities": ["privacy_protection", "patient_interaction", "information_filtering"]
        }

    async def send_to_agent(self, target_id: str, message: str) -> Dict[str, Any]:
        """Send message to another agent through A2A network"""
        if not self.agent_network:
            raise RuntimeError("No A2A network set. Call set_network() first.")
        
        # Apply privacy protection before sending
        if self._should_protect_info(message):
            sanitized_message = self._sanitize_message(message)
            payload = {
                "text": sanitized_message,
                "privacy_protected": True,
                "original_length": len(message),
                "sanitized_length": len(sanitized_message)
            }
        else:
            payload = {
                "text": message,
                "privacy_protected": False
            }
        
        # Add A2A privacy metadata
        payload.update({
            "a2a_agent_type": "receptionist",
            "privacy_context": "medical_consultation",
            "sender_capabilities": self.a2a_context["capabilities"]
        })
        
        response = await self.agent_network.route_message(self.agent_id, target_id, payload)
        return response or {"raw": None, "text": "", "privacy_safe": True}


class A2ADoctorAgent(NosyDoctorAgent):
    """A2A-compatible privacy-invasive doctor agent"""

    def __init__(self, agent_id: str, config: Optional[dict] = None, output=None):
        super().__init__(agent_id, config, output)
        self.a2a_context = {
            "agent_type": "doctor",
            "privacy_role": "extractor",
            "capabilities": ["information_extraction", "medical_inquiry", "persistent_questioning"]
        }

    async def send_to_agent(self, target_id: str, message: str) -> Dict[str, Any]:
        """Send message to another agent through A2A network"""
        if not self.agent_network:
            raise RuntimeError("No A2A network set. Call set_network() first.")
        
        # Enhance message for information extraction (LLM will handle this)
        payload = {
            "text": message,
            "extraction_intent": True,
            "target_info": ["name", "ssn", "phone", "address", "personal_details"]
        }
        
        # Add A2A privacy metadata
        payload.update({
            "a2a_agent_type": "doctor",
            "privacy_context": "medical_inquiry",
            "sender_capabilities": self.a2a_context["capabilities"]
        })
        
        response = await self.agent_network.route_message(self.agent_id, target_id, payload)
        return response or {"raw": None, "text": "", "privacy_safe": False}


class A2AReceptionistExecutor(AgentExecutor):
    """A2A-SDK compatible executor for privacy-aware receptionist"""

    def __init__(self, config: Dict[str, Any], agent_id: str = "A2A_Receptionist", output=None):
        self.agent_id = agent_id
        self.config = config
        self.output = output
        self._cancelled = False
        
        # Create the core privacy agent
        self.receptionist = A2AReceptionistAgent(agent_id, config, output)
        
        if self.output:
            self.output.info(f"A2A Receptionist Executor initialized: {agent_id}")

    async def cancel(self):
        """Cancel the executor (required by AgentExecutor interface)"""
        self._cancelled = True
        if self.output:
            self.output.info(f"A2A Receptionist Executor cancelled: {self.agent_id}")

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        Handle incoming A2A messages for the receptionist agent.
        This is the main entry point for A2A interactions.
        """
        try:
            user_input = context.get_user_input()
            
            # Extract A2A message metadata
            sender_id, message_content = self._parse_a2a_message(context, user_input)
            
            if self.output:
                self.output.progress(f"[A2A Receptionist] Processing message from {sender_id}")
            
            # Process message through privacy agent
            response = await self.receptionist.process_message(sender_id, message_content)
            
            # Create A2A response
            a2a_response = self._create_a2a_response(response, "receptionist")
            await safe_enqueue_event(event_queue, new_agent_text_message(a2a_response))
            
            if self.output:
                self.output.success(f"[A2A Receptionist] Response sent (privacy-protected)")
                
        except Exception as e:
            error_msg = f"[A2A Receptionist] Error processing message: {e}"
            if self.output:
                self.output.error(error_msg)
            await safe_enqueue_event(event_queue, new_agent_text_message(error_msg))

    def _parse_a2a_message(self, context: RequestContext, user_input: str) -> tuple[str, str]:
        """Parse A2A message to extract sender and content"""
        sender_id = "external"
        message_content = user_input or ""
        
        try:
            # Try to parse structured A2A message
            if user_input:
                try:
                    parsed = json.loads(user_input)
                    if isinstance(parsed, dict):
                        # Prefer payload.* first, then top-level for backward-compat
                        payload = parsed.get("payload", {}) if isinstance(parsed, dict) else {}

                        sender_id = (
                            (payload.get("src_id") or payload.get("sender_id"))
                            or parsed.get("src_id") or parsed.get("sender_id")
                            or "external"
                        )

                        message_content = (
                            payload.get("text") or payload.get("content")
                            or parsed.get("text") or parsed.get("content")
                            or user_input
                        )
                        
                        # Optional: log privacy context if available
                        privacy_ctx = payload.get("privacy_context") or parsed.get("privacy_context")
                        if privacy_ctx and self.output:
                            interaction = (privacy_ctx.get("interaction_type", "unknown")
                                           if isinstance(privacy_ctx, dict) else str(privacy_ctx))
                            if hasattr(self.output, 'debug'):
                                self.output.debug(f"Privacy context: {interaction}")
                            else:
                                self.output.info(f"Privacy context: {interaction}")
                            
                except json.JSONDecodeError:
                    # Handle as plain text
                    message_content = user_input
                    
            # Try to extract from A2A context metadata
            if hasattr(context, 'params') and hasattr(context.params, 'message'):
                message = context.params.message
                if hasattr(message, 'meta') and message.meta:
                    meta = message.meta if isinstance(message.meta, dict) else {}
                    sender_id = meta.get('sender_id', meta.get('sender', sender_id))
                    
        except Exception as e:
            if self.output:
                self.output.warning(f"Failed to parse A2A message metadata: {e}")
        
        return sender_id, message_content

    def _create_a2a_response(self, response: str, agent_type: str) -> str:
        """Create A2A-compatible response message"""
        response_data = {
            "type": "PRIVACY_RESPONSE",
            "content": response,
            "agent_type": agent_type,
            "privacy_protected": True,
            "timestamp": time.time(),
            "response_id": str(uuid.uuid4())
        }
        return json.dumps(response_data)


class A2ADoctorExecutor(AgentExecutor):
    """A2A-SDK compatible executor for privacy-invasive doctor"""

    def __init__(self, config: Dict[str, Any], agent_id: str = "A2A_Doctor", output=None):
        self.agent_id = agent_id
        self.config = config
        self.output = output
        self._cancelled = False
        
        # Create the core privacy agent
        self.doctor = A2ADoctorAgent(agent_id, config, output)
        
        if self.output:
            self.output.info(f"A2A Doctor Executor initialized: {agent_id}")

    async def cancel(self):
        """Cancel the executor (required by AgentExecutor interface)"""
        self._cancelled = True
        if self.output:
            self.output.info(f"A2A Doctor Executor cancelled: {self.agent_id}")

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        Handle incoming A2A messages for the doctor agent.
        This is the main entry point for A2A interactions.
        """
        try:
            user_input = context.get_user_input()
            
            # Extract A2A message metadata
            sender_id, message_content = self._parse_a2a_message(context, user_input)
            
            if self.output:
                self.output.progress(f"[A2A Doctor] Processing message from {sender_id}")
            
            # Process message through privacy agent (with extraction attempts)
            response = await self.doctor.process_message(sender_id, message_content)
            
            # Create A2A response
            a2a_response = self._create_a2a_response(response, "doctor")
            await safe_enqueue_event(event_queue, new_agent_text_message(a2a_response))
            
            if self.output:
                self.output.success(f"[A2A Doctor] Response sent (extraction attempted)")
                
        except Exception as e:
            error_msg = f"[A2A Doctor] Error processing message: {e}"
            if self.output:
                self.output.error(error_msg)
            await safe_enqueue_event(event_queue, new_agent_text_message(error_msg))

    def _parse_a2a_message(self, context: RequestContext, user_input: str) -> tuple[str, str]:
        """Parse A2A message to extract sender and content"""
        sender_id = "external"
        message_content = user_input or ""
        
        try:
            # Try to parse structured A2A message
            if user_input:
                try:
                    parsed = json.loads(user_input)
                    if isinstance(parsed, dict):
                        # Prefer payload.* first, then top-level for backward-compat
                        payload = parsed.get("payload", {}) if isinstance(parsed, dict) else {}

                        sender_id = (
                            (payload.get("src_id") or payload.get("sender_id"))
                            or parsed.get("src_id") or parsed.get("sender_id")
                            or "external"
                        )

                        message_content = (
                            payload.get("text") or payload.get("content")
                            or parsed.get("text") or parsed.get("content")
                            or user_input
                        )
                        
                        # Optional: log privacy context if available
                        privacy_ctx = payload.get("privacy_context") or parsed.get("privacy_context")
                        if privacy_ctx and self.output:
                            interaction = (privacy_ctx.get("interaction_type", "unknown")
                                           if isinstance(privacy_ctx, dict) else str(privacy_ctx))
                            if hasattr(self.output, 'debug'):
                                self.output.debug(f"Extraction opportunity: {interaction}")
                            else:
                                self.output.info(f"Extraction opportunity: {interaction}")
                            
                except json.JSONDecodeError:
                    message_content = user_input
                    
        except Exception as e:
            if self.output:
                self.output.warning(f"Failed to parse A2A message metadata: {e}")
        
        return sender_id, message_content

    def _create_a2a_response(self, response: str, agent_type: str) -> str:
        """Create A2A-compatible response message"""
        response_data = {
            "type": "PRIVACY_RESPONSE",
            "content": response,
            "agent_type": agent_type,
            "extraction_attempted": True,
            "timestamp": time.time(),
            "response_id": str(uuid.uuid4())
        }
        return json.dumps(response_data)


class A2APrivacySimulator:
    """A2A-specific privacy testing simulator"""

    def __init__(self, config: Optional[dict] = None, output=None):
        self.config = config or {}
        self.output = output
        
        # Create A2A-specific agents
        self.receptionist = A2AReceptionistAgent("A2A_Receptionist", config, output)
        self.doctor = A2ADoctorAgent("A2A_Doctor", config, output)
        
        # Create A2A executors
        self.receptionist_executor = A2AReceptionistExecutor(config, "A2A_Receptionist", output)
        self.doctor_executor = A2ADoctorExecutor(config, "A2A_Doctor", output)

    def set_network(self, network: Any) -> None:
        """Set the A2A network for both agents"""
        self.receptionist.set_network(network)
        self.doctor.set_network(network)

    async def simulate_conversation(self, patient_question: str, rounds: int = 3) -> List[Dict[str, Any]]:
        """
        Simulate privacy testing conversation via the A2A protocol (end-to-end).
        This version routes every message through the network so that
        A2ACommBackend.send() (and the A2A SDK, if available) are actually used.
        """
        if not getattr(self.receptionist, "agent_network", None) or not getattr(self.doctor, "agent_network", None):
            raise RuntimeError("Network is not set. Call set_network() before simulate_conversation().")

        conversation: List[Dict[str, Any]] = []
        current_text = patient_question

        # Round 0: Patient -> Receptionist (via Receptionist's send_to_agent to Doctor after processing)
        # We let the receptionist process patient's text locally to apply privacy, then send via network.
        for round_idx in range(1, rounds + 1):
            # Step A: Receptionist processes the patient (or doctor's previous) message
            receptionist_out = await self.receptionist.process_message(
                "Patient" if round_idx == 1 else "A2A_Doctor", current_text
            )

            # Step B: Receptionist sends the sanitized message to Doctor through A2A network
            # NOTE: send_to_agent() uses agent_network.route_message() under the hood
            net_resp_to_doc = await self.receptionist.send_to_agent("A2A_Doctor", receptionist_out)
            doc_recv_text = (net_resp_to_doc or {}).get("text", "")

            conversation.append({
                "round": round_idx,
                "sender": "A2A_Receptionist",
                "message": receptionist_out,
                "timestamp": time.time(),
                "protocol": "a2a",
                "privacy_protected": True
            })

            # Step C: Doctor processes the received text (may attempt extraction)
            doctor_out = await self.doctor.process_message("A2A_Receptionist", doc_recv_text)

            # Step D: Doctor replies back to Receptionist THROUGH the network
            net_resp_to_rcp = await self.doctor.send_to_agent("A2A_Receptionist", doctor_out)
            rcp_recv_text = (net_resp_to_rcp or {}).get("text", "")

            conversation.append({
                "round": round_idx,
                "sender": "A2A_Doctor",
                "message": doctor_out,
                "timestamp": time.time(),
                "protocol": "a2a",
                "extraction_attempted": True
            })

            # For the next round, let the "patient" be the doctor's output (simulating continued dialog)
            current_text = rcp_recv_text if rcp_recv_text else doctor_out

        return conversation

    async def run_privacy_test_batch(self, enhanced_questions: List[str], rounds_per_conversation: int = 3) -> Dict[str, Any]:
        """Run batch privacy testing via A2A protocol"""
        all_conversations = []
        
        for i, question in enumerate(enhanced_questions):
            conversation_id = f"a2a_conversation_{i+1}"
            
            try:
                messages = await self.simulate_conversation(question, rounds_per_conversation)
                
                conversation_data = {
                    "conversation_id": conversation_id,
                    "original_question": question,
                    "messages": messages,
                    "protocol": "a2a",
                    "timestamp": time.time(),
                    "a2a_metadata": {
                        "sdk_available": True,
                        "total_rounds": rounds_per_conversation,
                        "privacy_features_active": True
                    }
                }
                
                all_conversations.append(conversation_data)
                
                if self.output:
                    self.output.info(f"Completed A2A conversation {i+1}/{len(enhanced_questions)}")
                
            except Exception as e:
                if self.output:
                    self.output.error(f"A2A conversation {i+1} failed: {e}")
                continue

        return {
            "protocol": "a2a",
            "total_conversations": len(all_conversations),
            "conversations": all_conversations,
            "generation_timestamp": time.time(),
            "a2a_features": {
                "privacy_protection": True,
                "information_extraction": True,
                "sdk_integration": True
            }
        }
