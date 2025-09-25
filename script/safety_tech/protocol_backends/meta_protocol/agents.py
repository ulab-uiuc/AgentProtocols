# -*- coding: utf-8 -*-
"""
Meta Protocol Agent Adapters for Privacy Testing
Bridges between meta protocol routing and privacy testing agent logic.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional, List

# Import core agent classes
try:
    from ...core.privacy_agent_base import ReceptionistAgent, NosyDoctorAgent
except ImportError:
    try:
        from core.privacy_agent_base import ReceptionistAgent, NosyDoctorAgent
    except ImportError:
        # Create minimal base classes if not available
        class ReceptionistAgent:
            def __init__(self, agent_id):
                self.agent_id = agent_id
                self.agent_network = None
            async def process_patient_message(self, message):
                return f"Receptionist processed: {message}"
        
        class NosyDoctorAgent:
            def __init__(self, agent_id):
                self.agent_id = agent_id
                self.agent_network = None
            async def process_patient_message(self, message):
                return f"Doctor processed: {message}"


class MetaReceptionistAgent(ReceptionistAgent):
    """Meta protocol receptionist agent with intelligent routing."""

    def __init__(self, agent_id: str, selected_protocol: str, routing_decision=None):
        super().__init__(agent_id)
        self.selected_protocol = selected_protocol
        self.routing_decision = routing_decision
        self.underlying_agent = None  # Will be set by coordinator

    async def send_to_agent(self, target_id: str, message: str) -> Dict[str, Any]:
        """Send message through meta protocol network with intelligent routing."""
        if not self.agent_network:
            raise RuntimeError("No meta network set. Call set_network() first.")
        
        # Add meta protocol routing information
        payload = {
            "text": message,
            "meta_info": {
                "selected_protocol": self.selected_protocol,
                "routing_confidence": self.routing_decision.confidence if self.routing_decision else 0.8,
                "sender_type": "receptionist"
            }
        }
        
        response = await self.agent_network.route_message(self.agent_id, target_id, payload)
        return response or {"raw": None, "text": ""}

    def get_protocol_info(self) -> Dict[str, Any]:
        """Get meta protocol routing information."""
        return {
            "selected_protocol": self.selected_protocol,
            "routing_decision": {
                "confidence": self.routing_decision.confidence,
                "reasoning": self.routing_decision.reasoning,
                "strategy": self.routing_decision.strategy
            } if self.routing_decision else None,
            "agent_type": "meta_receptionist"
        }


class MetaNosyDoctorAgent(NosyDoctorAgent):
    """Meta protocol nosy doctor agent with intelligent routing."""

    def __init__(self, agent_id: str, selected_protocol: str, routing_decision=None):
        super().__init__(agent_id)
        self.selected_protocol = selected_protocol
        self.routing_decision = routing_decision
        self.underlying_agent = None  # Will be set by coordinator

    async def send_to_agent(self, target_id: str, message: str) -> Dict[str, Any]:
        """Send message through meta protocol network with intelligent routing."""
        if not self.agent_network:
            raise RuntimeError("No meta network set. Call set_network() first.")
        
        # Add meta protocol routing information
        payload = {
            "text": message,
            "meta_info": {
                "selected_protocol": self.selected_protocol,
                "routing_confidence": self.routing_decision.confidence if self.routing_decision else 0.8,
                "sender_type": "doctor"
            }
        }
        
        response = await self.agent_network.route_message(self.agent_id, target_id, payload)
        return response or {"raw": None, "text": ""}

    def get_protocol_info(self) -> Dict[str, Any]:
        """Get meta protocol routing information."""
        return {
            "selected_protocol": self.selected_protocol,
            "routing_decision": {
                "confidence": self.routing_decision.confidence,
                "reasoning": self.routing_decision.reasoning,
                "strategy": self.routing_decision.strategy
            } if self.routing_decision else None,
            "agent_type": "meta_doctor"
        }


class MetaReceptionistExecutor:
    """Meta protocol executor for receptionist agent."""

    def __init__(self, agent: MetaReceptionistAgent, output=None):
        self.agent = agent
        self.output = output
        self.message_count = 0

    async def execute(self, context, event_queue):
        """Execute receptionist logic with meta protocol routing."""
        try:
            # Get user input from context
            user_input = getattr(context, 'user_input', '') or getattr(context, 'text', '')
            if not user_input:
                await self._send_response(event_queue, "I'm here to help you. How can I assist you today?")
                return

            self.message_count += 1
            
            # Process with privacy-aware logic
            response = await self.agent.process_patient_message(user_input)
            
            # Add meta protocol information to response
            meta_response = f"[META-{self.agent.selected_protocol.upper()}] {response}"
            
            await self._send_response(event_queue, meta_response)
            
            if self.output:
                self.output.progress(f"[MetaReceptionist] Processed message via {self.agent.selected_protocol.upper()}")

        except Exception as e:
            error_msg = f"Meta receptionist error: {str(e)}"
            await self._send_response(event_queue, error_msg)
            if self.output:
                self.output.error(f"[MetaReceptionist] {error_msg}")

    async def _send_response(self, event_queue, message: str):
        """Send response through event queue."""
        event = {
            "type": "agent_text_message",
            "data": message,
            "protocol": f"meta_{self.agent.selected_protocol}",
            "timestamp": time.time()
        }
        await event_queue.enqueue_event(event)


class MetaDoctorExecutor:
    """Meta protocol executor for doctor agent."""

    def __init__(self, agent: MetaNosyDoctorAgent, output=None):
        self.agent = agent
        self.output = output
        self.message_count = 0

    async def execute(self, context, event_queue):
        """Execute doctor logic with meta protocol routing."""
        try:
            # Get user input from context
            user_input = getattr(context, 'user_input', '') or getattr(context, 'text', '')
            if not user_input:
                await self._send_response(event_queue, "I'm ready to provide medical assistance. What are your symptoms?")
                return

            self.message_count += 1
            
            # Process with nosy doctor logic (may leak privacy)
            response = await self.agent.process_patient_message(user_input)
            
            # Add meta protocol information to response
            meta_response = f"[META-{self.agent.selected_protocol.upper()}] {response}"
            
            await self._send_response(event_queue, meta_response)
            
            if self.output:
                self.output.progress(f"[MetaDoctor] Processed message via {self.agent.selected_protocol.upper()}")

        except Exception as e:
            error_msg = f"Meta doctor error: {str(e)}"
            await self._send_response(event_queue, error_msg)
            if self.output:
                self.output.error(f"[MetaDoctor] {error_msg}")

    async def _send_response(self, event_queue, message: str):
        """Send response through event queue."""
        event = {
            "type": "agent_text_message", 
            "data": message,
            "protocol": f"meta_{self.agent.selected_protocol}",
            "timestamp": time.time()
        }
        await event_queue.enqueue_event(event)


# Factory functions for creating meta protocol agents
def create_meta_receptionist(agent_id: str, selected_protocol: str, routing_decision=None) -> MetaReceptionistAgent:
    """Create a meta protocol receptionist agent."""
    return MetaReceptionistAgent(agent_id, selected_protocol, routing_decision)


def create_meta_doctor(agent_id: str, selected_protocol: str, routing_decision=None) -> MetaNosyDoctorAgent:
    """Create a meta protocol doctor agent.""" 
    return MetaNosyDoctorAgent(agent_id, selected_protocol, routing_decision)


def create_meta_receptionist_executor(agent: MetaReceptionistAgent, output=None) -> MetaReceptionistExecutor:
    """Create a meta protocol receptionist executor."""
    return MetaReceptionistExecutor(agent, output)


def create_meta_doctor_executor(agent: MetaNosyDoctorAgent, output=None) -> MetaDoctorExecutor:
    """Create a meta protocol doctor executor."""
    return MetaDoctorExecutor(agent, output)
