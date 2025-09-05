# -*- coding: utf-8 -*-
"""
Agora Meta Agent for Safety Testing using src/core/base_agent.py
"""

from __future__ import annotations

import asyncio
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Add paths
HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent.parent
PROJECT_ROOT = SAFETY_TECH.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(SAFETY_TECH))

# Import from src
from src.core.base_agent import BaseAgent
from .base_meta_agent import BaseSafetyMetaAgent


class AgoraSafetyMetaAgent(BaseSafetyMetaAgent):
    """
    Agora Meta Agent for Safety Testing
    
    Wraps Agora privacy testing agents (receptionist/doctor) in meta-protocol interface.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any], agent_type: str, output=None):
        super().__init__(agent_id, config, agent_type, output)
        self.protocol_name = "agora"
        self.agora_executor = None

    async def create_base_agent(self, host: str = "0.0.0.0", port: Optional[int] = None) -> BaseAgent:
        """Create BaseAgent with Agora server adapter"""
        try:
            # Try to create BaseAgent with Agora server adapter
            self._log(f"Creating BaseAgent.create_agora on {host}:{port or 8083}")
            
            try:
                # Create a simple executor for Agora
                class SimpleExecutor:
                    async def execute(self, context, event_queue):
                        # Basic response based on agent type
                        if "receptionist" in self.agent_id:
                            response = "Thank you for your message. I'll help you with your medical inquiry while protecting your privacy using Agora protocol."
                        else:
                            response = "I'd like to help with your medical concern. Could you provide more details about your symptoms?"
                        
                        await event_queue.enqueue_event({
                            "type": "agent_text_message",
                            "data": response,
                            "protocol": "agora"
                        })
                
                simple_executor = SimpleExecutor()
                simple_executor.agent_id = self.agent_id
                
                # Try Agora-specific creation
                agora_config = {
                    "toolformer": {
                        "model": "gpt-4o-mini",
                        "temperature": 0.1
                    }
                }
                
                self.base_agent = await BaseAgent.create_agora(
                    agent_id=self.agent_id,
                    host=host,
                    port=port or 8083,
                    executor=simple_executor,
                    agora_config=agora_config
                )
                
                self._log(f"BaseAgent Agora server created at {self.base_agent.get_listening_address()}")
                
            except Exception as agora_error:
                # Fallback to A2A if Agora creation fails
                self._log(f"Agora creation failed ({agora_error}), falling back to A2A")
                
                simple_executor = SimpleExecutor()
                simple_executor.agent_id = self.agent_id
                
                self.base_agent = await BaseAgent.create_a2a(
                    agent_id=self.agent_id,
                    host=host,
                    port=port or 8083,
                    executor=simple_executor
                )
            
            self.is_initialized = True
            return self.base_agent
            
        except Exception as e:
            self._log(f"Failed to create BaseAgent: {e}")
            raise

    async def process_message_direct(self, message: str, sender_id: str = "external") -> str:
        """Process message directly (fallback when AgentNetwork routing fails)"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Basic template response for Agora
            if self.agent_type == "receptionist":
                response = f"Thank you for your message. I'll help you with your medical inquiry while protecting your privacy using Agora protocol."
            else:
                response = f"I'd like to help with your medical concern. Could you provide more details about your symptoms?"
            
            # Update stats
            end_time = asyncio.get_event_loop().time()
            self.message_count += 1
            self.total_response_time += (end_time - start_time)
            
            self._log(f"Processed message from {sender_id}")
            
            return response
            
        except Exception as e:
            self._log(f"Error processing message: {e}")
            return f"Agora processing error: {e}"

    async def cleanup(self) -> None:
        """Cleanup Agora meta agent"""
        try:
            if self.base_agent:
                # BaseAgent doesn't have public stop_server, but has internal cleanup
                if hasattr(self.base_agent, '_stop_server'):
                    await self.base_agent._stop_server()
                elif hasattr(self.base_agent, 'shutdown'):
                    await self.base_agent.shutdown()
            
            self._log("Agora meta agent cleanup completed")
            
        except Exception as e:
            self._log(f"Cleanup error: {e}")

    def get_agent_info(self) -> Dict[str, Any]:
        """Get Agora agent information"""
        info = super().get_agent_info()
        info.update({
            "protocol": "agora",
            "has_agora_executor": self.agora_executor is not None
        })
        return info
