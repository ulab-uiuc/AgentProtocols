# -*- coding: utf-8 -*-
"""
A2A Meta Agent for Safety Testing using src/core/base_agent.py
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SAFETY_TECH) not in sys.path:
    sys.path.insert(0, str(SAFETY_TECH))

# Import from src
from src.core.base_agent import BaseAgent
from .base_meta_agent import BaseSafetyMetaAgent

# Import streaming_queue A2A worker (reuse existing implementation)
try:
    STREAMING_QUEUE_PATH = PROJECT_ROOT / "script" / "streaming_queue"
    sys.path.insert(0, str(STREAMING_QUEUE_PATH))
    from protocol_backend.a2a.worker import QAAgentExecutor
    A2A_WORKER_AVAILABLE = True
except ImportError:
    A2A_WORKER_AVAILABLE = False


class A2ASafetyMetaAgent(BaseSafetyMetaAgent):
    """
    A2A Meta Agent for Safety Testing
    
    Wraps A2A privacy testing agents (receptionist/doctor) in meta-protocol interface.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any], agent_type: str, output=None):
        super().__init__(agent_id, config, agent_type, output)
        self.protocol_name = "a2a"
        self.a2a_executor: Optional[QAAgentExecutor] = None

    async def create_base_agent(self, host: str = "0.0.0.0", port: Optional[int] = None) -> BaseAgent:
        """Create BaseAgent with A2A server adapter"""
        try:
            # Convert config for A2A worker
            qa_config = self._convert_config_for_executor()
            
            if A2A_WORKER_AVAILABLE:
                # Create A2A worker executor
                self.a2a_executor = QAAgentExecutor(qa_config)
                
                # Create BaseAgent with A2A server adapter (default)
                self._log(f"Creating BaseAgent.create_a2a on {host}:{port or 8085}")
                self.base_agent = await BaseAgent.create_a2a(
                    agent_id=self.agent_id,
                    host=host,
                    port=port or 8085,
                    executor=self.a2a_executor
                )
                
                self._log(f"BaseAgent A2A server created at {self.base_agent.get_listening_address()}")
            else:
                # Fallback: create basic A2A BaseAgent
                self._log("A2A Worker not available, using basic A2A BaseAgent")
                
                # Create a simple executor for fallback
                class SimpleExecutor:
                    async def execute(self, context, event_queue):
                        # Basic response based on agent type
                        if "receptionist" in self.agent_id:
                            response = "Thank you for your message. I'll help you with your medical inquiry while protecting your privacy using A2A protocol."
                        else:
                            response = "I'd like to help with your medical concern. Could you provide more details about your symptoms?"
                        
                        await event_queue.enqueue_event({
                            "type": "agent_text_message",
                            "data": response,
                            "protocol": "a2a_fallback"
                        })
                
                simple_executor = SimpleExecutor()
                simple_executor.agent_id = self.agent_id
                
                self.base_agent = await BaseAgent.create_a2a(
                    agent_id=self.agent_id,
                    host=host,
                    port=port or 8085,
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
            
            # Use A2A executor if available
            if self.a2a_executor and hasattr(self.a2a_executor, '_worker'):
                response = await self.a2a_executor._worker.answer(message)
            else:
                # Basic template response for A2A
                if self.agent_type == "receptionist":
                    response = f"Thank you for your message. I'll help you with your medical inquiry while protecting your privacy using A2A protocol."
                else:
                    response = f"I'd like to help with your medical concern. Could you provide more details about your symptoms?"
            
            # Update stats
            end_time = asyncio.get_event_loop().time()
            self.message_count += 1
            self.total_response_time += (end_time - start_time)
            
            self._log(f"Processed message from {sender_id}")
            
            return response or "A2A agent response"
            
        except Exception as e:
            self._log(f"Error processing message: {e}")
            return f"A2A processing error: {e}"

    async def cleanup(self) -> None:
        """Cleanup A2A meta agent"""
        try:
            if self.base_agent:
                # BaseAgent doesn't have public stop_server, but has internal cleanup
                if hasattr(self.base_agent, '_stop_server'):
                    await self.base_agent._stop_server()
                elif hasattr(self.base_agent, 'shutdown'):
                    await self.base_agent.shutdown()
            
            self._log("A2A meta agent cleanup completed")
            
        except Exception as e:
            self._log(f"Cleanup error: {e}")

    def get_agent_info(self) -> Dict[str, Any]:
        """Get A2A agent information"""
        info = super().get_agent_info()
        info.update({
            "protocol": "a2a",
            "has_a2a_executor": self.a2a_executor is not None,
            "a2a_worker_available": A2A_WORKER_AVAILABLE
        })
        return info
