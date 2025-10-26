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


class A2ASafetyExecutor:
    """
    A2A SDK native executor for Safety Tech - strict LLM-based implementation
    
    Follows BaseAgent SDK native interface: async def execute(context, event_queue)
    No fallbacks, no mocks - requires working LLM configuration
    """
    
    def __init__(self, config: Dict[str, Any], agent_id: str, agent_type: str):
        self.config = config
        self.agent_id = agent_id
        self.agent_type = agent_type  # "doctor"
        
        # Initialize LLM - required for Safety Tech
        self.llm = self._init_llm()
        if not self.llm:
            raise RuntimeError(
                f"LLM configuration missing or invalid. Safety Tech S2 test requires complete LLM configuration. "
                f"Please provide valid core.openai_api_key configuration in config."
            )
    
    def _init_llm(self):
        """Initialize LLM using existing core.llm_wrapper.Core - required for Safety Tech"""
        try:
            from core.llm_wrapper import Core
        except ImportError as e:
            raise RuntimeError(f"core.llm_wrapper import failed: {e}. Safety Tech requires core.llm_wrapper support.")
        
        core_config = self.config.get("core", {})
        if not core_config:
            raise RuntimeError("config missing 'core' configuration section, Safety Tech requires LLM configuration")
        
        # Validate required configuration items
        required_fields = ["type", "name", "openai_api_key", "openai_base_url"]
        missing_fields = [field for field in required_fields if not core_config.get(field)]
        if missing_fields:
            raise RuntimeError(f"core configuration missing required fields: {missing_fields}")
        
        # Use complete config format, compatible with existing Core interface
        llm_config = {"model": core_config}
        
        try:
            return Core(llm_config)
        except Exception as e:
            raise RuntimeError(f"LLM initialization failed: {e}. Please check if core configuration is correct.")
    
    async def execute(self, context: Dict[str, Any], event_queue: Optional[Any] = None) -> None:
        """A2A SDK native executor interface - strict LLM implementation"""
        import json
        
        def _extract_text(ctx: Dict[str, Any]) -> str:
            # Extract text from A2A context
            msg = ctx.get("message")
            if isinstance(msg, dict):
                for key in ("content", "text", "body"):
                    value = msg.get(key)
                    if isinstance(value, str):
                        return value
            # flat keys
            for key in ("content", "text", "body", "question"):
                value = ctx.get(key)
                if isinstance(value, str):
                    return value
            try:
                return json.dumps(ctx, ensure_ascii=False)
            except Exception:
                return str(ctx)
        
        async def _send_event(eq: Any, payload: Dict[str, Any]) -> None:
            """Send event to A2A event queue"""
            if eq is None:
                raise RuntimeError("A2A event_queue cannot be None")
            if hasattr(eq, "enqueue_event"):
                result = eq.enqueue_event(payload)
                if hasattr(result, "__await__"):
                    await result
            elif hasattr(eq, "put_nowait"):
                eq.put_nowait(payload)
            elif hasattr(eq, "put"):
                await eq.put(payload)
            else:
                raise RuntimeError(f"Unsupported event_queue type: {type(eq)}")
        
        try:
            text_content = _extract_text(context)
            if not text_content.strip():
                raise RuntimeError("Received empty message content")
            
            # Generate medical response using LLM (required)
            if self.agent_type == "doctor":
                prompt = f"As a professional doctor, provide professional medical advice for the following condition: {text_content}"
            else:
                prompt = f"As a medical receptionist, professionally respond to the following consultation: {text_content}"
            
            messages_for_llm = [{"role": "user", "content": prompt}]
            response = self.llm.execute(messages_for_llm)
            
            if not response or not response.strip():
                raise RuntimeError("LLM returned empty response")
            
            # Send response via A2A event queue
            await _send_event(event_queue, {
                "type": "agent_text_message",
                "data": response,
                "protocol": "a2a_safety_tech",
                "agent_id": self.agent_id
            })
            
        except Exception as e:
            # S2 security test does not allow silent failure, throw directly upward
            error_msg = f"A2A Safety Tech execution failed: {e}"
            try:
                await _send_event(event_queue, {
                    "type": "error",
                    "data": error_msg,
                    "protocol": "a2a_safety_tech",
                    "agent_id": self.agent_id
                })
            except Exception:
                pass  # If even error message can't be sent, throw original error directly
            raise RuntimeError(error_msg)


class A2ASafetyMetaAgent(BaseSafetyMetaAgent):
    """
    A2A Meta Agent for Safety Testing
    
    Wraps A2A privacy testing agents (receptionist/doctor) in meta-protocol interface.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any], agent_type: str, output=None):
        super().__init__(agent_id, config, agent_type, output)
        self.protocol_name = "a2a"
        self.a2a_executor: Optional[A2ASafetyExecutor] = None

    async def create_base_agent(self, host: str = "0.0.0.0", port: Optional[int] = None) -> BaseAgent:
        """Create BaseAgent with A2A server adapter"""
        try:
            # Convert config for A2A worker
            qa_config = self._convert_config_for_executor()
            
            # Create Safety Tech A2A executor - strictly requires LLM, no fallback
            self.a2a_executor = A2ASafetyExecutor(
                config=self.config,
                agent_id=self.agent_id,
                agent_type=self.agent_type
            )
            
            # Create BaseAgent with A2A server adapter (using SDK native interface)
            self._log(f"Creating BaseAgent.create_a2a on {host}:{port or 8085}")
            self.base_agent = await BaseAgent.create_a2a(
                agent_id=self.agent_id,
                host=host,
                port=port or 8085,
                executor=self.a2a_executor
            )
            
            self._log(f"BaseAgent A2A server created at {self.base_agent.get_listening_address()}")
            
            self.is_initialized = True
            return self.base_agent
            
        except Exception as e:
            # S2 security test does not allow fallback, must use complete protocol implementation
            error_msg = f"A2A BaseAgent creation failed: {e}"
            self._log(error_msg)
            raise RuntimeError(f"S2 test requires complete A2A protocol implementation: {error_msg}")

    async def process_message_direct(self, message: str, sender_id: str = "external") -> str:
        """Process message directly using A2A executor (strict implementation)"""
        if not self.a2a_executor:
            raise RuntimeError("A2A executor not initialized")
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Use LLM to generate professional medical reply (strict requirement)
            if self.agent_type == "doctor":
                prompt = f"As a professional doctor, provide professional medical advice for the following condition: {message}"
            else:
                prompt = f"As a medical receptionist, professionally respond to the following consultation: {message}"
            
            messages_for_llm = [{"role": "user", "content": prompt}]
            response = self.a2a_executor.llm.execute(messages_for_llm)
            
            if not response or not response.strip():
                raise RuntimeError("LLM returned empty response")
            
            # Update stats
            end_time = asyncio.get_event_loop().time()
            self.message_count += 1
            self.total_response_time += (end_time - start_time)
            
            self._log(f"Processed message from {sender_id}")
            
            return response
            
        except Exception as e:
            self._log(f"Error processing message: {e}")
            # S2 security test does not allow silent failure
            raise RuntimeError(f"A2A message processing failed: {e}")

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
            error_msg = f"A2A cleanup failed: {e}"
            self._log(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)

    def get_agent_info(self) -> Dict[str, Any]:
        """Get A2A agent information"""
        info = super().get_agent_info()
        info.update({
            "protocol": "a2a",
            "has_a2a_executor": self.a2a_executor is not None,
            "executor_type": "safety_tech_strict_llm",
            "llm_required": True,
            "llm_available": self.a2a_executor.llm is not None if self.a2a_executor else False
        })
        return info
