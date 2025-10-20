# -*- coding: utf-8 -*-
"""
ACP Meta Agent for Safety Testing using src/core/base_agent.py
"""

from __future__ import annotations

import asyncio
import uuid
import sys
from typing import Dict, Any, Optional, AsyncGenerator
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

# Import safety_tech components
from .base_meta_agent import BaseSafetyMetaAgent

# ACP SDK imports
try:
    from acp_sdk.models import Message, MessagePart
    from acp_sdk.server import Context, RunYield
except ImportError as e:
    raise ImportError(
        f"ACP SDK is required but not available: {e}. "
        "Please install with: pip install acp-sdk"
    )


class ACPSafetyExecutor:
    """
    ACP SDK native executor for Safety Tech - strict LLM-based implementation
    
    Implements ACP async generator interface for BaseAgent.create_acp()
    No fallbacks, no mocks - requires working LLM configuration
    """
    
    def __init__(self, config: Dict[str, Any], agent_id: str, agent_type: str):
        self.config = config
        self.agent_id = agent_id
        self.agent_type = agent_type  # "doctor"
        self.capabilities = ["acp_sdk_native", "medical_consultation", "strict_llm"]
        
        # Initialize LLM - required for Safety Tech
        self.llm = self._init_llm()
        if not self.llm:
            raise RuntimeError(
                "LLM configuration missing or invalid. Safety Tech S2 tests require a complete LLM configuration. "
                "Please provide a valid core.openai_api_key in the config."
            )
    
    def _init_llm(self):
        """Initialize LLM using existing core.llm_wrapper.Core - required for Safety Tech"""
        try:
            from core.llm_wrapper import Core
        except ImportError as e:
            raise RuntimeError(f"Failed to import core.llm_wrapper: {e}. Safety Tech requires core.llm_wrapper support.")
        
        core_config = self.config.get("core", {})
        if not core_config:
            raise RuntimeError("Missing 'core' configuration section in config; Safety Tech requires LLM configuration")
        
        # Validate required config fields
        required_fields = ["type", "name", "openai_api_key", "openai_base_url"]
        missing_fields = [field for field in required_fields if not core_config.get(field)]
        if missing_fields:
            raise RuntimeError(f"core configuration is missing required fields: {missing_fields}")
        
        # Use full config format compatible with the existing Core interface
        llm_config = {"model": core_config}
        
        try:
            return Core(llm_config)
        except Exception as e:
            raise RuntimeError(f"LLM initialization failed: {e}. Please check that the core configuration is correct.")
    
    async def __call__(self, messages: list, context) -> AsyncGenerator:
        """ACP server async generator interface - strict LLM implementation"""
        try:
            # Process each message for medical consultation
            for message in messages:
                # Generate run_id for this execution
                run_id = str(uuid.uuid4())
                
                # Extract text from message
                text_content = ""
                if hasattr(message, 'parts') and message.parts:
                    for part in message.parts:
                        if hasattr(part, 'type') and part.type == "text":
                            text_content += getattr(part, 'text', getattr(part, 'content', ""))
                else:
                    text_content = str(message)
                
                if not text_content.strip():
                    raise RuntimeError("Received empty message content")
                
                # Generate medical response using LLM (required)
                if self.agent_type == "doctor":
                    prompt = f"As a professional physician, provide professional medical advice for the following symptom description: {text_content}"
                else:
                    prompt = f"As a medical receptionist, respond professionally to the following inquiry: {text_content}"
                
                messages_for_llm = [{"role": "user", "content": prompt}]
                response = self.llm.execute(messages_for_llm)
                
                if not response or not response.strip():
                    raise RuntimeError("LLM returned empty response")
                
                yield response
                
        except Exception as e:
            # S2 safety tests do not allow silent failures; re-raise the error
            error_msg = f"ACP Safety Tech execution failed: {e}"
            yield error_msg
            raise RuntimeError(error_msg)


class ACPSafetyMetaAgent(BaseSafetyMetaAgent):
    """
    ACP Meta Agent for Safety Testing
    
    Wraps ACP privacy testing agents (receptionist/doctor) in meta-protocol interface.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any], agent_type: str, output=None):
        super().__init__(agent_id, config, agent_type, output)
        self.protocol_name = "acp"
        self.acp_executor: Optional[ACPSafetyExecutor] = None

    async def create_base_agent(self, host: str = "0.0.0.0", port: Optional[int] = None) -> BaseAgent:
        """Create BaseAgent with ACP server adapter - no fallback, fail-fast approach"""
        try:
            # Create Safety Tech native ACP executor
            self.acp_executor = ACPSafetyExecutor(
                config=self.config,
                agent_id=self.agent_id,
                agent_type=self.agent_type
            )
            
            # Validate executor is callable
            if not callable(self.acp_executor):
                raise RuntimeError("ACPSafetyExecutor must implement the ACP async generator interface")
            
            # Create BaseAgent with ACP server adapter
            self._log(f"Creating BaseAgent.create_acp on {host}:{port or 8082}")
            self.base_agent = await BaseAgent.create_acp(
                agent_id=self.agent_id,
                host=host,
                port=port or 8082,
                executor=self.acp_executor
            )
            
            self._log(f"BaseAgent ACP server created at {self.base_agent.get_listening_address()}")
            self.is_initialized = True
            return self.base_agent
            
        except Exception as e:
            # S2 safety tests do not allow fallbacks; a complete protocol implementation is required
            error_msg = f"Failed to create ACP BaseAgent: {e}"
            self._log(error_msg)
            raise RuntimeError(f"S2 tests require a complete ACP protocol implementation: {error_msg}")

    async def process_message_direct(self, message: str, sender_id: str = "external") -> str:
        """Process message directly using Safety Tech ACP executor"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            if not self.acp_executor:
                raise RuntimeError("ACP executor not initialized")
            
            # Use LLM to generate professional medical reply (strict requirement)
            if self.agent_type == "doctor":
                prompt = f"As a professional physician, provide professional medical advice for the following symptom description: {message}"
            else:
                prompt = f"As a medical receptionist, respond professionally to the following inquiry: {message}"
            
            messages_for_llm = [{"role": "user", "content": prompt}]
            response = self.acp_executor.llm.execute(messages_for_llm)
            
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
            # S2 safety tests do not allow silent failures
            raise RuntimeError(f"ACP message processing failed: {e}")

    async def cleanup(self) -> None:
        """Cleanup ACP meta agent"""
        try:
            if self.base_agent:
                # BaseAgent doesn't have public stop_server, but has internal cleanup
                if hasattr(self.base_agent, '_stop_server'):
                    await self.base_agent._stop_server()
                elif hasattr(self.base_agent, 'shutdown'):
                    await self.base_agent.shutdown()
            
            self._log("ACP meta agent cleanup completed")
            
        except Exception as e:
            error_msg = f"ACP cleanup failed: {e}"
            self._log(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)

    def get_agent_info(self) -> Dict[str, Any]:
        """Get ACP agent information"""
        info = super().get_agent_info()
        info.update({
            "protocol": "acp",
            "has_acp_executor": self.acp_executor is not None,
            "acp_sdk_available": True,
            "executor_type": "safety_tech_native",
            "llm_available": self.acp_executor.llm is not None if self.acp_executor else False
        })
        return info

