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

# Import streaming_queue ACP worker (reuse existing implementation)
try:
    STREAMING_QUEUE_PATH = PROJECT_ROOT / "script" / "streaming_queue"
    sys.path.insert(0, str(STREAMING_QUEUE_PATH))
    from protocol_backend.acp.worker import ACPWorkerExecutor
    ACP_WORKER_AVAILABLE = True
except ImportError:
    ACP_WORKER_AVAILABLE = False

# ACP SDK imports
try:
    from acp_sdk.models import Message, MessagePart
    from acp_sdk.server import Context, RunYield
    ACP_SDK_AVAILABLE = True
except ImportError:
    ACP_SDK_AVAILABLE = False


class ACPExecutorWrapper:
    """
    Wrapper to adapt ACPWorkerExecutor to ACP server async generator interface
    """
    
    def __init__(self, acp_worker_executor):
        self.acp_worker_executor = acp_worker_executor
        self.capabilities = ["text_processing", "async_generation", "acp_sdk_1.0.3"]
    
    async def __call__(self, messages: list, context) -> AsyncGenerator:
        """ACP server async generator interface"""
        try:
            # Process each message
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
                
                # Call LLM through worker
                if hasattr(self.acp_worker_executor, '_worker') and hasattr(self.acp_worker_executor._worker, 'answer'):
                    llm_result = await self.acp_worker_executor._worker.answer(text_content)
                    yield llm_result
                else:
                    # Fallback
                    if ACP_SDK_AVAILABLE:
                        result_message = await self.acp_worker_executor.process_message(message, run_id)
                        if isinstance(result_message, Message):
                            yield result_message
                        else:
                            yield "ACP processing completed"
                    else:
                        yield f"Processed: {text_content[:100]}..."
                
        except Exception as e:
            yield f"ACP processing error: {e}"


class ACPSafetyMetaAgent(BaseSafetyMetaAgent):
    """
    ACP Meta Agent for Safety Testing
    
    Wraps ACP privacy testing agents (receptionist/doctor) in meta-protocol interface.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any], agent_type: str, output=None):
        super().__init__(agent_id, config, agent_type, output)
        self.protocol_name = "acp"
        self.acp_executor: Optional[ACPWorkerExecutor] = None
        self.executor_wrapper: Optional[ACPExecutorWrapper] = None

    async def create_base_agent(self, host: str = "0.0.0.0", port: Optional[int] = None) -> BaseAgent:
        """Create BaseAgent with ACP server adapter"""
        try:
            # Convert config for ACP worker
            qa_config = self._convert_config_for_executor()
            
            if ACP_WORKER_AVAILABLE:
                # Create ACP worker executor
                self.acp_executor = ACPWorkerExecutor(qa_config)
                
                # Wrap in async generator interface
                self.executor_wrapper = ACPExecutorWrapper(self.acp_executor)
                
                if not callable(self.executor_wrapper):
                    raise RuntimeError("ACPExecutorWrapper must be callable as async generator")
                
                # Create BaseAgent with ACP server adapter
                self._log(f"Creating BaseAgent.create_acp on {host}:{port or 8082}")
                self.base_agent = await BaseAgent.create_acp(
                    agent_id=self.agent_id,
                    host=host,
                    port=port or 8082,
                    executor=self.executor_wrapper
                )
                
                self._log(f"BaseAgent ACP server created at {self.base_agent.get_listening_address()}")
            else:
                # Fallback: create basic A2A BaseAgent (default)
                self._log("ACP Worker not available, using basic A2A BaseAgent")
                
                # Create a simple executor for fallback
                class SimpleExecutor:
                    async def execute(self, context, event_queue):
                        # Basic response based on agent type
                        if "receptionist" in self.agent_id:
                            response = "Thank you for your message. I'll help you with your medical inquiry while protecting your privacy."
                        else:
                            response = "I'd like to help with your medical concern. Could you provide more details about your symptoms?"
                        
                        await event_queue.enqueue_event({
                            "type": "agent_text_message",
                            "data": response,
                            "protocol": "acp_fallback"
                        })
                
                simple_executor = SimpleExecutor()
                simple_executor.agent_id = self.agent_id
                
                self.base_agent = await BaseAgent.create_a2a(
                    agent_id=self.agent_id,
                    host=host,
                    port=port or 8082,
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
            
            # Use ACP executor if available
            if self.acp_executor and hasattr(self.acp_executor, '_worker'):
                response = await self.acp_executor._worker.answer(message)
            else:
                # Basic template response
                if self.agent_type == "receptionist":
                    response = f"Thank you for your message. I'll help you with your medical inquiry while protecting your privacy."
                else:
                    response = f"I'd like to help with your medical concern. Could you provide more details about your symptoms?"
            
            # Update stats
            end_time = asyncio.get_event_loop().time()
            self.message_count += 1
            self.total_response_time += (end_time - start_time)
            
            self._log(f"Processed message from {sender_id}")
            
            return response or "ACP agent response"
            
        except Exception as e:
            self._log(f"Error processing message: {e}")
            return f"ACP processing error: {e}"

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
            self._log(f"Cleanup error: {e}")

    def get_agent_info(self) -> Dict[str, Any]:
        """Get ACP agent information"""
        info = super().get_agent_info()
        info.update({
            "protocol": "acp",
            "has_acp_executor": self.acp_executor is not None,
            "has_executor_wrapper": self.executor_wrapper is not None,
            "acp_worker_available": ACP_WORKER_AVAILABLE,
            "acp_sdk_available": ACP_SDK_AVAILABLE
        })
        return info
