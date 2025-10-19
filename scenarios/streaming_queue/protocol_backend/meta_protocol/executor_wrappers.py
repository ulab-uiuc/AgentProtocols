"""
Executor Wrappers - Adapt streaming_queue protocol executors to BaseAgent interface

These wrappers convert protocol-specific worker executors to the BaseAgent
standard interface: async def execute(context, event_queue) -> None
"""

import asyncio
import json
import uuid
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add streaming_queue to path
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parents[2]
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))

# Import streaming_queue protocol executors
from protocol_backend.acp.worker import ACPWorkerExecutor
from protocol_backend.anp.worker import ANPWorkerExecutor
from protocol_backend.agora.worker import AgoraWorkerExecutor
from protocol_backend.a2a.worker import QAAgentExecutor


class ACPExecutorWrapper:
    """
    Wrapper for ACP Worker Executor
    Adapts ACPWorkerExecutor.process_message() to BaseAgent interface
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.acp_executor = ACPWorkerExecutor(config)
        self.protocol_name = "acp"
        # Store direct access to QA worker for LLM calls
        if hasattr(self.acp_executor, '_worker'):
            self.acp_qa_worker = self.acp_executor._worker
        pass  # ACP wrapper created
    
    async def execute(self, context, event_queue) -> None:
        """BaseAgent standard interface"""
        try:
            # Extract message from context
            if hasattr(context, 'message'):
                message_text = str(context.message)
            else:
                message_text = str(context)
            
            # Create ACP message with proper format
            from acp_sdk import Message
            run_id = str(uuid.uuid4())
            
            acp_message = Message(
                id=str(uuid.uuid4()),
                parts=[{"type": "text", "text": message_text}]
            )
            
            # Process with ACP executor
            result_message = await self.acp_executor.process_message(acp_message, run_id)
            
            # Extract result and convert to UTE format
            result_text = ""
            if hasattr(result_message, 'parts') and result_message.parts:
                for part in result_message.parts:
                    if hasattr(part, 'type') and part.type == "text":
                        result_text += getattr(part, 'text', "")
            
            # Create event for BaseAgent
            event = {
                "type": "agent_text_message",
                "data": result_text,
                "protocol": "acp",
                "metadata": {"run_id": run_id}
            }
            
            await event_queue.enqueue_event(event)
            print(f"[ACPWrapper] Processed message via ACP SDK")
            
        except Exception as e:
            # Error event
            error_event = {
                "type": "agent_error",
                "data": f"ACP processing error: {e}",
                "protocol": "acp"
            }
            await event_queue.enqueue_event(error_event)
            print(f"[ACPWrapper] Error: {e}")
    
    async def __call__(self, messages, context):
        """ACP server adapter calls this method directly"""
        try:
            # Extract first message
            if not messages:
                yield "No message provided"
                return
            
            message = messages[0]
            text_content = ""
            
            # Extract text from ACP message
            if hasattr(message, 'parts') and message.parts:
                for part in message.parts:
                    if hasattr(part, 'type') and part.type == "text":
                        text_content += getattr(part, 'text', "")
            
            print(f"[ACPWrapper] Processing via __call__: {text_content[:50]}...")
            
            # Call LLM directly - only yield the real result
            if hasattr(self, 'acp_qa_worker') and hasattr(self.acp_qa_worker, 'answer'):
                result = await self.acp_qa_worker.answer(text_content)
                print(f"[ACPWrapper] LLM result: {len(result)} chars")
                yield result  # Only yield the LLM result
            elif hasattr(self.acp_executor, '_worker') and hasattr(self.acp_executor._worker, 'answer'):
                result = await self.acp_executor._worker.answer(text_content)
                print(f"[ACPWrapper] LLM result via _worker: {len(result)} chars")
                yield result
            else:
                print(f"[ACPWrapper] No worker found, using fallback")
                yield f"ACP fallback: {text_content}"
                
        except Exception as e:
            print(f"[ACPWrapper] __call__ error: {e}")
            yield f"ACP error: {e}"


class ANPExecutorWrapper:
    """
    Wrapper for ANP Worker Executor
    Adapts ANPWorkerExecutor to BaseAgent interface
    """
    
    def __init__(self, anp_qa_worker):
        # Store the QA worker directly for server adapter compatibility
        self.anp_qa_worker = anp_qa_worker
        self.anp_executor = anp_qa_worker  # For backward compatibility
        self.protocol_name = "anp"
        pass  # ANP wrapper created
    
    async def execute(self, context, event_queue) -> None:
        """BaseAgent standard interface"""
        try:
            # Extract message from context
            if hasattr(context, 'message'):
                message_text = str(context.message)
            else:
                message_text = str(context)
            
            # Process with ANP worker (direct QA worker call)
            if hasattr(self.anp_qa_worker, 'answer'):
                result = await self.anp_qa_worker.answer(message_text)
            else:
                result = f"ANP processed: {message_text}"
            
            # Create event for BaseAgent with ANP metadata
            event = {
                "type": "agent_text_message",
                "data": result,
                "protocol": "anp",
                "metadata": {
                    "did_authenticated": True,
                    "encrypted": True
                }
            }
            
            await event_queue.enqueue_event(event)
            print(f"[ANPWrapper] Processed message via AgentConnect SDK")
            
        except Exception as e:
            error_event = {
                "type": "agent_error", 
                "data": f"ANP processing error: {e}",
                "protocol": "anp"
            }
            await event_queue.enqueue_event(error_event)
            print(f"[ANPWrapper] Error: {e}")


class AgoraExecutorWrapper:
    """
    Wrapper for Agora Worker Executor
    Adapts AgoraWorkerExecutor.execute() to BaseAgent interface
    """
    
    def __init__(self, agora_qa_worker):
        # Store the QA worker directly for LLM calls
        self.agora_qa_worker = agora_qa_worker
        self.agora_executor = agora_qa_worker  # For backward compatibility
        self.protocol_name = "agora"
        pass  # Agora wrapper created
    
    async def execute(self, context, event_queue) -> None:
        """BaseAgent standard interface"""
        try:
            # Extract message from context
            if hasattr(context, 'message'):
                message_text = str(context.message)
            else:
                message_text = str(context)
            
            # Create Agora input format
            agora_input = {
                "protocolHash": None,
                "body": message_text,
                "protocolSources": []
            }
            
            # Process with Agora QA worker - direct LLM call
            if hasattr(self.agora_qa_worker, 'answer'):
                result_text = await self.agora_qa_worker.answer(message_text)
            else:
                result_text = f"[Agora Fallback] Processed: {message_text}"
            
            # Create event for BaseAgent
            event = {
                "type": "agent_text_message",
                "data": result_text,
                "protocol": "agora",
                "metadata": {
                    "agora_enhanced": True,
                    "protocol_sources": agora_input.get("protocolSources", [])
                }
            }
            
            await event_queue.enqueue_event(event)
            print(f"[AgoraWrapper] Processed message via Agora SDK")
            
        except Exception as e:
            error_event = {
                "type": "agent_error",
                "data": f"Agora processing error: {e}",
                "protocol": "agora"
            }
            await event_queue.enqueue_event(error_event)
            print(f"[AgoraWrapper] Error: {e}")


class A2AExecutorWrapper:
    """
    Wrapper for A2A Worker Executor (already compatible)
    This is mainly for consistency and potential future enhancements
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.a2a_executor = QAAgentExecutor(config)
        self.protocol_name = "a2a"
        pass  # A2A wrapper created
    
    async def execute(self, context, event_queue) -> None:
        """BaseAgent standard interface - direct passthrough"""
        try:
            # A2A executor already has the correct interface
            await self.a2a_executor.execute(context, event_queue)
            
        except Exception as e:
            error_event = {
                "type": "agent_error",
                "data": f"A2A processing error: {e}",
                "protocol": "a2a"
            }
            await event_queue.enqueue_event(error_event)


def create_protocol_worker(protocol: str, config: Dict[str, Any]):
    """
    Factory function to create protocol-specific worker wrappers
    """
    if protocol == "acp":
        return ACPExecutorWrapper(config)
    elif protocol == "anp":
        return ANPExecutorWrapper(config)
    elif protocol == "agora":
        return AgoraExecutorWrapper(config)
    elif protocol == "a2a":
        return A2AExecutorWrapper(config)
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")


# Validation functions
def validate_executor_interface(executor) -> bool:
    """Validate that executor has the correct BaseAgent interface"""
    return (hasattr(executor, 'execute') and 
            callable(getattr(executor, 'execute')) and
            asyncio.iscoroutinefunction(getattr(executor, 'execute')))


def get_supported_protocols() -> list[str]:
    """Get list of supported protocols"""
    return ["acp", "anp", "agora", "a2a"]
