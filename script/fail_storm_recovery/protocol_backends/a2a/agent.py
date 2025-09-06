#!/usr/bin/env python3
"""
A2A Agent implementation for Fail-Storm Recovery scenario.

This module provides A2A agent implementation using real A2A SDK,
strictly following SDK requirements.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path
import sys

# Add src path for BaseAgent
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import real BaseAgent and server adapters
try:
    from core.base_agent import BaseAgent
    from server_adapters.a2a_adapter import A2AServerAdapter
    REAL_BASEAGENT_AVAILABLE = True
except ImportError:
    # Fallback to simple base agent
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core"))
    from simple_base_agent import SimpleBaseAgent as BaseAgent
    REAL_BASEAGENT_AVAILABLE = False
    
    # Create a mock A2AServerAdapter
    class A2AServerAdapter:
        def __init__(self):
            pass

# Import A2A SDK
try:
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.utils import new_agent_text_message
    from a2a.types import Message, MessageSendParams
    A2A_AVAILABLE = True
except ImportError as e:
    print(f"Warning: A2A SDK not available: {e}")
    print("Using fallback implementation")
    A2A_AVAILABLE = False
    
    # Create mock classes for fallback
    class AgentExecutor:
        async def execute(self, context, event_queue):
            pass
        async def cancel(self, context, event_queue):
            pass
    
    class RequestContext:
        def __init__(self, params=None):
            self.params = params
        def get_user_input(self):
            return "Mock input"
    
    class EventQueue:
        def __init__(self):
            self.events = []
        async def enqueue_event(self, event):
            self.events.append(event)
    
    class Message:
        def __init__(self, **kwargs):
            pass
    
    class MessageSendParams:
        def __init__(self, message=None):
            self.message = message
    
    def new_agent_text_message(text, role="user"):
        return {"type": "text", "content": text, "role": str(role)}


class A2AExecutorWrapper(AgentExecutor):
    """
    Wrapper to convert ShardWorkerExecutor to A2A SDK native executor interface.
    
    A2A SDK expects: async def execute(context: RequestContext, event_queue: EventQueue) -> None
    """
    
    def __init__(self, shard_worker_executor: Any):
        """
        Initialize with shard worker executor.
        
        Args:
            shard_worker_executor: ShardWorkerExecutor instance
        """
        self.shard_worker_executor = shard_worker_executor
        self.logger = logging.getLogger("A2AExecutorWrapper")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        A2A SDK native executor interface implementation.
        
        Args:
            context: A2A SDK RequestContext object
            event_queue: A2A SDK EventQueue object
        """
        try:
            if not A2A_AVAILABLE:
                await event_queue.enqueue_event(new_agent_text_message("A2A SDK not available"))
                return
            
            # Extract user input from context
            user_input = context.get_user_input()
            
            # Use shard worker to process the input
            if self.shard_worker_executor and hasattr(self.shard_worker_executor, 'worker'):
                try:
                    # Start QA task using group 0
                    result = await self.shard_worker_executor.worker.start_task(0)
                    
                    # Send result to event queue
                    result_message = new_agent_text_message(str(result) if result else "No result")
                    await event_queue.enqueue_event(result_message)
                    
                except Exception as e:
                    self.logger.error(f"Error in shard worker execution: {e}")
                    error_message = new_agent_text_message(f"Execution error: {e}")
                    await event_queue.enqueue_event(error_message)
            else:
                # No executor available
                no_exec_message = new_agent_text_message("Shard worker executor not available")
                await event_queue.enqueue_event(no_exec_message)
                
        except Exception as e:
            self.logger.error(f"Error in A2A executor: {e}")
            try:
                error_message = new_agent_text_message(f"A2A executor error: {e}")
                await event_queue.enqueue_event(error_message)
            except:
                pass  # If we can't even send error message, just log it
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel execution if needed."""
        try:
            cancel_message = new_agent_text_message("Execution cancelled")
            await event_queue.enqueue_event(cancel_message)
        except Exception as e:
            self.logger.error(f"Error cancelling A2A execution: {e}")


async def create_a2a_agent(agent_id: str, host: str, port: int, executor: Any) -> BaseAgent:
    """
    Factory function to create A2A agent using real BaseAgent and A2A SDK.
    
    Args:
        agent_id: Unique identifier for the agent
        host: Host address to bind to
        port: Port number for HTTP server
        executor: ShardWorkerExecutor instance
        
    Returns:
        BaseAgent instance configured with A2A SDK
    """
    if REAL_BASEAGENT_AVAILABLE and A2A_AVAILABLE:
        # Use real BaseAgent with A2A SDK
        a2a_executor = A2AExecutorWrapper(executor)
        
        agent = await BaseAgent.create_a2a(
            agent_id=agent_id,
            host=host,
            port=port,
            executor=a2a_executor,
            server_adapter=A2AServerAdapter()
        )
        
        return agent
    else:
        # Use simple BaseAgent fallback
        agent = await BaseAgent.create_a2a(
            agent_id=agent_id,
            host=host,
            port=port,
            executor=executor
        )
        
        return agent