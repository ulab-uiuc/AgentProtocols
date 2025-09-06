#!/usr/bin/env python3
"""
A2A Agent implementation for Fail-Storm Recovery scenario.

This module provides A2A agent implementation using real A2A SDK,
following the correct implementation pattern from main branch.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path
import sys

# Add fail_storm_recovery core path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import SimpleBaseAgent from fail-storm core
from core.simple_base_agent import SimpleBaseAgent as BaseAgent

# Import A2A SDK (required)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.types import Message, MessageSendParams


class A2AExecutorWrapper(AgentExecutor):
    """
    Wrapper to convert ShardWorkerExecutor to A2A SDK native executor interface.
    
    Based on the correct implementation from main branch.
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
            # Extract user input from context
            user_input = context.get_user_input() if hasattr(context, "get_user_input") else None
            
            # Extract text content
            if user_input:
                if isinstance(user_input, str):
                    question = user_input
                elif isinstance(user_input, dict):
                    question = user_input.get("text", str(user_input))
                else:
                    question = str(user_input)
            else:
                question = "What is artificial intelligence?"  # Default question
            
            # Use shard worker to process the question
            if self.shard_worker_executor and hasattr(self.shard_worker_executor, 'worker'):
                try:
                    # Call LLM directly through worker.answer() if available
                    if hasattr(self.shard_worker_executor.worker, 'answer'):
                        result = await self.shard_worker_executor.worker.answer(question)
                    else:
                        # Fallback to start_task
                        result = await self.shard_worker_executor.worker.start_task(0)
                    
                    # Send result to event queue using A2A SDK format
                    result_message = new_agent_text_message(str(result) if result else "No result")
                    
                    # Try both sync and async enqueue_event (based on main branch pattern)
                    try:
                        # First try async (correct way based on RuntimeWarning)
                        await event_queue.enqueue_event(result_message)
                    except Exception:
                        try:
                            # Fallback to sync
                            event_queue.enqueue_event(result_message)
                        except Exception:
                            # Last resort - try simple text event
                            event_queue.enqueue_event(new_agent_text_message(f"A2A result: {result}"))
                    
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
    Factory function to create A2A agent using SimpleBaseAgent.
    
    Args:
        agent_id: Unique identifier for the agent
        host: Host address to bind to
        port: Port number for HTTP server
        executor: ShardWorkerExecutor instance
        
    Returns:
        SimpleBaseAgent instance configured for A2A protocol
    """
    # In fail-storm environment, use SimpleBaseAgent.create_a2a()
    agent = await BaseAgent.create_a2a(
        agent_id=agent_id,
        host=host,
        port=port,
        executor=executor
    )
    
    return agent