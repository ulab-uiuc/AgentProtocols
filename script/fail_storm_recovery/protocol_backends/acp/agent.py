#!/usr/bin/env python3
"""
ACP Agent implementation for Fail-Storm Recovery scenario.

This module provides ACP agent implementation using real ACP SDK,
following the correct implementation pattern from main branch.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, Optional, List, AsyncGenerator
from pathlib import Path
import sys

# Add fail_storm_recovery core path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import SimpleBaseAgent from fail-storm core
from core.simple_base_agent import SimpleBaseAgent as BaseAgent

# Import ACP SDK (required)
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield


class ACPExecutorWrapper:
    """
    Wrapper to convert ShardWorkerExecutor to ACP SDK native executor interface.
    
    Based on the correct implementation from main branch.
    ACP SDK expects: async def executor(messages: List[Message], context: Context) -> AsyncGenerator[RunYield, None]
    """
    
    def __init__(self, shard_worker_executor: Any):
        """
        Initialize with shard worker executor.
        
        Args:
            shard_worker_executor: ShardWorkerExecutor instance
        """
        self.shard_worker_executor = shard_worker_executor
        self.capabilities = ["text_processing", "async_generation", "acp_sdk_1.0.3"]
        self.logger = logging.getLogger("ACPExecutorWrapper")
    
    async def __call__(self, messages: List[Message], context: Context) -> AsyncGenerator[RunYield, None]:
        """
        ACP SDK native executor interface implementation.
        
        Args:
            messages: List of ACP SDK Message objects
            context: ACP SDK Context object
            
        Yields:
            RunYield objects as required by ACP SDK
        """
        self.logger.debug(f"[ACP] ACPExecutorWrapper called with {len(messages)} messages")
        
        try:
            # Process each message (typically just one)
            for i, message in enumerate(messages):
                # Generate run_id for this execution
                run_id = str(uuid.uuid4())
                
                self.logger.debug(f"[ACP] Processing message {i+1}/{len(messages)} with run_id: {run_id}")
                
                # Extract text content from message
                text_content = ""
                if hasattr(message, 'parts') and message.parts:
                    for part in message.parts:
                        if hasattr(part, 'type') and part.type == "text":
                            text_content += getattr(part, 'text', getattr(part, 'content', ""))
                else:
                    text_content = str(message)
                
                # Use shard worker to process the content
                if self.shard_worker_executor and hasattr(self.shard_worker_executor, 'worker'):
                    try:
                        # Call LLM directly through worker.answer() if available
                        if hasattr(self.shard_worker_executor.worker, 'answer'):
                            result = await self.shard_worker_executor.worker.answer(text_content)
                            yield result  # Yield LLM result as string
                            self.logger.debug(f"[ACP] LLM result: {len(str(result))} chars")
                        else:
                            # Fallback to start_task
                            result = await self.shard_worker_executor.worker.start_task(0)
                            yield str(result) if result else "No result"
                            
                    except Exception as e:
                        self.logger.error(f"Error in shard worker execution: {e}")
                        yield f"Execution error: {e}"
                else:
                    yield "Shard worker executor not available"
                    
                self.logger.debug(f"[ACP] Executor yielded result for message {i+1}")
                
        except Exception as e:
            # Yield error as string
            error_msg = f"ACP processing error: {e}"
            yield error_msg
            self.logger.error(f"[ACP] Executor error: {e}", exc_info=True)


async def create_acp_agent(agent_id: str, host: str, port: int, executor: Any) -> BaseAgent:
    """
    Factory function to create ACP agent using SimpleBaseAgent.
    
    Args:
        agent_id: Unique identifier for the agent
        host: Host address to bind to
        port: Port number for HTTP server
        executor: ShardWorkerExecutor instance
        
    Returns:
        SimpleBaseAgent instance configured for ACP protocol
    """
    # In fail-storm environment, use SimpleBaseAgent.create_acp()
    agent = await BaseAgent.create_acp(
        agent_id=agent_id,
        host=host,
        port=port,
        executor=executor
    )
    
    return agent