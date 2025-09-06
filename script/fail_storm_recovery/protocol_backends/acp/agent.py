#!/usr/bin/env python3
"""
ACP Agent implementation for Fail-Storm Recovery scenario.

This module provides ACP agent implementation using real ACP SDK,
strictly following SDK requirements.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional, List, AsyncGenerator
from pathlib import Path
import sys

# Add src path for BaseAgent
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import real BaseAgent and server adapters
try:
    from core.base_agent import BaseAgent
    from server_adapters.acp_adapter import ACPServerAdapter
    REAL_BASEAGENT_AVAILABLE = True
except ImportError:
    # Fallback to simple base agent
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core"))
    from simple_base_agent import SimpleBaseAgent as BaseAgent
    REAL_BASEAGENT_AVAILABLE = False
    
    # Create a mock ACPServerAdapter
    class ACPServerAdapter:
        def __init__(self):
            pass

# Import ACP SDK
try:
    from acp_sdk.models import Message, MessagePart
    from acp_sdk.server import Context, RunYield
    ACP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ACP SDK not available: {e}")
    print("Using fallback implementation")
    ACP_AVAILABLE = False
    
    # Create mock classes for fallback
    class Message:
        def __init__(self, **kwargs):
            self.parts = kwargs.get('parts', [])
            self.content = kwargs.get('content', '')
    
    class MessagePart:
        def __init__(self, type: str, text: str):
            self.type = type
            self.text = text
    
    class Context:
        def __init__(self, **kwargs):
            pass
    
    class RunYield:
        def __init__(self, type: str, content: str):
            self.type = type
            self.content = content


class ACPExecutorWrapper:
    """
    Wrapper to convert ShardWorkerExecutor to ACP SDK native executor interface.
    
    ACP SDK expects: async def executor(messages: List[Message], context: Context) -> AsyncGenerator[RunYield, None]
    """
    
    def __init__(self, shard_worker_executor: Any):
        """
        Initialize with shard worker executor.
        
        Args:
            shard_worker_executor: ShardWorkerExecutor instance
        """
        self.shard_worker_executor = shard_worker_executor
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
        try:
            if not ACP_AVAILABLE:
                yield RunYield(type="text", content="ACP SDK not available")
                return
            
            # Extract text content from messages
            text_content = ""
            for message in messages:
                if hasattr(message, 'parts'):
                    for part in message.parts:
                        if hasattr(part, 'text'):
                            text_content += part.text + " "
                elif hasattr(message, 'content'):
                    text_content += str(message.content) + " "
                else:
                    text_content += str(message) + " "
            
            text_content = text_content.strip()
            
            # Use shard worker to process the content
            if self.shard_worker_executor and hasattr(self.shard_worker_executor, 'worker'):
                try:
                    # Start QA task using group 0
                    result = await self.shard_worker_executor.worker.start_task(0)
                    
                    # Yield the result
                    yield RunYield(type="text", content=str(result) if result else "No result")
                    
                except Exception as e:
                    self.logger.error(f"Error in shard worker execution: {e}")
                    yield RunYield(type="error", content=f"Execution error: {e}")
            else:
                yield RunYield(type="text", content="Shard worker executor not available")
                
        except Exception as e:
            self.logger.error(f"Error in ACP executor: {e}")
            yield RunYield(type="error", content=f"ACP executor error: {e}")


async def create_acp_agent(agent_id: str, host: str, port: int, executor: Any) -> BaseAgent:
    """
    Factory function to create ACP agent using real BaseAgent and ACP SDK.
    
    Args:
        agent_id: Unique identifier for the agent
        host: Host address to bind to
        port: Port number for HTTP server
        executor: ShardWorkerExecutor instance
        
    Returns:
        BaseAgent instance configured with ACP SDK
    """
    if REAL_BASEAGENT_AVAILABLE and ACP_AVAILABLE:
        # Use real BaseAgent with ACP SDK
        acp_executor = ACPExecutorWrapper(executor)
        
        agent = await BaseAgent.create_acp(
            agent_id=agent_id,
            host=host,
            port=port,
            executor=acp_executor,
            server_adapter=ACPServerAdapter()
        )
        
        return agent
    else:
        # Use simple BaseAgent fallback
        agent = await BaseAgent.create_acp(
            agent_id=agent_id,
            host=host,
            port=port,
            executor=executor
        )
        
        return agent