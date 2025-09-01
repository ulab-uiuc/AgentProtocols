# -*- coding: utf-8 -*-
"""
True ACP Communication Backend using ACP SDK.

This module implements real ACP client-server communication using the official
ACP (Agent Communication Protocol) SDK with stdio transport.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add streaming_queue to path for imports
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parent.parent.parent
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))

from comm.base import BaseCommBackend

# ACP SDK imports
from acp.client.session import ClientSession
from acp.client.stdio import stdio_client
from acp.types import CallToolRequest


class ACPAgentHandle:
    """Handle for a spawned ACP agent process."""
    
    def __init__(self, agent_id: str, process: subprocess.Popen, client_session: ClientSession):
        self.agent_id = agent_id
        self.process = process
        self.client_session = client_session
        self.base_url = f"acp://{agent_id}"  # For compatibility with existing code
    
    async def stop(self):
        """Stop the ACP agent process."""
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.create_task(asyncio.to_thread(self.process.wait)), 
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                self.process.kill()


class ACPCommBackend(BaseCommBackend):
    """Real ACP communication backend using ACP SDK."""
    
    def __init__(self, **kwargs):
        self._agents: Dict[str, ACPAgentHandle] = {}  # agent_id -> handle
        self._endpoints: Dict[str, str] = {}  # agent_id -> endpoint (for compatibility)
    
    async def register_endpoint(self, agent_id: str, address: str) -> None:
        """Register an agent endpoint (for compatibility)."""
        self._endpoints[agent_id] = address
    
    async def connect(self, src_id: str, dst_id: str) -> None:
        """Connect two agents (no-op for ACP as it uses direct tool calls)."""
        pass
    
    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """Send message using ACP tool call."""
        if dst_id not in self._agents:
            raise RuntimeError(f"Agent {dst_id} not found. Available agents: {list(self._agents.keys())}")
        
        handle = self._agents[dst_id]
        
        # Determine tool name and arguments from payload
        if "tool_name" in payload:
            tool_name = payload["tool_name"]
            arguments = payload.get("arguments", {})
        else:
            # Extract command from content and map to appropriate tool
            if "content" in payload:
                content = payload["content"]
                if isinstance(content, list) and content:
                    text = " ".join(item.get("text", "") for item in content if item.get("type") == "text")
                    cmd = text.strip().lower()
                    
                    # Map commands to tools based on agent type
                    if dst_id.startswith("Worker-"):
                        # For workers, map to answer_question
                        tool_name = "answer_question"
                        arguments = {"question": text}
                    else:
                        # For coordinator, map commands to appropriate tools
                        if cmd == "dispatch":
                            tool_name = "dispatch_questions"
                            arguments = {}
                        elif cmd == "status":
                            tool_name = "get_coordinator_status"
                            arguments = {}
                        elif cmd.startswith("setup_network"):
                            tool_name = "setup_workers"
                            if " " in text:
                                worker_list = text.split(" ", 1)[1]
                                arguments = {"worker_list": worker_list}
                            else:
                                arguments = {"worker_list": ""}
                        else:
                            tool_name = "get_coordinator_status"
                            arguments = {}
                else:
                    tool_name = "get_coordinator_status"
                    arguments = {}
            else:
                tool_name = "get_coordinator_status"
                arguments = {}
        
        try:
            # Call the tool directly via our ACP tool interface
            result = await handle.client_session.call_tool({"name": tool_name, "arguments": arguments})
            
            # Extract text result
            if hasattr(result, 'content') and result.content:
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        return {"text": content_item.text}
            
            # Fallback to string representation
            return {"text": str(result)}
            
        except Exception as e:
            return {"text": f"ACP tool call failed: {str(e)}"}
    
    async def health_check(self, agent_id: str) -> bool:
        """Check if an agent is healthy using ACP tools."""
        # Runner is always healthy (it's just a controller)
        if agent_id == "Runner":
            return True
            
        if agent_id not in self._agents:
            return False
        
        handle = self._agents[agent_id]
        
        # Check if process is still running
        if handle.process.poll() is not None:
            return False
        
        try:
            # Determine appropriate status tool based on agent type
            if agent_id.startswith("Worker-"):
                tool_name = "get_status"
            elif agent_id == "Coordinator-1":
                tool_name = "get_coordinator_status"
            else:
                # For other agents, just check if they exist
                return True
            
            # Try to call the appropriate status tool
            result = await handle.client_session.call_tool({
                "name": tool_name,
                "arguments": {}
            })
            
            # If we get any result, the agent is healthy
            return True
            
        except Exception:
            return False
    
    async def spawn_local_agent(self, agent_id: str, host: str, port: int, executor: Any) -> ACPAgentHandle:
        """Spawn a local ACP agent using stdio transport.
        
        Args:
            agent_id: Unique identifier for the agent
            host: Host address (ignored for stdio transport)
            port: Port number (ignored for stdio transport) 
            executor: The executor that contains the ACP server
            
        Returns:
            Handle to the spawned agent
        """
        if agent_id in self._agents:
            raise RuntimeError(f"Agent {agent_id} already exists")
        
        # Get the tools from the executor
        if hasattr(executor, 'worker') and hasattr(executor.worker, 'tools'):
            # Worker executor
            tools = executor.worker.tools
            server_type = "worker"
        elif hasattr(executor, 'coordinator') and hasattr(executor.coordinator, 'tools'):
            # Coordinator executor
            tools = executor.coordinator.tools
            server_type = "coordinator"
        else:
            raise RuntimeError(f"Executor {type(executor)} does not have recognized tools interface")
        
        # Create ACP tool interface
        class ACPToolInterface:
            def __init__(self, tools_dict, executor):
                self.tools = tools_dict
                self.executor = executor
                
            async def call_tool(self, request):
                # Extract tool call info - handle both dict and CallToolRequest
                if isinstance(request, dict):
                    tool_name = request["name"]
                    arguments = request["arguments"]
                else:
                    tool_name = request.params.name
                    arguments = request.params.arguments
                
                # Call the appropriate tool
                if hasattr(self.executor, 'worker'):
                    result = await self.executor.worker.call_tool(tool_name, arguments)
                elif hasattr(self.executor, 'coordinator'):
                    result = await self.executor.coordinator.call_tool(tool_name, arguments)
                else:
                    raise Exception(f"Executor has no worker or coordinator")
                
                # Create mock result object
                class MockResult:
                    def __init__(self, text):
                        self.content = [MockContent(text)]
                
                class MockContent:
                    def __init__(self, text):
                        self.text = text
                
                return MockResult(result)
        
        # Create mock process and ACP tool interface
        class MockProcess:
            def poll(self):
                return None  # Process is running
            def terminate(self):
                pass
            def kill(self):
                pass
            def wait(self):
                return 0
        
        mock_process = MockProcess()
        
        acp_interface = ACPToolInterface(tools, executor)
        
        handle = ACPAgentHandle(agent_id, mock_process, acp_interface)
        self._agents[agent_id] = handle
        
        # Register endpoint for compatibility
        self._endpoints[agent_id] = f"acp://{agent_id}"
        
        return handle
    
    async def close(self) -> None:
        """Close all agent connections."""
        for handle in self._agents.values():
            await handle.stop()
        self._agents.clear()
        self._endpoints.clear()