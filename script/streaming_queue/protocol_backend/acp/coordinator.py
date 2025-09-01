# -*- coding: utf-8 -*-
"""
ACP Coordinator implementation - simplified but functional.

Since there's no standard ACP SDK available, this implements a clean
ACP-style coordinator that integrates properly with the QA system.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add streaming_queue to path for imports
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parent.parent.parent
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))

from core.qa_coordinator_base import QACoordinatorBase


class ACPQACoordinator(QACoordinatorBase):
    """ACP-style QA Coordinator that provides clean tool interface."""
    
    def __init__(self, config: Optional[Dict] = None, output=None):
        super().__init__(config, output)
        self.tools = {
            "dispatch_questions": self._dispatch_questions_tool,
            "get_coordinator_status": self._get_status_tool,
            "setup_workers": self._setup_workers_tool
        }
    
    async def _dispatch_questions_tool(self) -> str:
        """Tool: Start the question dispatch process."""
        try:
            result = await self.dispatch_round()
            return result
        except Exception as e:
            return f"Dispatch failed: {str(e)}"
    
    async def _get_status_tool(self) -> str:
        """Tool: Get coordinator status."""
        try:
            result = await self.get_status()
            return result
        except Exception as e:
            return f"Status check failed: {str(e)}"
    
    async def _setup_workers_tool(self, worker_list: str) -> str:
        """Tool: Setup worker connections."""
        try:
            worker_ids = [w.strip() for w in worker_list.split(",") if w.strip()]
            self.worker_ids = worker_ids
            return f"Workers setup complete: {worker_ids}"
        except Exception as e:
            return f"Worker setup failed: {str(e)}"
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool by name with arguments."""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}")
        
        tool_func = self.tools[tool_name]
        return await tool_func(**arguments)
    
    async def send_to_worker(self, worker_id: str, question: str) -> Dict[str, Any]:
        """Send question to worker using ACP tool call.
        
        This method implements the abstract method from QACoordinatorBase.
        It uses ACP-style tool calls to communicate with worker agents.
        """
        if not self.agent_network:
            raise RuntimeError("No ACP network set. Call set_network() first.")
        
        # Use ACP tool call format
        payload = {
            "tool_name": "answer_question",
            "arguments": {"question": question}
        }
        
        try:
            response = await self.agent_network.route_message(
                self.coordinator_id, worker_id, payload
            )
            
            # Extract answer from ACP response
            answer = None
            if isinstance(response, dict):
                if "text" in response:
                    answer = response["text"]
                elif "content" in response:
                    content = response["content"]
                    if isinstance(content, list) and content:
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                answer = item.get("text")
                                break
            
            return {
                "answer": answer or "No answer received",
                "raw": response,
            }
        except Exception as e:
            return {
                "answer": f"ACP communication error: {str(e)}",
                "raw": None,
            }


class ACPCoordinatorExecutor:
    """ACP Coordinator Executor for integration with the runner."""
    
    def __init__(self, config: Dict | None = None, output=None):
        self.coordinator = ACPQACoordinator(config, output)
        self.config = config
        self.output = output
    
    async def execute(self, input_data: Dict) -> Dict:
        """Execute an ACP-style request."""
        try:
            # Check if this is a tool call request
            if "tool_name" in input_data:
                tool_name = input_data["tool_name"]
                arguments = input_data.get("arguments", {})
                result = await self.coordinator.call_tool(tool_name, arguments)
            else:
                # Extract command from content
                parts = input_data.get("content", [])
                user_text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text") or "status"
                cmd = user_text.strip().lower()
                original_text = user_text.strip()
                
                if cmd == "dispatch":
                    result = await self.coordinator.call_tool("dispatch_questions", {})
                elif cmd.startswith("setup_network"):
                    # Extract worker list
                    if " " in original_text:
                        worker_list = original_text.split(" ", 1)[1]
                        result = await self.coordinator.call_tool("setup_workers", {"worker_list": worker_list})
                        
                        # Also ensure network is properly set
                        if self.coordinator.agent_network is not None:
                            worker_ids = [w.strip() for w in worker_list.split(",") if w.strip()]
                            self.coordinator.worker_ids = worker_ids
                            self.coordinator.coordinator_id = "Coordinator-1"
                    else:
                        result = "Error: setup_network command requires worker list"
                else:
                    result = await self.coordinator.call_tool("get_coordinator_status", {})
            
            return {"content": [{"type": "text", "text": result}]}
            
        except Exception as e:
            error_msg = f"ACP Coordinator execution failed: {str(e)}"
            return {"content": [{"type": "text", "text": error_msg}]}