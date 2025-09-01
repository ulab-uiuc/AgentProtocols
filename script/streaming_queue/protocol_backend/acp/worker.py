# -*- coding: utf-8 -*-
"""
ACP Worker implementation - simplified but functional.

Since there's no standard ACP SDK available, this implements a clean
ACP-style worker that integrates properly with the QA system.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add streaming_queue to path for imports
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parent.parent.parent
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))

from core.qa_worker_base import QAWorkerBase


class ACPQAWorker(QAWorkerBase):
    """ACP-style QA Worker that provides clean tool interface."""
    
    def __init__(self, config: Optional[Dict] = None, output=None):
        super().__init__(config, output)
        self.tools = {
            "answer_question": self._answer_question_tool,
            "get_status": self._get_status_tool
        }
    
    async def _answer_question_tool(self, question: str) -> str:
        """Tool: Answer a question using LLM."""
        try:
            result = await self.answer(question)
            return result
        except Exception as e:
            return f"Error answering question: {str(e)}"
    
    async def _get_status_tool(self) -> str:
        """Tool: Get worker status."""
        try:
            llm_available = hasattr(self, 'llm') and self.llm is not None
            return f"ACP QA Worker ready - LLM available: {llm_available}"
        except Exception as e:
            return f"ACP QA Worker status check failed: {str(e)}"
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool by name with arguments."""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}")
        
        tool_func = self.tools[tool_name]
        return await tool_func(**arguments)


class ACPWorkerExecutor:
    """ACP Worker Executor for integration with the runner."""
    
    def __init__(self, config: Optional[Dict] = None, output=None):
        self.worker = ACPQAWorker(config, output)
        self.config = config
        self.output = output
    
    async def execute(self, input_data: Dict) -> Dict:
        """Execute an ACP-style request."""
        try:
            # Check if this is a tool call request
            if "tool_name" in input_data:
                tool_name = input_data["tool_name"]
                arguments = input_data.get("arguments", {})
                result = await self.worker.call_tool(tool_name, arguments)
            else:
                # Extract question from content
                parts = input_data.get("content", [])
                question = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
                
                if not question:
                    question = "What is artificial intelligence?"
                
                result = await self.worker.call_tool("answer_question", {"question": question})
            
            return {"content": [{"type": "text", "text": result}]}
            
        except Exception as e:
            error_msg = f"ACP Worker execution failed: {str(e)}"
            return {"content": [{"type": "text", "text": error_msg}]}