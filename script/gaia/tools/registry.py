"""Tool registry system for dynamic tool registration and management."""
from typing import Any, Dict, List, Optional, Type

from .base import BaseTool
from .tool_collection import ToolCollection

# Import available tools
from .ask_human import AskHuman
# from .python_execute import PythonExecute
from .sandbox_python_execute import SandboxPythonExecute
from .str_replace_editor import StrReplaceEditor
from .create_chat_completion import CreateChatCompletion
import sys

# Import BrowserUseTool with proper version checking
try:
    from .browser_use_tool import BrowserUseTool
except (ImportError, RuntimeError) as e:
    raise ImportError(f"BrowserUseTool is required for tool registry. Please install browser_use package and ensure compatibility. Error: {e}")


class ToolRegistry:
    """Registry for managing and discovering available tools."""
    
    def __init__(self):
        # Tool collections storage
        self._collections: Dict[str, ToolCollection] = {}
        
        # Available tools collection - add new tools here directly
        tools = [
            SandboxPythonExecute(),
            StrReplaceEditor(),
            CreateChatCompletion(),
            BrowserUseTool(),
        ]

        self.available_tools: ToolCollection = ToolCollection(*tools)
    
    def add_tool(self, tool: BaseTool):
        """Add a tool instance to the available tools collection."""
        self.available_tools.add_tools(tool)
    
    def create_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool instance by name."""
        return self.available_tools.tool_map.get(name)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.available_tools.tool_map.keys())
    
    def create_collection(self, tool_names: List[str], collection_name: str = "default") -> ToolCollection:
        """Create a tool collection from tool names."""
        tools = []
        for name in tool_names:
            tool = self.create_tool(name)
            if tool:
                tools.append(tool)
        
        collection = ToolCollection(*tools)
        self._collections[collection_name] = collection
        return collection
    
    def get_collection(self, name: str) -> Optional[ToolCollection]:
        """Get a tool collection by name."""
        return self._collections.get(name)
    
    def get_tool_info(self, name: str) -> Dict[str, Any]:
        """Get information about a tool."""
        tool = self.create_tool(name)
        if not tool:
            return {}
        
        return {
            "name": name,
            "class": tool.__class__.__name__,
            "description": getattr(tool, "description", "No description available"),
            "parameters": getattr(tool, "parameters", None)
        }
    
    def validate_tool_name(self, name: str) -> bool:
        """Validate if a tool name is available."""
        return name.lower() in [tool.lower() for tool in self.available_tools.tool_map.keys()]


# Global tool registry instance
registry = ToolRegistry()
