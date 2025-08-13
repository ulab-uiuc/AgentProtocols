"""Tool registry system for dynamic tool registration and management."""
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .base import BaseTool
from .tool_collection import ToolCollection

# Import all available tools
try:
    from .web_search_simple import WebSearch
    from .python_execute import PythonExecute
    from .file_operators_simple import FileOperators
    from .create_chat_completion_simple import CreateChatCompletion
except ImportError as e:
    print(f"Warning: Could not import some tools: {e}")


class ToolRegistry:
    """Registry for managing and discovering available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Type[BaseTool]] = {}
        self._collections: Dict[str, ToolCollection] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register all available tools."""
        # Define the canonical tool mappings to avoid duplicates
        tool_classes = [
            ("web_search", WebSearch),
            ("python_execute", PythonExecute), 
            ("file_operators", FileOperators),
            ("create_chat_completion", CreateChatCompletion),
        ]
        
        registered_tools = []
        for tool_name, tool_class in tool_classes:
            if tool_class is not None:
                try:
                    # Only register with the canonical name
                    self._tools[tool_name] = tool_class
                    registered_tools.append(tool_name)
                except Exception as e:
                    print(f"Warning: Could not register tool {tool_name}: {e}")
        
        print(f"Registered {len(registered_tools)} tools: {registered_tools}")
    
    def register_tool(self, tool_class: Type[BaseTool], name: Optional[str] = None):
        """Register a tool class."""
        tool_name = name or getattr(tool_class, 'name', tool_class.__name__.lower())
        self._tools[tool_name] = tool_class
    
    def get_tool_class(self, name: str) -> Optional[Type[BaseTool]]:
        """Get a tool class by name."""
        return self._tools.get(name.lower())
    
    def create_tool(self, name: str, **kwargs) -> Optional[BaseTool]:
        """Create a tool instance by name."""
        tool_class = self.get_tool_class(name)
        if tool_class:
            try:
                return tool_class(**kwargs)
            except Exception as e:
                print(f"Warning: Could not create tool {name}: {e}")
                return None
        return None
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self._tools.keys())
    
    def get_tool_mapping(self) -> Dict[str, str]:
        """Get mapping of logical tool names to actual tool names."""
        # Return identity mapping since we now use canonical names
        return {tool_name: tool_name for tool_name in self._tools.keys()}
    
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
        tool_class = self.get_tool_class(name)
        if not tool_class:
            return {}
        
        return {
            "name": name,
            "class": tool_class.__name__,
            "description": getattr(tool_class, "description", "No description available"),
            "parameters": getattr(tool_class, "parameters", None)
        }
    
    def validate_tool_name(self, name: str) -> bool:
        """Validate if a tool name is available."""
        return name.lower() in [tool.lower() for tool in self._tools.keys()]
