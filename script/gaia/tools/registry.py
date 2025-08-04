"""Tool registry system for dynamic tool registration and management."""
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .base import BaseTool
from .tool_collection import ToolCollection

# Import all available tools
from .planning import PlanningTool
from .web_search_simple import WebSearch
from .python_execute import PythonExecute
from .file_operators_simple import FileOperators
from .create_chat_completion_simple import CreateChatCompletion


class ToolRegistry:
    """Registry for managing and discovering available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Type[BaseTool]] = {}
        self._collections: Dict[str, ToolCollection] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register all available tools."""
        default_tools = [
            PlanningTool,
            WebSearch,
            PythonExecute,
            FileOperators,
            CreateChatCompletion
        ]
        
        for tool_class in default_tools:
            # Use the tool's name attribute or class name
            tool_name = getattr(tool_class, 'name', tool_class.__name__.lower())
            self._tools[tool_name] = tool_class
            
            # Also register by class name for compatibility
            self._tools[tool_class.__name__.lower()] = tool_class
    
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
            return tool_class(**kwargs)
        return None
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self._tools.keys())
    
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
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .base import BaseTool
from .tool_collection import ToolCollection
from .adapter import get_tool_adapter


class ToolRegistry:
    """Registry for managing and discovering available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Type[BaseTool]] = {}
        self._collections: Dict[str, ToolCollection] = {}
        self.adapter = get_tool_adapter() 
        self._auto_discover()
    
    def _auto_discover(self):
        """Automatically discover tools using the adapter."""
        # Use adapter to get available tools
        available_tools = self.adapter.list_tools()
        
        for tool_name in available_tools:
            tool_instance = self.adapter.get_tool(tool_name)
            if tool_instance:
                self._tools[tool_name] = type(tool_instance)
        
        print(f"Discovered {len(self._tools)} tools: {list(self._tools.keys())}")
    
    def register_tool(self, tool_class: Type[BaseTool], name: Optional[str] = None):
        """Register a tool class."""
        tool_name = name or tool_class.__name__.lower()
        self._tools[tool_name] = tool_class
    
    def get_tool_class(self, name: str) -> Optional[Type[BaseTool]]:
        """Get a tool class by name."""
        return self._tools.get(name.lower())
    
    def create_tool(self, name: str, **kwargs) -> Optional[BaseTool]:
        """Create a tool instance by name using the adapter."""
        return self.adapter.get_tool(name)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self._tools.keys())
    
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
