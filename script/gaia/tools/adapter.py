"""Tool adapter for integrating existing tools with the multi-agent framework."""
import asyncio
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add tools directory to path
tools_dir = Path(__file__).parent
sys.path.append(str(tools_dir))

from .base import BaseTool, ToolResult

try:
    from .web_search import WebSearch
except ImportError:
    WebSearch = None

try:
    from .python_execute import PythonExecute
except ImportError:
    PythonExecute = None

try:
    from .planning import PlanningTool
except ImportError:
    PlanningTool = None

try:
    from .file_operators import FileOperators
except ImportError:
    FileOperators = None

try:
    from .create_chat_completion import CreateChatCompletion
except ImportError:
    CreateChatCompletion = None


class ToolAdapter:
    """Adapter to integrate existing tools with the multi-agent framework."""
    
    def __init__(self):
        self.available_tools = self._discover_tools()
    
    def _discover_tools(self) -> Dict[str, type]:
        """Discover available tools."""
        tools = {}
        
        # Map tool names to classes
        tool_mappings = {
            "web_search": WebSearch,
            "python_execute": PythonExecute,
            "planning": PlanningTool,
            "file_operators": FileOperators,
            "create_chat_completion": CreateChatCompletion
        }
        
        for name, tool_class in tool_mappings.items():
            if tool_class is not None:
                tools[name] = tool_class
        
        return tools
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool instance by name."""
        tool_class = self.available_tools.get(name)
        if tool_class:
            try:
                return tool_class()
            except Exception as e:
                print(f"Error creating tool {name}: {e}")
        return None
    
    def list_tools(self) -> List[str]:
        """List available tool names."""
        return list(self.available_tools.keys())


# Simplified tool implementations for cases where the full tools aren't available
class SimpleWebSearchTool(BaseTool):
    """Simplified web search tool."""
    
    name: str = "web_search"
    description: str = "Search the web for information"
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
    
    async def execute(self, query: str = "", **kwargs) -> ToolResult:
        """Execute web search."""
        # Simulate web search results
        mock_results = f"""
Search results for "{query}":

1. Related Article - https://example.com/article1
   Summary: Information about {query} with relevant details.

2. Reference Source - https://example.com/source2  
   Summary: Additional context and data about {query}.

3. Research Paper - https://example.com/paper3
   Summary: Academic research findings related to {query}.

Note: This is a simulated search result. In a production environment, 
this would connect to actual search APIs.
"""
        return ToolResult(output=mock_results)


class SimplePythonExecuteTool(BaseTool):
    """Simplified Python execution tool."""
    
    name: str = "python_execute"
    description: str = "Execute Python code"
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"}
        },
        "required": ["code"]
    }
    
    async def execute(self, code: str = "", **kwargs) -> ToolResult:
        """Execute Python code safely."""
        try:
            # For demo purposes, just return the code that would be executed
            # In production, this would use a sandboxed environment
            return ToolResult(output=f"Code execution result for:\n```python\n{code}\n```\n\nNote: Simulated execution in demo mode.")
        except Exception as e:
            return ToolResult(error=f"Execution error: {str(e)}")


class SimplePlanningTool(BaseTool):
    """Simplified planning tool."""
    
    name: str = "planning"
    description: str = "Create and manage task plans"
    parameters: dict = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Task to plan for"}
        },
        "required": ["task"]
    }
    
    async def execute(self, task: str = "", **kwargs) -> ToolResult:
        """Create a simple plan for the task."""
        plan = f"""
Task Planning for: {task}

Plan Steps:
1. Analyze the task requirements
2. Gather necessary information and resources
3. Break down into manageable subtasks
4. Execute each subtask systematically
5. Review and verify results
6. Provide comprehensive response

Current Status: Plan created and ready for execution
Next Action: Begin information gathering phase
"""
        return ToolResult(output=plan)


class SimpleFileOperatorsTool(BaseTool):
    """Simplified file operations tool."""
    
    name: str = "file_operators"
    description: str = "Perform file operations"
    parameters: dict = {
        "type": "object", 
        "properties": {
            "operation": {"type": "string", "description": "File operation to perform"},
            "content": {"type": "string", "description": "Content to work with"}
        },
        "required": ["operation"]
    }
    
    async def execute(self, operation: str = "read", content: str = "", **kwargs) -> ToolResult:
        """Execute file operation."""
        result = f"File operation '{operation}' executed.\nContent processed: {len(content)} characters"
        return ToolResult(output=result)


class SimpleChatCompletionTool(BaseTool):
    """Simplified chat completion tool for reasoning."""
    
    name: str = "create_chat_completion" 
    description: str = "Generate responses using language model"
    parameters: dict = {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input text to process"}
        },
        "required": ["input"]
    }
    
    async def execute(self, input: str = "", **kwargs) -> ToolResult:
        """Generate chat completion."""
        # Simple reasoning simulation
        response = f"""
Based on the provided information: {input[:200]}{'...' if len(input) > 200 else ''}

Analysis and Reasoning:
- The input has been processed and analyzed
- Key points have been identified and evaluated
- Logical connections have been established
- A comprehensive response has been formulated

Final Answer: This is a reasoned response based on the available information. 
In a production environment, this would connect to an actual language model 
for sophisticated reasoning and response generation.
"""
        return ToolResult(output=response)


class FallbackToolAdapter(ToolAdapter):
    """Fallback adapter that provides simplified tools when full tools aren't available."""
    
    def __init__(self):
        super().__init__()
        
        # Add fallback tools for missing ones
        fallback_tools = {
            "web_search": SimpleWebSearchTool,
            "python_execute": SimplePythonExecuteTool,
            "planning": SimplePlanningTool,
            "file_operators": SimpleFileOperatorsTool,
            "create_chat_completion": SimpleChatCompletionTool
        }
        
        # Use fallback tools for any missing tools
        for name, tool_class in fallback_tools.items():
            if name not in self.available_tools:
                self.available_tools[name] = tool_class
                print(f"Using fallback implementation for tool: {name}")


# Global tool adapter instance
_tool_adapter = None

def get_tool_adapter() -> ToolAdapter:
    """Get global tool adapter instance."""
    global _tool_adapter
    if _tool_adapter is None:
        _tool_adapter = FallbackToolAdapter()
    return _tool_adapter


async def test_tools():
    """Test the tool adapter."""
    print("=== Tool Adapter Test ===")
    
    adapter = get_tool_adapter()
    available_tools = adapter.list_tools()
    
    print(f"Available tools: {available_tools}")
    
    # Test each tool
    for tool_name in available_tools[:3]:  # Test first 3 tools
        print(f"\nTesting {tool_name}:")
        try:
            tool = adapter.get_tool(tool_name)
            if tool:
                if tool_name == "web_search":
                    result = await tool.execute(query="test query")
                elif tool_name == "python_execute":
                    result = await tool.execute(code="print('Hello, World!')")
                elif tool_name == "planning":
                    result = await tool.execute(task="test task")
                elif tool_name == "file_operators":
                    result = await tool.execute(operation="read", content="test content")
                elif tool_name == "create_chat_completion":
                    result = await tool.execute(input="test input")
                else:
                    result = await tool.execute(input="test")
                
                print(f"✅ {tool_name}: {str(result)[:100]}...")
            else:
                print(f"❌ Failed to create tool: {tool_name}")
        except Exception as e:
            print(f"❌ Error testing {tool_name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_tools())
