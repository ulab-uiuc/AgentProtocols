"""Simplified file operations tool."""
import asyncio
import os
from pathlib import Path
from typing import Optional, Union

from .base import BaseTool, ToolResult
from .exceptions import ToolError


class FileOperators(BaseTool):
    """Tool for basic file operations."""

    name: str = "file_operators"
    description: str = "Perform file operations like reading, writing, and listing files."
    parameters: dict = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "The operation to perform: read, write, list, exists",
                "enum": ["read", "write", "list", "exists"]
            },
            "path": {
                "type": "string",
                "description": "The file or directory path"
            },
            "content": {
                "type": "string",
                "description": "Content to write (for write operation)"
            }
        },
        "required": ["operation", "path"]
    }

    async def execute(self, operation: str, path: str, content: Optional[str] = None, **kwargs) -> ToolResult:
        """
        Execute file operation.
        
        Args:
            operation: The operation to perform
            path: File or directory path
            content: Content for write operations
            
        Returns:
            ToolResult with operation result
        """
        try:
            if operation == "read":
                return await self._read_file(path)
            elif operation == "write":
                return await self._write_file(path, content or "")
            elif operation == "list":
                return await self._list_directory(path)
            elif operation == "exists":
                return await self._check_exists(path)
            else:
                raise ToolError(f"Unknown operation: {operation}")
                
        except Exception as e:
            return ToolResult(error=str(e))

    async def _read_file(self, path: str) -> ToolResult:
        """Read file content."""
        try:
            file_path = Path(path)
            if not file_path.exists():
                return ToolResult(error=f"File not found: {path}")
            
            content = file_path.read_text(encoding='utf-8')
            return ToolResult(output=f"File content of {path}:\n{content}")
        except Exception as e:
            return ToolResult(error=f"Error reading file: {e}")

    async def _write_file(self, path: str, content: str) -> ToolResult:
        """Write content to file."""
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
            return ToolResult(output=f"Successfully wrote to {path}")
        except Exception as e:
            return ToolResult(error=f"Error writing file: {e}")

    async def _list_directory(self, path: str) -> ToolResult:
        """List directory contents."""
        try:
            dir_path = Path(path)
            if not dir_path.exists():
                return ToolResult(error=f"Directory not found: {path}")
            
            if not dir_path.is_dir():
                return ToolResult(error=f"Path is not a directory: {path}")
            
            items = []
            for item in dir_path.iterdir():
                item_type = "DIR" if item.is_dir() else "FILE"
                items.append(f"{item_type}: {item.name}")
            
            if not items:
                return ToolResult(output=f"Directory {path} is empty")
            
            return ToolResult(output=f"Contents of {path}:\n" + "\n".join(items))
        except Exception as e:
            return ToolResult(error=f"Error listing directory: {e}")

    async def _check_exists(self, path: str) -> ToolResult:
        """Check if path exists."""
        try:
            exists = Path(path).exists()
            return ToolResult(output=f"Path {path} {'exists' if exists else 'does not exist'}")
        except Exception as e:
            return ToolResult(error=f"Error checking path: {e}")
