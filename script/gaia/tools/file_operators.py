"""Simplified file operations tool.

Enhancements:
- Resolve relative paths using GAIA_WORKSPACE_DIR / GAIA_DATASET_DIR / GAIA_ROOT
- If only a bare filename is provided (e.g., bec7...jsonld), attempt to find
    it under the dataset/workspace directories via a shallow recursive search.
- If an absolute path is provided but doesn't exist (e.g., /documents/foo.txt),
    rebase it into workspace/dataset roots so LLM-provided paths still work.
"""
import asyncio
import os
from pathlib import Path
from typing import Optional, Union

from .base import BaseTool, ToolResult
from .exceptions import ToolError
from .utils.config import PROJECT_ROOT as GAIA_ROOT_DEFAULT  # 新增：用于默认回退


class FileOperators(BaseTool):
    """Tool for basic file operations."""

    name: str = "file_operators"
    description: str = "Perform file operations like reading, writing, and listing files."
    parameters: dict = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "The operation to perform: read, write, list, exists, is_dir, run_command",
                "enum": ["read", "write", "list", "exists", "is_dir", "run_command"]
            },
            "path": {
                "type": "string",
                "description": "The file or directory path"
            },
            "command": {
                "type": "string",
                "description": "Shell command to run (for run_command)."
            },
            "content": {
                "type": "string",
                "description": "Content to write (for write operation)"
            }
        },
        "required": ["operation"]
    }

    async def execute(self, operation: str, path: Optional[str] = None, content: Optional[str] = None, **kwargs) -> ToolResult:
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
            elif operation == "is_dir":
                return await self._check_is_dir(path)
            elif operation == "run_command":
                return await self._run_command(kwargs.get("command"))
            else:
                raise ToolError(f"Unknown operation: {operation}")
                
        except Exception as e:
            return ToolResult(error=str(e))

    async def _read_file(self, path: str) -> ToolResult:
        """Read file content."""
        try:
            file_path = self._resolve_path(Path(path))
            if not file_path.exists():
                return ToolResult(error=f"File not found: {path}")
            
            content = file_path.read_text(encoding='utf-8')
            return ToolResult(output=f"File content of {file_path}::\n{content}")
        except Exception as e:
            return ToolResult(error=f"Error reading file: {e}")

    async def _write_file(self, path: str, content: str) -> ToolResult:
        """Write content to file."""
        try:
            file_path = self._resolve_path(Path(path), prefer_as_given=True)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
            return ToolResult(output=f"Successfully wrote to {file_path}")
        except Exception as e:
            return ToolResult(error=f"Error writing file: {e}")

    async def _list_directory(self, path: str) -> ToolResult:
        """List directory contents."""
        try:
            dir_path = self._resolve_path(Path(path))
            if not dir_path.exists():
                return ToolResult(error=f"Directory not found: {path}")
            
            if not dir_path.is_dir():
                return ToolResult(error=f"Path is not a directory: {path}")
            
            items = []
            for item in dir_path.iterdir():
                item_type = "DIR" if item.is_dir() else "FILE"
                items.append(f"{item_type}: {item.name}")
            
            if not items:
                return ToolResult(output=f"Directory {dir_path} is empty")
            
            return ToolResult(output=f"Contents of {dir_path}:\n" + "\n".join(items))
        except Exception as e:
            return ToolResult(error=f"Error listing directory: {e}")

    async def _check_exists(self, path: str) -> ToolResult:
        """Check if path exists."""
        try:
            resolved = self._resolve_path(Path(path))
            exists = resolved.exists()
            return ToolResult(output=f"Path {resolved} {'exists' if exists else 'does not exist'}")
        except Exception as e:
            return ToolResult(error=f"Error checking path: {e}")

    async def _check_is_dir(self, path: str) -> ToolResult:
        """Check if path is a directory."""
        try:
            p = self._resolve_path(Path(path))
            if not p.exists():
                return ToolResult(output=f"Path {p} does not exist")
            return ToolResult(output=f"Path {p} {'is a directory' if p.is_dir() else 'is not a directory'}")
        except Exception as e:
            return ToolResult(error=f"Error checking directory: {e}")

    async def _run_command(self, command: Optional[str]) -> ToolResult:
        """Run a shell command and capture stdout/stderr."""
        try:
            if not command:
                return ToolResult(error="No command provided")
            import subprocess
            proc = subprocess.run(command, shell=True, capture_output=True, text=True)
            return ToolResult(output=f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        except Exception as e:
            return ToolResult(error=f"Error running command: {e}")

    # ---------------- Internal helpers ----------------
    def _resolve_path(self, p: Path, prefer_as_given: bool = False) -> Path:
        """Resolve a path for GAIA task workspace.

        Strategy for relative paths (including ./path):
        1. Remove leading './' for cleaner handling
        2. If prefer_as_given and path exists relative to CWD -> use it
        3. Try GAIA_WORKSPACE_DIR (primary, set by runner to workspaces/<protocol>/<task_id>/)
        4. Try GAIA_AGENT_WORKSPACE_DIR (legacy fallback)
        5. Try other GAIA environment variables as backup
        6. Fallback to CWD
        
        For absolute paths:
        1. If exists -> return as-is
        2. Try to rebase into workspace (for write operations)
        """
        try:
            # Handle relative paths
            if not p.is_absolute():
                # Remove leading './' for cleaner path handling
                path_str = str(p)
                if path_str.startswith('./'):
                    p = Path(path_str[2:])
                
                # For prefer_as_given (writes), try CWD first if it exists
                if prefer_as_given:
                    cwd_candidate = (Path.cwd() / p).resolve()
                    if cwd_candidate.exists():
                        return cwd_candidate
                
                # Primary: Try GAIA_WORKSPACE_DIR (workspaces/<protocol>/<task_id>/)
                workspace_dir = os.environ.get("GAIA_WORKSPACE_DIR")
                if workspace_dir:
                    candidate = (Path(workspace_dir) / p).resolve()
                    if candidate.exists() or prefer_as_given:
                        return candidate
                
                # Fallback: Try GAIA_AGENT_WORKSPACE_DIR (legacy)
                agent_workspace = os.environ.get("GAIA_AGENT_WORKSPACE_DIR")
                if agent_workspace:
                    candidate = (Path(agent_workspace) / p).resolve()
                    if candidate.exists() or prefer_as_given:
                        return candidate
                
                # Additional fallbacks for other GAIA environment variables
                for env_var in ["GAIA_DATASET_DIR", "GAIA_ROOT"]:
                    env_path = os.environ.get(env_var)
                    if env_path:
                        candidate = (Path(env_path) / p).resolve()
                        if candidate.exists():
                            return candidate
                
                # Try default workspace structure relative to GAIA_ROOT
                gaia_root = os.environ.get("GAIA_ROOT")
                if gaia_root:
                    default_workspace = Path(gaia_root) / "workspaces"
                    if default_workspace.exists():
                        candidate = (default_workspace / p).resolve()
                        if candidate.exists():
                            return candidate
                
                # Last resort: resolve relative to CWD
                return (Path.cwd() / p).resolve()
            
            # Handle absolute paths
            if p.exists():
                return p
            
            # For non-existing absolute paths, try to rebase into workspace for write operations
            if prefer_as_given:
                workspace_dir = os.environ.get("GAIA_WORKSPACE_DIR")
                if workspace_dir:
                    # Extract the filename/relative part from absolute path
                    try:
                        relative_part = Path(*p.parts[1:]) if len(p.parts) > 1 else Path(p.name)
                        return (Path(workspace_dir) / relative_part).resolve()
                    except Exception:
                        pass
            
            # Fallback to original path
            return p
            
        except Exception:
            # Safety fallback
            return p

    def _find_by_name(self, root: Path, name: str, max_depth: int = 2) -> Optional[Path]:
        """Find a file by name under root up to a max depth."""
        try:
            if max_depth < 0:
                return None
            for entry in root.iterdir():
                if entry.name == name:
                    return entry.resolve()
                if entry.is_dir() and max_depth > 0:
                    res = self._find_by_name(entry, name, max_depth - 1)
                    if res is not None:
                        return res
            return None
        except Exception:
            return None
