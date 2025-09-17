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
        """Resolve a path robustly for agent tools.

        Strategy:
        1) If absolute and exists -> return as-is
           If absolute but not exists -> try to rebase into workspace/dataset roots.
        2) If prefer_as_given -> try relative to CWD first
        3) Try GAIA_WORKSPACE_DIR, then GAIA_DATASET_DIR, then GAIA_ROOT
           If none provided, fallback to <project>/workspaces or <project>/workspace
        4) If only a filename (no parents) and not found, perform a shallow
           recursive search under dataset then workspace for a matching name
        5) Fallback to CWD
        """
        try:
            # Collect base candidates from env and sane defaults
            env_workspace = os.environ.get("GAIA_WORKSPACE_DIR")
            env_dataset = os.environ.get("GAIA_DATASET_DIR")
            env_root = os.environ.get("GAIA_ROOT")

            default_workspace = (GAIA_ROOT_DEFAULT / "workspaces")
            if not default_workspace.exists():
                # fallback to singular form
                default_workspace = (GAIA_ROOT_DEFAULT / "workspace")

            base_candidates = []
            if env_workspace:
                base_candidates.append(Path(env_workspace))
            # prefer an existing default workspace root
            if default_workspace:
                base_candidates.append(default_workspace)
            if env_dataset:
                base_candidates.append(Path(env_dataset))
            if env_root:
                base_candidates.append(Path(env_root))
            # always include GAIA root default at the end for safety
            if GAIA_ROOT_DEFAULT:
                base_candidates.append(GAIA_ROOT_DEFAULT)

            # 1) Absolute path handling
            if p.is_absolute():
                if p.exists():
                    return p
                # Try to rebase absolute path into our known bases when it doesn't exist
                parts = p.parts[1:]  # drop leading '/'
                relative = Path(*parts) if parts else Path(p.name)
                # For write scenario, allow creating under the first candidate
                if prefer_as_given and base_candidates:
                    return (base_candidates[0] / relative).resolve()
                # For read scenario, try to find an existing rebased path
                for base in base_candidates:
                    candidate = (base / relative).resolve()
                    if candidate.exists():
                        return candidate
                # As a last try, fall through to filename search below

            # 2) Use as-given relative path if requested and exists
            if prefer_as_given:
                ag = (Path.cwd() / p).resolve()
                if ag.exists():
                    return ag

            # 3) Try each base in order
            envs = {
                "GAIA_WORKSPACE_DIR": env_workspace,
                "GAIA_DATASET_DIR": env_dataset,
                "GAIA_ROOT": env_root,
            }
            for key in ("GAIA_WORKSPACE_DIR", "GAIA_DATASET_DIR", "GAIA_ROOT"):
                base = envs.get(key)
                if base:
                    candidate = (Path(base) / p).resolve()
                    if candidate.exists():
                        return candidate

            # Also try default workspace root
            if default_workspace:
                candidate = (default_workspace / p).resolve()
                if candidate.exists():
                    return candidate

            # 4) If it's just a filename, attempt shallow search (depth<=3) under dataset then workspace
            if len(p.parts) == 1:
                for base in (env_dataset, env_workspace, str(default_workspace)):
                    if not base:
                        continue
                    base_path = Path(base)
                    try:
                        found = self._find_by_name(base_path, p.name, max_depth=3)
                        if found:
                            return found
                    except Exception:
                        continue

            # 5) Fallback to CWD
            return (Path.cwd() / p).resolve()
        except Exception:
            # As a last resort, return the original
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
