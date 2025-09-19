"""
Docker Sandbox Python Execution Tool

Provides isolated Python code execution with package installation support.
"""
import asyncio
import json
import os
from typing import Dict, Optional

from .base import BaseTool, ToolResult
from .utils.config import config
from .utils.logger import logger
from sandbox.client import create_sandbox_client


class SandboxPythonExecute(BaseTool):
    """A tool for executing Python code in a Docker sandbox with package installation support."""

    name: str = "sandbox_python_execute" 
    description: str = "Executes Python code in an isolated Docker container with ability to install packages. Safer than direct execution."
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
            "packages": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of Python packages to install before execution (e.g., ['biopython', 'numpy'])",
                "default": []
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds",
                "default": 60
            }
        },
        "required": ["code"],
    }

    def __init__(self):
        super().__init__()
        self._sandbox_client = None
        self._container_ready = False

    async def _ensure_sandbox(self):
        """Ensure sandbox is ready for execution."""
        if self._sandbox_client is None:
            from .utils.config import SandboxSettings
            
            # Configure sandbox with Python scientific computing image
            sandbox_config = SandboxSettings(
                use_sandbox=True,
                image="python:3.11",  # Use stable Python 3.11 image
                work_dir="/workspace", 
                memory_limit="2g",  # More memory for scientific computing
                cpu_limit=2.0,
                timeout=300,
                network_enabled=True  # Allow package installation
            )
            
            # Get workspace binding - improved with task-specific support
            workspace_bindings = {}
            
            # Try multiple workspace environment variables in priority order
            workspace_candidates = [
                os.environ.get("GAIA_WORKSPACE_DIR"),          # Primary: task-specific workspace
                os.environ.get("GAIA_AGENT_WORKSPACE_DIR"),    # Legacy: agent-specific workspace
            ]
            
            # Add dataset directory for multimodal tasks
            dataset_dir = os.environ.get("GAIA_DATASET_DIR")
            if dataset_dir and os.path.isdir(dataset_dir):
                workspace_bindings[dataset_dir] = "/dataset"
                logger.info(f"📁 Mounting dataset directory: {dataset_dir} -> /dataset")
            
            # Mount the primary workspace
            for agent_ws in workspace_candidates:
                if agent_ws and os.path.isdir(agent_ws):
                    workspace_bindings[agent_ws] = "/workspace"
                    logger.info(f"🏗️ Mounting workspace: {agent_ws} -> /workspace")
                    break
            else:
                # Create default workspace if none found
                task_id = os.environ.get("GAIA_TASK_ID", "default")
                protocol_name = os.environ.get("GAIA_PROTOCOL_NAME", "default")
                gaia_root = os.environ.get("GAIA_ROOT", "/tmp")
                default_ws = os.path.join(gaia_root, "workspaces", protocol_name, task_id)
                os.makedirs(default_ws, exist_ok=True)
                workspace_bindings[default_ws] = "/workspace"
                logger.info(f"🆕 Created and mounted default workspace: {default_ws} -> /workspace")
            
            self._sandbox_client = create_sandbox_client()
            await self._sandbox_client.create(sandbox_config, workspace_bindings)
            self._container_ready = True

    async def _install_packages(self, packages: list) -> str:
        """Install Python packages in the sandbox."""
        if not packages:
            return "No packages to install."
        
        try:
            # Update pip first
            await self._sandbox_client.run_command("python -m pip install --upgrade pip", timeout=60)
            
            # Install packages
            packages_str = " ".join(packages)
            install_cmd = f"python -m pip install {packages_str}"
            result = await self._sandbox_client.run_command(install_cmd, timeout=180)
            return f"Successfully installed packages: {packages}\n{result}"
        except Exception as e:
            return f"Failed to install packages {packages}: {str(e)}"

    async def execute(
        self,
        code: str,
        packages: Optional[list] = None,
        timeout: int = 60,
    ) -> ToolResult:
        """
        Executes Python code in a Docker sandbox.

        Args:
            code (str): The Python code to execute.
            packages (list): Python packages to install before execution.
            timeout (int): Execution timeout in seconds.

        Returns:
            ToolResult: Contains execution output or error message.
        """
        packages = packages or []
        
        try:
            # Ensure sandbox is ready
            await self._ensure_sandbox()
            
            # Install packages if needed
            install_output = ""
            if packages:
                install_output = await self._install_packages(packages)
            
            # Create Python script with enhanced path handling
            script_content = f"""
import sys
import traceback
import os
from io import StringIO

# Capture stdout
old_stdout = sys.stdout
sys.stdout = captured_output = StringIO()

# Add dataset directory to sys.path if available
if os.path.exists('/dataset'):
    sys.path.insert(0, '/dataset')
    os.environ['DATASET_DIR'] = '/dataset'
    print("Dataset directory available at /dataset")

# Add workspace directory to sys.path
if os.path.exists('/workspace'):
    sys.path.insert(0, '/workspace')
    os.environ['WORKSPACE_DIR'] = '/workspace'
    print("Workspace directory available at /workspace")
    # Set working directory to workspace
    os.chdir('/workspace')

# Import file path resolver functionality
def resolve_dataset_file(filename):
    \"\"\"Helper function to resolve dataset file paths.\"\"\"
    if os.path.exists('/dataset'):
        candidate = os.path.join('/dataset', filename)
        if os.path.exists(candidate):
            return candidate
        # Try case-insensitive search
        for f in os.listdir('/dataset'):
            if f.lower() == filename.lower():
                return os.path.join('/dataset', f)
    return filename

# Make resolve_dataset_file available globally
__builtins__['resolve_dataset_file'] = resolve_dataset_file

try:
    # Execute user code
{self._indent_code(code, '    ')}
    
    # Get the output and restore stdout
    output = captured_output.getvalue()
    sys.stdout = old_stdout
    
    print("EXECUTION_SUCCESS")
    print("OUTPUT_START")
    if output.strip():
        print(output.strip())
    else:
        print("(No output)")
    print("OUTPUT_END")
    
except Exception as e:
    # Restore stdout first
    sys.stdout = old_stdout
    print("EXECUTION_ERROR") 
    print("ERROR_START")
    print(f"{{type(e).__name__}}: {{str(e)}}")
    print("TRACEBACK_START")
    traceback.print_exc()
    print("TRACEBACK_END")
    print("ERROR_END")
"""
            
            # Write script to container
            await self._sandbox_client.write_file("execute_code.py", script_content)
            
            # Execute the script
            result = await self._sandbox_client.run_command(
                "cd /workspace && python execute_code.py", 
                timeout=timeout
            )
            
            # Parse result
            output_lines = result.split('\n')
            
            if "EXECUTION_SUCCESS" in result:
                # Extract output between OUTPUT_START and OUTPUT_END
                try:
                    start_idx = output_lines.index("OUTPUT_START") + 1
                    end_idx = output_lines.index("OUTPUT_END")
                    execution_output = '\n'.join(output_lines[start_idx:end_idx])
                    
                    full_output = ""
                    if install_output:
                        full_output += f"Package Installation:\n{install_output}\n\n"
                    full_output += f"Code Execution Output:\n{execution_output}"
                    
                    return ToolResult(output=full_output)
                except (ValueError, IndexError):
                    return ToolResult(output=f"Execution completed but output parsing failed:\n{result}")
                    
            elif "EXECUTION_ERROR" in result:
                # Extract error information
                try:
                    error_start = output_lines.index("ERROR_START") + 1
                    error_end = output_lines.index("ERROR_END")
                    error_info = '\n'.join(output_lines[error_start:error_end])
                    
                    error_output = ""
                    if install_output:
                        error_output += f"Package Installation:\n{install_output}\n\n"
                    error_output += f"Execution Error:\n{error_info}"
                    
                    return ToolResult(output="", error=error_output)
                except (ValueError, IndexError):
                    return ToolResult(output="", error=f"Execution failed:\n{result}")
            else:
                # Unexpected result format
                full_output = ""
                if install_output:
                    full_output += f"Package Installation:\n{install_output}\n\n"
                full_output += f"Raw Output:\n{result}"
                return ToolResult(output=full_output)
                
        except asyncio.TimeoutError:
            return ToolResult(
                output="", 
                error=f"Execution timeout after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                output="",
                error=f"Sandbox execution failed: {str(e)}"
            )

    def _indent_code(self, code: str, indent: str) -> str:
        """Add indentation to each line of code."""
        lines = code.split('\n')
        return '\n'.join(indent + line if line.strip() else line for line in lines)

    async def cleanup(self):
        """Clean up sandbox resources."""
        if self._sandbox_client:
            await self._sandbox_client.cleanup()
            self._sandbox_client = None
            self._container_ready = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()