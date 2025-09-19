"""
Docker Sandbox Python Execution Tool

Provides isolated Python code execution with package installation support and intelligent error recovery.
"""
import asyncio
import json
import os
import re
from typing import Dict, Optional, Tuple

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

    async def _fix_code_with_llm(self, original_code: str, error_message: str, attempt: int) -> Tuple[str, list]:
        """
        Use LLM to analyze error and generate improved code.
        
        Args:
            original_code: The code that failed
            error_message: The error message from execution
            attempt: Current attempt number
            
        Returns:
            Tuple of (improved_code, suggested_packages)
        """
        try:
            # First try rule-based fixes for common issues
            fixed_code, packages = await self._apply_rule_based_fixes(original_code, error_message)
            if fixed_code != original_code:
                logger.info(f"🔧 Applied rule-based fix")
                return fixed_code, packages
            
            # If rule-based fixes didn't help, try LLM (if available)
            try:
                # Import here to avoid circular imports
                import sys
                import os
                
                # Add core directory to path for imports
                core_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core')
                if core_dir not in sys.path:
                    sys.path.insert(0, core_dir)
                
                from llm import LLM
                
                llm_client = LLM()
                
                # Create a prompt for code fixing
                fix_prompt = f"""You are an expert Python programmer. The following code failed with an error:

ORIGINAL CODE:
```python
{original_code}
```

ERROR MESSAGE:
{error_message}

ATTEMPT: {attempt + 1}

Please analyze the error and provide a corrected version of the code. Focus on:
1. Fixing the specific error mentioned
2. Adding proper error handling and defensive programming
3. Adding missing imports if needed
4. Suggesting required packages if ModuleNotFoundError occurs
5. Using safer coding practices (None checks, exception handling)

Respond with JSON format:
{{
    "analysis": "Brief explanation of what went wrong and how you fixed it",
    "fixed_code": "The corrected Python code",
    "packages": ["list", "of", "required", "packages"],
    "confidence": "high|medium|low"
}}

Make sure the fixed code is complete and self-contained. If the error is about missing packages, include them in the packages list.
"""

                messages = [
                    {"role": "system", "content": "You are an expert Python programmer who fixes code errors intelligently."},
                    {"role": "user", "content": fix_prompt}
                ]
                
                response = await llm_client.call_llm(
                    messages=messages,
                    model="gpt-4o-mini",  # Use fast model for code fixing
                    temperature=0.1,  # Low temperature for consistent fixes
                    max_tokens=2000
                )
                
                if response and response.get('choices'):
                    content = response['choices'][0]['message']['content']
                    
                    # Try to extract JSON response
                    try:
                        # Look for JSON block in response
                        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            # Look for JSON without code blocks
                            json_match = re.search(r'\{.*\}', content, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(0)
                            else:
                                raise ValueError("No JSON found in response")
                        
                        result = json.loads(json_str)
                        
                        fixed_code = result.get('fixed_code', original_code)
                        packages = result.get('packages', [])
                        analysis = result.get('analysis', 'No analysis provided')
                        confidence = result.get('confidence', 'unknown')
                        
                        logger.info(f"🔧 LLM code fix analysis (confidence: {confidence}): {analysis}")
                        logger.info(f"📦 LLM suggested packages: {packages}")
                        
                        return fixed_code, packages
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse LLM response as JSON: {e}")
                        logger.info(f"Raw LLM response: {content}")
                        
                        # Fallback: try to extract code from response
                        code_match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
                        if code_match:
                            extracted_code = code_match.group(1)
                            logger.info("🔧 Extracted code from non-JSON LLM response")
                            return extracted_code, []
                
                logger.warning("LLM code fixing failed, returning original code")
                return original_code, []
                
            except Exception as e:
                logger.info(f"LLM not available ({e}), using rule-based fixes only")
                return original_code, []
            
        except Exception as e:
            logger.error(f"Error in code fixing: {e}")
            return original_code, []

    async def _apply_rule_based_fixes(self, original_code: str, error_message: str) -> Tuple[str, list]:
        """
        Apply rule-based fixes for common Python errors.
        
        Args:
            original_code: The code that failed
            error_message: The error message from execution
            
        Returns:
            Tuple of (fixed_code, suggested_packages)
        """
        fixed_code = original_code
        packages = []
        
        # Fix 1: Add None checks for AttributeError on NoneType
        if "AttributeError" in error_message and "NoneType" in error_message:
            # Look for patterns like data.get() or obj.method()
            lines = fixed_code.split('\n')
            fixed_lines = []
            
            for line in lines:
                # Simple pattern: if line contains direct attribute access, add None check
                if '.' in line and '=' in line and not line.strip().startswith('#'):
                    # Look for pattern: variable = something.method()
                    attr_match = re.search(r'(\w+)\s*=\s*(\w+)\.(\w+)', line)
                    if attr_match:
                        var_name, obj_name, method_name = attr_match.groups()
                        # Add defensive check
                        indent = len(line) - len(line.lstrip())
                        indent_str = ' ' * indent
                        
                        fixed_lines.append(f"{indent_str}# Added defensive None check")
                        fixed_lines.append(f"{indent_str}if {obj_name} is not None:")
                        fixed_lines.append(f"{indent_str}    {line.strip()}")
                        fixed_lines.append(f"{indent_str}else:")
                        fixed_lines.append(f"{indent_str}    {var_name} = None  # Handle None case safely")
                        continue
                
                fixed_lines.append(line)
            
            if len(fixed_lines) > len(lines):
                fixed_code = '\n'.join(fixed_lines)
                logger.info("🔧 Applied None-check fix for AttributeError")
        
        # Fix 2: Add missing colon for syntax errors
        if "SyntaxError" in error_message and ("expected ':'" in error_message or ":" in error_message):
            lines = fixed_code.split('\n')
            fixed_lines = []
            
            for line in lines:
                # Remove inline comments for pattern matching
                line_without_comment = line.split('#')[0].rstrip()
                line_content = line_without_comment.strip()
                
                # Check for common missing colon patterns
                if re.match(r'^\s*(if|for|while|def|class|try|except|finally|with)\s+.*[^:]$', line_content):
                    # Add colon at the end of the actual content, preserve comments
                    if '#' in line:
                        # Line has comment, insert colon before comment
                        comment_part = line[line.index('#'):]
                        fixed_line = line_without_comment + ':  ' + comment_part
                    else:
                        # No comment, just add colon
                        fixed_line = line + ':'
                    fixed_lines.append(fixed_line)
                    logger.info(f"🔧 Added missing colon to: {line_content}")
                else:
                    fixed_lines.append(line)
            
            fixed_code = '\n'.join(fixed_lines)
        
        # Fix 3: Add try-except wrapper for risky operations
        if any(keyword in error_message for keyword in ["FileNotFoundError", "KeyError", "IndexError", "ValueError"]):
            # Wrap the main code in try-except
            lines = fixed_code.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            if non_empty_lines:
                # Find the main execution part (skip imports and function definitions)
                main_start = 0
                for i, line in enumerate(lines):
                    if not (line.strip().startswith('import') or 
                           line.strip().startswith('from') or 
                           line.strip().startswith('def ') or
                           line.strip().startswith('class ') or
                           line.strip().startswith('#') or
                           not line.strip()):
                        main_start = i
                        break
                
                if main_start < len(lines):
                    # Wrap main code in try-except
                    before_main = lines[:main_start]
                    main_code = lines[main_start:]
                    
                    fixed_lines = before_main + [
                        "try:",
                        *[f"    {line}" for line in main_code],
                        "except Exception as e:",
                        "    print(f'Error: {e}')",
                        "    print('Attempting to handle the error gracefully...')"
                    ]
                    
                    fixed_code = '\n'.join(fixed_lines)
                    logger.info("🔧 Added try-except wrapper for error handling")
        
        return fixed_code, packages

    async def _analyze_error_for_packages(self, error_message: str) -> list:
        """
        Analyze error message to suggest missing packages.
        
        Args:
            error_message: The error message from code execution
            
        Returns:
            List of suggested package names
        """
        packages = []
        
        # Common package mappings for ModuleNotFoundError
        package_mappings = {
            'pandas': ['pandas'],
            'numpy': ['numpy'], 
            'matplotlib': ['matplotlib'],
            'seaborn': ['seaborn'],
            'sklearn': ['scikit-learn'],
            'cv2': ['opencv-python'],
            'PIL': ['pillow'],
            'openpyxl': ['openpyxl'],
            'xlrd': ['xlrd'],
            'requests': ['requests'],
            'beautifulsoup4': ['beautifulsoup4'],
            'bs4': ['beautifulsoup4'],
            'scipy': ['scipy'],
            'networkx': ['networkx'],
            'plotly': ['plotly'],
            'streamlit': ['streamlit'],
            'flask': ['flask'],
            'django': ['django'],
            'fastapi': ['fastapi']
        }
        
        # Look for ModuleNotFoundError patterns
        module_not_found_pattern = r"No module named ['\"]([^'\"]+)['\"]"
        matches = re.findall(module_not_found_pattern, error_message, re.IGNORECASE)
        
        for module_name in matches:
            # Handle submodule cases (e.g., 'sklearn.metrics' -> 'sklearn')
            base_module = module_name.split('.')[0]
            
            if base_module in package_mappings:
                packages.extend(package_mappings[base_module])
            else:
                # Try the module name as package name
                packages.append(base_module)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(packages))

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
        max_retries: int = 3,
    ) -> ToolResult:
        """
        Executes Python code in a Docker sandbox with intelligent retry and error recovery.

        Args:
            code (str): The Python code to execute.
            packages (list): Python packages to install before execution.
            timeout (int): Execution timeout in seconds.
            max_retries (int): Maximum number of retry attempts for failed executions.

        Returns:
            ToolResult: Contains execution output or error message.
        """
        packages = packages or []
        current_code = code  # Track the current version of code
        
        for attempt in range(max_retries + 1):
            try:
                # Ensure sandbox is ready
                await self._ensure_sandbox()
                
                # Install packages if needed
                install_output = ""
                if packages:
                    install_output = await self._install_packages(packages)
                
                # Create Python script with enhanced path handling and error recovery
                script_content = f"""
import sys
import traceback
import os
from io import StringIO

# Store original stdout for final output
original_stdout = sys.stdout

# Capture stdout for user code
captured_output = StringIO()

# Add dataset directory to sys.path if available
if os.path.exists('/dataset'):
    sys.path.insert(0, '/dataset')
    os.environ['DATASET_DIR'] = '/dataset'

# Add workspace directory to sys.path
if os.path.exists('/workspace'):
    sys.path.insert(0, '/workspace')
    os.environ['WORKSPACE_DIR'] = '/workspace'
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

# Make resolve_dataset_file available globally (with error handling)
try:
    # Try different ways to make function globally available
    if isinstance(__builtins__, dict):
        __builtins__['resolve_dataset_file'] = resolve_dataset_file
    elif hasattr(__builtins__, '__dict__'):
        __builtins__.__dict__['resolve_dataset_file'] = resolve_dataset_file
    else:
        # __builtins__ is a module in some environments
        setattr(__builtins__, 'resolve_dataset_file', resolve_dataset_file)
except Exception as builtins_error:
    # Fallback: add to globals and print warning
    globals()['resolve_dataset_file'] = resolve_dataset_file

try:
    # Redirect stdout to capture user output
    sys.stdout = captured_output
    
    # Print environment info to captured output
    if os.path.exists('/dataset'):
        print("Dataset directory available at /dataset")
    if os.path.exists('/workspace'):
        print("Workspace directory available at /workspace")
    
    # Create namespace for execution
    exec_globals = globals().copy()
    
    # 准备要执行的完整代码
    full_code = '''
{current_code}
'''.strip()

    # 将整个代码块作为一个整体来执行
    # 这会保留所有的缩进和代码结构
    exec(full_code, exec_globals)
    
    # Restore stdout and get captured output
    sys.stdout = original_stdout
    user_output = captured_output.getvalue()
    
    # Print success markers and output
    print("EXECUTION_SUCCESS")
    print("OUTPUT_START")
    if user_output.strip():
        print(user_output.strip())
    else:
        print("(No output)")
    print("OUTPUT_END")
    
except Exception as e:
    # Restore stdout first
    sys.stdout = original_stdout
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
                await self._sandbox_client.write_file("execute_code.py", script_content)
                
                # Execute the script
                result = await self._sandbox_client.run_command(
                    "cd /workspace && python execute_code.py", 
                    timeout=timeout
                )
                
                # Debug: log raw result for troubleshooting
                logger.debug(f"Raw sandbox result (attempt {attempt + 1}):\n{result}")
                
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
                        if attempt > 0:
                            full_output += f"Note: Succeeded on attempt {attempt + 1}/{max_retries + 1} after code improvements\n\n"
                        full_output += f"Code Execution Output:\n{execution_output}"
                        
                        return ToolResult(output=full_output)
                    except (ValueError, IndexError) as parse_error:
                        logger.warning(f"Failed to parse successful execution output: {parse_error}")
                        if attempt < max_retries:
                            continue  # Retry on parsing failure
                        return ToolResult(output=f"Execution completed but output parsing failed:\n{result}")
                        
                elif "EXECUTION_ERROR" in result:
                    # Extract error information
                    try:
                        error_start = output_lines.index("ERROR_START") + 1
                        error_end = output_lines.index("ERROR_END")
                        error_info = '\n'.join(output_lines[error_start:error_end])
                        
                        # If we have retries left, try to fix the code with LLM
                        if attempt < max_retries:
                            logger.info(f"🔧 Attempt {attempt + 1} failed, trying to fix code with LLM...")
                            
                            # First, check if we can fix with simple package analysis
                            suggested_packages = await self._analyze_error_for_packages(error_info)
                            if suggested_packages:
                                logger.info(f"📦 Adding packages based on error: {suggested_packages}")
                                # Add missing packages and retry with same code
                                packages.extend(suggested_packages)
                                packages = list(dict.fromkeys(packages))  # Remove duplicates
                                continue
                            
                            # Use LLM to fix the code
                            fixed_code, additional_packages = await self._fix_code_with_llm(current_code, error_info, attempt)
                            
                            if fixed_code != current_code:
                                logger.info(f"� Code improved by LLM for attempt {attempt + 2}")
                                current_code = fixed_code
                                
                                # Add any additional packages suggested by LLM
                                if additional_packages:
                                    packages.extend(additional_packages)
                                    packages = list(dict.fromkeys(packages))  # Remove duplicates
                                    logger.info(f"📦 Added LLM-suggested packages: {additional_packages}")
                            else:
                                logger.info(f"🔄 LLM couldn't improve code, retrying with same code")
                            
                            continue  # Retry with improved code
                        
                        # No more retries, return the error
                        error_output = ""
                        if install_output:
                            error_output += f"Package Installation:\n{install_output}\n\n"
                        error_output += f"Final attempt failed after {attempt + 1} attempts with intelligent fixes\n\n"
                        error_output += f"Execution Error:\n{error_info}"
                        
                        return ToolResult(output="", error=error_output)
                    except (ValueError, IndexError):
                        if attempt < max_retries:
                            continue  # Retry on parsing failure
                        return ToolResult(output="", error=f"Execution failed:\n{result}")
                else:
                    # Unexpected result format - retry if we have attempts left
                    if attempt < max_retries:
                        continue
                    
                    full_output = ""
                    if install_output:
                        full_output += f"Package Installation:\n{install_output}\n\n"
                    full_output += f"Raw Output:\n{result}"
                    return ToolResult(output=full_output)
                    
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    logger.info(f"🔄 Retrying execution (attempt {attempt + 1}/{max_retries + 1}) due to timeout")
                    continue
                return ToolResult(
                    output="", 
                    error=f"Execution timeout after {timeout} seconds (tried {max_retries + 1} times)"
                )
            except Exception as e:
                if attempt < max_retries:
                    logger.info(f"🔄 Retrying execution (attempt {attempt + 1}/{max_retries + 1}) due to: {str(e)}")
                    continue
                return ToolResult(
                    output="",
                    error=f"Sandbox execution failed after {max_retries + 1} attempts: {str(e)}"
                )
        
        # This should never be reached, but just in case
        return ToolResult(
            output="",
            error=f"Maximum retries ({max_retries}) exceeded"
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