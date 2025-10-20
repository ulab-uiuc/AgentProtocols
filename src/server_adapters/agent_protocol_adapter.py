"""
Agent Protocol Server Adapter - Agent Protocol server adapter for A2A framework
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Tuple, Optional, List
from uuid import uuid4
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, FileResponse
from starlette.routing import Route
from starlette.requests import Request

# Import base adapter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from server_adapters.base_adapter import BaseServerAdapter
except ImportError:
    # If import fails, define a minimal base class
    class BaseServerAdapter:
        def build(self, agent_card: Dict[str, Any], executor: Any) -> Starlette:
            raise NotImplementedError
        
        def get_default_agent_card(self, agent_id: str, host: str, port: int) -> Dict[str, Any]:
            raise NotImplementedError

logger = logging.getLogger(__name__)


class AgentProtocolTask:
    """Agent Protocol Task data structure"""
    
    def __init__(self, task_id: str, input_text: str, additional_input: Optional[Dict[str, Any]] = None):
        self.task_id = task_id
        self.input = input_text
        self.additional_input = additional_input or {}
        self.status = "created"
        self.steps: List[Dict[str, Any]] = []
        self.artifacts: List[Dict[str, Any]] = []


class AgentProtocolStep:
    """Agent Protocol Step data structure"""
    
    def __init__(self, step_id: str, task_id: str, name: str = "", input_text: str = "", 
                 additional_input: Optional[Dict[str, Any]] = None):
        self.step_id = step_id
        self.task_id = task_id
        self.name = name or f"step_{step_id[:8]}"
        self.input = input_text
        self.additional_input = additional_input or {}
        self.status = "created"
        self.output = ""
        self.additional_output: Dict[str, Any] = {}
        self.artifacts: List[Dict[str, Any]] = []
        self.is_last = False


class AgentProtocolStarletteApplication:
    """Agent Protocol server implementation compatible with Agent Protocol v1"""
    
    def __init__(self, agent_card: Dict[str, Any], executor: Any):
        """
        Initialize the Agent Protocol server application.
        
        Parameters
        ----------
        agent_card : Dict[str, Any]
            Agent card information
        executor : Any
            Agent executor instance
        """
        self.agent_card = agent_card
        self.executor = executor
        self.tasks: Dict[str, AgentProtocolTask] = {}
        self.steps: Dict[str, AgentProtocolStep] = {}

        
    def build(self) -> Starlette:
        """Build Starlette application"""
        routes = [
            # Standard agent endpoints
            Route("/.well-known/agent.json", self.get_agent_card, methods=["GET"]),
            Route("/health", self.health_check, methods=["GET"]),
            
            # Agent Protocol v1 endpoints
            Route("/ap/v1/agent/tasks", self.create_task, methods=["POST"]),
            Route("/ap/v1/agent/tasks/{task_id}", self.get_task, methods=["GET"]),
            Route("/ap/v1/agent/tasks/{task_id}/steps", self.create_step, methods=["POST"]),
            Route("/ap/v1/agent/tasks/{task_id}/steps", self.list_steps, methods=["GET"]), 
            Route("/ap/v1/agent/tasks/{task_id}/steps/{step_id}", self.get_step, methods=["GET"]),
            Route("/ap/v1/agent/tasks/{task_id}/artifacts", self.list_artifacts, methods=["GET"]),
            Route("/ap/v1/agent/tasks/{task_id}/artifacts/{artifact_id}", self.get_artifact, methods=["GET"]),
            
            # A2A compatible endpoint (optional)
            Route("/message", self.handle_a2a_message, methods=["POST"]),
        ]
        
        return Starlette(routes=routes)
    
    async def get_agent_card(self, request: Request) -> JSONResponse:
        """Return the agent card"""
        return JSONResponse(self.agent_card)
    
    async def health_check(self, request: Request) -> Response:
        """Health check endpoint"""
        return Response("OK", status_code=200)
    
    async def create_task(self, request: Request) -> JSONResponse:
        """Create new task - POST /ap/v1/agent/tasks"""
        try:
            task_data = await request.json()
            
            # Validate required fields
            if "input" not in task_data:
                return JSONResponse(
                    {"error": "Missing required field: input"}, 
                    status_code=400
                )
            
            # Create task
            task_id = str(uuid4())
            task = AgentProtocolTask(
                task_id=task_id,
                input_text=task_data["input"],
                additional_input=task_data.get("additional_input", {})
            )
            
            self.tasks[task_id] = task
            
            # If an executor is available, call task handling logic
            if hasattr(self.executor, 'handle_task_creation'):
                try:
                    await self.executor.handle_task_creation(task)
                except Exception as e:
                    logger.warning(f"Executor task creation failed: {e}")
            
            # Return task information
            return JSONResponse({
                "task_id": task.task_id,
                "input": task.input,
                "additional_input": task.additional_input,
                "status": task.status,
                "steps": task.steps,
                "artifacts": task.artifacts
            })
            
        except json.JSONDecodeError:
            return JSONResponse(
                {"error": "Invalid JSON"}, 
                status_code=400
            )
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            return JSONResponse(
                {"error": str(e)}, 
                status_code=500
            )
    
    async def get_task(self, request: Request) -> JSONResponse:
        """Get task information - GET /ap/v1/agent/tasks/{task_id}"""
        try:
            task_id = request.path_params["task_id"]
            task = self.tasks.get(task_id)
            
            if not task:
                return JSONResponse(
                    {"error": "Task not found"}, 
                    status_code=404
                )
            
            return JSONResponse({
                "task_id": task.task_id,
                "input": task.input,
                "additional_input": task.additional_input,
                "status": task.status,
                "steps": task.steps,
                "artifacts": task.artifacts
            })
            
        except Exception as e:
            logger.error(f"Error getting task: {e}")
            return JSONResponse(
                {"error": str(e)}, 
                status_code=500
            )
    
    async def create_step(self, request: Request) -> JSONResponse:
        """Create and execute step - POST /ap/v1/agent/tasks/{task_id}/steps"""
        try:
            task_id = request.path_params["task_id"]
            step_data = await request.json()
            
            # Validate task exists
            task = self.tasks.get(task_id)
            if not task:
                return JSONResponse(
                    {"error": "Task not found"}, 
                    status_code=404
                )
            
            # Create step
            step_id = str(uuid4())
            step = AgentProtocolStep(
                step_id=step_id,
                task_id=task_id,
                name=step_data.get("name", ""),
                input_text=step_data.get("input", ""),
                additional_input=step_data.get("additional_input", {})
            )
            
            self.steps[step_id] = step
            
            # Execute step
            if hasattr(self.executor, 'execute_step'):
                try:
                    result = await self.executor.execute_step(step)
                    if result:
                        step.output = result.get("output", "")
                        step.additional_output = result.get("additional_output", {})
                        step.status = result.get("status", "completed")
                        step.is_last = result.get("is_last", False)
                        step.artifacts = result.get("artifacts", [])
                except Exception as e:
                    logger.error(f"Step execution failed: {e}")
                    step.status = "failed"
                    step.output = f"Execution error: {str(e)}"
            else:
                # Default handling
                step.status = "completed"
                step.output = f"Step {step.name} executed"
            
            # Update the task's step list
            step_summary = {
                "step_id": step.step_id,
                "name": step.name,
                "status": step.status,
                "output": step.output,
                "is_last": step.is_last
            }
            task.steps.append(step_summary)
            
            # If it's the last step, update task status
            if step.is_last:
                task.status = "completed"
            
            return JSONResponse({
                "step_id": step.step_id,
                "task_id": step.task_id,
                "name": step.name,
                "status": step.status,
                "input": step.input,
                "additional_input": step.additional_input,
                "output": step.output,
                "additional_output": step.additional_output,
                "artifacts": step.artifacts,
                "is_last": step.is_last
            })
            
        except json.JSONDecodeError:
            return JSONResponse(
                {"error": "Invalid JSON"}, 
                status_code=400
            )
        except Exception as e:
            logger.error(f"Error creating step: {e}")
            return JSONResponse(
                {"error": str(e)}, 
                status_code=500
            )
    
    async def list_steps(self, request: Request) -> JSONResponse:
        """List all steps of a task - GET /ap/v1/agent/tasks/{task_id}/steps"""
        try:
            task_id = request.path_params["task_id"]
            
            # Validate task exists
            task = self.tasks.get(task_id)
            if not task:
                return JSONResponse(
                    {"error": "Task not found"}, 
                    status_code=404
                )
            
            # Get all steps for the task
            task_steps = [
                {
                    "step_id": step.step_id,
                    "task_id": step.task_id,
                    "name": step.name,
                    "status": step.status,
                    "input": step.input,
                    "additional_input": step.additional_input,
                    "output": step.output,
                    "additional_output": step.additional_output,
                    "artifacts": step.artifacts,
                    "is_last": step.is_last
                }
                for step in self.steps.values()
                if step.task_id == task_id
            ]
            
            return JSONResponse({"steps": task_steps})
            
        except Exception as e:
            logger.error(f"Error listing steps: {e}")
            return JSONResponse(
                {"error": str(e)}, 
                status_code=500
            )
    
    async def get_step(self, request: Request) -> JSONResponse:
        """Get a specific step - GET /ap/v1/agent/tasks/{task_id}/steps/{step_id}"""
        try:
            task_id = request.path_params["task_id"]
            step_id = request.path_params["step_id"]
            
            step = self.steps.get(step_id)
            
            if not step or step.task_id != task_id:
                return JSONResponse(
                    {"error": "Step not found"}, 
                    status_code=404
                )
            
            return JSONResponse({
                "step_id": step.step_id,
                "task_id": step.task_id,
                "name": step.name,
                "status": step.status,
                "input": step.input,
                "additional_input": step.additional_input,
                "output": step.output,
                "additional_output": step.additional_output,
                "artifacts": step.artifacts,
                "is_last": step.is_last
            })
            
        except Exception as e:
            logger.error(f"Error getting step: {e}")
            return JSONResponse(
                {"error": str(e)}, 
                status_code=500
            )
    
    async def list_artifacts(self, request: Request) -> JSONResponse:
        """List all artifacts for a task - GET /ap/v1/agent/tasks/{task_id}/artifacts"""
        try:
            task_id = request.path_params["task_id"]
            
            # Validate task exists
            task = self.tasks.get(task_id)
            if not task:
                return JSONResponse(
                    {"error": "Task not found"}, 
                    status_code=404
                )
            
            # Collect all artifacts
            all_artifacts = []
            all_artifacts.extend(task.artifacts)
            
            # Collect artifacts from steps
            for step in self.steps.values():
                if step.task_id == task_id:
                    all_artifacts.extend(step.artifacts)
            
            return JSONResponse({"artifacts": all_artifacts})
            
        except Exception as e:
            logger.error(f"Error listing artifacts: {e}")
            return JSONResponse(
                {"error": str(e)}, 
                status_code=500
            )
    
    async def get_artifact(self, request: Request) -> Response:
        """Download a specific artifact - GET /ap/v1/agent/tasks/{task_id}/artifacts/{artifact_id}"""
        try:
            task_id = request.path_params["task_id"]
            artifact_id = request.path_params["artifact_id"]
            
            # Validate task exists
            task = self.tasks.get(task_id)
            if not task:
                return JSONResponse(
                    {"error": "Task not found"}, 
                    status_code=404
                )
            
            # Find artifact
            artifact = None
            
            # Search in task artifacts
            for art in task.artifacts:
                if art.get("artifact_id") == artifact_id:
                    artifact = art
                    break
            
            # Search in step artifacts
            if not artifact:
                for step in self.steps.values():
                    if step.task_id == task_id:
                        for art in step.artifacts:
                            if art.get("artifact_id") == artifact_id:
                                artifact = art
                                break
                        if artifact:
                            break
            
            if not artifact:
                return JSONResponse(
                    {"error": "Artifact not found"}, 
                    status_code=404
                )
            
            # Return artifact content
            if "file_path" in artifact:
                # File artifact
                return FileResponse(
                    artifact["file_path"],
                    filename=artifact.get("file_name", "artifact")
                )
            elif "content" in artifact:
                # Content artifact
                return Response(
                    content=artifact["content"],
                    media_type=artifact.get("content_type", "text/plain")
                )
            else:
                return JSONResponse(artifact)
                
        except Exception as e:
            logger.error(f"Error getting artifact: {e}")
            return JSONResponse(
                {"error": str(e)}, 
                status_code=500
            )
    
    async def handle_a2a_message(self, request: Request) -> JSONResponse:
        """Handle A2A message (compatibility endpoint) - directly execute LLM and return response"""
        try:
            message_data = await request.json()
            
            # Extract content from A2A message
            message_content = message_data.get("params", {}).get("message", {})
            
            # Extract text content
            if isinstance(message_content, dict):
                # Try multiple formats
                if "parts" in message_content and message_content["parts"]:
                    # A2A format: {"parts": [{"type": "text", "text": "question"}]}
                    input_text = message_content["parts"][0].get("text", str(message_content))
                else:
                    # Direct format: {"input": "question"} or other
                    input_text = message_content.get("input", str(message_content))
                additional_input = message_content.get("additional_input", {})
            else:
                input_text = str(message_content)
                additional_input = {}
            
            # If an executor is available, directly invoke LLM for answering
            if hasattr(self.executor, 'execute_step'):
                try:
                    # Create a temporary step object
                    class TempStep:
                        def __init__(self, input_text):
                            self.input = input_text
                            self.step_id = str(uuid4())
                    
                    step = TempStep(input_text)
                    
                    # Directly execute step to get LLM response
                    result = await self.executor.execute_step(step)
                    
                    # Extract output text
                    output_text = result.get("output", "No response") if isinstance(result, dict) else str(result)
                    
                    # Return Agent Protocol formatted response with the LLM answer
                    return JSONResponse({
                        "id": message_data.get("id", str(uuid4())),
                        "result": {
                            "output": output_text,
                            "status": "completed",
                            "response_type": "agent_protocol"
                        }
                    })
                    
                except Exception as e:
                    logger.error(f"Error executing LLM: {e}")
                    # If LLM execution fails, return error info
                    return JSONResponse({
                        "id": message_data.get("id", str(uuid4())),
                        "result": {
                            "output": f"Error processing request: {str(e)}",
                            "status": "failed",
                            "response_type": "agent_protocol"
                        }
                    })
            
            # If no executor, create task (keep original logic as fallback)
            task_id = str(uuid4())
            task = AgentProtocolTask(
                task_id=task_id,
                input_text=input_text,
                additional_input=additional_input
            )
            
            self.tasks[task_id] = task
            
            # Return Agent Protocol formatted response
            return JSONResponse({
                "id": message_data.get("id", str(uuid4())),
                "result": {
                    "output": f"Task created: {task.task_id}",
                    "status": "created",
                    "task_id": task.task_id,
                    "response_type": "agent_protocol"
                }
            })
            
        except Exception as e:
            logger.error(f"Error handling A2A message: {e}")
            return JSONResponse(
                {"error": str(e)}, 
                status_code=500
            )


class AgentProtocolServerAdapter(BaseServerAdapter):
    """Agent Protocol server adapter"""
    
    protocol_name = "AgentProtocol"
    
    def build(
        self,
        host: str,
        port: int,
        agent_id: str,
        executor: Any,
        **kwargs
    ) -> Tuple[uvicorn.Server, Dict[str, Any]]:
        """
        Build an Agent Protocol server instance.
        
        Parameters
        ----------
        host : str
            Server host address
        port : int
            Server port
        agent_id : str
            Agent ID
        executor : Any
            Agent executor
        **kwargs : dict
            Additional configuration parameters
            
        Returns
        -------
        Tuple[uvicorn.Server, Dict[str, Any]]
            Server instance and agent card
        """
        import uvicorn
        
        # Generate default agent card
        agent_card = self.get_default_agent_card(agent_id, host, port)
        
        # Create Agent Protocol Starlette application
        app_builder = AgentProtocolStarletteApplication(agent_card, executor)
        app = app_builder.build()
        
        # Configure uvicorn server
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="error",  # Minimize server logs
            lifespan="off"      # Disable lifespan to avoid CancelledError
        )
        server = uvicorn.Server(config)
        
        return server, agent_card
    
    def get_default_agent_card(self, agent_id: str, host: str, port: int) -> Dict[str, Any]:
        """
        Get default agent card.
        
        Parameters
        ----------
        agent_id : str
            Agent ID
        host : str
            Host address
        port : int
            Port number
            
        Returns
        -------
        Dict[str, Any]
            Default agent card
        """
        return {
            "id": agent_id,
            "name": f"Agent Protocol Agent - {agent_id}",
            "description": "Agent Protocol v1 compatible agent supporting Task/Step/Artifact paradigm",
            "version": "1.0.0",
            "url": f"http://{host}:{port}",
            "protocolVersion": "v1",
            "capabilities": [
                "agent_protocol_v1",
                "task_management",
                "step_execution", 
                "artifact_handling",
                "a2a_compatibility"
            ],
            "protocols": ["agent_protocol", "a2a"],
            "endpoints": {
                "agent_card": "/.well-known/agent.json",
                "health": "/health",
                "tasks": "/ap/v1/agent/tasks",
                "steps": "/ap/v1/agent/tasks/{task_id}/steps",
                "artifacts": "/ap/v1/agent/tasks/{task_id}/artifacts",
                "a2a_message": "/message"
            },
            "supportedInputModes": ["text", "json"],
            "supportedOutputModes": ["text", "json", "artifacts"],
            "maxConcurrentTasks": 10,
            "supportsStreaming": False,
            "supportsAuthentication": True
        }