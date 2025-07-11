"""
Agent Protocol Server Adapter - A2A 框架的 Agent Protocol 服务器适配器
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Tuple, Optional, List
from uuid import uuid4
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, FileResponse
from starlette.routing import Route
from starlette.requests import Request

# 导入基础适配器
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from server_adapters.base_adapter import BaseServerAdapter
except ImportError:
    # 如果无法导入，创建一个简单的基类
    class BaseServerAdapter:
        def build_app(self, agent_card: Dict[str, Any], executor: Any) -> Starlette:
            raise NotImplementedError
        
        def get_default_agent_card(self, agent_id: str, host: str, port: int) -> Dict[str, Any]:
            raise NotImplementedError

logger = logging.getLogger(__name__)


class AgentProtocolTask:
    """Agent Protocol Task 数据结构"""
    
    def __init__(self, task_id: str, input_text: str, additional_input: Optional[Dict[str, Any]] = None):
        self.task_id = task_id
        self.input = input_text
        self.additional_input = additional_input or {}
        self.status = "created"
        self.steps: List[Dict[str, Any]] = []
        self.artifacts: List[Dict[str, Any]] = []


class AgentProtocolStep:
    """Agent Protocol Step 数据结构"""
    
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
    """Agent Protocol 服务器实现，兼容 Agent Protocol v1 规范"""
    
    def __init__(self, agent_card: Dict[str, Any], executor: Any):
        """
        初始化 Agent Protocol 服务器应用
        
        Parameters
        ----------
        agent_card : Dict[str, Any]
            智能体卡片信息
        executor : Any
            智能体执行器实例
        """
        self.agent_card = agent_card
        self.executor = executor
        self.tasks: Dict[str, AgentProtocolTask] = {}
        self.steps: Dict[str, AgentProtocolStep] = {}

        
    def build(self) -> Starlette:
        """构建 Starlette 应用"""
        routes = [
            # 标准智能体端点
            Route("/.well-known/agent.json", self.get_agent_card, methods=["GET"]),
            Route("/health", self.health_check, methods=["GET"]),
            
            # Agent Protocol v1 端点
            Route("/ap/v1/agent/tasks", self.create_task, methods=["POST"]),
            Route("/ap/v1/agent/tasks/{task_id}", self.get_task, methods=["GET"]),
            Route("/ap/v1/agent/tasks/{task_id}/steps", self.create_step, methods=["POST"]),
            Route("/ap/v1/agent/tasks/{task_id}/steps", self.list_steps, methods=["GET"]), 
            Route("/ap/v1/agent/tasks/{task_id}/steps/{step_id}", self.get_step, methods=["GET"]),
            Route("/ap/v1/agent/tasks/{task_id}/artifacts", self.list_artifacts, methods=["GET"]),
            Route("/ap/v1/agent/tasks/{task_id}/artifacts/{artifact_id}", self.get_artifact, methods=["GET"]),
            
            # A2A 兼容端点（可选）
            Route("/message", self.handle_a2a_message, methods=["POST"]),
        ]
        
        return Starlette(routes=routes)
    
    async def get_agent_card(self, request: Request) -> JSONResponse:
        """返回智能体卡片"""
        return JSONResponse(self.agent_card)
    
    async def health_check(self, request: Request) -> Response:
        """健康检查端点"""
        return Response("OK", status_code=200)
    
    async def create_task(self, request: Request) -> JSONResponse:
        """创建新任务 - POST /ap/v1/agent/tasks"""
        try:
            task_data = await request.json()
            
            # 验证必要字段
            if "input" not in task_data:
                return JSONResponse(
                    {"error": "Missing required field: input"}, 
                    status_code=400
                )
            
            # 创建任务
            task_id = str(uuid4())
            task = AgentProtocolTask(
                task_id=task_id,
                input_text=task_data["input"],
                additional_input=task_data.get("additional_input", {})
            )
            
            self.tasks[task_id] = task
            
            # 如果有执行器，调用任务处理逻辑
            if hasattr(self.executor, 'handle_task_creation'):
                try:
                    await self.executor.handle_task_creation(task)
                except Exception as e:
                    logger.warning(f"Executor task creation failed: {e}")
            
            # 返回任务信息
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
        """获取任务信息 - GET /ap/v1/agent/tasks/{task_id}"""
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
        """创建并执行步骤 - POST /ap/v1/agent/tasks/{task_id}/steps"""
        try:
            task_id = request.path_params["task_id"]
            step_data = await request.json()
            
            # 验证任务存在
            task = self.tasks.get(task_id)
            if not task:
                return JSONResponse(
                    {"error": "Task not found"}, 
                    status_code=404
                )
            
            # 创建步骤
            step_id = str(uuid4())
            step = AgentProtocolStep(
                step_id=step_id,
                task_id=task_id,
                name=step_data.get("name", ""),
                input_text=step_data.get("input", ""),
                additional_input=step_data.get("additional_input", {})
            )
            
            self.steps[step_id] = step
            
            # 执行步骤
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
                # 默认处理
                step.status = "completed"
                step.output = f"Step {step.name} executed"
            
            # 更新任务的步骤列表
            step_summary = {
                "step_id": step.step_id,
                "name": step.name,
                "status": step.status,
                "output": step.output,
                "is_last": step.is_last
            }
            task.steps.append(step_summary)
            
            # 如果是最后一步，更新任务状态
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
        """列出任务的所有步骤 - GET /ap/v1/agent/tasks/{task_id}/steps"""
        try:
            task_id = request.path_params["task_id"]
            
            # 验证任务存在
            task = self.tasks.get(task_id)
            if not task:
                return JSONResponse(
                    {"error": "Task not found"}, 
                    status_code=404
                )
            
            # 获取任务的所有步骤
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
        """获取特定步骤 - GET /ap/v1/agent/tasks/{task_id}/steps/{step_id}"""
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
        """列出任务的所有工件 - GET /ap/v1/agent/tasks/{task_id}/artifacts"""
        try:
            task_id = request.path_params["task_id"]
            
            # 验证任务存在
            task = self.tasks.get(task_id)
            if not task:
                return JSONResponse(
                    {"error": "Task not found"}, 
                    status_code=404
                )
            
            # 收集所有工件
            all_artifacts = []
            all_artifacts.extend(task.artifacts)
            
            # 从步骤中收集工件
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
        """下载特定工件 - GET /ap/v1/agent/tasks/{task_id}/artifacts/{artifact_id}"""
        try:
            task_id = request.path_params["task_id"]
            artifact_id = request.path_params["artifact_id"]
            
            # 验证任务存在
            task = self.tasks.get(task_id)
            if not task:
                return JSONResponse(
                    {"error": "Task not found"}, 
                    status_code=404
                )
            
            # 查找工件
            artifact = None
            
            # 在任务工件中查找
            for art in task.artifacts:
                if art.get("artifact_id") == artifact_id:
                    artifact = art
                    break
            
            # 在步骤工件中查找
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
            
            # 返回工件内容
            if "file_path" in artifact:
                # 文件工件
                return FileResponse(
                    artifact["file_path"],
                    filename=artifact.get("file_name", "artifact")
                )
            elif "content" in artifact:
                # 内容工件
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
        """处理 A2A 消息（兼容性端点）"""
        try:
            message_data = await request.json()
            
            # 从 A2A 消息中提取内容
            message_content = message_data.get("params", {}).get("message", {})
            
            # 将 A2A 消息转换为 Agent Protocol 任务
            if isinstance(message_content, dict):
                input_text = message_content.get("input", str(message_content))
                additional_input = message_content.get("additional_input", {})
            else:
                input_text = str(message_content)
                additional_input = {}
            
            # 创建任务
            task_id = str(uuid4())
            task = AgentProtocolTask(
                task_id=task_id,
                input_text=input_text,
                additional_input=additional_input
            )
            
            self.tasks[task_id] = task
            
            # 返回 A2A 格式响应
            return JSONResponse({
                "id": message_data.get("id", str(uuid4())),
                "result": {
                    "task_id": task.task_id,
                    "status": task.status,
                    "message": f"Task created: {task.task_id}"
                }
            })
            
        except Exception as e:
            logger.error(f"Error handling A2A message: {e}")
            return JSONResponse(
                {"error": str(e)}, 
                status_code=500
            )


class AgentProtocolServerAdapter(BaseServerAdapter):
    """Agent Protocol 服务器适配器"""
    
    def build_app(self, agent_card: Dict[str, Any], executor: Any) -> Starlette:
        """
        构建 Agent Protocol 服务器应用
        
        Parameters
        ----------
        agent_card : Dict[str, Any]
            智能体卡片
        executor : Any
            智能体执行器
            
        Returns
        -------
        Starlette
            配置好的 Starlette 应用
        """
        app_builder = AgentProtocolStarletteApplication(agent_card, executor)
        return app_builder.build()
    
    def get_default_agent_card(self, agent_id: str, host: str, port: int) -> Dict[str, Any]:
        """
        获取默认智能体卡片
        
        Parameters
        ----------
        agent_id : str
            智能体ID
        host : str
            主机地址
        port : int
            端口号
            
        Returns
        -------
        Dict[str, Any]
            默认智能体卡片
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