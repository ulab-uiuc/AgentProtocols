"""
Agent Protocol 客户端适配器 - 实现 Agent Protocol v1 规范的客户端
"""

import json
from typing import Any, Dict, Optional, AsyncIterator
from uuid import uuid4

import httpx
from .base_adapter import BaseProtocolAdapter


class AgentProtocolAdapter(BaseProtocolAdapter):
    """
    Agent Protocol v1 客户端适配器
    
    实现与 Agent Protocol 兼容服务器的通信，支持任务创建、步骤执行和工件处理
    """

    def __init__(
        self, 
        httpx_client: httpx.AsyncClient, 
        base_url: str,
        auth_headers: Optional[Dict[str, str]] = None,
        agent_card_path: str = "/.well-known/agent.json"
    ):
        super().__init__(base_url=base_url, auth_headers=auth_headers or {})
        self.httpx_client = httpx_client
        self.base_url = base_url.rstrip('/')
        self.auth_headers = auth_headers or {}
        self.agent_card_path = agent_card_path
        self.agent_card: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """初始化适配器并获取智能体卡片"""
        try:
            headers = {}
            headers.update(self.auth_headers)
            
            response = await self.httpx_client.get(
                f"{self.base_url}{self.agent_card_path}",
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            self.agent_card = response.json()
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Agent Protocol adapter: {e}") from e

    async def send_message(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        """
        发送消息到 Agent Protocol 服务器
        
        将消息转换为 Agent Protocol 任务并返回任务结果
        """
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(self.auth_headers)
            
            # 将消息转换为 Agent Protocol 任务格式
            task_data = {
                "input": payload.get("message", str(payload)),
                "additional_input": payload.get("context", {})
            }
            
            # 创建任务
            response = await self.httpx_client.post(
                f"{self.base_url}/ap/v1/agent/tasks",
                content=json.dumps(task_data, separators=(',', ':')),
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            task_result = response.json()
            
            return {
                "task_id": task_result.get("task_id"),
                "status": task_result.get("status"),
                "message": f"Task created: {task_result.get('task_id')}",
                "result": task_result
            }
            
        except httpx.HTTPStatusError as e:
            raise ConnectionError(f"Agent Protocol HTTP error {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            raise RuntimeError(f"Agent Protocol send failed: {e}") from e

    async def create_task(self, input_text: str, additional_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建 Agent Protocol 任务"""
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(self.auth_headers)
            
            task_data = {
                "input": input_text,
                "additional_input": additional_input or {}
            }
            
            response = await self.httpx_client.post(
                f"{self.base_url}/ap/v1/agent/tasks",
                content=json.dumps(task_data, separators=(',', ':')),
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            raise RuntimeError(f"Failed to create task: {e}") from e

    async def create_step(self, task_id: str, step_input: str, additional_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """为任务创建步骤"""
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(self.auth_headers)
            
            step_data = {
                "input": step_input,
                "additional_input": additional_input or {}
            }
            
            response = await self.httpx_client.post(
                f"{self.base_url}/ap/v1/agent/tasks/{task_id}/steps",
                content=json.dumps(step_data, separators=(',', ':')),
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            raise RuntimeError(f"Failed to create step: {e}") from e

    async def get_task(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        try:
            headers = {}
            headers.update(self.auth_headers)
            
            response = await self.httpx_client.get(
                f"{self.base_url}/ap/v1/agent/tasks/{task_id}",
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            raise RuntimeError(f"Failed to get task: {e}") from e

    async def list_steps(self, task_id: str) -> Dict[str, Any]:
        """列出任务的所有步骤"""
        try:
            headers = {}
            headers.update(self.auth_headers)
            
            response = await self.httpx_client.get(
                f"{self.base_url}/ap/v1/agent/tasks/{task_id}/steps",
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            raise RuntimeError(f"Failed to list steps: {e}") from e

    async def receive_message(self) -> Dict[str, Any]:
        """
        Agent Protocol 不支持主动接收消息
        返回空结果
        """
        return {"messages": []}

    def get_agent_card(self) -> Dict[str, Any]:
        """获取智能体卡片"""
        return self.agent_card

    async def health_check(self) -> bool:
        """检查智能体健康状态"""
        try:
            headers = {}
            headers.update(self.auth_headers)
            
            response = await self.httpx_client.get(
                f"{self.base_url}/health",
                headers=headers,
                timeout=5.0
            )
            return response.status_code == 200
        except Exception:
            return False

    async def cleanup(self) -> None:
        """清理资源"""
        # Agent Protocol 适配器通常不需要特殊清理
        pass

    def __repr__(self) -> str:
        """调试表示"""
        return (
            f"AgentProtocolAdapter(base_url='{self.base_url}', "
            f"auth={'enabled' if self.auth_headers else 'disabled'}, "
            f"card_loaded={bool(self.agent_card)})"
        )