"""
SimpleJSONAdapter - 简单JSON协议适配器实现
"""

import json
import time
from typing import Any, Dict, Optional, AsyncIterator
from uuid import uuid4

import httpx
from .base_adapter import BaseProtocolAdapter


class SimpleJSONAdapter(BaseProtocolAdapter):
    """
    简单JSON协议适配器
    
    使用最简单的JSON消息格式，专为测试和原型设计。
    无复杂验证，专注于消息传递的核心功能。
    """

    @property
    def protocol_name(self) -> str:
        return "simple_json"

    def __init__(
        self, 
        httpx_client: httpx.AsyncClient, 
        base_url: str,
        auth_headers: Optional[Dict[str, str]] = None,
        agent_id: str = "unknown"
    ):
        """
        初始化简单JSON适配器
        
        Parameters
        ----------
        httpx_client : httpx.AsyncClient
            HTTP客户端
        base_url : str
            目标Agent的基础URL
        auth_headers : Optional[Dict[str, str]]
            认证头（可选）
        agent_id : str
            当前Agent的ID
        """
        super().__init__(base_url=base_url, auth_headers=auth_headers or {})
        self.httpx_client = httpx_client
        self.base_url = base_url.rstrip('/')
        self.auth_headers = auth_headers or {}
        self.agent_id = agent_id

    async def initialize(self) -> None:
        """
        初始化适配器
        对于简单JSON协议，不需要特殊的初始化步骤
        """
        # 简单JSON协议不需要复杂的初始化
        pass

    async def send_message(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        """
        发送简单JSON消息
        
        Parameters
        ----------
        dst_id : str
            目标Agent ID
        payload : Dict[str, Any]
            消息负载
        
        Returns
        -------
        Any
            响应数据
        """
        # 构造简单JSON消息格式
        message_id = str(uuid4())
        simple_message = {
            "id": message_id,
            "from": self.agent_id,
            "to": dst_id,
            "timestamp": time.time(),
            "payload": payload
        }
        
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(self.auth_headers)
            
            response = await self.httpx_client.post(
                f"{self.base_url}/message",
                content=json.dumps(simple_message, separators=(',', ':')),
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.TimeoutException as e:
            raise TimeoutError(f"SimpleJSON message timeout to {dst_id} (msg_id: {message_id})") from e
        except httpx.HTTPStatusError as e:
            raise ConnectionError(f"SimpleJSON HTTP error {e.response.status_code}: {e.response.text} (msg_id: {message_id})") from e
        except Exception as e:
            raise RuntimeError(f"Failed to send SimpleJSON message to {dst_id}: {e}") from e

    async def send_message_streaming(self, dst_id: str, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        发送流式JSON消息（简化版本）
        """
        # 对于简单协议，流式处理返回单个响应
        result = await self.send_message(dst_id, payload)
        yield result

    async def receive_message(self) -> Dict[str, Any]:
        """接收消息（客户端适配器不适用）"""
        raise NotImplementedError("Client adapters do not receive messages directly")

    def convert_to_native(self, ute_message) -> Dict[str, Any]:
        """将UTE消息转换为简单JSON格式"""
        return {
            "id": ute_message.id,
            "from": ute_message.src,
            "to": ute_message.dst,
            "timestamp": ute_message.timestamp,
            "content": ute_message.content,
            "context": ute_message.context,
            "metadata": ute_message.metadata
        }

    def convert_from_native(self, native_message: Dict[str, Any]):
        """将简单JSON消息转换为UTE格式"""
        from ..core.unified_message import UTE
        
        return UTE(
            id=native_message.get("id", str(uuid4())),
            src=native_message.get("from", "unknown"),
            dst=native_message.get("to", "unknown"),
            timestamp=native_message.get("timestamp", time.time()),
            content=native_message.get("content", {}),
            context=native_message.get("context", {}),
            metadata=native_message.get("metadata", {})
        )

    def get_agent_card(self) -> Dict[str, Any]:
        """获取Agent能力和元数据"""
        return {
            "agent_id": self.agent_id,
            "protocol": "simple_json",
            "base_url": self.base_url,
            "capabilities": ["message_sending", "json_processing"],
            "supported_message_types": ["json"],
            "version": "1.0.0"
        }

    async def health_check(self) -> bool:
        """检查适配器是否健康和已连接"""
        try:
            # 尝试发送健康检查请求到目标URL
            response = await self.httpx_client.get(
                f"{self.base_url}/health",
                headers=self.auth_headers,
                timeout=5.0
            )
            return response.status_code == 200
        except Exception:
            return False