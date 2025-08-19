"""
SimpleJSONServerAdapter - 简单JSON协议服务器适配器
"""

import json
import asyncio
import logging
from typing import Any, Dict, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .base_adapter import BaseServerAdapter

logger = logging.getLogger(__name__)


class SimpleJSONStarletteApplication:
    """简单JSON协议的Starlette应用实现"""
    
    def __init__(self, executor: Any, agent_id: str):
        """
        初始化简单JSON应用
        
        Parameters
        ----------
        executor : Any
            SDK原生执行器
        agent_id : str
            Agent ID
        """
        self.executor = executor
        self.agent_id = agent_id

    async def health_check(self, request: Request) -> JSONResponse:
        """健康检查端点"""
        return JSONResponse({
            "status": "ok",
            "agent_id": self.agent_id,
            "protocol": "simple_json",
            "timestamp": asyncio.get_event_loop().time()
        })

    async def agent_card(self, request: Request) -> JSONResponse:
        """Agent card端点，返回Agent的能力和元数据"""
        return JSONResponse({
            "agent_id": self.agent_id,
            "protocol": "simple_json",
            "capabilities": ["message_processing", "health_check"],
            "supported_message_types": ["json"],
            "streaming": False,
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "message": "/message",
                "agent_card": "/.well-known/agent.json"
            }
        })

    async def handle_message(self, request: Request) -> JSONResponse:
        """
        处理传入的简单JSON消息
        
        使用SDK原生执行器接口处理消息
        """
        try:
            # 解析请求体
            body = await request.json()
            
            # 提取消息内容 - 支持多种协议格式
            message_content = None
            
            # 检查A2A格式: body["params"]["message"]
            if "params" in body and "message" in body["params"]:
                message_content = body["params"]["message"]
            
            # 检查直接payload格式
            elif "payload" in body:
                message_content = body["payload"]
            
            # 如果都没有，使用整个body
            else:
                message_content = body
            
            message_text = self._extract_text_content(message_content)
            
            # 创建模拟的RequestContext和EventQueue
            context = self._create_simple_context(message_text, body)
            event_queue = SimpleEventQueue()
            
            # 调用SDK原生执行器
            try:
                await self.executor.execute(context, event_queue)
                
                # 收集执行结果
                events = event_queue.get_events()
                result = self._format_response(events)
                
                return JSONResponse({
                    "success": True,
                    "response": result,
                    "agent_id": self.agent_id,
                    "message_id": body.get("id"),
                    "timestamp": asyncio.get_event_loop().time()
                })
                
            except Exception as exec_error:
                logger.error(f"Executor error: {exec_error}")
                return JSONResponse({
                    "success": False,
                    "error": f"Execution failed: {exec_error}",
                    "agent_id": self.agent_id,
                    "message_id": body.get("id")
                }, status_code=500)
                
        except Exception as e:
            logger.error(f"Message handling failed: {e}")
            return JSONResponse({
                "success": False,
                "error": f"Message handling failed: {e}",
                "agent_id": self.agent_id
            }, status_code=500)

    def _extract_text_content(self, payload: Dict[str, Any]) -> str:
        """从负载中提取文本内容"""
        if isinstance(payload, dict):
            # 处理Gaia文档广播
            if payload.get("type") == "gaia_document_init" and "document" in payload:
                document = payload["document"]
                print(f"[DEBUG] Gaia document received: type={type(document)}, keys={list(document.keys()) if isinstance(document, dict) else 'N/A'}")
                if isinstance(document, dict):
                    # 提取文档的主要内容
                    content_parts = []
                    if "title" in document:
                        content_parts.append(f"Title: {document['title']}")
                        print(f"[DEBUG] Found title: {document['title']}")
                    if "content" in document:
                        content_length = len(document['content']) if document['content'] else 0
                        content_parts.append(f"Content: {document['content'][:100]}..." if content_length > 100 else f"Content: {document['content']}")
                        print(f"[DEBUG] Found content: length={content_length}")
                    if "question" in document:
                        content_parts.append(f"Question: {document['question']}")
                        print(f"[DEBUG] Found question: {document['question']}")
                    result = "\n".join(content_parts) if content_parts else json.dumps(document)
                    print(f"[DEBUG] Extracted content result: {result[:100]}...")
                    return result
                else:
                    return str(document)
            
            # 尝试多种可能的文本字段
            for field in ["text", "content", "message", "query", "input"]:
                if field in payload:
                    return str(payload[field])
            # 如果没有找到特定字段，返回整个负载的字符串表示
            return json.dumps(payload)
        else:
            return str(payload)

    def _create_simple_context(self, text: str, original_message: Dict[str, Any]):
        """创建简单的RequestContext"""
        class SimpleContext:
            def __init__(self, text: str, original: Dict[str, Any]):
                self.text = text
                self.original = original
                
            def get_user_input(self) -> str:
                return self.text
                
            def get_message_data(self) -> Dict[str, Any]:
                return self.original
        
        return SimpleContext(text, original_message)

    def _format_response(self, events: list) -> str:
        """格式化响应事件"""
        if not events:
            return "No response generated"
        
        # 简单地连接所有事件的文本
        responses = []
        for event in events:
            if hasattr(event, 'text'):
                responses.append(event.text)
            elif isinstance(event, dict):
                responses.append(event.get('text', str(event)))
            else:
                responses.append(str(event))
        
        return "\n".join(responses) if responses else "Processing completed"


class SimpleEventQueue:
    """简单的事件队列实现"""
    
    def __init__(self):
        self.events = []
    
    async def enqueue_event(self, event):
        """添加事件到队列"""
        self.events.append(event)
    
    def get_events(self):
        """获取所有事件"""
        return self.events.copy()


class SimpleJSONServerAdapter(BaseServerAdapter):
    """
    简单JSON协议服务器适配器
    
    提供轻量级的JSON消息处理，无复杂验证
    """
    
    def create_app(self, executor: Any, agent_id: str, **kwargs) -> Starlette:
        """
        创建简单JSON Starlette应用
        
        Parameters
        ----------
        executor : Any
            SDK原生执行器
        agent_id : str
            Agent ID
        **kwargs
            额外配置参数
            
        Returns
        -------
        Starlette
            配置好的Starlette应用
        """
        app_instance = SimpleJSONStarletteApplication(executor, agent_id)
        
        # 定义路由
        routes = [
            Route("/health", app_instance.health_check, methods=["GET"]),
            Route("/message", app_instance.handle_message, methods=["POST"]),
            Route("/", app_instance.health_check, methods=["GET"]),  # 根路径也返回健康状态
            Route("/.well-known/agent.json", app_instance.agent_card, methods=["GET"]),  # Agent card端点
        ]
        
        # 创建Starlette应用
        app = Starlette(routes=routes)
        
        return app

    def get_protocol_name(self) -> str:
        """返回协议名称"""
        return "simple_json"

    def supports_streaming(self) -> bool:
        """简单JSON协议支持基本响应，不支持复杂流式处理"""
        return False

    def build(self, host: str, port: int, agent_id: str, executor: Any, **kwargs) -> tuple:
        """
        构建简单JSON协议的uvicorn服务器
        
        Parameters
        ----------
        host : str
            服务器主机地址
        port : int
            服务器端口
        agent_id : str
            Agent ID
        executor : Any
            业务逻辑执行器
        **kwargs
            额外配置参数
            
        Returns
        -------
        tuple
            (uvicorn.Server实例, agent_card字典)
        """
        import uvicorn
        
        # 创建Starlette应用
        app = self.create_app(executor, agent_id, **kwargs)
        
        # 创建uvicorn配置
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="error",  # 减少HTTP请求日志输出
            access_log=False,   # 关闭访问日志
            lifespan="off"     # 禁用lifespan避免CancelledError
        )
        
        # 创建服务器
        server = uvicorn.Server(config)
        
        # 创建agent card
        agent_card = {
            "agent_id": agent_id,
            "protocol": "simple_json",
            "host": host,
            "port": port,
            "capabilities": ["message_processing", "health_check"],
            "supported_message_types": ["json"],
            "streaming": False
        }
        
        return server, agent_card