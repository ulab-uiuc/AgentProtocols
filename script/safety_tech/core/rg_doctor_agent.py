# -*- coding: utf-8 -*-
"""
RG-Integrated Doctor Agents
真正通过RG注册并进行LLM对话的医生Agent
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
import uvicorn
import httpx
import logging
try:
    from ..protocol_backends.agora.registration_adapter import AgoraRegistrationAdapter
except ImportError:
    from protocol_backends.agora.registration_adapter import AgoraRegistrationAdapter

# 可选导入 ACP 适配器（根据配置选择）
try:
    from ..protocol_backends.acp.registration_adapter import ACPRegistrationAdapter
except ImportError:
    try:
        from protocol_backends.acp.registration_adapter import ACPRegistrationAdapter
    except Exception:
        ACPRegistrationAdapter = None  # 延迟失败：只有当配置要求acp时才会报错

# 导入基础Agent类
try:
    from .privacy_agent_base import DoctorAAgent, DoctorBAgent
except ImportError:
    from core.privacy_agent_base import DoctorAAgent, DoctorBAgent

logger = logging.getLogger(__name__)


class RGDoctorAAgent(DoctorAAgent):
    """通过RG注册的Doctor A Agent"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any], port: int):
        super().__init__(agent_id, config)
        self.port = port
        self.rg_endpoint = config.get('rg_endpoint', 'http://127.0.0.1:8001')
        self.conversation_id = config.get('conversation_id')
        self.endpoint = f"http://127.0.0.1:{port}"
        
        # FastAPI应用
        self.app = FastAPI(title=f"Doctor A Agent {agent_id}")
        self.setup_routes()
        
        # 注册状态
        self.registered = False
        self.session_token = None
        
        # 对话历史
        self.conversation_history = []
        
    def setup_routes(self):
        """设置FastAPI路由"""
        
        @self.app.post("/message")
        async def receive_message(payload: Dict[str, Any]):
            """接收消息并处理"""
            try:
                message_type = payload.get('type', 'normal')
                
                if message_type == 'mirror':
                    # Observer镜像消息，不需要回应
                    return {"status": "mirrored", "agent_id": self.agent_id}
                
                # 提取消息内容
                content = payload.get('text', payload.get('content', ''))
                sender_id = payload.get('sender_id', 'unknown')
                
                if not content:
                    return {"status": "no_content", "agent_id": self.agent_id}
                
                # 使用LLM处理消息
                response = await self.process_message(sender_id, content)
                
                # 保存对话历史
                self.conversation_history.append({
                    "timestamp": time.time(),
                    "sender": sender_id,
                    "received": content,
                    "response": response,
                    "type": "llm_conversation"
                })
                
                logger.debug(f"[{self.agent_id}] Processed message from {sender_id}, generated {len(response)} chars response")
                
                return {
                    "status": "processed",
                    "agent_id": self.agent_id,
                    "response": response,
                    "llm_used": True
                }
                
            except Exception as e:
                logger.error(f"[{self.agent_id}] Error processing message: {e}")
                return {"status": "error", "agent_id": self.agent_id, "error": str(e)}
        
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "registered": self.registered,
                "llm_available": self.use_llm,
                "conversation_turns": len(self.conversation_history)
            }
        
        @self.app.get("/conversation_history")
        async def get_conversation_history():
            """获取对话历史"""
            return {
                "agent_id": self.agent_id,
                "total_turns": len(self.conversation_history),
                "history": self.conversation_history
            }
    
    async def register_to_rg(self) -> bool:
        """注册到RG"""
        try:
            protocol = (self.config.get('protocol') or 'agora').lower()
            if protocol == 'acp':
                if ACPRegistrationAdapter is None:
                    raise RuntimeError("ACPRegistrationAdapter not available")
                adapter = ACPRegistrationAdapter({'rg_endpoint': self.rg_endpoint})
            else:
                adapter = AgoraRegistrationAdapter({'rg_endpoint': self.rg_endpoint, 'agora': {}, 'core': self.config.get('core', {})})

            result = await adapter.register_agent(
                agent_id=self.agent_id,
                endpoint=self.endpoint,
                conversation_id=self.conversation_id,
                role="doctor_a"
            )
            self.session_token = result.get('session_token')
            self.registered = True
            logger.info(f"[{self.agent_id}] Successfully registered to RG via Agora adapter")
            return True
                    
        except Exception as e:
            logger.error(f"[{self.agent_id}] Registration error: {e}")
            return False
    
    async def send_message_to_network(self, target_id: str, message: str) -> Dict[str, Any]:
        """通过RG网络发送消息"""
        if not self.registered:
            raise RuntimeError("Agent not registered to RG")
        
        # 通过协调器发送消息
        coordinator_endpoint = "http://127.0.0.1:8888"  # 协调器端点
        
        payload = {
            "sender_id": self.agent_id,
            "receiver_id": target_id,
            "text": message,
            "timestamp": time.time(),
            "llm_generated": True
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{coordinator_endpoint}/route_message",
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Message routing failed: {response.status_code}")
                    return {"error": f"Routing failed: {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Message sending error: {e}")
            return {"error": str(e)}
    
    def run_server(self):
        """运行FastAPI服务器"""
        uvicorn.run(self.app, host="127.0.0.1", port=self.port, log_level="warning", access_log=False)


class RGDoctorBAgent(DoctorBAgent):
    """通过RG注册的Doctor B Agent"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any], port: int):
        super().__init__(agent_id, config)
        self.port = port
        self.rg_endpoint = config.get('rg_endpoint', 'http://127.0.0.1:8001')
        self.conversation_id = config.get('conversation_id')
        self.endpoint = f"http://127.0.0.1:{port}"
        
        # FastAPI应用
        self.app = FastAPI(title=f"Doctor B Agent {agent_id}")
        self.setup_routes()
        
        # 注册状态
        self.registered = False
        self.session_token = None
        
        # 对话历史
        self.conversation_history = []
        
    def setup_routes(self):
        """设置FastAPI路由"""
        
        @self.app.post("/message")
        async def receive_message(payload: Dict[str, Any]):
            """接收消息并处理"""
            try:
                message_type = payload.get('type', 'normal')
                
                if message_type == 'mirror':
                    # Observer镜像消息，不需要回应
                    return {"status": "mirrored", "agent_id": self.agent_id}
                
                # 提取消息内容
                content = payload.get('text', payload.get('content', ''))
                sender_id = payload.get('sender_id', 'unknown')
                
                if not content:
                    return {"status": "no_content", "agent_id": self.agent_id}
                
                # 使用LLM处理消息
                response = await self.process_message(sender_id, content)
                
                # 保存对话历史
                self.conversation_history.append({
                    "timestamp": time.time(),
                    "sender": sender_id,
                    "received": content,
                    "response": response,
                    "type": "llm_conversation"
                })
                
                logger.debug(f"[{self.agent_id}] Processed message from {sender_id}, generated {len(response)} chars response")
                
                # 自动回复给发送者
                if sender_id != self.agent_id:
                    asyncio.create_task(self._auto_reply(sender_id, response))
                
                return {
                    "status": "processed",
                    "agent_id": self.agent_id,
                    "response": response,
                    "llm_used": True
                }
                
            except Exception as e:
                logger.error(f"[{self.agent_id}] Error processing message: {e}")
                return {"status": "error", "agent_id": self.agent_id, "error": str(e)}
        
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "registered": self.registered,
                "llm_available": self.use_llm,
                "conversation_turns": len(self.conversation_history)
            }
        
        @self.app.get("/conversation_history")
        async def get_conversation_history():
            """获取对话历史"""
            return {
                "agent_id": self.agent_id,
                "total_turns": len(self.conversation_history),
                "history": self.conversation_history
            }
    
    async def _auto_reply(self, target_id: str, message: str):
        """自动回复消息"""
        try:
            await asyncio.sleep(1)  # 短暂延迟
            await self.send_message_to_network(target_id, message)
        except Exception as e:
            logger.error(f"Auto reply failed: {e}")
    
    async def register_to_rg(self) -> bool:
        """注册到RG"""
        try:
            protocol = (self.config.get('protocol') or 'agora').lower()
            if protocol == 'acp':
                if ACPRegistrationAdapter is None:
                    raise RuntimeError("ACPRegistrationAdapter not available")
                adapter = ACPRegistrationAdapter({'rg_endpoint': self.rg_endpoint})
            else:
                adapter = AgoraRegistrationAdapter({'rg_endpoint': self.rg_endpoint, 'agora': {}, 'core': self.config.get('core', {})})

            result = await adapter.register_agent(
                agent_id=self.agent_id,
                endpoint=self.endpoint,
                conversation_id=self.conversation_id,
                role="doctor_b"
            )
            self.session_token = result.get('session_token')
            self.registered = True
            logger.info(f"[{self.agent_id}] Successfully registered to RG via Agora adapter")
            return True
                    
        except Exception as e:
            logger.error(f"[{self.agent_id}] Registration error: {e}")
            return False
    
    async def send_message_to_network(self, target_id: str, message: str) -> Dict[str, Any]:
        """通过RG网络发送消息"""
        if not self.registered:
            raise RuntimeError("Agent not registered to RG")
        
        # 通过协调器发送消息
        coordinator_endpoint = "http://127.0.0.1:8888"  # 协调器端点
        
        payload = {
            "sender_id": self.agent_id,
            "receiver_id": target_id,
            "text": message,
            "timestamp": time.time(),
            "llm_generated": True
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{coordinator_endpoint}/route_message",
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Message routing failed: {response.status_code}")
                    return {"error": f"Routing failed: {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Message sending error: {e}")
            return {"error": str(e)}
    
    def run_server(self):
        """运行FastAPI服务器"""
        uvicorn.run(self.app, host="127.0.0.1", port=self.port, log_level="warning", access_log=False)


async def create_and_start_doctor_agent(agent_class, agent_id: str, config: Dict[str, Any], port: int):
    """创建并启动医生Agent"""
    agent = agent_class(agent_id, config, port)
    
    # 在后台启动服务器
    import threading
    def run_server():
        agent.run_server()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # 等待服务器启动
    await asyncio.sleep(2)
    
    # 注册到RG
    success = await agent.register_to_rg()
    if not success:
        raise Exception(f"Failed to register {agent_id} to RG")
    
    return agent
