# -*- coding: utf-8 -*-
"""
RG-Integrated Coordinator
基于注册网关(RG)的协调器实现，支持动态路由和Observer角色
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Set
import httpx
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
import uvicorn
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParticipantInfo:
    """参与者信息"""
    agent_id: str
    protocol: str
    endpoint: str
    role: str
    verified: bool
    joined_at: float


@dataclass 
class ConversationMessage:
    """会话消息"""
    sender_id: str
    receiver_id: str
    content: str
    timestamp: float
    message_id: str
    role: str
    correlation_id: str | None = None


class RGCoordinator:
    """基于RG的协调器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rg_endpoint = config.get('rg_endpoint', 'http://127.0.0.1:8001')
        self.conversation_id = config.get('conversation_id', f'conv_{int(time.time())}')
        self.port = config.get('coordinator_port', 8888)
        
        # FastAPI应用
        self.app = FastAPI(title="RG Coordinator")
        self.setup_routes()
        
        # 参与者管理
        self.participants: Dict[str, ParticipantInfo] = {}
        # observers属性已移除 - 新S2设计不需要Observer机制
        
        # 消息历史（用于backfill）
        self.message_history: List[ConversationMessage] = []
        self.max_history_size = config.get('max_history_size', 100)
        
        # Bridge配置
        self.bridge_config = config.get('bridge', {})
        self.enable_live_mirror = self.bridge_config.get('enable_live_mirror', True)
        self.enable_backfill = self.bridge_config.get('enable_backfill', True)
        self.backfill_limit = self.bridge_config.get('backfill_limit', 10)
        
        # 轮询配置
        self.directory_poll_interval = config.get('directory_poll_interval', 5.0)
        self.running = False
        
        # 性能优化：缓存协议后端注册表
        self._backend_registry = None
        self._registry_initialized = False
    
    def setup_routes(self):
        """设置HTTP路由"""
        
        @self.app.post("/route_message")
        async def route_message_endpoint(payload: Dict[str, Any]):
            """消息路由端点"""
            try:
                sender_id = payload.get('sender_id')
                receiver_id = payload.get('receiver_id')
                
                if not sender_id:
                    raise HTTPException(status_code=400, detail="Missing sender_id")
                
                # 统一关联ID：如未提供则生成
                corr = payload.get('correlation_id')
                if not corr:
                    corr = f"corr_{int(time.time()*1000)}"
                    payload['correlation_id'] = corr

                result = await self.route_message(sender_id, receiver_id, payload)
                return result
                
            except Exception as e:
                # 对于未注册发送者的错误，使用debug级别（这通常是攻击测试）
                if "not registered" in str(e):
                    logger.debug(f"Message routing blocked: {e}")
                else:
                    logger.error(f"Message routing error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/deliver")
        async def deliver_message_endpoint(payload: Dict[str, Any]):
            """由目标Agent回投的业务回执。
            期望字段：sender_id, receiver_id, correlation_id, content/text/body 可三选一。
            """
            try:
                sender_id = payload.get('sender_id')
                receiver_id = payload.get('receiver_id')
                if not sender_id:
                    raise HTTPException(status_code=400, detail="Missing sender_id")
                # 内容抽取
                content = self._extract_message_content(payload)
                correlation_id = payload.get('correlation_id')
                # 角色推断
                role = self.participants.get(sender_id).role if sender_id in self.participants else 'unknown'
                message = ConversationMessage(
                    sender_id=sender_id,
                    receiver_id=receiver_id or 'broadcast',
                    content=content,
                    timestamp=time.time(),
                    message_id=f"deliver_{int(time.time()*1000)}",
                    role=role,
                    correlation_id=correlation_id
                )
                # 入库
                self._store_message(message)
                # 镜像给Observers
                if self.enable_live_mirror:
                    await self._broadcast_to_observers(message, payload)
                return {"status": "received", "message_id": message.message_id, "correlation_id": correlation_id}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Deliver handling error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/conversation_status")
        async def get_conversation_status_endpoint():
            """获取会话状态"""
            return await self.get_conversation_status()
        
        @self.app.get("/message_history")
        async def get_message_history_endpoint(limit: int = 50):
            """获取消息历史"""
            return await self.get_message_history(limit)
        
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            return {
                "status": "healthy",
                "conversation_id": self.conversation_id,
                "participants": len(self.participants),
                "observers": 0,  # Observer机制已移除
                "message_count": len(self.message_history)
            }
        
    async def start(self):
        """启动协调器"""
        self.running = True
        
        # 启动目录轮询任务
        asyncio.create_task(self._directory_polling_loop())
        
        # 启动HTTP服务器
        import threading
        def run_server():
            uvicorn.run(self.app, host="127.0.0.1", port=self.port, log_level="warning", access_log=False, lifespan="off", loop="asyncio", http="h11")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        await asyncio.sleep(2)  # 等待服务器启动
        
        logger.info(f"RG Coordinator started for conversation {self.conversation_id} on port {self.port}")
    
    async def stop(self):
        """停止协调器"""
        self.running = False
        logger.info("RG Coordinator stopped")
    
    async def _directory_polling_loop(self):
        """目录轮询循环"""
        while self.running:
            try:
                await self._refresh_participants()
                await asyncio.sleep(self.directory_poll_interval)
            except Exception as e:
                logger.error(f"Directory polling error: {e}")
                await asyncio.sleep(self.directory_poll_interval)
    
    async def _refresh_participants(self):
        """刷新参与者列表"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.rg_endpoint}/directory",
                    params={"conversation_id": self.conversation_id},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    directory = response.json()
                    await self._update_participants(directory['participants'])
                elif response.status_code != 404:  # 404表示会话不存在，正常情况
                    logger.warning(f"Directory query failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Failed to refresh participants: {e}")
    
    async def _update_participants(self, participants_data: List[Dict[str, Any]]):
        """更新参与者信息"""
        new_participants = {}
        # new_observers已移除 - 新S2设计不需要Observer机制
        
        for participant_data in participants_data:
            agent_id = participant_data['agent_id']
            role = participant_data['role']
            
            participant = ParticipantInfo(
                agent_id=agent_id,
                protocol=participant_data['protocol'],
                endpoint=participant_data['endpoint'],
                role=role,
                verified=participant_data['verified'],
                joined_at=participant_data['joined_at']
            )
            
            new_participants[agent_id] = participant
            
            if role == 'observer':
                # Observer处理已移除 - 新S2设计不需要Observer机制
                continue
        
        # 检测新加入的参与者
        for agent_id in new_participants:
            if agent_id not in self.participants:
                logger.info(f"New participant joined: {agent_id} ({new_participants[agent_id].role})")
        
        self.participants = new_participants
        # self.observers已移除 - 新S2设计不需要Observer机制
    
    async def route_message(self, sender_id: str, receiver_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """路由消息"""
        # 验证发送者
        if sender_id not in self.participants:
            raise ValueError(f"Sender {sender_id} not registered in conversation")
        
        sender = self.participants[sender_id]
        
        # 检查发送权限
        if sender.role == 'observer':
            raise ValueError("Observers cannot send messages")

        # 角色配对约束：仅允许 doctor_a ↔ doctor_b 之间通信
        if receiver_id:
            receiver = self.participants.get(receiver_id)
            if not receiver:
                raise ValueError(f"Receiver {receiver_id} not registered in conversation")
            allowed_pairs = {('doctor_a', 'doctor_b'), ('doctor_b', 'doctor_a')}
            if (sender.role, receiver.role) not in allowed_pairs and receiver.role != 'observer':
                raise ValueError(f"Routing not allowed between roles {sender.role} -> {receiver.role}")
        
        # 验证接收者（如果指定）
        if receiver_id and receiver_id not in self.participants:
            raise ValueError(f"Receiver {receiver_id} not registered in conversation")
        
        # 提取消息内容
        content = self._extract_message_content(payload)
        
        # 创建消息记录
        corr_id = payload.get('correlation_id')
        message = ConversationMessage(
            sender_id=sender_id,
            receiver_id=receiver_id or 'broadcast',
            content=content,
            timestamp=time.time(),
            message_id=f"msg_{int(time.time() * 1000)}",
            role=sender.role,
            correlation_id=corr_id
        )
        
        # 存储消息历史
        self._store_message(message)
        
        # 路由消息
        result = await self._deliver_message(message, payload)
        
        # 向Observers广播镜像
        if self.enable_live_mirror:
            await self._broadcast_to_observers(message, payload)
        
        return result
    
    def _get_backend_registry(self):
        """获取缓存的协议后端注册表"""
        if not self._registry_initialized:
            try:
                from script.safety_tech.protocol_backends.common.interfaces import get_registry
                self._backend_registry = get_registry()
                self._registry_initialized = True
                logger.info("协议后端注册表已缓存")
            except Exception as e:
                logger.error(f"协议后端注册表初始化失败: {e}")
                raise RuntimeError(f"Protocol backend registry not available: {e}")
        return self._backend_registry
    
    def _extract_message_content(self, payload: Dict[str, Any]) -> str:
        """提取消息内容"""
        # 支持多种payload格式
        if 'text' in payload:
            return payload['text']
        elif 'body' in payload:
            return payload['body']
        elif 'content' in payload:
            return payload['content']
        else:
            return json.dumps(payload)
    
    def _store_message(self, message: ConversationMessage):
        """存储消息到历史"""
        self.message_history.append(message)
        
        # 限制历史大小
        if len(self.message_history) > self.max_history_size:
            self.message_history = self.message_history[-self.max_history_size:]
    
    async def _deliver_message(self, message: ConversationMessage, original_payload: Dict[str, Any]) -> Dict[str, Any]:
        """投递消息到目标"""
        if message.receiver_id == 'broadcast':
            # 广播到所有非Observer参与者
            results = []
            for participant in self.participants.values():
                if participant.role != 'observer' and participant.agent_id != message.sender_id:
                    try:
                        result = await self._send_to_participant(participant, original_payload)
                        results.append({"agent_id": participant.agent_id, "result": result})
                    except Exception as e:
                        logger.error(f"Failed to send to {participant.agent_id}: {e}")
                        results.append({"agent_id": participant.agent_id, "error": str(e)})
            return {"broadcast_results": results}
        else:
            # 单播到指定接收者
            receiver = self.participants[message.receiver_id]
            try:
                result = await self._send_to_participant(receiver, original_payload)
                return result
            except Exception as e:
                logger.error(f"Failed to send to {message.receiver_id}: {e}")
                return {"error": str(e)}
    
    async def _broadcast_to_observers(self, message: ConversationMessage, original_payload: Dict[str, Any]):
        """向Observers广播镜像 - 已禁用"""
        # Observer广播已移除 - 新S2设计不需要Observer机制
        return
        
        # 构建Observer镜像payload
        mirror_payload = {
            "type": "mirror",
            "original_message": {
                "sender_id": message.sender_id,
                "receiver_id": message.receiver_id,
                "content": message.content,
                "timestamp": message.timestamp,
                "message_id": message.message_id,
                "sender_role": message.role,
                "correlation_id": message.correlation_id
            },
            "original_payload": original_payload
        }
        
        # Observer镜像逻辑已移除 - 新S2设计不需要Observer机制
    
    async def _provide_backfill(self, observer_id: str):
        """为Observer提供历史回填"""
        if not self.message_history:
            return
        
        observer = self.participants.get(observer_id)
        if not observer:
            return
        
        # 获取最近的消息
        recent_messages = self.message_history[-self.backfill_limit:]
        
        backfill_payload = {
            "type": "backfill",
            "conversation_id": self.conversation_id,
            "message_count": len(recent_messages),
            "messages": []
        }
        
        for msg in recent_messages:
            backfill_payload["messages"].append({
                "sender_id": msg.sender_id,
                "receiver_id": msg.receiver_id,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "message_id": msg.message_id,
                "sender_role": msg.role
            })
        
        try:
            # 直接向Observer的/message端点发送回填
            await self._send_to_endpoint(observer.endpoint, backfill_payload)
            logger.info(f"Provided backfill to observer {observer_id}: {len(recent_messages)} messages")
        except Exception as e:
            logger.error(f"Failed to provide backfill to observer {observer_id}: {e}")

    async def _send_to_endpoint(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """协议无关直连HTTP发送。
        约定：Observer等通用HTTP接收方暴露 /message 接口。
        """
        url = (endpoint or '').rstrip('/') + '/message'
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=10.0)
                if resp.status_code in (200, 202):
                    try:
                        return resp.json()
                    except Exception:
                        return {"status": "received"}
                raise RuntimeError(f"Endpoint {url} returned {resp.status_code}: {resp.text}")
        except Exception as e:
            raise RuntimeError(f"Failed to send to endpoint {url}: {e}")
    
    async def _send_to_participant(self, participant: ParticipantInfo, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发送消息到参与者：通过协议注册表分发到对应后端客户端。

        要求：
        - 后端实现必须为原生协议，不允许mock/fallback
        - 协调器不再拼装各协议负载细节，由各client实现负责
        """
        # 使用缓存的注册表，避免重复查询
        registry = self._get_backend_registry()
        backend = registry.get(participant.protocol)
        if backend is None:
            raise RuntimeError(f"No backend registered for protocol: {participant.protocol}")

        # 从payload中提取correlation_id（兼容现有逻辑）
        correlation_id = payload.get('correlation_id')
        
        # 添加超时控制，避免长时间阻塞
        try:
            result = await asyncio.wait_for(
                backend.send(participant.endpoint, payload, correlation_id),
                timeout=35.0  # 35秒超时，给各协议后端足够时间处理
            )
            return result
        except asyncio.TimeoutError:
            error_msg = f"Backend send timeout for {participant.protocol} to {participant.endpoint}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    # 管理接口
    async def get_conversation_status(self) -> Dict[str, Any]:
        """获取会话状态"""
        return {
            "conversation_id": self.conversation_id,
            "participants": {
                agent_id: {
                    "protocol": p.protocol,
                    "role": p.role,
                    "verified": p.verified,
                    "joined_at": p.joined_at
                }
                for agent_id, p in self.participants.items()
            },
            "observers": [],  # Observer机制已移除
            "message_count": len(self.message_history),
            "last_activity": self.message_history[-1].timestamp if self.message_history else None,
            "bridge_config": {
                "live_mirror_enabled": self.enable_live_mirror,
                "backfill_enabled": self.enable_backfill,
                "backfill_limit": self.backfill_limit
            }
        }
    
    async def force_refresh_directory(self):
        """强制刷新目录"""
        await self._refresh_participants()
    
    async def get_message_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取消息历史"""
        messages = self.message_history
        if limit:
            messages = messages[-limit:]
        
        return [
            {
                "sender_id": msg.sender_id,
                "receiver_id": msg.receiver_id,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "message_id": msg.message_id,
                "sender_role": msg.role,
                "correlation_id": msg.correlation_id
            }
            for msg in messages
        ]
    
    async def request_observer_backfill(self, observer_id: str, limit: Optional[int] = None) -> bool:
        """为Observer请求历史回填 - 已禁用"""
        # Observer回填已移除 - 新S2设计不需要Observer机制
        return False
        
        if not self.enable_backfill:
            return False
        
        # 使用指定限制或默认限制
        backfill_limit = limit or self.backfill_limit
        original_limit = self.backfill_limit
        self.backfill_limit = backfill_limit
        
        try:
            await self._provide_backfill(observer_id)
            return True
        except Exception as e:
            logger.error(f"Manual backfill failed for {observer_id}: {e}")
            return False
        finally:
            self.backfill_limit = original_limit
