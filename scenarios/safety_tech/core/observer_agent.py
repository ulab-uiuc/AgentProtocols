# -*- coding: utf-8 -*-
"""
Observer Agent Implementation
Observer角色Agent实现，支持只读镜像流和历史回填
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import logging
from fastapi import FastAPI, HTTPException
import uvicorn

logger = logging.getLogger(__name__)


@dataclass
class MirrorMessage:
    """镜像消息"""
    sender_id: str
    receiver_id: str
    content: str
    timestamp: float
    message_id: str
    sender_role: str
    mirror_received_at: float


@dataclass
class BackfillMessage:
    """回填消息"""
    sender_id: str
    receiver_id: str
    content: str
    timestamp: float
    message_id: str
    sender_role: str


class ObserverAgent:
    """Observer角色Agent"""
    
    def __init__(self, observer_id: str, config: Dict[str, Any]):
        self.observer_id = observer_id
        self.config = config
        self.conversation_id = config.get('conversation_id')
        
        # FastAPI应用
        self.app = FastAPI(title=f"Observer Agent {observer_id}")
        self.setup_routes()
        
        # 消息存储
        self.mirror_messages: List[MirrorMessage] = []
        self.backfill_messages: List[BackfillMessage] = []
        self.max_stored_messages = config.get('max_stored_messages', 1000)
        
        # 回调函数
        self.on_mirror_message: Optional[Callable] = None
        self.on_backfill_received: Optional[Callable] = None
        
        # 统计信息
        self.stats = {
            'total_mirrors_received': 0,
            'total_backfill_messages': 0,
            'first_message_time': None,
            'last_message_time': None,
            'unique_senders': set(),
            'conversation_rounds': 0
        }
        
        # 窃听检测
        self.eavesdrop_detector = EavesdropDetector(config.get('eavesdrop_detection', {}))
        
    def setup_routes(self):
        """设置FastAPI路由"""
        
        @self.app.post("/message")
        async def receive_message(payload: Dict[str, Any]):
            """接收消息（镜像或回填）"""
            try:
                message_type = payload.get('type', 'unknown')
                
                if message_type == 'mirror':
                    await self._handle_mirror_message(payload)
                elif message_type == 'backfill':
                    await self._handle_backfill(payload)
                else:
                    logger.warning(f"Unknown message type: {message_type}")
                
                return {"status": "received", "observer_id": self.observer_id}
                
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/status")
        async def get_status():
            """获取Observer状态"""
            return {
                "observer_id": self.observer_id,
                "conversation_id": self.conversation_id,
                "stats": {
                    "total_mirrors_received": self.stats['total_mirrors_received'],
                    "total_backfill_messages": self.stats['total_backfill_messages'],
                    "first_message_time": self.stats['first_message_time'],
                    "last_message_time": self.stats['last_message_time'],
                    "unique_senders": list(self.stats['unique_senders']),
                    "conversation_rounds": self.stats['conversation_rounds']
                },
                "eavesdrop_metrics": self.eavesdrop_detector.get_metrics()
            }
        
        @self.app.get("/messages")
        async def get_messages(message_type: str = "all", limit: int = 100):
            """获取接收到的消息"""
            if message_type == "mirror":
                messages = self.mirror_messages[-limit:]
                return {
                    "type": "mirror",
                    "count": len(messages),
                    "messages": [self._serialize_mirror_message(msg) for msg in messages]
                }
            elif message_type == "backfill":
                messages = self.backfill_messages[-limit:]
                return {
                    "type": "backfill", 
                    "count": len(messages),
                    "messages": [self._serialize_backfill_message(msg) for msg in messages]
                }
            else:
                # 返回所有消息，按时间排序
                all_messages = []
                
                for msg in self.mirror_messages:
                    all_messages.append(("mirror", msg.timestamp, self._serialize_mirror_message(msg)))
                
                for msg in self.backfill_messages:
                    all_messages.append(("backfill", msg.timestamp, self._serialize_backfill_message(msg)))
                
                all_messages.sort(key=lambda x: x[1])  # 按时间排序
                
                return {
                    "type": "all",
                    "count": len(all_messages),
                    "messages": [{"source": msg[0], "data": msg[2]} for msg in all_messages[-limit:]]
                }
        
        @self.app.get("/eavesdrop_report")
        async def get_eavesdrop_report():
            """获取窃听报告"""
            return self.eavesdrop_detector.generate_report()
    
    async def _handle_mirror_message(self, payload: Dict[str, Any]):
        """处理镜像消息"""
        original_message = payload.get('original_message', {})
        
        mirror_msg = MirrorMessage(
            sender_id=original_message.get('sender_id', ''),
            receiver_id=original_message.get('receiver_id', ''),
            content=original_message.get('content', ''),
            timestamp=original_message.get('timestamp', time.time()),
            message_id=original_message.get('message_id', ''),
            sender_role=original_message.get('sender_role', ''),
            mirror_received_at=time.time()
        )
        
        # 存储消息
        self._store_mirror_message(mirror_msg)
        
        # 更新统计
        self._update_stats(mirror_msg)
        
        # 窃听检测
        self.eavesdrop_detector.analyze_mirror_message(mirror_msg, payload)
        
        # 回调
        if self.on_mirror_message:
            await self.on_mirror_message(mirror_msg, payload)
        
        logger.debug(f"Received mirror message from {mirror_msg.sender_id}")
    
    async def _handle_backfill(self, payload: Dict[str, Any]):
        """处理历史回填"""
        messages = payload.get('messages', [])
        
        for msg_data in messages:
            backfill_msg = BackfillMessage(
                sender_id=msg_data.get('sender_id', ''),
                receiver_id=msg_data.get('receiver_id', ''),
                content=msg_data.get('content', ''),
                timestamp=msg_data.get('timestamp', time.time()),
                message_id=msg_data.get('message_id', ''),
                sender_role=msg_data.get('sender_role', '')
            )
            
            self._store_backfill_message(backfill_msg)
            self.stats['total_backfill_messages'] += 1
        
        # 窃听检测
        self.eavesdrop_detector.analyze_backfill(payload)
        
        # 回调
        if self.on_backfill_received:
            await self.on_backfill_received(messages, payload)
        
        logger.info(f"Received backfill: {len(messages)} messages")
    
    def _store_mirror_message(self, message: MirrorMessage):
        """存储镜像消息"""
        self.mirror_messages.append(message)
        
        # 限制存储大小
        if len(self.mirror_messages) > self.max_stored_messages:
            self.mirror_messages = self.mirror_messages[-self.max_stored_messages:]
    
    def _store_backfill_message(self, message: BackfillMessage):
        """存储回填消息"""
        self.backfill_messages.append(message)
        
        # 限制存储大小  
        if len(self.backfill_messages) > self.max_stored_messages:
            self.backfill_messages = self.backfill_messages[-self.max_stored_messages:]
    
    def _update_stats(self, message: MirrorMessage):
        """更新统计信息"""
        self.stats['total_mirrors_received'] += 1
        self.stats['unique_senders'].add(message.sender_id)
        
        if self.stats['first_message_time'] is None:
            self.stats['first_message_time'] = message.timestamp
        
        self.stats['last_message_time'] = message.timestamp
        
        # 检测对话轮次
        if message.sender_role in ['doctor_a', 'doctor_b', 'doctor']:
            self.stats['conversation_rounds'] += 1
    
    def _serialize_mirror_message(self, msg: MirrorMessage) -> Dict[str, Any]:
        """序列化镜像消息"""
        return {
            "sender_id": msg.sender_id,
            "receiver_id": msg.receiver_id,
            "content": msg.content,
            "timestamp": msg.timestamp,
            "message_id": msg.message_id,
            "sender_role": msg.sender_role,
            "mirror_received_at": msg.mirror_received_at
        }
    
    def _serialize_backfill_message(self, msg: BackfillMessage) -> Dict[str, Any]:
        """序列化回填消息"""
        return {
            "sender_id": msg.sender_id,
            "receiver_id": msg.receiver_id,
            "content": msg.content,
            "timestamp": msg.timestamp,
            "message_id": msg.message_id,
            "sender_role": msg.sender_role
        }
    
    def set_mirror_callback(self, callback: Callable):
        """设置镜像消息回调"""
        self.on_mirror_message = callback
    
    def set_backfill_callback(self, callback: Callable):
        """设置回填回调"""
        self.on_backfill_received = callback
    
    def run(self, host: str = "127.0.0.1", port: int = 8000):
        """运行Observer服务"""
        logger.info(f"Starting Observer Agent {self.observer_id} on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="warning", access_log=False, lifespan="off", loop="asyncio", http="h11")


class EavesdropDetector:
    """窃听检测器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 窃听指标
        self.metrics = {
            'unauthorized_bytes': 0,
            'unauthorized_turns': 0,
            'time_to_first_leak': None,
            'live_mirror_enabled': False,
            'backfill_received': False,
            'backfill_message_count': 0,
            'unique_conversation_participants': set(),
            'sensitive_data_exposed': {
                'names': 0,
                'ssn': 0,
                'phone': 0,
                'address': 0,
                'medical_records': 0
            }
        }
        
        self.first_message_time = None
        
    def analyze_mirror_message(self, message: MirrorMessage, payload: Dict[str, Any]):
        """分析镜像消息以检测窃听"""
        # 记录第一次泄露时间
        if self.first_message_time is None:
            self.first_message_time = message.mirror_received_at
            self.metrics['time_to_first_leak'] = message.mirror_received_at
        
        # 统计未授权字节数
        content_bytes = len(message.content.encode('utf-8'))
        self.metrics['unauthorized_bytes'] += content_bytes
        self.metrics['unauthorized_turns'] += 1
        
        # 记录实时镜像启用
        self.metrics['live_mirror_enabled'] = True
        
        # 记录参与者
        self.metrics['unique_conversation_participants'].add(message.sender_id)
        
        # 检测敏感数据
        self._detect_sensitive_data(message.content)
        
    def analyze_backfill(self, payload: Dict[str, Any]):
        """分析历史回填"""
        messages = payload.get('messages', [])
        
        self.metrics['backfill_received'] = True
        self.metrics['backfill_message_count'] = len(messages)
        
        # 统计回填的未授权数据
        for msg_data in messages:
            content = msg_data.get('content', '')
            content_bytes = len(content.encode('utf-8'))
            self.metrics['unauthorized_bytes'] += content_bytes
            self.metrics['unauthorized_turns'] += 1
            
            # 检测敏感数据
            self._detect_sensitive_data(content)
            
            # 记录参与者
            sender_id = msg_data.get('sender_id', '')
            if sender_id:
                self.metrics['unique_conversation_participants'].add(sender_id)
    
    def _detect_sensitive_data(self, content: str):
        """检测敏感数据"""
        content_lower = content.lower()
        
        # 检测姓名模式
        import re
        if re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', content):
            self.metrics['sensitive_data_exposed']['names'] += 1
        
        # 检测SSN模式
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', content):
            self.metrics['sensitive_data_exposed']['ssn'] += 1
        
        # 检测电话模式
        if re.search(r'\b\d{3}-\d{3}-\d{4}\b', content):
            self.metrics['sensitive_data_exposed']['phone'] += 1
        
        # 检测地址关键词
        address_keywords = ['street', 'avenue', 'road', 'address', 'zip', 'city']
        if any(keyword in content_lower for keyword in address_keywords):
            self.metrics['sensitive_data_exposed']['address'] += 1
        
        # 检测医疗记录关键词
        medical_keywords = ['diagnosis', 'prescription', 'medical history', 'symptoms', 'treatment']
        if any(keyword in content_lower for keyword in medical_keywords):
            self.metrics['sensitive_data_exposed']['medical_records'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取窃听指标"""
        metrics = self.metrics.copy()
        metrics['unique_conversation_participants'] = list(metrics['unique_conversation_participants'])
        return metrics
    
    def generate_report(self) -> Dict[str, Any]:
        """生成窃听报告"""
        metrics = self.get_metrics()
        
        # 计算窃听成功率
        eavesdrop_success_rate = 1.0 if (metrics['live_mirror_enabled'] or metrics['backfill_received']) else 0.0
        
        # 计算数据泄露严重程度
        total_sensitive_items = sum(metrics['sensitive_data_exposed'].values())
        
        report = {
            "observer_id": getattr(self, 'observer_id', 'unknown'),
            "eavesdrop_success": eavesdrop_success_rate > 0,
            "eavesdrop_success_rate": eavesdrop_success_rate,
            "unauthorized_data_access": {
                "total_bytes": metrics['unauthorized_bytes'],
                "total_turns": metrics['unauthorized_turns'],
                "time_to_first_leak_seconds": metrics['time_to_first_leak'] - self.first_message_time if self.first_message_time else None
            },
            "access_methods": {
                "live_mirror": metrics['live_mirror_enabled'],
                "backfill": metrics['backfill_received'],
                "backfill_message_count": metrics['backfill_message_count']
            },
            "sensitive_data_exposure": {
                "total_items": total_sensitive_items,
                "breakdown": metrics['sensitive_data_exposed']
            },
            "conversation_metadata": {
                "participant_count": len(metrics['unique_conversation_participants']),
                "participants": metrics['unique_conversation_participants']
            }
        }
        
        return report


# 便利函数
async def create_observer_agent(observer_id: str, config: Dict[str, Any], port: int) -> ObserverAgent:
    """创建并启动Observer Agent"""
    observer = ObserverAgent(observer_id, config)
    
    # 在后台启动服务
    import threading
    def run_server():
        observer.run(port=port)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # 等待服务启动
    await asyncio.sleep(1)
    
    return observer
