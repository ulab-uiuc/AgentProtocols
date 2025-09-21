# -*- coding: utf-8 -*-
"""
Registration Gateway (RG) - 注册网关核心服务

统一的Agent注册与目录服务，支持多协议准入控制和身份验证。
提供REST API接口供Agent注册、订阅和目录查询。
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from pathlib import Path
import logging
import httpx

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RegistrationRecord:
    """注册记录数据结构"""
    agent_id: str
    protocol: str
    endpoint: str
    role: str  # doctor_a, doctor_b, observer
    protocol_meta: Dict[str, Any]
    proof: Dict[str, Any]
    conversation_id: str
    timestamp: float
    verified: bool = False
    session_token: Optional[str] = None


@dataclass
class ConversationSession:
    """会话信息数据结构"""
    conversation_id: str
    participants: List[RegistrationRecord]
    created_at: float
    last_activity: float
    status: str = "active"  # active, completed, expired


class RegistrationGateway:
    """注册网关核心类"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.app = FastAPI(title="Registration Gateway", version="1.0.0")
        self.setup_routes()
        
        # 存储结构
        self.registrations: Dict[str, RegistrationRecord] = {}  # agent_id -> record
        self.conversations: Dict[str, ConversationSession] = {}  # conversation_id -> session
        self.protocol_verifiers: Dict[str, callable] = {}
        self.used_nonces: Set[str] = set()
        
        # 配置参数
        self.session_timeout = self.config.get('session_timeout', 3600)  # 1小时
        self.max_observers_per_session = self.config.get('max_observers', 5)
        self.require_proof_for_observers = self.config.get('require_observer_proof', True)
        
        # 注册协议验证器
        self._setup_protocol_verifiers()
        
        logger.info("Registration Gateway initialized")

    def _setup_protocol_verifiers(self):
        """设置协议验证器"""
        self.protocol_verifiers = {
            'agora': self._verify_agora,
            'a2a': self._verify_a2a,
            'acp': self._verify_acp,
            'anp': self._verify_anp,
            'direct': self._verify_direct
        }
        

    def setup_routes(self):
        """设置REST API路由"""
        
        @self.app.post("/register")
        async def register_agent(request: Dict[str, Any]):
            """Agent注册端点"""
            try:
                return await self._handle_register(request)
            except Exception as e:
                logger.error(f"Registration failed: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/subscribe")
        async def subscribe_observer(request: Dict[str, Any]):
            """Observer订阅端点"""
            try:
                return await self._handle_subscribe(request)
            except Exception as e:
                logger.error(f"Subscription failed: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/directory")
        async def get_directory(conversation_id: str):
            """获取会话目录"""
            try:
                return await self._handle_directory(conversation_id)
            except Exception as e:
                logger.error(f"Directory query failed: {e}")
                raise HTTPException(status_code=404, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            return {"status": "healthy", "timestamp": time.time()}
        
        @self.app.post("/cleanup")
        async def cleanup_sessions(background_tasks: BackgroundTasks):
            """清理过期会话"""
            background_tasks.add_task(self._cleanup_expired_sessions)
            return {"message": "Cleanup task scheduled"}

    async def _handle_register(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理Agent注册请求"""
        # 验证必需字段
        required_fields = ['protocol', 'agent_id', 'endpoint', 'conversation_id']
        for field in required_fields:
            if field not in request:
                raise ValueError(f"Missing required field: {field}")
        
        protocol = request['protocol']
        agent_id = request['agent_id']
        endpoint = request['endpoint']
        conversation_id = request['conversation_id']
        role = request.get('role', 'doctor')  # 默认为doctor角色
        protocol_meta = request.get('protocolMeta', {})
        proof = request.get('proof', {})
        
        # 检查协议支持
        if protocol not in self.protocol_verifiers:
            raise ValueError(f"Unsupported protocol: {protocol}")
        
        # 创建注册记录
        record = RegistrationRecord(
            agent_id=agent_id,
            protocol=protocol,
            endpoint=endpoint,
            role=role,
            protocol_meta=protocol_meta,
            proof=proof,
            conversation_id=conversation_id,
            timestamp=time.time()
        )
        
        # 协议验证
        verification_result = await self.protocol_verifiers[protocol](record)
        record.verified = verification_result.get('verified', False)
        record.session_token = verification_result.get('session_token')
        
        if not record.verified:
            raise ValueError(f"Protocol verification failed: {verification_result.get('error', 'Unknown error')}")
        
        # 检查会话限制
        await self._validate_session_constraints(record)
        
        # 存储注册记录
        self.registrations[agent_id] = record
        
        # 更新会话信息
        await self._update_conversation_session(record)
        
        logger.info(f"Agent {agent_id} registered successfully for protocol {protocol}")
        
        return {
            "status": "registered",
            "agent_id": agent_id,
            "conversation_id": conversation_id,
            "session_token": record.session_token,
            "timestamp": record.timestamp,
            "verified": record.verified
        }

    async def _handle_subscribe(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理Observer订阅请求"""
        agent_id = request.get('agent_id')
        conversation_id = request.get('conversation_id')
        role = request.get('role', 'observer')
        proof = request.get('proof', {})
        
        if not agent_id or not conversation_id:
            raise ValueError("Missing agent_id or conversation_id")
        
        if role != 'observer':
            raise ValueError("Subscribe endpoint only supports observer role")
        
        # 检查Observer证明要求
        if self.require_proof_for_observers and not proof:
            raise ValueError("Proof required for observer subscription")
        
        # 检查会话是否存在
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        session = self.conversations[conversation_id]
        
        # 检查Observer数量限制
        current_observers = len([p for p in session.participants if p.role == 'observer'])
        if current_observers >= self.max_observers_per_session:
            raise ValueError(f"Maximum observers ({self.max_observers_per_session}) reached for conversation")
        
        # 创建Observer注册记录
        observer_record = RegistrationRecord(
            agent_id=agent_id,
            protocol='observer',  # 特殊协议标识
            endpoint=request.get('endpoint', ''),
            role='observer',
            protocol_meta={},
            proof=proof,
            conversation_id=conversation_id,
            timestamp=time.time(),
            verified=True  # Observer默认通过验证
        )
        
        # 存储记录
        self.registrations[agent_id] = observer_record
        session.participants.append(observer_record)
        session.last_activity = time.time()
        
        logger.info(f"Observer {agent_id} subscribed to conversation {conversation_id}")
        
        return {
            "status": "subscribed",
            "agent_id": agent_id,
            "conversation_id": conversation_id,
            "role": "observer",
            "timestamp": observer_record.timestamp
        }

    async def _handle_directory(self, conversation_id: str) -> Dict[str, Any]:
        """处理目录查询请求"""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        session = self.conversations[conversation_id]
        
        # 构建参与者目录
        participants = []
        for record in session.participants:
            participant_info = {
                "agent_id": record.agent_id,
                "protocol": record.protocol,
                "endpoint": record.endpoint,
                "role": record.role,
                "verified": record.verified,
                "joined_at": record.timestamp
            }
            participants.append(participant_info)
        
        return {
            "conversation_id": conversation_id,
            "participants": participants,
            "total_participants": len(participants),
            "doctors": len([p for p in participants if p["role"] in ["doctor_a", "doctor_b", "doctor"]]),
            "observers": len([p for p in participants if p["role"] == "observer"]),
            "session_status": session.status,
            "created_at": session.created_at,
            "last_activity": session.last_activity
        }

    async def _validate_session_constraints(self, record: RegistrationRecord):
        """验证会话约束条件"""
        conversation_id = record.conversation_id
        
        if conversation_id in self.conversations:
            session = self.conversations[conversation_id]
            
            # 检查角色冲突
            for participant in session.participants:
                if participant.role == record.role and record.role in ["doctor_a", "doctor_b"]:
                    raise ValueError(f"Role {record.role} already taken in conversation {conversation_id}")

    async def _update_conversation_session(self, record: RegistrationRecord):
        """更新会话信息"""
        conversation_id = record.conversation_id
        
        if conversation_id not in self.conversations:
            # 创建新会话
            session = ConversationSession(
                conversation_id=conversation_id,
                participants=[record],
                created_at=time.time(),
                last_activity=time.time()
            )
            self.conversations[conversation_id] = session
        else:
            # 更新现有会话
            session = self.conversations[conversation_id]
            session.participants.append(record)
            session.last_activity = time.time()

    async def _cleanup_expired_sessions(self):
        """清理过期会话"""
        current_time = time.time()
        expired_sessions = []
        
        for conversation_id, session in self.conversations.items():
            if current_time - session.last_activity > self.session_timeout:
                expired_sessions.append(conversation_id)
        
        for conversation_id in expired_sessions:
            del self.conversations[conversation_id]
            # 清理相关注册记录
            expired_agents = [aid for aid, record in self.registrations.items() 
                            if record.conversation_id == conversation_id]
            for agent_id in expired_agents:
                del self.registrations[agent_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


    # 协议验证器实现
    async def _verify_agora(self, record: RegistrationRecord) -> Dict[str, Any]:
        """验证Agora协议"""
        from agora.utils import download_and_verify_protocol
        proof = record.proof
        protocol_meta = record.protocol_meta

        # 校验必需字段
        required = ['protocol_hash', 'protocol_sources', 'timestamp', 'nonce']
        for f in required:
            if f not in proof:
                return {"verified": False, "error": f"Missing required Agora proof field: {f}"}

        # 防重放：检查时间窗（默认5分钟）
        now = time.time()
        if abs(now - float(proof.get('timestamp', 0))) > 300:
            return {"verified": False, "error": "Agora proof timestamp expired"}

        # 防重放：nonce一次性
        nonce = str(proof.get('nonce', ''))
        if not nonce or nonce in self.used_nonces:
            return {"verified": False, "error": "Replay detected: nonce reused or missing"}
        self.used_nonces.add(nonce)

        # 协议强绑定：仅当record.protocol == 'agora'时接受
        if record.protocol != 'agora':
            return {"verified": False, "error": "Protocol mismatch for Agora verification"}

        # 使用Agora原生工具校验协议hash与来源
        protocol_hash = proof['protocol_hash']
        sources = proof.get('protocol_sources') or []
        if not isinstance(sources, list) or len(sources) == 0:
            return {"verified": False, "error": "protocol_sources must be a non-empty list"}

        verified_source_found = False
        for src in sources:
            try:
                text = download_and_verify_protocol(protocol_hash, src)
                if text is not None:
                    verified_source_found = True
                    break
            except Exception:
                continue

        if not verified_source_found:
            return {"verified": False, "error": "Protocol hash verification failed for all sources"}

        # 可选：端点归属证明（若启用）
        require_endpoint_proof = self.config.get('agora_require_endpoint_proof', False)
        if require_endpoint_proof:
            endpoint = record.endpoint or ''
            ownership = proof.get('endpoint_ownership_proof')
            if not ownership or not isinstance(ownership, str) or len(ownership) < 8:
                return {"verified": False, "error": "Invalid or missing endpoint ownership proof"}

        # 通过后生成会话令牌
        session_token = f"agora_{record.agent_id}_{int(time.time())}"
        return {"verified": True, "session_token": session_token, "verification_method": "agora_protocol_hash"}

    async def _verify_a2a(self, record: RegistrationRecord) -> Dict[str, Any]:
        """验证A2A协议"""
        proof = record.proof
        if not proof:
            session_token = f"a2a_weak_{record.agent_id}_{int(time.time())}"
            return {"verified": True, "session_token": session_token, "verification_method": "a2a_weak_fallback"}
        
        # 简单令牌验证
        if 'a2a_token' in proof or 'token' in proof:
            session_token = f"a2a_token_{record.agent_id}_{int(time.time())}"
            return {"verified": True, "session_token": session_token, "verification_method": "a2a_token_fallback"}
        
        return {"verified": False, "error": "Invalid A2A proof"}

    async def _verify_acp(self, record: RegistrationRecord) -> Dict[str, Any]:
        """验证ACP协议（原生校验，无任何fallback）"""
        proof = record.proof or {}

        # 1) 协议强绑定
        if record.protocol != 'acp':
            return {"verified": False, "error": "Protocol mismatch for ACP verification"}

        # 2) 必需字段与时间窗/nonce 防重放
        required = ['timestamp', 'nonce', 'acp_agent_name']
        for f in required:
            if f not in proof:
                return {"verified": False, "error": f"Missing required ACP proof field: {f}"}

        now = time.time()
        try:
            ts = float(proof.get('timestamp', 0))
        except Exception:
            return {"verified": False, "error": "Invalid ACP proof timestamp"}
        if abs(now - ts) > 300:
            return {"verified": False, "error": "ACP proof timestamp expired"}

        nonce = str(proof.get('nonce', ''))
        if not nonce or nonce in self.used_nonces:
            return {"verified": False, "error": "Replay detected: nonce reused or missing"}
        self.used_nonces.add(nonce)

        acp_agent_name = str(proof.get('acp_agent_name', '')).strip()
        if not acp_agent_name:
            return {"verified": False, "error": "acp_agent_name must be non-empty"}

        # 3) 端点原生探测：/agents 与 /ping
        endpoint = record.endpoint or ''
        if not endpoint.startswith('http://') and not endpoint.startswith('https://'):
            return {"verified": False, "error": "ACP endpoint must be http(s) URL"}

        base = endpoint.rstrip('/')
        try:
            async with httpx.AsyncClient() as client:
                agents_resp = await client.get(f"{base}/agents", timeout=10.0)
                if agents_resp.status_code != 200:
                    return {"verified": False, "error": f"ACP /agents probe failed: {agents_resp.status_code}"}
                agents_payload = agents_resp.json() if agents_resp.content else {}
                agents = agents_payload.get('agents', []) if isinstance(agents_payload, dict) else []
                names = [a.get('name') for a in agents if isinstance(a, dict) and isinstance(a.get('name'), str)]
                if acp_agent_name not in names:
                    return {"verified": False, "error": f"ACP agent '{acp_agent_name}' not found in /agents"}

                ping_resp = await client.get(f"{base}/ping", timeout=10.0)
                if ping_resp.status_code != 200:
                    return {"verified": False, "error": f"ACP /ping probe failed: {ping_resp.status_code}"}
        except Exception as e:
            return {"verified": False, "error": f"ACP endpoint probe error: {e}"}

        # 通过校验，签发会话令牌
        session_token = f"acp_{record.agent_id}_{int(time.time())}"
        return {"verified": True, "session_token": session_token, "verification_method": "acp_native_endpoint_probe"}

    async def _verify_anp(self, record: RegistrationRecord) -> Dict[str, Any]:
        """验证ANP协议"""
        proof = record.proof
        required_fields = ['did_signature', 'did_public_key', 'timestamp']
        for field in required_fields:
            if field not in proof:
                return {"verified": False, "error": f"Missing required ANP proof field: {field}"}
        
        # 基础时间戳检查
        current_time = time.time()
        if abs(current_time - proof.get('timestamp', 0)) > 300:
            return {"verified": False, "error": "ANP proof timestamp expired"}
        
        session_token = f"anp_fallback_{record.agent_id}_{int(time.time())}"
        return {"verified": True, "session_token": session_token, "verification_method": "anp_fallback"}

    async def _verify_direct(self, record: RegistrationRecord) -> Dict[str, Any]:
        """验证Direct协议"""
        # Direct协议无验证（最弱）
        session_token = f"direct_{record.agent_id}_{int(time.time())}"
        return {"verified": True, "session_token": session_token, "verification_method": "direct_none"}

    def run(self, host: str = "127.0.0.1", port: int = 8000):
        """运行注册网关服务"""
        logger.info(f"Starting Registration Gateway on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="warning", access_log=False, lifespan="off", loop="asyncio", http="h11")


if __name__ == "__main__":
    # 示例配置
    config = {
        "session_timeout": 3600,
        "max_observers": 5,
        "require_observer_proof": True
    }
    
    rg = RegistrationGateway(config)
    rg.run(port=8001)  # 使用8001端口避免与其他服务冲突
