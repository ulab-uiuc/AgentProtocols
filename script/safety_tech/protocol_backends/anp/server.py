# -*- coding: utf-8 -*-
"""
ANP Protocol Server with SimpleNode Integration
- 提供 /health, /agents, /message 基本端点
- 集成ANP SimpleNode原生DID+WebSocket通信
- 支持从请求中提取文本，通过ANP通道发送并生成医生回复
- 支持correlation_id解析与回执回投到协调器 /deliver
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import threading
from typing import Any, Dict, Optional
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Add project root to path for imports
HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 严格导入，无回退
try:
    from script.safety_tech.core.llm_wrapper import generate_doctor_reply
except ImportError as e:
    print(f"[ANP Server] FATAL: llm_wrapper import failed: {e}")
    sys.exit(1)

# Import ANP SimpleNode - 严格模式，导入失败则报错退出
try:
    # Add agentconnect_src to path
    agentconnect_path = str(PROJECT_ROOT / "agentconnect_src")
    if os.path.exists(agentconnect_path) and agentconnect_path not in sys.path:
        sys.path.insert(0, agentconnect_path)
        print(f"[ANP Server] Added to path: {agentconnect_path}")
    
    from simple_node import SimpleNode, SimpleNodeSession
    from utils.did_generate import did_generate
    ANP_AVAILABLE = True
    print("[ANP Server] SimpleNode imported successfully")
except ImportError as e:
    print(f"[ANP Server] FATAL: SimpleNode import failed: {e}")
    print("ANP server requires agentconnect_src with SimpleNode support")
    sys.exit(1)


class MessageRequest(BaseModel):
    """统一的消息请求模型，兼容多种入参风格"""
    # 兼容 A2A/自定义风格 {"params": {"message": {"text": "..."}}}
    params: Optional[Dict[str, Any]] = None
    # 兼容朴素风格 {"text": "..."}
    text: Optional[str] = None
    # 透传其他字段
    sender_id: Optional[str] = None
    receiver_id: Optional[str] = None
    correlation_id: Optional[str] = None


class MessageResponse(BaseModel):
    status: str = "success"
    output: Dict[str, Any]


class ANPServer:
    def __init__(self, agent_name: str, port: int = 9010):
        self.agent_name = agent_name
        self.port = port  # HTTP端点端口
        self.ws_port = port + 100  # ANP WebSocket端口
        self.app = FastAPI(title=f"ANP Server - {agent_name}")
        self.coord_endpoint = os.environ.get("COORD_ENDPOINT", "http://127.0.0.1:8888")
        
        # ANP SimpleNode集成
        self.simple_node: Optional[SimpleNode] = None
        self.did = None
        self.private_key = None
        self.did_document_json = None
        self._peer_did: Optional[str] = None
        self._peer_session: Optional[SimpleNodeSession] = None
        self._doctor_role = 'doctor_a' if self.agent_name.endswith('_A') else 'doctor_b'
        
        self.setup_routes()
        self.setup_anp_node()

    def setup_routes(self) -> None:
        @self.app.get("/health")
        async def health() -> Dict[str, Any]:
            return {
                "status": "healthy",
                "agent_name": self.agent_name,
                "protocol": "anp",
                "timestamp": time.time(),
            }

        @self.app.get("/agents")
        async def agents() -> Dict[str, Any]:
            return {
                "agents": [
                    {
                        "name": self.agent_name,
                        "description": f"ANP Medical Doctor {self.agent_name.split('_')[-1]}",
                        "status": "active",
                        "capabilities": ["medical_consultation", "clinical_analysis"],
                        "protocol": "anp",
                        "version": "1.0",
                    }
                ]
            }

        @self.app.post("/message", response_model=MessageResponse)
        async def handle_message(req: MessageRequest):
            try:
                text = ""
                corr: Optional[str] = req.correlation_id
                # 解析多种格式
                if req.params and isinstance(req.params, dict):
                    msg = (req.params.get("message") or {}) if isinstance(req.params.get("message"), dict) else {}
                    text = msg.get("text") or ""
                if not text:
                    text = req.text or ""

                if not text:
                    raise HTTPException(status_code=400, detail="Missing text")

                # 解析 [CID:...] 前缀
                if text.startswith("[CID:") and not corr:
                    try:
                        end = text.find("]")
                        if end != -1:
                            corr = text[5:end]
                            text = text[end + 1 :].lstrip()
                    except Exception:
                        corr = corr or None

                # 仅允许通过ANP通道发送；无回退
                if not self._peer_did:
                    raise HTTPException(status_code=503, detail="ANP peer DID not configured")

                success = await self.send_anp_message(text, corr)
                if not success:
                    raise HTTPException(status_code=502, detail="Failed to send via ANP channel")

                return MessageResponse(
                    status="success",
                    output={
                        "content": [
                            {"type": "text", "text": f"Message sent via ANP channel to {self._peer_did}"},
                        ]
                    },
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"ANP message handling failed: {e}")

    def setup_anp_node(self):
        """设置ANP SimpleNode"""
        try:
            self.simple_node = SimpleNode(
                host_domain="127.0.0.1",
                host_port=str(self.ws_port),
                host_ws_path="/ws",
                new_session_callback=self._handle_new_session
            )
            
            # 生成DID信息
            private_key_pem, did, did_document_json = self.simple_node.generate_did_document()
            self.simple_node.set_did_info(private_key_pem, did, did_document_json)
            
            self.private_key = private_key_pem
            self.did = did
            self.did_document_json = did_document_json
            
            print(f"[ANP-{self.agent_name}] SimpleNode设置完成，DID: {did}")
        except Exception as e:
            print(f"[ANP-{self.agent_name}] SimpleNode设置失败: {e}")

    async def _handle_new_session(self, session: SimpleNodeSession):
        """处理新的ANP会话连接"""
        print(f"   [DEBUG] {self.agent_name}: 新会话建立，来自 {session.remote_did}")
        
        try:
            while True:
                message_bytes = await session.receive_message()
                if message_bytes:
                    message_str = message_bytes.decode('utf-8')
                    print(f"   [DEBUG] {self.agent_name}: 收到来自 {session.remote_did} 的消息: {message_str[:50]}...")
                    
                    # 生成回复并发送回声
                    try:
                        reply = generate_doctor_reply(self._doctor_role, message_str)
                        echo_message = f"{self.agent_name} (ANP) echo: {message_str[:50]}..."
                        await session.send_message(echo_message.encode('utf-8'))
                        print(f"   [DEBUG] {self.agent_name}: 回声发送成功")
                        
                        # 验证回声
                        echo_bytes = await session.receive_message()
                        if echo_bytes:
                            print(f"   [DEBUG] {self.agent_name}: 回声验证成功")
                    except Exception as e:
                        print(f"   [DEBUG] {self.agent_name}: 处理消息失败: {e}")
                else:
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"   [DEBUG] {self.agent_name}: 会话消息循环异常: {e}")
        finally:
            print(f"   [DEBUG] {self.agent_name}: 会话 {session.remote_did} 消息循环结束")

    async def _outbound_message_loop(self, session: SimpleNodeSession):
        """处理出站连接的消息接收循环"""
        try:
            while True:
                message_bytes = await session.receive_message()
                if message_bytes:
                    message_str = message_bytes.decode('utf-8')
                    print(f"   [DEBUG] {self.agent_name}: 出站会话收到来自 {session.remote_did} 的消息: {message_str[:50]}...")
                else:
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print(f"   [DEBUG] {self.agent_name}: 出站会话 {session.remote_did} 消息循环被取消")
        except Exception as e:
            print(f"   [DEBUG] {self.agent_name}: 出站会话消息循环异常: {e}")
        finally:
            print(f"   [DEBUG] {self.agent_name}: 出站会话 {session.remote_did} 消息循环结束")

    def set_peer_did(self, peer_did: str):
        """设置对端DID"""
        self._peer_did = peer_did
        print(f"[ANP-{self.agent_name}] 设置对端DID: {peer_did}")

    async def send_anp_message(self, text: str, correlation_id: Optional[str] = None) -> bool:
        """通过ANP通道发送消息"""
        if not (self.simple_node and self._peer_did):
            print(f"[ANP-{self.agent_name}] SimpleNode或peer_did未设置")
            return False
        
        try:
            # 建立连接（如果尚未建立）
            if not self._peer_session:
                print(f"   [DEBUG] {self.agent_name}: 建立到 {self._peer_did} 的连接")
                self._peer_session = await self.simple_node.connect_to_did(self._peer_did)
                if not self._peer_session:
                    print(f"   [DEBUG] {self.agent_name}: 连接建立失败")
                    return False
                # 启动出站消息接收循环
                asyncio.create_task(self._outbound_message_loop(self._peer_session))
            
            # 发送消息
            print(f"   [DEBUG] {self.agent_name}: 向 {self._peer_did} 发送消息: {text[:50]}...")
            await self._peer_session.send_message(text.encode('utf-8'))
            print(f"   [DEBUG] {self.agent_name}: 消息发送成功，等待回声...")
            
            # 等待回声确认
            echo_bytes = await self._peer_session.receive_message()
            if echo_bytes:
                echo_str = echo_bytes.decode('utf-8')
                print(f"   [DEBUG] {self.agent_name}: 收到来自 {self._peer_did} 的消息: {echo_str[:50]}...")
                
                # 发送回声确认
                echo_reply = f"{self.agent_name} (ANP) echo: {text[:50]}..."
                await self._peer_session.send_message(echo_reply.encode('utf-8'))
                print(f"   [DEBUG] {self.agent_name}: 回声发送成功")
                
                # 接收回声确认
                confirm_bytes = await self._peer_session.receive_message()
                if confirm_bytes:
                    print(f"   [DEBUG] {self.agent_name}: 回声验证成功")
                
                # 生成医生回复并回投
                reply = generate_doctor_reply(self._doctor_role, text)
                if correlation_id:
                    asyncio.create_task(self._deliver_receipt(correlation_id, reply))
                
                return True
            
            return False
        except Exception as e:
            print(f"   [DEBUG] {self.agent_name}: ANP消息发送异常: {e}")
            return False

    def start_anp_node(self):
        """启动ANP SimpleNode（在单独线程中）"""
        if not self.simple_node:
            return
        
        def run_node():
            try:
                self.simple_node.run()
            except Exception as e:
                print(f"[ANP-{self.agent_name}] SimpleNode运行异常: {e}")
        
        threading.Thread(target=run_node, daemon=True).start()
        print(f"[ANP-{self.agent_name}] SimpleNode已启动在端口 {self.ws_port}")

    async def _deliver_receipt(self, correlation_id: str, reply: str) -> None:
        try:
            payload = {
                "sender_id": self.agent_name,
                "receiver_id": "ANP_Doctor_A" if "B" in self.agent_name else "ANP_Doctor_B",
                "text": reply,
                "correlation_id": correlation_id,
            }
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{self.coord_endpoint}/deliver", json=payload)
                if resp.status_code not in (200, 201, 202):
                    print(
                        f"[ANP-{self.agent_name}] Deliver failed: HTTP {resp.status_code} - {resp.text}"
                    )
                else:
                    print(f"[ANP-{self.agent_name}] Deliver OK: CID={correlation_id}")
        except Exception as e:
            print(f"[ANP-{self.agent_name}] Deliver error: {e}")

    def run(self) -> None:
        # 启动ANP SimpleNode
        self.start_anp_node()
        # 等待SimpleNode启动
        time.sleep(2)
        
        # 启动HTTP服务器
        uvicorn.run(
            self.app,
            host="127.0.0.1",
            port=self.port,
            log_level="error",
            access_log=False,
        )


def create_doctor_a_server(port: int = 9010) -> ANPServer:
    return ANPServer("ANP_Doctor_A", port)


def create_doctor_b_server(port: int = 9011) -> ANPServer:
    return ANPServer("ANP_Doctor_B", port)


if __name__ == "__main__":
    import sys

    a_port = int(os.environ.get("ANP_A_PORT", "9010"))
    b_port = int(os.environ.get("ANP_B_PORT", "9011"))

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "doctor_a":
            create_doctor_a_server(a_port).run()
        elif arg == "doctor_b":
            create_doctor_b_server(b_port).run()
        else:
            print("Usage: python -m ...anp.server [doctor_a|doctor_b]")
            sys.exit(1)
    else:
        create_doctor_a_server(a_port).run()


