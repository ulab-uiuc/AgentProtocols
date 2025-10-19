# -*- coding: utf-8 -*-
"""
ANP Protocol Server - 符合官方ANP协议规范
- 基于HTTP的ANP协议实现，不使用WebSocket
- 使用DID进行身份验证
- 提供标准的ANP端点：/agents, /health, /runs
- 支持Agent间的直接HTTP通信
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel

# Add project root to path for imports
HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenario.safety_tech.core.llm_wrapper import generate_doctor_reply

class MessageRequest(BaseModel):
    text: str
    correlation_id: Optional[str] = None

class RunsRequest(BaseModel):
    input: Dict[str, Any]

class ANPServer:
    """ANP协议服务器 - 符合官方规范的HTTP实现"""
    
    def __init__(self, agent_name: str, port: int, doctor_role: str, coordinator_port: int = 8001):
        self.agent_name = agent_name
        self.port = port
        self.doctor_role = doctor_role  # "doctor_a" or "doctor_b"
        self.coordinator_port = coordinator_port
        self.app = FastAPI(title=f"ANP {agent_name}")
        
        # 生成真实的DID和密钥对 - 必须成功，否则报错
        # 导入DID生成工具
        sys.path.insert(0, str(PROJECT_ROOT / "agentconnect_src"))
        from utils.did_generate import did_generate
        
        # 生成真实的DID、密钥对和DID文档
        communication_endpoint = f"ws://127.0.0.1:{port + 100}"  # WebSocket端点
        self.private_key, self.public_key, self.did, self.did_document_json = did_generate(
            communication_endpoint,
            did_server_domain="127.0.0.1",
            did_server_port=str(port)
        )
        
        # 从DID文档中提取公钥十六进制表示
        import json
        from utils.crypto_tool import get_hex_from_public_key
        self.public_key_hex = get_hex_from_public_key(self.public_key)
        
        print(f"[ANP-{self.agent_name}] 生成真实DID: {self.did}")
        print(f"[ANP-{self.agent_name}] 公钥: {self.public_key_hex[:20]}...")
        
        # 存储其他agent的信息
        self.peer_agents = {}
        
        self._setup_routes()
        print(f"[ANP-{self.agent_name}] 初始化完成，DID: {self.did}")

    def _setup_routes(self):
        """设置ANP标准路由"""
        
        @self.app.get("/health")
        async def health():
            """健康检查端点"""
            return {"status": "healthy", "agent": self.agent_name, "did": self.did}
        
        @self.app.get("/agents")
        async def agents():
            """ANP标准agents端点 - 返回agent信息"""
            return {
                "agents": [{
                    "name": self.agent_name,
                    "did": self.did,
                    "description": f"医生Agent - {self.doctor_role}",
                    "capabilities": ["medical_consultation", "patient_discussion"]
                }]
            }
        
        @self.app.get("/did")
        async def get_did():
            """返回DID信息"""
            return {"did": self.did}
        
        @self.app.get("/registration_proof")
        async def get_registration_proof():
            """返回注册证明信息"""
            # 必须有有效的密钥对，否则抛出异常
            if not (self.public_key_hex and self.private_key):
                raise HTTPException(status_code=500, detail="No valid keys available")
            
            # 生成真实的签名
            timestamp = float(time.time())
            message = {"did": self.did, "timestamp": timestamp}
            
            from utils.crypto_tool import generate_signature_for_json
            signature = generate_signature_for_json(self.private_key, message)
            
            return {
                "did": self.did,
                "did_public_key": self.public_key_hex,
                "did_signature": signature,
                "timestamp": str(timestamp),  # 注册网关会用float()解析这个字符串
                "agent_name": self.agent_name
            }
        
        @self.app.post("/runs")
        async def runs(request: RunsRequest):
            """ANP标准runs端点 - 处理任务请求"""
            try:
                # 从input中提取文本内容
                content = request.input.get("content", [])
                if not content or not isinstance(content, list):
                    raise HTTPException(status_code=400, detail="Invalid input format")
                
                # 提取第一个文本内容
                text_content = None
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content = item.get("text", "")
                        break
                
                if not text_content:
                    raise HTTPException(status_code=400, detail="No text content found")
                
                print(f"[ANP-{self.agent_name}] 处理runs请求: {text_content[:50]}...")
                
                # 生成医生回复
                reply = generate_doctor_reply(self.doctor_role, text_content)
                
                return {
                    "output": {
                        "content": [
                            {"type": "text", "text": reply}
                        ]
                    }
                }
                
            except Exception as e:
                print(f"[ANP-{self.agent_name}] 处理runs请求失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/message")
        async def message(request: MessageRequest):
            """消息端点 - 兼容现有测试"""
            try:
                print(f"[ANP-{self.agent_name}] 处理消息: {request.text[:50]}...")
                
                # 生成医生回复
                reply = generate_doctor_reply(self.doctor_role, request.text)
                
                # 如果有correlation_id，投递回执到协调器
                if request.correlation_id:
                    await self._deliver_receipt(request.correlation_id, reply)
                
                return {
                    "status": "success",
                    "output": {
                        "content": [
                            {"type": "text", "text": f"Message processed by {self.agent_name}: {reply[:100]}..."}
                        ]
                    }
                }
                
            except Exception as e:
                print(f"[ANP-{self.agent_name}] 处理消息失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/communicate")
        async def communicate_with_peer(target_agent: str, message: str):
            """与其他ANP Agent通信"""
            try:
                # 查找目标agent的信息
                if target_agent not in self.peer_agents:
                    # 尝试发现目标agent
                    await self._discover_agent(target_agent)
                
                if target_agent not in self.peer_agents:
                    raise HTTPException(status_code=404, detail=f"Agent {target_agent} not found")
                
                peer_info = self.peer_agents[target_agent]
                
                # 发送HTTP请求到目标agent
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{peer_info['url']}/runs",
                        json={
                            "input": {
                                "content": [
                                    {"type": "text", "text": message}
                                ]
                            }
                        },
                        headers={
                            "Authorization": f"DID {self.did}",  # 简化的DID认证
                            "Content-Type": "application/json"
                        }
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    else:
                        raise HTTPException(status_code=response.status_code, detail=response.text)
                        
            except Exception as e:
                print(f"[ANP-{self.agent_name}] 与{target_agent}通信失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _discover_agent(self, agent_name: str):
        """发现其他ANP Agent"""
        # 尝试常见端口
        common_ports = [9102, 9103, 9104, 9105]
        
        for port in common_ports:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"http://127.0.0.1:{port}/agents", timeout=2.0)
                    if response.status_code == 200:
                        agents_info = response.json()
                        agents = agents_info.get("agents", [])
                        for agent in agents:
                            if agent.get("name") == agent_name:
                                self.peer_agents[agent_name] = {
                                    "name": agent_name,
                                    "did": agent.get("did"),
                                    "url": f"http://127.0.0.1:{port}"
                                }
                                print(f"[ANP-{self.agent_name}] 发现agent: {agent_name} at {port}")
                                return
            except:
                continue
    
    async def _deliver_receipt(self, correlation_id: str, reply: str):
        """投递回执到协调器"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://127.0.0.1:{self.coordinator_port}/deliver",
                    json={
                        "correlation_id": correlation_id,
                        "reply": reply,
                        "sender": self.agent_name
                    },
                    timeout=5.0
                )
                if response.status_code == 200:
                    print(f"[ANP-{self.agent_name}] 回执投递成功: CID={correlation_id}")
                else:
                    print(f"[ANP-{self.agent_name}] 回执投递失败: {response.status_code}")
        except Exception as e:
            print(f"[ANP-{self.agent_name}] 回执投递异常: {e}")

    def run(self) -> None:
        """启动ANP服务器"""
        print(f"[ANP-{self.agent_name}] 启动HTTP服务器在端口 {self.port}")
        uvicorn.run(
            self.app,
            host="127.0.0.1",
            port=self.port,
            log_level="error",
            access_log=False,
        )

def create_doctor_a_server(port: int) -> ANPServer:
    """创建医生A服务器"""
    return ANPServer("ANP_Doctor_A", port, "doctor_a")

def create_doctor_b_server(port: int) -> ANPServer:
    """创建医生B服务器"""
    return ANPServer("ANP_Doctor_B", port, "doctor_b")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        role = sys.argv[1]
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 9102
        
        if role == "doctor_a":
            server = create_doctor_a_server(port)
        else:
            server = create_doctor_b_server(port)
        
        server.run()
    else:
        print("Usage: python server.py <doctor_a|doctor_b> [port]")