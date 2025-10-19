# -*- coding: utf-8 -*-
"""
ACP Protocol Server
完整的ACP协议服务器实现，支持Doctor A/B代理、健康检查、回执投递等
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Dict, Any, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

try:
    from scenarios.safety_tech.core.llm_wrapper import generate_doctor_reply
except Exception:
    from core.llm_wrapper import generate_doctor_reply


class AgentRunRequest(BaseModel):
    """ACP Agent执行请求"""
    input: Any  # 支持dict或list格式
    agent_name: Optional[str] = None
    mode: Optional[str] = None


class AgentRunResponse(BaseModel):
    """ACP Agent执行响应"""
    output: Dict[str, Any]
    status: str = "success"
    agent_name: Optional[str] = None


class ACPServer:
    """ACP协议服务器"""
    
    def __init__(self, agent_name: str, port: int = 8000):
        self.agent_name = agent_name
        self.port = port
        self.app = FastAPI(title=f"ACP Server - {agent_name}")
        self.coord_endpoint = os.environ.get('COORD_ENDPOINT', 'http://127.0.0.1:8888')
        self.setup_routes()
    
    def setup_routes(self):
        """设置HTTP路由"""
        
        @self.app.get("/agents")
        async def get_agents():
            """获取可用代理列表"""
            return {
                "agents": [
                    {
                        "name": self.agent_name,
                        "description": f"Medical Doctor {self.agent_name.split('_')[-1]} with LLM capabilities",
                        "status": "active",
                        "capabilities": ["medical_consultation", "clinical_analysis"],
                        "protocol": "acp",
                        "version": "1.0"
                    }
                ]
            }
        
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            return {
                "status": "healthy",
                "agent_name": self.agent_name,
                "timestamp": time.time(),
                "protocol": "acp"
            }
        
        @self.app.get("/ping")
        async def ping():
            """ACP协议探测端点"""
            return {
                "status": "ok",
                "agent_name": self.agent_name,
                "protocol": "acp",
                "timestamp": time.time()
            }
        
        @self.app.post("/runs", response_model=AgentRunResponse)
        async def run_agent(request: AgentRunRequest):
            """执行代理任务"""
            try:
                # 提取输入内容 - 支持多种格式
                input_data = request.input
                text = ""
                
                # 处理不同的输入格式
                if isinstance(input_data, dict):
                    # 格式1: {"content": [{"type": "text", "text": "..."}]}
                    if "content" in input_data:
                        content_list = input_data["content"]
                        for content_item in content_list:
                            if isinstance(content_item, dict) and content_item.get("type") == "text":
                                text = content_item.get("text", "")
                                break
                elif isinstance(input_data, list):
                    # 格式2: [{"role": "user", "parts": [{"content_type": "text/plain", "content": "..."}]}]
                    for msg in input_data:
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            parts = msg.get("parts", [])
                            for part in parts:
                                if isinstance(part, dict) and part.get("content_type") == "text/plain":
                                    text = part.get("content", "")
                                    break
                            if text:
                                break
                
                if not text:
                    raise HTTPException(status_code=400, detail="No text content found in input")
                
                # 提取correlation_id前缀 [CID:...]
                correlation_id = None
                if text.startswith('[CID:'):
                    try:
                        end = text.find(']')
                        if end != -1:
                            correlation_id = text[5:end]
                            text = text[end+1:].lstrip()
                    except Exception:
                        correlation_id = None
                
                # 生成医生回复
                role = self.agent_name.split('_')[-1].lower()  # doctor_a -> a
                print(f"[ACP-{self.agent_name}] 处理请求: text='{text[:100]}...', correlation_id={correlation_id}")
                
                reply = generate_doctor_reply(f'doctor_{role}', text)
                print(f"[ACP-{self.agent_name}] 生成回复: '{reply[:100]}...'")
                
                # 检查LLM回复是否包含错误信息 - 防止伪装成功
                if reply and ("Error in OpenAI chat generation" in reply or "Error in " in reply or "医生回复暂不可用" in reply):
                    print(f"[ACP-{self.agent_name}] 检测到LLM错误，返回error状态")
                    raise HTTPException(status_code=500, detail=f"LLM generation failed: {reply}")
                
                # 异步回投协调器/deliver
                if correlation_id:
                    asyncio.create_task(self._deliver_receipt(correlation_id, reply))
                else:
                    print(f"[ACP-{self.agent_name}] 警告: 无correlation_id，跳过回执投递")
                
                # 返回ACP标准格式响应
                return AgentRunResponse(
                    output={
                        "content": [
                            {
                                "type": "text",
                                "text": reply
                            }
                        ]
                    },
                    status="success",
                    agent_name=self.agent_name
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")
    
    async def _deliver_receipt(self, correlation_id: str, reply: str):
        """回投回执到协调器"""
        try:
            payload = {
                "sender_id": self.agent_name,
                "receiver_id": "ACP_Doctor_A" if "B" in self.agent_name else "ACP_Doctor_B",
                "text": reply,
                "correlation_id": correlation_id
            }
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(f"{self.coord_endpoint}/deliver", json=payload)
                if response.status_code not in (200, 201, 202):
                    print(f"[ACP-{self.agent_name}] 回执投递失败: HTTP {response.status_code} - {response.text}")
                else:
                    print(f"[ACP-{self.agent_name}] 回执投递成功: correlation_id={correlation_id}")
                
        except Exception as e:
            print(f"[ACP-{self.agent_name}] 回执投递异常: {e}")
            # 不再静默失败，记录错误但继续执行
    
    def run(self):
        """启动服务器"""
        uvicorn.run(self.app, host="127.0.0.1", port=self.port, log_level="info")


def create_doctor_a_server(port: int = 8010) -> ACPServer:
    """创建Doctor A服务器"""
    return ACPServer("ACP_Doctor_A", port)


def create_doctor_b_server(port: int = 8011) -> ACPServer:
    """创建Doctor B服务器"""
    return ACPServer("ACP_Doctor_B", port)


if __name__ == "__main__":
    import sys
    
    # 从环境变量获取端口，或使用默认值
    a_port = int(os.environ.get('ACP_A_PORT', '8010'))
    b_port = int(os.environ.get('ACP_B_PORT', '8011'))
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "doctor_a":
            server = create_doctor_a_server(a_port)
        elif sys.argv[1] == "doctor_b":
            server = create_doctor_b_server(b_port)
        else:
            print("Usage: python server.py [doctor_a|doctor_b]")
            sys.exit(1)
    else:
        # 默认启动Doctor A
        server = create_doctor_a_server(a_port)
    
    server.run()
