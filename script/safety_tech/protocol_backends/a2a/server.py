# -*- coding: utf-8 -*-
"""
A2A Protocol Server for Safety Tech
简化的A2A协议服务器实现，支持Doctor A/B代理、健康检查、回执投递等
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Dict, Any, List, Optional
import httpx

# A2A SDK imports
try:
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentProvider
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    A2A_SDK_AVAILABLE = True
    print("[A2A Server] A2A SDK available")
except ImportError as e:
    A2A_SDK_AVAILABLE = False
    print(f"[A2A Server] A2A SDK not available: {e}")

try:
    from script.safety_tech.core.llm_wrapper import generate_doctor_reply
except Exception:
    from core.llm_wrapper import generate_doctor_reply


class A2ADoctorExecutor(AgentExecutor if A2A_SDK_AVAILABLE else object):
    """A2A医生代理执行器"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.coord_endpoint = os.environ.get('COORD_ENDPOINT', 'http://127.0.0.1:8888')
    
    async def execute(self, context: "RequestContext", event_queue: "EventQueue") -> None:
        """执行A2A代理逻辑"""
        try:
            # 从context获取输入消息（优先使用A2A SDK提供的API）
            task_input = ""
            try:
                if hasattr(context, 'get_user_input'):
                    task_input = context.get_user_input() or ""
            except Exception:
                task_input = ""
            if not task_input:
                if hasattr(context, 'message') and context.message:
                    # 从消息中提取文本内容
                    if hasattr(context.message, 'text'):
                        task_input = context.message.text
                    elif hasattr(context.message, 'content'):
                        task_input = str(context.message.content)
                    else:
                        task_input = str(context.message)
            
            # 提取correlation_id前缀 [CID:...]
            correlation_id = None
            text = task_input
            
            if text.startswith('[CID:'):
                try:
                    end = text.find(']')
                    if end != -1:
                        correlation_id = text[5:end]
                        text = text[end+1:].lstrip()
                except Exception:
                    correlation_id = None
            
            role = self.agent_name.split('_')[-1].lower()  # A2A_Doctor_A -> a
            print(f"[A2A-{self.agent_name}] 处理请求: text='{text[:100]}...', correlation_id={correlation_id}")
            
            # 生成医生回复
            reply = generate_doctor_reply(f'doctor_{role}', text)
            print(f"[A2A-{self.agent_name}] 生成回复: '{reply[:100]}...'")
            
            # 发送回复到事件队列
            from a2a.utils import new_agent_text_message
            # A2A SDK EventQueue may return awaitable
            res = event_queue.enqueue_event(new_agent_text_message(reply))
            if hasattr(res, "__await__"):
                await res
            
            # 异步回投协调器/deliver
            if correlation_id:
                asyncio.create_task(self._deliver_receipt(correlation_id, reply))
            else:
                print(f"[A2A-{self.agent_name}] 警告: 无correlation_id，跳过回执投递")
            
        except Exception as e:
            print(f"[A2A-{self.agent_name}] 执行异常: {e}")
            import traceback
            traceback.print_exc()
            # 发送错误消息
            from a2a.utils import new_agent_text_message
            from a2a.utils import new_agent_text_message
            res = event_queue.enqueue_event(new_agent_text_message(f"Error: {str(e)}"))
            if hasattr(res, "__await__"):
                await res
    
    async def cancel(self, context: "RequestContext", event_queue: "EventQueue") -> None:
        """取消任务"""
        print(f"[A2A-{self.agent_name}] 取消任务请求")
        # 简单实现，不需要特殊处理
    
    async def _deliver_receipt(self, correlation_id: str, reply: str):
        """回投回执到协调器"""
        try:
            payload = {
                "sender_id": self.agent_name,
                "receiver_id": "A2A_Doctor_A" if "B" in self.agent_name else "A2A_Doctor_B",
                "text": reply,
                "correlation_id": correlation_id
            }
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(f"{self.coord_endpoint}/deliver", json=payload)
                if response.status_code not in (200, 201, 202):
                    print(f"[A2A-{self.agent_name}] 回执投递失败: HTTP {response.status_code} - {response.text}")
                else:
                    print(f"[A2A-{self.agent_name}] 回执投递成功: correlation_id={correlation_id}")
                    
        except Exception as e:
            print(f"[A2A-{self.agent_name}] 回执投递异常: {e}")


def create_doctor_app(agent_name: str, port: int):
    """创建A2A医生代理应用（使用本项目A2A适配器，内置/health与/message）"""
    if not A2A_SDK_AVAILABLE:
        raise RuntimeError("A2A SDK not available")

    # 创建执行器
    executor = A2ADoctorExecutor(agent_name)

    # 构造Agent Card（传给适配器用于/.well-known）
    agent_card = {
        "name": f"Medical Doctor {agent_name.split('_')[-1]}",
        "description": f"A2A-enabled medical doctor {agent_name} for safety testing",
        "url": f"http://127.0.0.1:{port}/",
        "version": "1.0.0",
        "provider": {
            "name": "Safety Testing Framework",
            "organization": "Agent Protocol Benchmark",
            "url": f"http://127.0.0.1:{port}/",
            "email": "safety@example.com",
        },
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "capabilities": {"streaming": False},
        "skills": [
            {
                "id": "medical_consultation",
                "name": "Medical Consultation",
                "description": "Provides medical consultation and diagnosis",
                "tags": ["medical", "consultation", "diagnosis"],
                "inputModes": ["text"],
                "outputModes": ["text"],
                "examples": [
                    "Diagnose patient symptoms",
                    "Provide treatment recommendations",
                ],
            }
        ],
    }

    # 使用项目自带A2A Starlette应用（内置/health）
    try:
        from src.server_adapters.a2a_adapter import A2AStarletteApplication as WrappedA2AApp
    except Exception:
        from script.safety_tech.src.server_adapters.a2a_adapter import A2AStarletteApplication as WrappedA2AApp  # 兜底

    app_builder = WrappedA2AApp(agent_card=agent_card, executor=executor)
    return app_builder


def run_server(agent_name: str, port: int):
    """运行A2A服务器"""
    import uvicorn
    
    print(f"[A2A Server] 启动 {agent_name} 在端口 {port}")
    
    try:
        app = create_doctor_app(agent_name, port)
        asgi_app = app.build()

        print(f"[A2A Server] {agent_name} 服务器已创建，准备启动...")
        uvicorn.run(
            asgi_app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            access_log=False,
        )
        
    except Exception as e:
        print(f"[A2A Server] 启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if not A2A_SDK_AVAILABLE:
        print("Cannot start A2A server without SDK")
        sys.exit(1)
    
    # 从环境变量获取配置
    a_port = int(os.environ.get('A2A_A_PORT', '8010'))
    b_port = int(os.environ.get('A2A_B_PORT', '8011'))
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "doctor_a":
            run_server("A2A_Doctor_A", a_port)
        elif sys.argv[1] == "doctor_b":
            run_server("A2A_Doctor_B", b_port)
        else:
            print("Usage: python server.py [doctor_a|doctor_b]")
            sys.exit(1)
    else:
        # 默认启动Doctor A
        run_server("A2A_Doctor_A", a_port)
