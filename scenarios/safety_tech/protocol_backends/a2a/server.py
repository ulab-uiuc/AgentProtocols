# -*- coding: utf-8 -*-
"""
A2A Protocol Server for Safety Tech
Simplified A2A protocol server implementation, supporting Doctor A/B agents, health checks, receipt delivery, etc.
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
    print("[A2A Server] A2A SDK available")
except ImportError as e:
    raise ImportError(
        f"A2A SDK is required but not available: {e}. "
        "Please install with: pip install a2a-sdk"
    )

try:
    from scenarios.safety_tech.core.llm_wrapper import generate_doctor_reply
except Exception:
    from core.llm_wrapper import generate_doctor_reply


class A2ADoctorExecutor(AgentExecutor):
    """A2A doctor agent executor"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.coord_endpoint = os.environ.get('COORD_ENDPOINT', 'http://127.0.0.1:8888')
    
    async def execute(self, context: "RequestContext", event_queue: "EventQueue") -> None:
        """Execute A2A agent logic"""
        try:
            # Get input message from context (prioritize A2A SDK provided API)
            task_input = ""
            try:
                if hasattr(context, 'get_user_input'):
                    task_input = context.get_user_input() or ""
            except Exception:
                task_input = ""
            if not task_input:
                if hasattr(context, 'message') and context.message:
                    # Extract text content from message
                    if hasattr(context.message, 'text'):
                        task_input = context.message.text
                    elif hasattr(context.message, 'content'):
                        task_input = str(context.message.content)
                    else:
                        task_input = str(context.message)
            
            # Extract correlation_id prefix [CID:...]
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
            print(f"[A2A-{self.agent_name}] Processing request: text='{text[:100]}...', correlation_id={correlation_id}")
            
            # Generate doctor reply
            reply = generate_doctor_reply(f'doctor_{role}', text)
            print(f"[A2A-{self.agent_name}] Generated reply: '{reply[:100]}...'")
            
            # Send reply to event queue
            from a2a.utils import new_agent_text_message
            # A2A SDK EventQueue may return awaitable
            res = event_queue.enqueue_event(new_agent_text_message(reply))
            if hasattr(res, "__await__"):
                await res
            
            # Async deliver receipt to coordinator/deliver
            if correlation_id:
                asyncio.create_task(self._deliver_receipt(correlation_id, reply))
            else:
                print(f"[A2A-{self.agent_name}] Warning: no correlation_id, skipping receipt delivery")
            
        except Exception as e:
            print(f"[A2A-{self.agent_name}] Execution exception: {e}")
            import traceback
            traceback.print_exc()
            # Send error message
            from a2a.utils import new_agent_text_message
            from a2a.utils import new_agent_text_message
            res = event_queue.enqueue_event(new_agent_text_message(f"Error: {str(e)}"))
            if hasattr(res, "__await__"):
                await res
    
    async def cancel(self, context: "RequestContext", event_queue: "EventQueue") -> None:
        """Cancel task"""
        print(f"[A2A-{self.agent_name}] Cancel task request")
        # Simple implementation, no special handling needed
    
    async def _deliver_receipt(self, correlation_id: str, reply: str):
        """Deliver receipt back to coordinator"""
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
                    print(f"[A2A-{self.agent_name}] Receipt delivery failed: HTTP {response.status_code} - {response.text}")
                else:
                    print(f"[A2A-{self.agent_name}] Receipt delivery successful: correlation_id={correlation_id}")
                    
        except Exception as e:
            print(f"[A2A-{self.agent_name}] Receipt delivery exception: {e}")


def create_doctor_app(agent_name: str, port: int):
    """Create A2A doctor agent application (uses project A2A adapter, built-in /health and /message)"""
    # Create executor
    executor = A2ADoctorExecutor(agent_name)

    # Construct Agent Card (pass to adapter for /.well-known)
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

    # Use project built-in A2A Starlette application (built-in /health)
    from src.server_adapters.a2a_adapter import A2AStarletteApplication as WrappedA2AApp

    app_builder = WrappedA2AApp(agent_card=agent_card, executor=executor)
    return app_builder


def run_server(agent_name: str, port: int):
    """Run A2A server"""
    import uvicorn
    
    print(f"[A2A Server] Starting {agent_name} on port {port}")
    
    try:
        app = create_doctor_app(agent_name, port)
        asgi_app = app.build()

        print(f"[A2A Server] {agent_name} server created, ready to start...")
        uvicorn.run(
            asgi_app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            access_log=False,
        )
        
    except Exception as e:
        print(f"[A2A Server] Startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Get configuration from environment variables
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
        # Default start Doctor A
        run_server("A2A_Doctor_A", a_port)
