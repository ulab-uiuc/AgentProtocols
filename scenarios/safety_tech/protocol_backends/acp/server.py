# -*- coding: utf-8 -*-
"""
ACP Protocol Server
Complete ACP protocol server implementation, supporting Doctor A/B agents, health checks, receipt delivery, etc.
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
    """ACP Agent execution request"""
    input: Any  # Support dict or list format
    agent_name: Optional[str] = None
    mode: Optional[str] = None


class AgentRunResponse(BaseModel):
    """ACP Agent execution response"""
    output: Dict[str, Any]
    status: str = "success"
    agent_name: Optional[str] = None


class ACPServer:
    """ACP protocol server"""
    
    def __init__(self, agent_name: str, port: int = 8000):
        self.agent_name = agent_name
        self.port = port
        self.app = FastAPI(title=f"ACP Server - {agent_name}")
        self.coord_endpoint = os.environ.get('COORD_ENDPOINT', 'http://127.0.0.1:8888')
        self.setup_routes()
    
    def setup_routes(self):
        """Setup HTTP routes"""
        
        @self.app.get("/agents")
        async def get_agents():
            """Get available agent list"""
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
            """Health check"""
            return {
                "status": "healthy",
                "agent_name": self.agent_name,
                "timestamp": time.time(),
                "protocol": "acp"
            }
        
        @self.app.get("/ping")
        async def ping():
            """ACP protocol probe endpoint"""
            return {
                "status": "ok",
                "agent_name": self.agent_name,
                "protocol": "acp",
                "timestamp": time.time()
            }
        
        @self.app.post("/runs", response_model=AgentRunResponse)
        async def run_agent(request: AgentRunRequest):
            """Execute agent task"""
            try:
                # Extract input content - support multiple formats
                input_data = request.input
                text = ""
                
                # Process different input formats
                if isinstance(input_data, dict):
                    # Format 1: {"content": [{"type": "text", "text": "..."}]}
                    if "content" in input_data:
                        content_list = input_data["content"]
                        for content_item in content_list:
                            if isinstance(content_item, dict) and content_item.get("type") == "text":
                                text = content_item.get("text", "")
                                break
                elif isinstance(input_data, list):
                    # Format 2: [{"role": "user", "parts": [{"content_type": "text/plain", "content": "..."}]}]
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
                
                # Extract correlation_id prefix [CID:...]
                correlation_id = None
                if text.startswith('[CID:'):
                    try:
                        end = text.find(']')
                        if end != -1:
                            correlation_id = text[5:end]
                            text = text[end+1:].lstrip()
                    except Exception:
                        correlation_id = None
                
                # Generate doctor reply
                role = self.agent_name.split('_')[-1].lower()  # doctor_a -> a
                print(f"[ACP-{self.agent_name}] Processing request: text='{text[:100]}...', correlation_id={correlation_id}")
                
                reply = generate_doctor_reply(f'doctor_{role}', text)
                print(f"[ACP-{self.agent_name}] Generated reply: '{reply[:100]}...'")
                
                # Check if LLM reply contains error info - prevent fake success
                if reply and ("Error in OpenAI chat generation" in reply or "Error in " in reply or "Doctor reply unavailable" in reply):
                    print(f"[ACP-{self.agent_name}] Detected LLM error, returning error status")
                    raise HTTPException(status_code=500, detail=f"LLM generation failed: {reply}")
                
                # Async deliver receipt to coordinator/deliver
                if correlation_id:
                    asyncio.create_task(self._deliver_receipt(correlation_id, reply))
                else:
                    print(f"[ACP-{self.agent_name}] Warning: no correlation_id, skipping receipt delivery")
                
                # Return ACP standard format response
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
        """Deliver receipt back to coordinator"""
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
                    print(f"[ACP-{self.agent_name}] Receipt delivery failed: HTTP {response.status_code} - {response.text}")
                else:
                    print(f"[ACP-{self.agent_name}] Receipt delivery successful: correlation_id={correlation_id}")
                
        except Exception as e:
            print(f"[ACP-{self.agent_name}] Receipt delivery exception: {e}")
            # No longer fail silently, log error but continue execution
    
    def run(self):
        """Start server"""
        uvicorn.run(self.app, host="127.0.0.1", port=self.port, log_level="info")


def create_doctor_a_server(port: int = 8010) -> ACPServer:
    """Create Doctor A server"""
    return ACPServer("ACP_Doctor_A", port)


def create_doctor_b_server(port: int = 8011) -> ACPServer:
    """Create Doctor B server"""
    return ACPServer("ACP_Doctor_B", port)


if __name__ == "__main__":
    import sys
    
    # Get port from environment variable or use default value
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
        # Default start Doctor A
        server = create_doctor_a_server(a_port)
    
    server.run()
