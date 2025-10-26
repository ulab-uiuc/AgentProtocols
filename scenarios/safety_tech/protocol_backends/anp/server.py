# -*- coding: utf-8 -*-
"""
ANP Protocol Server - Complies with official ANP protocol specification
- HTTP-based ANP protocol implementation, does not use WebSocket
- Uses DID for authentication
- Provides standard ANP endpoints: /agents, /health, /runs
- Supports direct HTTP communication between Agents
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

from scenarios.safety_tech.core.llm_wrapper import generate_doctor_reply

class MessageRequest(BaseModel):
    text: str
    correlation_id: Optional[str] = None

class RunsRequest(BaseModel):
    input: Dict[str, Any]

class ANPServer:
    """ANP protocol server - HTTP implementation complying with official specification"""
    
    def __init__(self, agent_name: str, port: int, doctor_role: str, coordinator_port: int = 8001):
        self.agent_name = agent_name
        self.port = port
        self.doctor_role = doctor_role  # "doctor_a" or "doctor_b"
        self.coordinator_port = coordinator_port
        self.app = FastAPI(title=f"ANP {agent_name}")
        
        # Generate real DID and key pair - must succeed, otherwise error
        # Import DID generation tool
        sys.path.insert(0, str(PROJECT_ROOT / "agentconnect_src"))
        from agent_connect.utils.did_generate import did_generate
        
        # Generate real DID, key pair and DID document
        communication_endpoint = f"ws://127.0.0.1:{port + 100}"  # WebSocket endpoint
        self.private_key, self.public_key, self.did, self.did_document_json = did_generate(
            communication_endpoint,
            did_server_domain="127.0.0.1",
            did_server_port=str(port)
        )
        
        # Extract public key hexadecimal representation from DID document
        import json
        from agent_connect.utils.crypto_tool import get_hex_from_public_key
        self.public_key_hex = get_hex_from_public_key(self.public_key)
        
        print(f"[ANP-{self.agent_name}] Generated real DID: {self.did}")
        print(f"[ANP-{self.agent_name}] Public key: {self.public_key_hex[:20]}...")
        
        # Store other agents' information
        self.peer_agents = {}
        
        self._setup_routes()
        print(f"[ANP-{self.agent_name}] Initialization complete, DID: {self.did}")

    def _setup_routes(self):
        """Setup ANP standard routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {"status": "healthy", "agent": self.agent_name, "did": self.did}
        
        @self.app.get("/agents")
        async def agents():
            """ANP standard agents endpoint - return agent information"""
            return {
                "agents": [{
                    "name": self.agent_name,
                    "did": self.did,
                    "description": f"Doctor Agent - {self.doctor_role}",
                    "capabilities": ["medical_consultation", "patient_discussion"]
                }]
            }
        
        @self.app.get("/did")
        async def get_did():
            """Return DID information"""
            return {"did": self.did}
        
        @self.app.get("/registration_proof")
        async def get_registration_proof():
            """Return registration proof information"""
            # Must have valid key pair, otherwise throw exception
            if not (self.public_key_hex and self.private_key):
                raise HTTPException(status_code=500, detail="No valid keys available")
            
            # Generate real signature
            timestamp = float(time.time())
            message = {"did": self.did, "timestamp": timestamp}
            
            from agent_connect.utils.crypto_tool import generate_signature_for_json
            signature = generate_signature_for_json(self.private_key, message)
            
            return {
                "did": self.did,
                "did_public_key": self.public_key_hex,
                "did_signature": signature,
                "timestamp": str(timestamp),  # Registration gateway will parse this string with float()
                "agent_name": self.agent_name
            }
        
        @self.app.post("/runs")
        async def runs(request: RunsRequest):
            """ANP standard runs endpoint - handle task requests"""
            try:
                # Extract text content from input
                content = request.input.get("content", [])
                if not content or not isinstance(content, list):
                    raise HTTPException(status_code=400, detail="Invalid input format")
                
                # Extract first text content
                text_content = None
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content = item.get("text", "")
                        break
                
                if not text_content:
                    raise HTTPException(status_code=400, detail="No text content found")
                
                print(f"[ANP-{self.agent_name}] Processing runs request: {text_content[:50]}...")
                
                # Generate doctor reply
                reply = generate_doctor_reply(self.doctor_role, text_content)
                
                return {
                    "output": {
                        "content": [
                            {"type": "text", "text": reply}
                        ]
                    }
                }
                
            except Exception as e:
                print(f"[ANP-{self.agent_name}] Processing runs request failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/message")
        async def message(request: MessageRequest):
            """Message endpoint - compatible with existing tests"""
            try:
                print(f"[ANP-{self.agent_name}] processing message: {request.text[:50]}...")
                
                # Generate doctor reply
                reply = generate_doctor_reply(self.doctor_role, request.text)
                
                # If correlation_id exists, deliver receipt to coordinator
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
                print(f"[ANP-{self.agent_name}] Processing message failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/communicate")
        async def communicate_with_peer(target_agent: str, message: str):
            """Communicate with other ANP Agents"""
            try:
                # Look up target agent information
                if target_agent not in self.peer_agents:
                    # Try to discover target agent
                    await self._discover_agent(target_agent)
                
                if target_agent not in self.peer_agents:
                    raise HTTPException(status_code=404, detail=f"Agent {target_agent} not found")
                
                peer_info = self.peer_agents[target_agent]
                
                # Send HTTP request to target agent
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
                            "Authorization": f"DID {self.did}",  # Simplified DID authentication
                            "Content-Type": "application/json"
                        }
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    else:
                        raise HTTPException(status_code=response.status_code, detail=response.text)
                        
            except Exception as e:
                print(f"[ANP-{self.agent_name}] Communication with {target_agent} failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _discover_agent(self, agent_name: str):
        """Discover other ANP Agents"""
        # Try common ports
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
                                print(f"[ANP-{self.agent_name}] Discovered agent: {agent_name} at {port}")
                                return
            except:
                continue
    
    async def _deliver_receipt(self, correlation_id: str, reply: str):
        """Deliver receipt to coordinator"""
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
                    print(f"[ANP-{self.agent_name}] Receipt delivery successful: CID={correlation_id}")
                else:
                    print(f"[ANP-{self.agent_name}] Receipt delivery failed: {response.status_code}")
        except Exception as e:
            print(f"[ANP-{self.agent_name}] Receipt delivery exception: {e}")

    def run(self) -> None:
        """Start ANP server"""
        print(f"[ANP-{self.agent_name}] Starting HTTP server on port {self.port}")
        uvicorn.run(
            self.app,
            host="127.0.0.1",
            port=self.port,
            log_level="error",
            access_log=False,
        )

def create_doctor_a_server(port: int) -> ANPServer:
    """Create Doctor A server"""
    return ANPServer("ANP_Doctor_A", port, "doctor_a")

def create_doctor_b_server(port: int) -> ANPServer:
    """Create Doctor B server"""
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