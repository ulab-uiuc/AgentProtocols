# -*- coding: utf-8 -*-
"""
RG-Integrated Doctor Agents
Doctor Agents that truly register via RG and conduct LLM conversations.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
import uvicorn
import httpx
import logging
import os
try:
    from ..protocol_backends.agora.registration_adapter import AgoraRegistrationAdapter
except ImportError:
    from protocol_backends.agora.registration_adapter import AgoraRegistrationAdapter

# Optionally import the ACP adapter (based on configuration)
try:
    from ..protocol_backends.acp.registration_adapter import ACPRegistrationAdapter
except ImportError:
    try:
        from protocol_backends.acp.registration_adapter import ACPRegistrationAdapter
    except Exception:
        ACPRegistrationAdapter = None  # Deferred failure: only raise if ACP is explicitly required by config

# Import base Agent classes
try:
    from .privacy_agent_base import DoctorAAgent, DoctorBAgent
except ImportError:
    from core.privacy_agent_base import DoctorAAgent, DoctorBAgent

# Unified LLM reply wrapper
try:
    from .llm_wrapper import generate_doctor_reply
except Exception:
    from core.llm_wrapper import generate_doctor_reply

logger = logging.getLogger(__name__)


class RGDoctorAAgent(DoctorAAgent):
    """Doctor A Agent registered via RG"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any], port: int):
        super().__init__(agent_id, config)
        self.port = port
        self.rg_endpoint = config.get('rg_endpoint', 'http://127.0.0.1:8001')
        self.conversation_id = config.get('conversation_id')
        self.endpoint = f"http://127.0.0.1:{port}"
        
    # FastAPI application
        self.app = FastAPI(title=f"Doctor A Agent {agent_id}")
        self.setup_routes()
        
    # Registration state
        self.registered = False
        self.session_token = None
        self.verification_method: Optional[str] = None
        self.verification_latency_ms: Optional[int] = None
        
    # Conversation history
        self.conversation_history = []
        
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        def _build_model_config() -> Optional[Dict[str, Any]]:
            core_cfg = (self.config or {}).get('core', {}) if isinstance(self.config, dict) else {}
            if not isinstance(core_cfg, dict) or not core_cfg:
                return None
            base_url = core_cfg.get('openai_base_url') or core_cfg.get('nvidia_base_url') or os.getenv('OPENAI_BASE_URL')
            api_key = core_cfg.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
            name = core_cfg.get('name') or os.getenv('OPENAI_MODEL') or 'gpt-4o'
            temperature = core_cfg.get('temperature', 0.3)
            return {
                'model': {
                    'type': core_cfg.get('type', 'openai'),
                    'name': name,
                    'temperature': temperature,
                    'openai_api_key': api_key,
                    'openai_base_url': base_url,
                }
            }

        @self.app.post("/message")
        async def receive_message(payload: Dict[str, Any]):
            """Receive and process message"""
            try:
                message_type = payload.get('type', 'normal')
                
                if message_type == 'mirror':
                    # Observer mirror message, no response required
                    return {"status": "mirrored", "agent_id": self.agent_id}
                
                # Extract message content
                content = payload.get('text', payload.get('content', ''))
                sender_id = payload.get('sender_id', 'unknown')
                
                if not content:
                    return {"status": "no_content", "agent_id": self.agent_id}
                
                # Generate reply using the unified LLM wrapper
                response = generate_doctor_reply('doctor_a', str(content), model_config=_build_model_config())
                
                # Save conversation history
                self.conversation_history.append({
                    "timestamp": time.time(),
                    "sender": sender_id,
                    "received": content,
                    "response": response,
                    "type": "llm_conversation"
                })
                
                logger.debug(f"[{self.agent_id}] Processed message from {sender_id}, generated {len(response)} chars response")
                
                return {
                    "status": "processed",
                    "agent_id": self.agent_id,
                    "response": response,
                    "llm_used": True
                }
                
            except Exception as e:
                logger.error(f"[{self.agent_id}] Error processing message: {e}")
                return {"status": "error", "agent_id": self.agent_id, "error": str(e)}
        
        @self.app.get("/health")
        async def health_check():
            """Health check"""
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "registered": self.registered,
                "llm_available": True,
                "conversation_turns": len(self.conversation_history),
                "verification_method": self.verification_method,
                "verification_latency_ms": self.verification_latency_ms
            }
        
        @self.app.get("/conversation_history")
        async def get_conversation_history():
            """Get conversation history"""
            return {
                "agent_id": self.agent_id,
                "total_turns": len(self.conversation_history),
                "history": self.conversation_history
            }
    
    async def register_to_rg(self) -> bool:
        """Register to RG"""
        try:
            protocol = (self.config.get('protocol') or 'agora').lower()
            if protocol == 'acp':
                if ACPRegistrationAdapter is None:
                    raise RuntimeError("ACPRegistrationAdapter not available")
                adapter = ACPRegistrationAdapter({'rg_endpoint': self.rg_endpoint})
            else:
                adapter = AgoraRegistrationAdapter({'rg_endpoint': self.rg_endpoint, 'agora': {}, 'core': self.config.get('core', {})})

            result = await adapter.register_agent(
                agent_id=self.agent_id,
                endpoint=self.endpoint,
                conversation_id=self.conversation_id,
                role="doctor_a"
            )
            self.session_token = result.get('session_token')
            self.verification_method = result.get('verification_method')
            self.verification_latency_ms = result.get('verification_latency_ms')
            self.registered = True
            logger.info(f"[{self.agent_id}] Successfully registered to RG via Agora adapter")
            return True
                    
        except Exception as e:
            logger.error(f"[{self.agent_id}] Registration error: {e}")
            return False
    
    async def send_message_to_network(self, target_id: str, message: str) -> Dict[str, Any]:
        """Send message via RG network"""
        if not self.registered:
            raise RuntimeError("Agent not registered to RG")
        
        # Send message via coordinator
        coordinator_endpoint = "http://127.0.0.1:8888"  # Coordinator endpoint
        
        payload = {
            "sender_id": self.agent_id,
            "receiver_id": target_id,
            "text": message,
            "timestamp": time.time(),
            "llm_generated": True
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{coordinator_endpoint}/route_message",
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Message routing failed: {response.status_code}")
                    return {"error": f"Routing failed: {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Message sending error: {e}")
            return {"error": str(e)}
    
    def run_server(self):
        """Run FastAPI server"""
        uvicorn.run(self.app, host="127.0.0.1", port=self.port, log_level="warning", access_log=False, lifespan="off", loop="asyncio", http="h11")


class RGDoctorBAgent(DoctorBAgent):
    """Doctor B Agent registered via RG"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any], port: int):
        super().__init__(agent_id, config)
        self.port = port
        self.rg_endpoint = config.get('rg_endpoint', 'http://127.0.0.1:8001')
        self.conversation_id = config.get('conversation_id')
        self.endpoint = f"http://127.0.0.1:{port}"
        
        # FastAPI application
        self.app = FastAPI(title=f"Doctor B Agent {agent_id}")
        self.setup_routes()
        
        # Registration state
        self.registered = False
        self.session_token = None
        
        # Conversation history
        self.conversation_history = []
        self.verification_method: Optional[str] = None
        self.verification_latency_ms: Optional[int] = None
        
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        def _build_model_config() -> Optional[Dict[str, Any]]:
            core_cfg = (self.config or {}).get('core', {}) if isinstance(self.config, dict) else {}
            if not isinstance(core_cfg, dict) or not core_cfg:
                return None
            base_url = core_cfg.get('openai_base_url') or core_cfg.get('nvidia_base_url') or os.getenv('OPENAI_BASE_URL')
            api_key = core_cfg.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
            name = core_cfg.get('name') or os.getenv('OPENAI_MODEL') or 'gpt-4o'
            temperature = core_cfg.get('temperature', 0.3)
            return {
                'model': {
                    'type': core_cfg.get('type', 'openai'),
                    'name': name,
                    'temperature': temperature,
                    'openai_api_key': api_key,
                    'openai_base_url': base_url,
                }
            }

        @self.app.post("/message")
        async def receive_message(payload: Dict[str, Any]):
            """Receive and process message"""
            try:
                message_type = payload.get('type', 'normal')
                
                if message_type == 'mirror':
                    # Observer mirror message, no response required
                    return {"status": "mirrored", "agent_id": self.agent_id}
                
                # Extract message content
                content = payload.get('text', payload.get('content', ''))
                sender_id = payload.get('sender_id', 'unknown')
                
                if not content:
                    return {"status": "no_content", "agent_id": self.agent_id}
                
                # Generate reply using the unified LLM wrapper
                response = generate_doctor_reply('doctor_b', str(content), model_config=_build_model_config())
                
                # Save conversation history
                self.conversation_history.append({
                    "timestamp": time.time(),
                    "sender": sender_id,
                    "received": content,
                    "response": response,
                    "type": "llm_conversation"
                })
                
                logger.debug(f"[{self.agent_id}] Processed message from {sender_id}, generated {len(response)} chars response")
                
                # Auto-reply to the sender
                if sender_id != self.agent_id:
                    asyncio.create_task(self._auto_reply(sender_id, response))
                
                return {
                    "status": "processed",
                    "agent_id": self.agent_id,
                    "response": response,
                    "llm_used": True
                }
                
            except Exception as e:
                logger.error(f"[{self.agent_id}] Error processing message: {e}")
                return {"status": "error", "agent_id": self.agent_id, "error": str(e)}
        
        @self.app.get("/health")
        async def health_check():
            """Health check"""
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "registered": self.registered,
                "llm_available": True,
                "conversation_turns": len(self.conversation_history),
                "verification_method": self.verification_method,
                "verification_latency_ms": self.verification_latency_ms
            }
        
        @self.app.get("/conversation_history")
        async def get_conversation_history():
            """Get conversation history"""
            return {
                "agent_id": self.agent_id,
                "total_turns": len(self.conversation_history),
                "history": self.conversation_history
            }
    
    async def _auto_reply(self, target_id: str, message: str):
        """Automatically reply to a message"""
        try:
            await asyncio.sleep(1)  # Short delay
            await self.send_message_to_network(target_id, message)
        except Exception as e:
            logger.error(f"Auto reply failed: {e}")
    
    async def register_to_rg(self) -> bool:
        """Register to RG"""
        try:
            protocol = (self.config.get('protocol') or 'agora').lower()
            if protocol == 'acp':
                if ACPRegistrationAdapter is None:
                    raise RuntimeError("ACPRegistrationAdapter not available")
                adapter = ACPRegistrationAdapter({'rg_endpoint': self.rg_endpoint})
            else:
                adapter = AgoraRegistrationAdapter({'rg_endpoint': self.rg_endpoint, 'agora': {}, 'core': self.config.get('core', {})})

            result = await adapter.register_agent(
                agent_id=self.agent_id,
                endpoint=self.endpoint,
                conversation_id=self.conversation_id,
                role="doctor_b"
            )
            self.session_token = result.get('session_token')
            self.verification_method = result.get('verification_method')
            self.verification_latency_ms = result.get('verification_latency_ms')
            self.registered = True
            logger.info(f"[{self.agent_id}] Successfully registered to RG via Agora adapter")
            return True
                    
        except Exception as e:
            logger.error(f"[{self.agent_id}] Registration error: {e}")
            return False
    
    async def send_message_to_network(self, target_id: str, message: str) -> Dict[str, Any]:
        """Send message via RG network"""
        if not self.registered:
            raise RuntimeError("Agent not registered to RG")
        
        # Send message via coordinator
        coordinator_endpoint = "http://127.0.0.1:8888"  # Coordinator endpoint
        
        payload = {
            "sender_id": self.agent_id,
            "receiver_id": target_id,
            "text": message,
            "timestamp": time.time(),
            "llm_generated": True
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{coordinator_endpoint}/route_message",
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Message routing failed: {response.status_code}")
                    return {"error": f"Routing failed: {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Message sending error: {e}")
            return {"error": str(e)}
    
    def run_server(self):
        """Run FastAPI server"""
        uvicorn.run(self.app, host="127.0.0.1", port=self.port, log_level="warning", access_log=False, lifespan="off", loop="asyncio", http="h11")


async def create_and_start_doctor_agent(agent_class, agent_id: str, config: Dict[str, Any], port: int):
    """Create and start a doctor agent"""
    agent = agent_class(agent_id, config, port)
    
    # Start server in background
    import threading
    def run_server():
        agent.run_server()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    await asyncio.sleep(2)
    
    # Register to RG
    success = await agent.register_to_rg()
    if not success:
        raise Exception(f"Failed to register {agent_id} to RG")
    
    return agent
