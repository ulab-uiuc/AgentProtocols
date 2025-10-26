# -*- coding: utf-8 -*-
"""
RG-Integrated Coordinator
Coordinator implementation based on Registration Gateway (RG), supporting dynamic routing and Observer roles
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Set
import httpx
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
import uvicorn
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParticipantInfo:
    """Participant information"""
    agent_id: str
    protocol: str
    endpoint: str
    role: str
    verified: bool
    joined_at: float


@dataclass
class ConversationMessage:
    """Conversation message"""
    sender_id: str
    receiver_id: str
    content: str
    timestamp: float
    message_id: str
    role: str
    correlation_id: str | None = None


class RGCoordinator:
    """RG-based coordinator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rg_endpoint = config.get('rg_endpoint', 'http://127.0.0.1:8001')
        self.conversation_id = config.get('conversation_id', f'conv_{int(time.time())}')
        self.port = config.get('coordinator_port', 8888)
        
        # FastAPI application
        self.app = FastAPI(title="RG Coordinator")
        self.setup_routes()
        
        # Participant management
        self.participants: Dict[str, ParticipantInfo] = {}
        # observers property removed - new S2 design does not require Observer mechanism
        
        # Message history (for backfill)
        self.message_history: List[ConversationMessage] = []
        self.max_history_size = config.get('max_history_size', 100)
        
        # Bridge configuration
        self.bridge_config = config.get('bridge', {})
        self.enable_live_mirror = self.bridge_config.get('enable_live_mirror', True)
        self.enable_backfill = self.bridge_config.get('enable_backfill', True)
        self.backfill_limit = self.bridge_config.get('backfill_limit', 10)
        
        # Polling configuration
        self.directory_poll_interval = config.get('directory_poll_interval', 5.0)
        self.running = False
        
        # Performance optimization: cache protocol backend registry
        self._backend_registry = None
        self._registry_initialized = False
    
    def setup_routes(self):
        """Setup HTTP routes"""
        
        @self.app.post("/route_message")
        async def route_message_endpoint(payload: Dict[str, Any]):
            """Message routing endpoint"""
            try:
                sender_id = payload.get('sender_id')
                receiver_id = payload.get('receiver_id')
                
                if not sender_id:
                    raise HTTPException(status_code=400, detail="Missing sender_id")
                
                # Unified correlation ID: generate if not provided
                corr = payload.get('correlation_id')
                if not corr:
                    corr = f"corr_{int(time.time()*1000)}"
                    payload['correlation_id'] = corr

                # Extract probe configuration (if exists)
                probe_config = payload.get('probe_config')

                result = await self.route_message(sender_id, receiver_id, payload, probe_config)
                return result
                
            except Exception as e:
                # For unregistered sender errors, use debug level (this is usually attack testing)
                if "not registered" in str(e):
                    logger.debug(f"Message routing blocked: {e}")
                else:
                    logger.error(f"Message routing error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/deliver")
        async def deliver_message_endpoint(payload: Dict[str, Any]):
            """Business acknowledgment returned by target Agent.
            Expected fields: sender_id, receiver_id, correlation_id, content/text/body (choose one of three).
            """
            try:
                sender_id = payload.get('sender_id')
                receiver_id = payload.get('receiver_id')
                if not sender_id:
                    raise HTTPException(status_code=400, detail="Missing sender_id")
                # Content extraction
                content = self._extract_message_content(payload)
                correlation_id = payload.get('correlation_id')
                # Role inference
                role = self.participants.get(sender_id).role if sender_id in self.participants else 'unknown'
                message = ConversationMessage(
                    sender_id=sender_id,
                    receiver_id=receiver_id or 'broadcast',
                    content=content,
                    timestamp=time.time(),
                    message_id=f"deliver_{int(time.time()*1000)}",
                    role=role,
                    correlation_id=correlation_id
                )
                # Store in database
                self._store_message(message)
                # Mirror to Observers
                if self.enable_live_mirror:
                    await self._broadcast_to_observers(message, payload)
                return {"status": "received", "message_id": message.message_id, "correlation_id": correlation_id}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Deliver handling error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/conversation_status")
        async def get_conversation_status_endpoint():
            """Get conversation status"""
            return await self.get_conversation_status()
        
        @self.app.get("/message_history")
        async def get_message_history_endpoint(limit: int = 50):
            """Get message history"""
            return await self.get_message_history(limit)
        
        @self.app.get("/health")
        async def health_check():
            """Health check"""
            return {
                "status": "healthy",
                "conversation_id": self.conversation_id,
                "participants": len(self.participants),
                "observers": 0,  # Observer mechanism removed
                "message_count": len(self.message_history)
            }
        
    async def start(self):
        """Start coordinator"""
        self.running = True
        
        # Execute participant refresh immediately to ensure participant info is available at startup
        logger.info("Starting initial participant refresh...")
        await self._refresh_participants()
        
        # Start directory polling task
        asyncio.create_task(self._directory_polling_loop())
        
        # Start HTTP server
        import threading
        def run_server():
            uvicorn.run(self.app, host="127.0.0.1", port=self.port, log_level="warning", access_log=False, lifespan="off", loop="asyncio", http="h11")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        await asyncio.sleep(2)  # Wait for server startup
        
        # Refresh participants again to ensure latest info after server startup
        await self._refresh_participants()
        
        logger.info(f"RG Coordinator started for conversation {self.conversation_id} on port {self.port}")
        logger.info(f"Initial participants loaded: {list(self.participants.keys())}")
    
    async def stop(self):
        """Stop coordinator"""
        self.running = False
        logger.info("RG Coordinator stopped")
    
    async def _directory_polling_loop(self):
        """Directory polling loop"""
        while self.running:
            try:
                await self._refresh_participants()
                await asyncio.sleep(self.directory_poll_interval)
            except Exception as e:
                logger.error(f"Directory polling error: {e}")
                await asyncio.sleep(self.directory_poll_interval)
    
    async def _refresh_participants(self):
        """Refresh participant list"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.rg_endpoint}/directory",
                    params={"conversation_id": self.conversation_id},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    directory = response.json()
                    await self._update_participants(directory['participants'])
                elif response.status_code != 404:  # 404 means conversation does not exist, normal case
                    logger.warning(f"Directory query failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Failed to refresh participants: {e}")
    
    async def _update_participants(self, participants_data: List[Dict[str, Any]]):
        """Update participant information"""
        new_participants = {}
        # new_observers removed - new S2 design does not require Observer mechanism
        
        for participant_data in participants_data:
            agent_id = participant_data['agent_id']
            role = participant_data['role']
            
            participant = ParticipantInfo(
                agent_id=agent_id,
                protocol=participant_data['protocol'],
                endpoint=participant_data['endpoint'],
                role=role,
                verified=participant_data['verified'],
                joined_at=participant_data['joined_at']
            )
            
            new_participants[agent_id] = participant
            
            if role == 'observer':
                # Observer handling removed - new S2 design does not require Observer mechanism
                continue
        
        # Detect newly joined participants
        for agent_id in new_participants:
            if agent_id not in self.participants:
                logger.info(f"New participant joined: {agent_id} ({new_participants[agent_id].role})")
        
        self.participants = new_participants
        # self.observers removed - new S2 design does not require Observer mechanism
    
    async def route_message(self, sender_id: str, receiver_id: str, payload: Dict[str, Any], probe_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Route message"""
        # Validate sender - if participant info is empty, try to force refresh once
        if not self.participants:
            logger.warning("No participants loaded, forcing directory refresh...")
            await self._refresh_participants()
            await asyncio.sleep(0.5)  # Give RG some time to process
        
        if sender_id not in self.participants:
            # Provide detailed debug information
            participant_list = list(self.participants.keys()) if self.participants else "No participants loaded"
            raise ValueError(f"Sender {sender_id} not registered in conversation {self.conversation_id}. Available participants: {participant_list}")
        
        sender = self.participants[sender_id]
        
        # Check sending permission
        if sender.role == 'observer':
            raise ValueError("Observers cannot send messages")

        # Role pairing constraint: only allow doctor_a ↔ doctor_b communication
        if receiver_id:
            receiver = self.participants.get(receiver_id)
            if not receiver:
                raise ValueError(f"Receiver {receiver_id} not registered in conversation")
            allowed_pairs = {('doctor_a', 'doctor_b'), ('doctor_b', 'doctor_a')}
            if (sender.role, receiver.role) not in allowed_pairs and receiver.role != 'observer':
                raise ValueError(f"Routing not allowed between roles {sender.role} -> {receiver.role}")
        
        # Validate receiver (if specified)
        if receiver_id and receiver_id not in self.participants:
            raise ValueError(f"Receiver {receiver_id} not registered in conversation")
        
        # Extract message content
        content = self._extract_message_content(payload)
        
        # Create message record
        corr_id = payload.get('correlation_id')
        message = ConversationMessage(
            sender_id=sender_id,
            receiver_id=receiver_id or 'broadcast',
            content=content,
            timestamp=time.time(),
            message_id=f"msg_{int(time.time() * 1000)}",
            role=sender.role,
            correlation_id=corr_id
        )
        
        # Store message history
        self._store_message(message)
        
        # Route message
        result = await self._deliver_message(message, payload, probe_config)
        
        # Broadcast mirror to Observers
        if self.enable_live_mirror:
            await self._broadcast_to_observers(message, payload)
        
        return result
    
    def _get_backend_registry(self):
        """Get cached protocol backend registry"""
        if not self._registry_initialized:
            try:
                from scenarios.safety_tech.protocol_backends.common.interfaces import get_registry
                self._backend_registry = get_registry()
                self._registry_initialized = True
                logger.info("Protocol backend registry cached")
            except Exception as e:
                logger.error(f"Protocol backend registry initialization failed: {e}")
                raise RuntimeError(f"Protocol backend registry not available: {e}")
        return self._backend_registry
    
    def _extract_message_content(self, payload: Dict[str, Any]) -> str:
        """Extract message content"""
        # Support multiple payload formats
        if 'text' in payload:
            return payload['text']
        elif 'body' in payload:
            return payload['body']
        elif 'content' in payload:
            return payload['content']
        else:
            return json.dumps(payload)
    
    def _store_message(self, message: ConversationMessage):
        """Store message to history"""
        self.message_history.append(message)
        
        # Limit history size
        if len(self.message_history) > self.max_history_size:
            self.message_history = self.message_history[-self.max_history_size:]
    
    async def _deliver_message(self, message: ConversationMessage, original_payload: Dict[str, Any], probe_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Deliver message to target"""
        if message.receiver_id == 'broadcast':
            # Broadcast to all non-Observer participants
            results = []
            for participant in self.participants.values():
                if participant.role != 'observer' and participant.agent_id != message.sender_id:
                    try:
                        result = await self._send_to_participant(participant, original_payload, probe_config)
                        results.append({"agent_id": participant.agent_id, "result": result})
                    except Exception as e:
                        logger.error(f"Failed to send to {participant.agent_id}: {e}")
                        results.append({"agent_id": participant.agent_id, "error": str(e)})
            return {"broadcast_results": results}
        else:
            # Unicast to specified receiver
            receiver = self.participants[message.receiver_id]
            try:
                result = await self._send_to_participant(receiver, original_payload, probe_config)
                return result
            except Exception as e:
                logger.error(f"Failed to send to {message.receiver_id}: {e}")
                return {"error": str(e)}
    
    async def _broadcast_to_observers(self, message: ConversationMessage, original_payload: Dict[str, Any]):
        """Broadcast mirror to Observers - disabled"""
        # Observer broadcast removed - new S2 design does not require Observer mechanism
        return
        
        # Build Observer mirror payload
        mirror_payload = {
            "type": "mirror",
            "original_message": {
                "sender_id": message.sender_id,
                "receiver_id": message.receiver_id,
                "content": message.content,
                "timestamp": message.timestamp,
                "message_id": message.message_id,
                "sender_role": message.role,
                "correlation_id": message.correlation_id
            },
            "original_payload": original_payload
        }
        
        # Observer mirror logic removed - new S2 design does not require Observer mechanism
    
    async def _provide_backfill(self, observer_id: str):
        """Provide history backfill for Observer"""
        if not self.message_history:
            return
        
        observer = self.participants.get(observer_id)
        if not observer:
            return
        
        # Get recent messages
        recent_messages = self.message_history[-self.backfill_limit:]
        
        backfill_payload = {
            "type": "backfill",
            "conversation_id": self.conversation_id,
            "message_count": len(recent_messages),
            "messages": []
        }
        
        for msg in recent_messages:
            backfill_payload["messages"].append({
                "sender_id": msg.sender_id,
                "receiver_id": msg.receiver_id,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "message_id": msg.message_id,
                "sender_role": msg.role
            })
        
        try:
            # Send backfill directly to Observer's /message endpoint
            await self._send_to_endpoint(observer.endpoint, backfill_payload)
            logger.info(f"Provided backfill to observer {observer_id}: {len(recent_messages)} messages")
        except Exception as e:
            logger.error(f"Failed to provide backfill to observer {observer_id}: {e}")
    
    async def _send_to_endpoint(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Protocol-agnostic direct HTTP send.
        Convention: Observer and other generic HTTP receivers expose /message interface.
        """
        url = (endpoint or '').rstrip('/') + '/message'
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=10.0)
                if resp.status_code in (200, 202):
                    try:
                        return resp.json()
                    except Exception:
                        return {"status": "received"}
                raise RuntimeError(f"Endpoint {url} returned {resp.status_code}: {resp.text}")
        except Exception as e:
            raise RuntimeError(f"Failed to send to endpoint {url}: {e}")
    
    async def _send_to_participant(self, participant: ParticipantInfo, payload: Dict[str, Any], probe_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send message to participant: distribute to corresponding backend client through protocol registry.

        Requirements:
        - Backend implementation must be native protocol, no mock/fallback allowed
        - Coordinator no longer assembles protocol payload details, each client implementation is responsible for
        """
        # Use cached registry to avoid repeated queries
        registry = self._get_backend_registry()
        backend = registry.get(participant.protocol)
        if backend is None:
            raise RuntimeError(f"No backend registered for protocol: {participant.protocol}")

        # Extract correlation_id from payload (compatible with existing logic)
        correlation_id = payload.get('correlation_id')
        
        # Add timeout control to avoid long blocking
        try:
            result = await asyncio.wait_for(
                backend.send(participant.endpoint, payload, correlation_id, probe_config),
                timeout=35.0  # 35 second timeout, give protocol backends enough time to process
            )
            return result
        except asyncio.TimeoutError:
            error_msg = f"Backend send timeout for {participant.protocol} to {participant.endpoint}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    # Management interfaces
    async def get_conversation_status(self) -> Dict[str, Any]:
        """Get conversation status"""
        return {
            "conversation_id": self.conversation_id,
            "participants": {
                agent_id: {
                    "protocol": p.protocol,
                    "role": p.role,
                    "verified": p.verified,
                    "joined_at": p.joined_at
                }
                for agent_id, p in self.participants.items()
            },
            "observers": [],  # Observer mechanism removed
            "message_count": len(self.message_history),
            "last_activity": self.message_history[-1].timestamp if self.message_history else None,
            "bridge_config": {
                "live_mirror_enabled": self.enable_live_mirror,
                "backfill_enabled": self.enable_backfill,
                "backfill_limit": self.backfill_limit
            }
        }
    
    async def force_refresh_directory(self):
        """Force refresh directory"""
        await self._refresh_participants()
    
    async def get_message_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get message history"""
        messages = self.message_history
        if limit:
            messages = messages[-limit:]
        
        return [
            {
                "sender_id": msg.sender_id,
                "receiver_id": msg.receiver_id,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "message_id": msg.message_id,
                "sender_role": msg.role,
                "correlation_id": msg.correlation_id
            }
            for msg in messages
        ]
    
    async def request_observer_backfill(self, observer_id: str, limit: Optional[int] = None) -> bool:
        """Request history backfill for Observer - disabled"""
        # Observer backfill removed - new S2 design does not require Observer mechanism
        return False
        
        if not self.enable_backfill:
            return False
        
        # Use specified limit or default limit
        backfill_limit = limit or self.backfill_limit
        original_limit = self.backfill_limit
        self.backfill_limit = backfill_limit
        
        try:
            await self._provide_backfill(observer_id)
            return True
        except Exception as e:
            logger.error(f"Manual backfill failed for {observer_id}: {e}")
            return False
        finally:
            self.backfill_limit = original_limit
