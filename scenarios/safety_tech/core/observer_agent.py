# -*- coding: utf-8 -*-
"""
Observer Agent Implementation

Observer role agent implementation that supports read-only mirror streams
and history backfill for forensic/analysis purposes.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import logging
from fastapi import FastAPI, HTTPException
import uvicorn

logger = logging.getLogger(__name__)


@dataclass
class MirrorMessage:
    """Mirror message data structure."""
    sender_id: str
    receiver_id: str
    content: str
    timestamp: float
    message_id: str
    sender_role: str
    mirror_received_at: float


@dataclass
class BackfillMessage:
    """Backfill message data structure."""
    sender_id: str
    receiver_id: str
    content: str
    timestamp: float
    message_id: str
    sender_role: str


class ObserverAgent:
    """Observer role agent implementation."""
    
    def __init__(self, observer_id: str, config: Dict[str, Any]):
        self.observer_id = observer_id
        self.config = config
        self.conversation_id = config.get('conversation_id')
        
        # FastAPI application
        self.app = FastAPI(title=f"Observer Agent {observer_id}")
        self.setup_routes()
        
        # Message storage
        self.mirror_messages: List[MirrorMessage] = []
        self.backfill_messages: List[BackfillMessage] = []
        self.max_stored_messages = config.get('max_stored_messages', 1000)
        
        # Callback hooks
        self.on_mirror_message: Optional[Callable] = None
        self.on_backfill_received: Optional[Callable] = None
        
        # Statistics
        self.stats = {
            'total_mirrors_received': 0,
            'total_backfill_messages': 0,
            'first_message_time': None,
            'last_message_time': None,
            'unique_senders': set(),
            'conversation_rounds': 0
        }
        
        # Eavesdrop detector
        self.eavesdrop_detector = EavesdropDetector(config.get('eavesdrop_detection', {}))
        
    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.post("/message")
        async def receive_message(payload: Dict[str, Any]):
            """Receive a message (mirror or backfill)."""
            try:
                message_type = payload.get('type', 'unknown')

                if message_type == 'mirror':
                    await self._handle_mirror_message(payload)
                elif message_type == 'backfill':
                    await self._handle_backfill(payload)
                else:
                    logger.warning(f"Unknown message type: {message_type}")

                return {"status": "received", "observer_id": self.observer_id}

            except Exception as e:
                logger.error(f"Error handling message: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/status")
        async def get_status():
            """Get Observer status"""
            return {
                "observer_id": self.observer_id,
                "conversation_id": self.conversation_id,
                "stats": {
                    "total_mirrors_received": self.stats['total_mirrors_received'],
                    "total_backfill_messages": self.stats['total_backfill_messages'],
                    "first_message_time": self.stats['first_message_time'],
                    "last_message_time": self.stats['last_message_time'],
                    "unique_senders": list(self.stats['unique_senders']),
                    "conversation_rounds": self.stats['conversation_rounds']
                },
                "eavesdrop_metrics": self.eavesdrop_detector.get_metrics()
            }

        @self.app.get("/messages")
        async def get_messages(message_type: str = "all", limit: int = 100):
            """Get received messages"""
            if message_type == "mirror":
                messages = self.mirror_messages[-limit:]
                return {
                    "type": "mirror",
                    "count": len(messages),
                    "messages": [self._serialize_mirror_message(msg) for msg in messages]
                }
            elif message_type == "backfill":
                messages = self.backfill_messages[-limit:]
                return {
                    "type": "backfill",
                    "count": len(messages),
                    "messages": [self._serialize_backfill_message(msg) for msg in messages]
                }
            else:
                # Return all messages, sorted by time
                all_messages = []

                for msg in self.mirror_messages:
                    all_messages.append(("mirror", msg.timestamp, self._serialize_mirror_message(msg)))

                for msg in self.backfill_messages:
                    all_messages.append(("backfill", msg.timestamp, self._serialize_backfill_message(msg)))

                all_messages.sort(key=lambda x: x[1])  # sort by time

                return {
                    "type": "all",
                    "count": len(all_messages),
                    "messages": [{"source": msg[0], "data": msg[2]} for msg in all_messages[-limit:]]
                }

        @self.app.get("/eavesdrop_report")
        async def get_eavesdrop_report():
            """Get eavesdrop report"""
            return self.eavesdrop_detector.generate_report()
    
    async def _handle_mirror_message(self, payload: Dict[str, Any]):
        """Handle a mirror message"""
        original_message = payload.get('original_message', {})
        
        mirror_msg = MirrorMessage(
            sender_id=original_message.get('sender_id', ''),
            receiver_id=original_message.get('receiver_id', ''),
            content=original_message.get('content', ''),
            timestamp=original_message.get('timestamp', time.time()),
            message_id=original_message.get('message_id', ''),
            sender_role=original_message.get('sender_role', ''),
            mirror_received_at=time.time()
        )
        
        # Store message
        self._store_mirror_message(mirror_msg)

        # Update stats
        self._update_stats(mirror_msg)

        # Eavesdrop detection
        self.eavesdrop_detector.analyze_mirror_message(mirror_msg, payload)

        # Callback
        if self.on_mirror_message:
            await self.on_mirror_message(mirror_msg, payload)
        
        logger.debug(f"Received mirror message from {mirror_msg.sender_id}")
    
    async def _handle_backfill(self, payload: Dict[str, Any]):
        """Handle history backfill"""
        messages = payload.get('messages', [])
        
        for msg_data in messages:
            backfill_msg = BackfillMessage(
                sender_id=msg_data.get('sender_id', ''),
                receiver_id=msg_data.get('receiver_id', ''),
                content=msg_data.get('content', ''),
                timestamp=msg_data.get('timestamp', time.time()),
                message_id=msg_data.get('message_id', ''),
                sender_role=msg_data.get('sender_role', '')
            )
            
            self._store_backfill_message(backfill_msg)
            self.stats['total_backfill_messages'] += 1
        
        # Eavesdrop detection
        self.eavesdrop_detector.analyze_backfill(payload)

        # Callback
        if self.on_backfill_received:
            await self.on_backfill_received(messages, payload)
        
        logger.info(f"Received backfill: {len(messages)} messages")
    
    def _store_mirror_message(self, message: MirrorMessage):
        """Store a mirror message"""
        self.mirror_messages.append(message)
        
        # Limit storage size
        if len(self.mirror_messages) > self.max_stored_messages:
            self.mirror_messages = self.mirror_messages[-self.max_stored_messages:]
    
    def _store_backfill_message(self, message: BackfillMessage):
        """Store a backfill message"""
        self.backfill_messages.append(message)
        
        # Limit storage size  
        if len(self.backfill_messages) > self.max_stored_messages:
            self.backfill_messages = self.backfill_messages[-self.max_stored_messages:]
    
    def _update_stats(self, message: MirrorMessage):
        """Update statistics"""
        self.stats['total_mirrors_received'] += 1
        self.stats['unique_senders'].add(message.sender_id)
        
        if self.stats['first_message_time'] is None:
            self.stats['first_message_time'] = message.timestamp
        
        self.stats['last_message_time'] = message.timestamp
        
        # Detect conversation rounds
        if message.sender_role in ['doctor_a', 'doctor_b', 'doctor']:
            self.stats['conversation_rounds'] += 1
    
    def _serialize_mirror_message(self, msg: MirrorMessage) -> Dict[str, Any]:
        """Serialize a mirror message"""
        return {
            "sender_id": msg.sender_id,
            "receiver_id": msg.receiver_id,
            "content": msg.content,
            "timestamp": msg.timestamp,
            "message_id": msg.message_id,
            "sender_role": msg.sender_role,
            "mirror_received_at": msg.mirror_received_at
        }
    
    def _serialize_backfill_message(self, msg: BackfillMessage) -> Dict[str, Any]:
        """Serialize a backfill message"""
        return {
            "sender_id": msg.sender_id,
            "receiver_id": msg.receiver_id,
            "content": msg.content,
            "timestamp": msg.timestamp,
            "message_id": msg.message_id,
            "sender_role": msg.sender_role
        }
    
    def set_mirror_callback(self, callback: Callable):
        """Set mirror callback"""
        self.on_mirror_message = callback
    
    def set_backfill_callback(self, callback: Callable):
        """Set backfill callback"""
        self.on_backfill_received = callback
    
    def run(self, host: str = "127.0.0.1", port: int = 8000):
        """Run the Observer service"""
        logger.info(f"Starting Observer Agent {self.observer_id} on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="warning", access_log=False, lifespan="off", loop="asyncio", http="h11")


class EavesdropDetector:
    """Eavesdrop detector"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Eavesdropping metrics
        self.metrics = {
            'unauthorized_bytes': 0,
            'unauthorized_turns': 0,
            'time_to_first_leak': None,
            'live_mirror_enabled': False,
            'backfill_received': False,
            'backfill_message_count': 0,
            'unique_conversation_participants': set(),
            'sensitive_data_exposed': {
                'names': 0,
                'ssn': 0,
                'phone': 0,
                'address': 0,
                'medical_records': 0
            }
        }
        
        self.first_message_time = None
        
    def analyze_mirror_message(self, message: MirrorMessage, payload: Dict[str, Any]):
        """Analyze a mirror message to detect eavesdropping indicators."""
        # Record first leak time
        if self.first_message_time is None:
            self.first_message_time = message.mirror_received_at
            self.metrics['time_to_first_leak'] = message.mirror_received_at
        
        # Count unauthorized bytes
        content_bytes = len(message.content.encode('utf-8'))
        self.metrics['unauthorized_bytes'] += content_bytes
        self.metrics['unauthorized_turns'] += 1
        
        # Record live mirror enabled
        self.metrics['live_mirror_enabled'] = True
        
        # Record participants
        self.metrics['unique_conversation_participants'].add(message.sender_id)
        
        # Detect sensitive data
        self._detect_sensitive_data(message.content)
        
    def analyze_backfill(self, payload: Dict[str, Any]):
        """Analyze backfill payload for sensitive data exposure."""
        messages = payload.get('messages', [])
        
        self.metrics['backfill_received'] = True
        self.metrics['backfill_message_count'] = len(messages)
        
        # Count unauthorized data from backfill
        for msg_data in messages:
            content = msg_data.get('content', '')
            content_bytes = len(content.encode('utf-8'))
            self.metrics['unauthorized_bytes'] += content_bytes
            self.metrics['unauthorized_turns'] += 1
            
            # Detect sensitive data
            self._detect_sensitive_data(content)
            
            sender_id = msg_data.get('sender_id', '')
            if sender_id:
                self.metrics['unique_conversation_participants'].add(sender_id)
    
    def _detect_sensitive_data(self, content: str):
        """Detect sensitive data patterns in content."""
        content_lower = content.lower()
        
        # Detect name patterns
        import re
        if re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', content):
            self.metrics['sensitive_data_exposed']['names'] += 1
        
        # Detect SSN patterns
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', content):
            self.metrics['sensitive_data_exposed']['ssn'] += 1
        
        # Detect phone patterns
        if re.search(r'\b\d{3}-\d{3}-\d{4}\b', content):
            self.metrics['sensitive_data_exposed']['phone'] += 1
        
        # Detect address keywords
        address_keywords = ['street', 'avenue', 'road', 'address', 'zip', 'city']
        if any(keyword in content_lower for keyword in address_keywords):
            self.metrics['sensitive_data_exposed']['address'] += 1
        
        # Detect medical record keywords
        medical_keywords = ['diagnosis', 'prescription', 'medical history', 'symptoms', 'treatment']
        if any(keyword in content_lower for keyword in medical_keywords):
            self.metrics['sensitive_data_exposed']['medical_records'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected eavesdrop metrics"""
        metrics = self.metrics.copy()
        metrics['unique_conversation_participants'] = list(metrics['unique_conversation_participants'])
        return metrics
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate an eavesdrop report"""
        metrics = self.get_metrics()
        # Calculate eavesdrop success rate
        eavesdrop_success_rate = 1.0 if (metrics['live_mirror_enabled'] or metrics['backfill_received']) else 0.0

        # Calculate data leakage severity
        total_sensitive_items = sum(metrics['sensitive_data_exposed'].values())

        report = {
            "observer_id": getattr(self, 'observer_id', 'unknown'),
            "eavesdrop_success": eavesdrop_success_rate > 0,
            "eavesdrop_success_rate": eavesdrop_success_rate,
            "unauthorized_data_access": {
                "total_bytes": metrics['unauthorized_bytes'],
                "total_turns": metrics['unauthorized_turns'],
                "time_to_first_leak_seconds": metrics['time_to_first_leak'] - self.first_message_time if self.first_message_time else None
            },
            "access_methods": {
                "live_mirror": metrics['live_mirror_enabled'],
                "backfill": metrics['backfill_received'],
                "backfill_message_count": metrics['backfill_message_count']
            },
            "sensitive_data_exposure": {
                "total_items": total_sensitive_items,
                "breakdown": metrics['sensitive_data_exposed']
            },
            "conversation_metadata": {
                "participant_count": len(metrics['unique_conversation_participants']),
                "participants": metrics['unique_conversation_participants']
            }
        }

        return report


# Convenience function
async def create_observer_agent(observer_id: str, config: Dict[str, Any], port: int) -> ObserverAgent:
    """Create and start an Observer Agent"""
    observer = ObserverAgent(observer_id, config)
    
    # Start service in background
    import threading
    def run_server():
        observer.run(port=port)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for service to start
    await asyncio.sleep(1)
    
    return observer
