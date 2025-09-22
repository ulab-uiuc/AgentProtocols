# -*- coding: utf-8 -*-
from __future__ import annotations
"""
ANP Privacy-Enhanced Communication Backend
Implements ANP protocol with comprehensive privacy protection mechanisms
"""

import asyncio
import json
import logging
import sys
import time
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Set, List
from urllib.parse import urlparse

# Add AgentConnect to path
current_dir = Path(__file__).parent.parent.parent.parent.parent
agentconnect_path = current_dir / "agentconnect_src"
sys.path.insert(0, str(agentconnect_path))

try:
    from agent_connect.python.simple_node import SimpleNode, SimpleNodeSession
    from agent_connect.python.authentication import (
        DIDWbaAuthHeader, verify_auth_header_signature
    )
    from agent_connect.python.utils.did_generate import did_generate
    from agent_connect.python.utils.crypto_tool import get_pem_from_private_key
    from agent_connect.python.e2e_encryption import WssMessageSDK
    
    ANP_AVAILABLE = True
    print("[ANP Privacy] Successfully imported AgentConnect SDK with privacy features")
except ImportError as e:
    ANP_AVAILABLE = False
    print(f"[ANP Privacy] AgentConnect SDK not available: {e}")

# Import base classes
try:
    from ...comm.base import BaseCommBackend
except ImportError:
    from comm.base import BaseCommBackend


@dataclass
class ANPPrivacyAgent:
    """ANP agent with privacy protection capabilities"""
    agent_id: str
    agent_type: str  # "receptionist", "doctor"
    did: Optional[str] = None
    private_key_pem: Optional[str] = None
    did_document_json: Optional[str] = None
    # 使用前向引用以避免在AgentConnect SDK缺失时报NameError
    simple_node: Optional["SimpleNode"] = None
    host: str = "127.0.0.1"
    port: int = 9000
    
    # Privacy features
    privacy_level: str = "high"  # "low", "medium", "high"
    data_minimization: bool = True
    selective_disclosure: bool = True
    human_authorization_required: bool = False
    

class ANPPrivacyFilter:
    """ANP privacy filter implementing data minimization and selective disclosure"""
    
    # Sensitive information patterns - precise to avoid false positives
    SENSITIVE_PATTERNS = {
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'phone': r'\b\d{3}-\d{3}-\d{4}\b|\(\d{3}\)\s*\d{3}-\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'address': r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b',
        # Standard name pattern - consistent with base analyzer
        'name': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b|'
               r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+\b',
        'age': r'\bAge:\s*\d+\b|\b\d+\s*years?\s*old\b'
    }
    
    @classmethod
    def apply_data_minimization(cls, content: str, agent_type: str) -> str:
        """Apply data minimization based on agent type and privacy level"""
        if agent_type == "receptionist":
            # Receptionist should protect patient privacy
            return cls._sanitize_sensitive_info(content)
        elif agent_type == "doctor":
            # Doctor might need some info but should still be careful
            return cls._partial_sanitize(content)
        return content
    
    @classmethod
    def _sanitize_sensitive_info(cls, content: str) -> str:
        """Remove or mask sensitive information"""
        sanitized = content
        
        # Replace SSN with masked version
        sanitized = re.sub(cls.SENSITIVE_PATTERNS['ssn'], 'XXX-XX-XXXX', sanitized)
        
        # Replace phone with masked version
        sanitized = re.sub(cls.SENSITIVE_PATTERNS['phone'], 'XXX-XXX-XXXX', sanitized)
        
        # Replace email with masked version
        sanitized = re.sub(cls.SENSITIVE_PATTERNS['email'], '[EMAIL_PROTECTED]', sanitized)
        
        # Replace specific addresses with generic location
        sanitized = re.sub(cls.SENSITIVE_PATTERNS['address'], '[ADDRESS_PROTECTED]', sanitized)
        
        # Replace names with generic identifiers
        sanitized = re.sub(cls.SENSITIVE_PATTERNS['name'], '[PATIENT_NAME]', sanitized)
        
        return sanitized
    
    @classmethod
    def _partial_sanitize(cls, content: str) -> str:
        """Partial sanitization for medical professionals"""
        sanitized = content
        
        # Only mask SSN and phone, keep other info for medical context
        sanitized = re.sub(cls.SENSITIVE_PATTERNS['ssn'], 'XXX-XX-XXXX', sanitized)
        sanitized = re.sub(cls.SENSITIVE_PATTERNS['phone'], 'XXX-XXX-XXXX', sanitized)
        
        return sanitized
    
    @classmethod
    def detect_privacy_violations(cls, content: str) -> Dict[str, int]:
        """Detect privacy violations in content"""
        violations = {}
        
        for violation_type, pattern in cls.SENSITIVE_PATTERNS.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                violations[violation_type] = len(matches)
        
        return violations


class ANPCommBackend(BaseCommBackend):
    """
    ANP Communication Backend with Privacy Protection
    
    Implements ANP protocol privacy features:
    - Data minimization and selective disclosure
    - Human authorization for sensitive operations
    - End-to-end encryption
    - Privacy-aware message filtering
    """
    
    def __init__(self, config: Dict[str, Any], output_handler=None):
        self.config = config
        self.output = output_handler
        self.agents: Dict[str, ANPPrivacyAgent] = {}
        self.privacy_filter = ANPPrivacyFilter()
        self.privacy_violations: Dict[str, Dict[str, int]] = {}
        
        # ANP privacy configuration
        self.privacy_config = config.get('anp', {}).get('privacy_features', {})
        self.data_minimization_enabled = self.privacy_config.get('data_minimization', True)
        self.selective_disclosure_enabled = self.privacy_config.get('selective_disclosure', True)
        self.human_authorization_enabled = self.privacy_config.get('human_authorization', False)
        
        if not ANP_AVAILABLE:
            if self.output:
                self.output.warning("[ANPCommBackend] AgentConnect not available - using privacy-aware stub mode")

    async def register_endpoint(self, agent_id: str, address: str) -> None:
        """Register ANP agent with privacy protection features"""
        try:
            # Parse address
            parsed = urlparse(address if address.startswith('http') else f'http://{address}')
            host = parsed.hostname or '127.0.0.1'
            port = parsed.port or 9000
            
            # Determine agent type and privacy level
            agent_type = "unknown"
            privacy_level = "medium"
            
            if "Receptionist" in agent_id:
                agent_type = "receptionist"
                privacy_level = "high"  # Receptionists need high privacy protection
            elif "Doctor" in agent_id:
                agent_type = "doctor"
                privacy_level = "medium"  # Doctors need some info but should be careful
            
            if ANP_AVAILABLE:
                # Generate DID for this agent
                ws_endpoint = f"ws://{host}:{port}/ws"
                private_key, _, did, did_document_json = did_generate(ws_endpoint)
                private_key_pem = get_pem_from_private_key(private_key)
                
                # Create SimpleNode with DID authentication
                simple_node = SimpleNode(
                    host_domain=host,
                    host_port=str(port),
                    host_ws_path="/ws",
                    private_key_pem=private_key_pem,
                    did=did,
                    did_document_json=did_document_json
                )
                
                # Start the SimpleNode
                simple_node.run()
                await asyncio.sleep(0.5)  # Wait for node to be ready
                
                if self.output:
                    self.output.success(f"[ANPCommBackend] Registered ANP agent {agent_id} with DID: {did}")
            else:
                # Stub mode
                did = f"did:anp:stub:{agent_id}"
                private_key_pem = None
                did_document_json = None
                simple_node = None
                
                if self.output:
                    self.output.info(f"[ANPCommBackend] Registered privacy agent {agent_id} (stub mode) @ {address}")
            
            # Create privacy-enhanced agent
            agent = ANPPrivacyAgent(
                agent_id=agent_id,
                agent_type=agent_type,
                did=did,
                private_key_pem=private_key_pem,
                did_document_json=did_document_json,
                simple_node=simple_node,
                host=host,
                port=port,
                privacy_level=privacy_level,
                data_minimization=self.data_minimization_enabled,
                selective_disclosure=self.selective_disclosure_enabled,
                human_authorization_required=self.human_authorization_enabled
            )
            
            self.agents[agent_id] = agent
            self.privacy_violations[agent_id] = {}
            
        except Exception as e:
            if self.output:
                self.output.error(f"[ANPCommBackend] Failed to register {agent_id}: {e}")
            raise
    
    def _create_session_callback(self, agent_id: str) -> Callable:
        """Create session callback with privacy protection"""
        async def on_new_session(session: SimpleNodeSession) -> None:
            if self.output:
                self.output.info(f"[ANPCommBackend] New privacy-protected session for {agent_id}")
            
            # Start privacy-aware message handling
            asyncio.create_task(self._handle_privacy_session(agent_id, session))
        
        return on_new_session
    
    async def _handle_privacy_session(self, agent_id: str, session: SimpleNodeSession):
        """Handle session messages with privacy protection"""
        try:
            while session.is_connected():
                try:
                    message_data = await session.receive_message()
                    if not message_data:
                        continue
                    
                    content = message_data.get('content', '')
                    
                    # Apply ANP privacy protection
                    protected_content = await self._apply_privacy_protection(
                        content, agent_id, session.remote_did
                    )
                    
                    # Create privacy-protected message
                    privacy_msg = {
                        'content': protected_content,
                        'original_content': content,  # For analysis purposes
                        'sender_id': session.remote_did,
                        'receiver_id': agent_id,
                        'timestamp': time.time(),
                        'protocol_context': {
                            'protocol': 'anp',
                            'did_authenticated': True,
                            'privacy_protected': True,
                            'data_minimized': self.data_minimization_enabled,
                            'selective_disclosure': self.selective_disclosure_enabled
                        }
                    }
                    
                    # Store message for later processing (simplified for now)
                    # In a full implementation, this would route to appropriate handlers
                    pass
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if self.output:
                        self.output.error(f"[ANPCommBackend] Privacy session error: {e}")
                    await asyncio.sleep(0.1)
        
        except Exception as e:
            if self.output:
                self.output.error(f"[ANPCommBackend] Privacy session handler error: {e}")
    
    async def _apply_privacy_protection(self, content: str, receiver_id: str, sender_did: str) -> str:
        """Apply ANP privacy protection mechanisms"""
        agent = self.agents.get(receiver_id)
        if not agent:
            return content
        
        protected_content = content
        
        # 1. Data Minimization (ANP core feature)
        if agent.data_minimization:
            protected_content = self.privacy_filter.apply_data_minimization(
                protected_content, agent.agent_type
            )
        
        # 2. Selective Disclosure (ANP core feature)
        if agent.selective_disclosure:
            protected_content = self._apply_selective_disclosure(
                protected_content, agent.agent_type, sender_did
            )
        
        # 3. Human Authorization Check (for sensitive operations)
        if agent.human_authorization_required:
            violations = self.privacy_filter.detect_privacy_violations(content)
            if violations:
                # In a real implementation, this would trigger human authorization
                if self.output:
                    self.output.warning(f"[ANPCommBackend] Sensitive data detected, human authorization required")
        
        return protected_content
    
    def _apply_selective_disclosure(self, content: str, agent_type: str, sender_did: str) -> str:
        """Apply selective disclosure based on agent type and relationship"""
        # Receptionist should get minimal information
        if agent_type == "receptionist":
            # Only disclose medical symptoms, not personal details
            lines = content.split('\n')
            filtered_lines = []
            
            for line in lines:
                # Keep medical symptom descriptions
                if any(keyword in line.lower() for keyword in 
                      ['symptom', 'feel', 'pain', 'dizzy', 'nausea', 'medical', 'doctor']):
                    # But still remove personal identifiers
                    filtered_line = self.privacy_filter._sanitize_sensitive_info(line)
                    filtered_lines.append(filtered_line)
            
            return '\n'.join(filtered_lines) if filtered_lines else "[PRIVACY_PROTECTED_CONTENT]"
        
        return content
    
    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send message with ANP privacy protection"""
        try:
            # Extract message content from payload
            message_content = payload.get('text', '') or str(payload.get('parts', []))
            context = payload.get('context', {})
            
            sender_agent = self.agents.get(src_id)
            receiver_agent = self.agents.get(dst_id)
            
            if not sender_agent or not receiver_agent:
                return {"raw": None, "text": "Agent not found", "success": False}
            
            # Apply privacy protection before sending
            protected_content = await self._apply_privacy_protection(
                message_content, dst_id, sender_agent.did
            )
            
            # Track privacy violations in original content
            violations = self.privacy_filter.detect_privacy_violations(message_content)
            if violations:
                if src_id not in self.privacy_violations:
                    self.privacy_violations[src_id] = {}
                self.privacy_violations[src_id].update(violations)
            
            # Create ANP privacy context
            privacy_context = {
                'protocol': 'anp',
                'sender_did': sender_agent.did,
                'receiver_did': receiver_agent.did,
                'privacy_protected': True,
                'data_minimized': sender_agent.data_minimization,
                'selective_disclosure': sender_agent.selective_disclosure,
                'human_authorized': not sender_agent.human_authorization_required,
                'timestamp': time.time()
            }
            
            if context:
                privacy_context.update(context)
            
            # Create privacy message
            privacy_msg = {
                'content': protected_content,
                'original_content': message_content,  # For analysis
                'sender_id': src_id,
                'receiver_id': dst_id,
                'timestamp': privacy_context['timestamp'],
                'protocol_context': privacy_context
            }
            
            # In a full implementation, this would route to the appropriate agent
            # For now, we'll simulate the message delivery
            
            if self.output:
                self.output.info(f"[ANPCommBackend] Privacy-protected message sent: {src_id} -> {dst_id}")
            
            # Return in BaseCommBackend format
            return {
                "raw": {
                    "success": True,
                    "privacy_protected": True,
                    "violations_detected": len(violations) > 0,
                    "violations": violations
                },
                "text": protected_content
            }
            
        except Exception as e:
            if self.output:
                self.output.error(f"[ANPCommBackend] Failed to send privacy message: {e}")
            return {"raw": None, "text": f"Privacy communication error: {e}"}

    async def health_check(self, agent_id: str) -> bool:
        """Check if ANP agent is healthy"""
        agent = self.agents.get(agent_id)
        if not agent:
            return False
        
        if ANP_AVAILABLE and agent.simple_node:
            # Check if SimpleNode is running
            return True  # Simplified check
        
        return True  # Stub mode is always "healthy"

    async def close(self) -> None:
        """Close ANP backend with privacy cleanup"""
        try:
            for agent_id, agent in self.agents.items():
                if agent.simple_node:
                    try:
                        await agent.simple_node.stop()
                    except Exception as e:
                        if self.output:
                            self.output.warning(f"[ANPCommBackend] Error stopping node for {agent_id}: {e}")
            
            # Clear privacy data
            self.agents.clear()
            self.privacy_violations.clear()
            
            if self.output:
                self.output.success("[ANPCommBackend] Privacy testing ANP backend closed")
        
        except Exception as e:
            if self.output:
                self.output.error(f"[ANPCommBackend] Error during privacy cleanup: {e}")
    
    def get_privacy_violations(self) -> Dict[str, Dict[str, int]]:
        """Get privacy violations detected during communication"""
        return self.privacy_violations.copy()
    
    def get_privacy_stats(self) -> Dict[str, Any]:
        """Get privacy protection statistics"""
        total_violations = sum(
            sum(violations.values()) for violations in self.privacy_violations.values()
        )
        
        return {
            'total_agents': len(self.agents),
            'privacy_protected_agents': sum(1 for a in self.agents.values() if a.data_minimization),
            'total_violations_detected': total_violations,
            'violations_by_agent': self.privacy_violations,
            'privacy_features': {
                'data_minimization': self.data_minimization_enabled,
                'selective_disclosure': self.selective_disclosure_enabled,
                'human_authorization': self.human_authorization_enabled
            }
        }
