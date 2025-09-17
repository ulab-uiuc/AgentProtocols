"""
Dummy Protocol Agent Implementation for GAIA Framework.
This agent simulates network communication for testing purposes.
"""

import asyncio
import json
import random
import time
from typing import Dict, Any, Optional
import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.agent import MeshAgent


class DummyClient:
    """
    Dummy client that simulates network communication with configurable parameters.
    """
    
    def __init__(self, router_url: str, agent_id: str):
        self.router_url = router_url
        self.agent_id = agent_id
        self._connected = False
        
        # Simulation parameters
        self.message_loss_rate = 0.05  # 5% message loss
        self.delivery_delay = 0.1  # 100ms network delay
        self.max_message_size = 1024 * 1024 * 1024  # 1GB max message size
        
        # Statistics
        self.sent_messages = 0
        self.received_messages = 0
        self.dropped_messages = 0
        
        # Global shared mailboxes for agent communication
        if not hasattr(DummyClient, '_global_mailboxes'):
            DummyClient._global_mailboxes = {}
        
        # Create mailbox for this agent
        if agent_id not in DummyClient._global_mailboxes:
            DummyClient._global_mailboxes[agent_id] = asyncio.Queue()
        
        self._mailbox = DummyClient._global_mailboxes[agent_id]
        self._next_message_id = 0
        
        print(f"[{agent_id}] DummyClient initialized for router: {router_url}")
    
    async def connect(self):
        """Initialize dummy connection (immediate success for simulation)"""
        try:
            self._connected = True
            print(f"[{self.agent_id}] DummyClient connected to: {self.router_url}")
        except Exception as e:
            print(f"[{self.agent_id}] Failed to connect DummyClient: {e}")
            raise ConnectionError(f"Cannot connect DummyClient to {self.router_url}") from e
    
    async def disconnect(self):
        """Disconnect from dummy system"""
        self._connected = False
        print(f"[{self.agent_id}] DummyClient disconnected")
    
    async def send_message(self, dst_id: str, message: Dict[str, Any]) -> bool:
        """
        Send message via dummy protocol with simulation features.
        
        Args:
            dst_id: Destination agent ID
            message: Message content
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        try:
            # Simulate message size check
            message_size = len(json.dumps(message).encode('utf-8'))
            if message_size > self.max_message_size:
                print(f"DummyClient {self.agent_id}: Message too large ({message_size} bytes)")
                self.dropped_messages += 1
                return False
            
            # Simulate message loss
            if random.random() < self.message_loss_rate:
                print(f"DummyClient {self.agent_id}: Message to {dst_id} lost (simulated)")
                self.dropped_messages += 1
                return False
            
            # Prepare message with metadata
            self._next_message_id += 1
            enhanced_message = {
                **message,
                "_meta": {
                    "sender_id": self.agent_id,
                    "recipient_id": dst_id,
                    "message_id": self._next_message_id,
                    "timestamp": time.time(),
                    "protocol": "dummy"
                }
            }
            
            # Create target mailbox if it doesn't exist
            if dst_id not in DummyClient._global_mailboxes:
                DummyClient._global_mailboxes[dst_id] = asyncio.Queue()
            
            # Simulate network delay
            if self.delivery_delay > 0:
                await asyncio.sleep(self.delivery_delay)
            
            # Deliver message
            await DummyClient._global_mailboxes[dst_id].put(enhanced_message)
            self.sent_messages += 1
            
            print(f"DummyClient {self.agent_id}: Sent message to {dst_id} (ID: {self._next_message_id})")
            return True
            
        except Exception as e:
            print(f"DummyClient {self.agent_id}: Failed to send message to {dst_id}: {e}")
            self.dropped_messages += 1
            return False
    
    async def receive_message(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Receive message from dummy protocol with timeout.
        
        Args:
            timeout: Maximum wait time in seconds (0 = non-blocking)
            
        Returns:
            Received message or None if timeout
        """
        try:
            if timeout == 0.0:
                # Non-blocking receive
                if self._mailbox.empty():
                    return None
                message = self._mailbox.get_nowait()
            else:
                # Blocking receive with timeout
                message = await asyncio.wait_for(self._mailbox.get(), timeout=timeout)
            
            self.received_messages += 1
            
            # Remove metadata before returning
            if isinstance(message, dict) and "_meta" in message:
                meta = message["_meta"]
                clean_message = {k: v for k, v in message.items() if k != "_meta"}
                print(f"DummyClient {self.agent_id}: Received message from {meta.get('sender_id')} (ID: {meta.get('message_id')})")
                return clean_message
            else:
                return message
                
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            print(f"DummyClient {self.agent_id}: Failed to receive message: {e}")
            return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get communication statistics."""
        return {
            "sent_messages": self.sent_messages,
            "received_messages": self.received_messages,
            "dropped_messages": self.dropped_messages
        }


class DummyAgent(MeshAgent):
    """
    Dummy Protocol Agent that inherits from MeshAgent and implements dummy communication.
    """
    
    def __init__(self, node_id: int, name: str, tool: str, port: int, 
                 config: Dict[str, Any], task_id: Optional[str] = None, 
                 router_url: str = "dummy://localhost:8000"):
        """
        Initialize DummyAgent.
        
        Args:
            node_id: Unique agent identifier
            name: Human-readable agent name
            tool: Tool name
            port: Listening port (not used in dummy protocol)
            config: Configuration dictionary
            task_id: Optional task identifier
            router_url: Dummy router URL for simulation
        """
        super().__init__(node_id, name, tool, port, config, task_id)
        
        # Get router_url from config
        self._router_url = router_url
        self._agent_id = str(node_id)  # Convert to string for dummy client (private field)
        
        # Validate router URL
        if not self._router_url:
            raise ValueError("router_url is required for Dummy communication")
        
        # Initialize dummy client
        self._client = DummyClient(self._router_url, self._agent_id)
        self._connected = False
        # Expose a conventional endpoint for health probing (dummy scheme)
        self._endpoint = f"{self._router_url}/health_check"
        
        # Dummy protocol specific settings (use private variables to avoid Pydantic field conflicts)
        self._message_loss_rate = config.get("message_loss_rate", 0.05)
        self._delivery_delay = config.get("delivery_delay", 0.1)
        self._max_message_size = config.get("max_message_size", 1024 * 1024)
        
        # Update client settings
        self._client.message_loss_rate = self._message_loss_rate
        self._client.delivery_delay = self._delivery_delay
        self._client.max_message_size = self._max_message_size
        
        self._log(f"DummyAgent initialized with router: {self._router_url}")
    
    async def connect(self):
        """Initialize dummy connection"""
        if self._client and not self._connected:
            try:
                await self._client.connect()
                self._connected = True
                self._log(f"DummyAgent connected successfully")
            except Exception as e:
                self._log(f"Failed to connect DummyAgent: {e}")
                raise ConnectionError(f"Cannot connect DummyAgent") from e
    
    async def disconnect(self):
        """Disconnect from dummy system"""
        if self._client:
            await self._client.disconnect()
        
        self._connected = False
        self._log("DummyAgent disconnected")
    
    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """
        Concrete implementation of send_msg using dummy protocol.
        
        Args:
            dst: Destination agent ID
            payload: Message content
        """
        try:
            if not self._connected:
                await self.connect()
            
            # Convert dst to string for dummy client
            dst_str = str(dst)
            
            # Send message via dummy client
            success = await self._client.send_message(dst_str, payload)
            
            if success:
                self._log(f"Successfully sent message to agent {dst}: {payload.get('type', 'unknown')}")
            else:
                self._log(f"Failed to send message to agent {dst} (dropped or lost)")
                
        except Exception as e:
            self._log(f"Error sending message to agent {dst}: {e}")
    
    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Concrete implementation of recv_msg using dummy protocol.
        
        Args:
            timeout: Maximum wait time in seconds (0 = non-blocking)
            
        Returns:
            Received message or None if timeout
        """
        try:
            if not self._connected:
                await self.connect()
            
            # Receive message via dummy client
            message = await self._client.receive_message(timeout)
            
            if message:
                self._log(f"Received message: {message.get('type', 'unknown')}")
                return message
            else:
                return None
                
        except Exception as e:
            self._log(f"Error receiving message: {e}")
            return None
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get dummy connection status and statistics."""
        stats = self._client.get_stats() if self._client else {}
        
        return {
            "agent_id": self.id,
            "agent_name": self.name,
            "protocol": "dummy",
            "connected": self._connected,
            "router_url": self._router_url,
            "endpoint": getattr(self, "_endpoint", None),
            "message_loss_rate": self._message_loss_rate,
            "delivery_delay": self._delivery_delay,
            "max_message_size": self._max_message_size,
            "statistics": stats
        }
    
    async def start(self):
        """Override start to include connection setup."""
        # Connect before starting the main loop
        await self.connect()
        
        # Call parent start method
        await super().start()
    
    async def stop(self):
        """Override stop to include disconnection."""
        # Disconnect before stopping
        await self.disconnect()
        
        # Call parent stop method
        await super().stop()

    async def health_check(self) -> bool:
        """Health check for DummyAgent using router/endpoint semantics.
        - endpoint is set to `<dummy-router>/health_check`
        - For dummy protocol, treat connectivity and mailbox availability as health
        - Note: does NOT consider token usage
        """
        try:
            # Ensure client exists
            if not hasattr(self, "_client") or self._client is None:
                self._log("DummyClient not initialized")
                return False

            # Ensure connection
            if not getattr(self, "_connected", False):
                try:
                    await self._client.connect()
                    self._connected = True
                except Exception as e:
                    self._log(f"Health check connect failed: {e}")
                    return False

            # Ensure mailbox exists for this agent (best-effort)
            try:
                if self._agent_id not in getattr(DummyClient, "_global_mailboxes", {}):
                    DummyClient._global_mailboxes[self._agent_id] = asyncio.Queue()
            except Exception:
                # Not fatal for dummy health
                pass

            # For dummy, if connected and mailbox present, consider healthy
            return True
        except Exception as e:
            self._log(f"Health check error: {e}")
            return False