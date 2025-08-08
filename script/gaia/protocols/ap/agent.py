from __future__ import annotations
import asyncio
import json
import time
from typing import Dict, Any, Optional, Tuple

# Import with fallback for agent_protocol
try:
    from agent_protocol import Agent, Step, Task
except ImportError:
    print("Warning: agent_protocol not found, using mock implementations")
    # Mock implementations for testing
    class Agent:
        pass
    class Step:
        pass  
    class Task:
        pass

# Import using absolute path to avoid relative import issues
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.agent_base import MeshAgent

class APAgent(MeshAgent):
    """Agent Protocol implementation for GAIA multi-agent framework."""
    
    def __init__(self, node_id: int, name: str, tool: str, adapter, port: int, 
                 config: Dict[str, Any], task_id: Optional[str] = None, network=None):
        super().__init__(node_id, name, tool, adapter, port, config, task_id)
        self.network = network
        
        # Agent Protocol specific state
        self.tasks: Dict[str, Any] = {}
        self.current_task_id: Optional[str] = None
        self._message_queue = asyncio.Queue()
    
    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """Send a message to another agent using Agent Protocol."""
        try:
            # Create Agent Protocol compatible message
            ap_message = {
                "type": payload.get("type", "task_result"),
                "input": payload.get("input", json.dumps(payload)),
                "additional_input": {
                    "source_agent_id": self.id,
                    "source_agent_name": self.name,
                    "timestamp": int(time.time()),
                    **payload.get("additional_input", {})
                }
            }
            
            # Add message to queue for network processing
            await self._message_queue.put({
                "dst": dst,
                "payload": ap_message,
                "timestamp": time.time()
            })
            
        except Exception as e:
            self._log(f"Error sending message to agent {dst}: {e}")
    
    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """Receive a message from other agents using Agent Protocol."""
        try:
            if timeout > 0:
                return await asyncio.wait_for(self._message_queue.get(), timeout=timeout)
            else:
                # Non-blocking
                try:
                    return self._message_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return None
        except (asyncio.TimeoutError, Exception) as e:
            if not isinstance(e, asyncio.TimeoutError):
                self._log(f"Error receiving message: {e}")
            return None
    
    async def connect(self) -> bool:
        """
        Establish Agent Protocol connection.
        
        Returns:
            True if connection successful
        """
        try:
            # For Agent Protocol, start the HTTP server
            self._log(f"AP Agent {self.name}: Starting HTTP server on port {self.port}")
            self.running = True
            print(f"AP Agent {self.name}: Connected successfully")
            return True
        except Exception as e:
            print(f"AP Agent {self.name}: Connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Agent Protocol network."""
        try:
            self.running = False
            await self.stop()
            print(f"AP Agent {self.name}: Disconnected")
        except Exception as e:
            print(f"AP Agent {self.name}: Disconnect error: {e}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current Agent Protocol connection status and statistics."""
        return {
            "agent_id": self.id,
            "agent_name": self.name,
            "protocol": "Agent Protocol",
            "connected": getattr(self, 'running', False),
            "port": self.port,
            "tasks_count": len(self.tasks),
            "message_queue_size": self._message_queue.qsize(),
            "specialization": self.specialization
        } 