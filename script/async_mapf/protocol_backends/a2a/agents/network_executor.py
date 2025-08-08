import asyncio
import json
import inspect
import time
from typing import Dict, Any

# A2A SDK imports
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.client import A2AClient
from a2a.types import SendMessageRequest
import httpx
import uuid

# Import NetworkBase
from script.async_mapf.core.network_base import NetworkBase
from script.async_mapf.core.types import MoveCmd, MoveFeedback

# Safe wrapper for EventQueue.enqueue_event (same as used in shard_qa)
async def safe_enqueue_event(event_queue, event):
    """Safely enqueue event, handling both sync and async EventQueue implementations."""
    try:
        result = event_queue.enqueue_event(event)
        if inspect.isawaitable(result):
            return await result
        return result
    except Exception as e:
        print(f"Error enqueuing event: {e}")
        return None

class NetworkBaseExecutor(AgentExecutor):
    """A2A-SDK compatible executor for NetworkBase coordinator."""

    def __init__(self, network_base: NetworkBase, agent_urls: dict = None, output=None):
        self.network_base = network_base
        self.agent_urls = agent_urls or {}  # agent_id -> A2A URL mapping
        self.output = output
        self.a2a_clients = {}  # agent_id -> A2AClient mapping
        
        # Initialize A2A clients for each agent
        self._init_agent_clients()
        
        # Set up A2A callback for NetworkBase
        if self.network_base:
            self.network_base.set_a2a_send_callback(self.send_to_agent)
        
        if self.output:
            self.output.info("NetworkBase A2A Executor initialized")

    def _init_agent_clients(self):
        """Initialize A2A clients for each agent"""
        for agent_id, url in self.agent_urls.items():
            try:
                httpx_client = httpx.AsyncClient()
                a2a_client = A2AClient(httpx_client, url=url)
                self.a2a_clients[agent_id] = a2a_client
                if self.output:
                    self.output.info(f"Initialized A2A client for agent {agent_id}: {url} (Router-based, will use /api/jsonrpc)")
            except Exception as e:
                if self.output:
                    self.output.error(f"Failed to init A2A client for agent {agent_id}: {e}")
        
        # Add special mapping for "network-base" to route back to agents 
        if self.output:
            self.output.info(f"NetworkBase executor ready to handle MOVE_REQUEST messages from agents")

    async def send_to_agent(self, agent_id: int, message_data: dict):
        """Send A2A message to specific agent via Router"""
        if agent_id not in self.a2a_clients:
            if self.output:
                self.output.warning(f"No A2A client for agent {agent_id}")
            return

        try:
            # 🔧 FIX: Put CONTROL info in message text since A2A SDK doesn't support meta parameter
            if message_data.get("type") == "CONTROL":
                # For CONTROL messages, serialize the entire message_data as JSON in text
                message_text = json.dumps(message_data)
                
                if self.output:
                    self.output.info(f"🚀 Sending CONTROL message to agent {agent_id}")
                    self.output.info(f"   Serialized payload: {message_text}")
                    
                try:
                    a2a_message = new_agent_text_message(message_text)
                    if self.output:
                        self.output.success(f"   ✅ A2A message created successfully")
                except Exception as msg_error:
                    if self.output:
                        self.output.error(f"   ❌ Failed to create A2A message: {msg_error}")
                    raise
                    
            else:
                # For other messages, serialize as JSON
                message_text = json.dumps(message_data)
                a2a_message = new_agent_text_message(message_text)
            
            # Create SendMessageRequest with receiver_id for proper routing
            request = SendMessageRequest(
                id=str(uuid.uuid4()),
                jsonrpc="2.0",
                method="message/send",
                params={
                    "message": a2a_message,
                    "receiver_id": str(agent_id)  # Add receiver_id for Router routing
                }
            )
            
            if self.output:
                self.output.info(f"   Created SendMessageRequest with receiver_id: {agent_id}")
            
            # Send via A2A Router (all clients now point to Router)
            client = self.a2a_clients[agent_id]
            response = await client.send_message(request)
            
            if self.output:
                self.output.success(f"✅ Sent A2A message via Router to agent {agent_id}: {message_data.get('type', 'unknown')}")
                self.output.info(f"   Response status: {getattr(response, 'status', 'unknown')}")
                
        except Exception as e:
            if self.output:
                self.output.error(f"❌ Failed to send A2A message to agent {agent_id}: {e}")
                self.output.error(f"   Message data: {message_data}")
                self.output.error(f"   Error type: {type(e).__name__}")
            # Don't re-raise to avoid stopping the flow, just log the error

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        Handle incoming A2A messages for NetworkBase.
        This receives messages from MAPF agents.
        """
        user_input = context.get_user_input()
        
        if self.output:
            self.output.progress(f"[NetworkBase] Received A2A message: {user_input[:100]}...")
        
        try:
            # Parse the incoming A2A message
            if user_input:
                try:
                    # Try to parse as JSON (our serialized message format)
                    message_data = json.loads(user_input)
                    
                    if isinstance(message_data, dict):
                        await self._handle_structured_message(message_data, event_queue)
                    else:
                        await self._handle_text_message(user_input, event_queue)
                        
                except json.JSONDecodeError:
                    # Handle as plain text message
                    await self._handle_text_message(user_input, event_queue)
            else:
                result = "[NetworkBase] No input received"
                await safe_enqueue_event(event_queue, new_agent_text_message(result))
                
        except Exception as e:
            error_msg = f"[NetworkBase] Error processing message: {e}"
            if self.output:
                self.output.error(error_msg)
            await safe_enqueue_event(event_queue, new_agent_text_message(error_msg))

    async def _handle_structured_message(self, message_data: dict, event_queue: EventQueue):
        """Handle structured JSON message from agents"""
        msg_type = message_data.get("type")
        payload = message_data.get("payload", {})
        
        if msg_type == "MOVE_REQUEST":
            # Handle concurrent move request from agent
            try:
                move_payload = payload
                agent_id = move_payload.get("agent_id")
                new_pos = tuple(move_payload.get("new_pos", [0, 0]))
                eta_ms = move_payload.get("eta_ms", 100)
                time_window_ms = move_payload.get("time_window_ms", 50)
                move_id = move_payload.get("move_id", "")
                priority = move_payload.get("priority", 1)
                
                # Create concurrent move command
                from script.async_mapf.core.concurrent_types import ConcurrentMoveCmd
                concurrent_move = ConcurrentMoveCmd(
                    agent_id=agent_id,
                    new_pos=new_pos,
                    eta_ms=eta_ms,
                    time_window_ms=time_window_ms,
                    move_id=move_id,
                    priority=priority
                )
                
                # Check bounds first (unified validation in NetworkBase)
                grid_size = self.network_base.grid_size
                if not (0 <= new_pos[0] < grid_size and 0 <= new_pos[1] < grid_size):
                    # Out of bounds - return immediate rejection
                    response_data = {
                        "type": "MOVE_RESPONSE", 
                        "payload": {
                            "agent_id": agent_id,
                            "move_id": move_id,
                            "status": "REJECT",
                            "reason": f"Move out of bounds: {new_pos}. Grid size is {grid_size}x{grid_size}.",
                            "conflicting_agents": [],
                            "suggested_eta_ms": None
                        }
                    }
                else:
                    # Valid bounds - use built-in conflict detection instead of conflict_manager
                    exec_ts = int(time.time() * 1000)
                    
                    # 🔧 CRITICAL FIX: Add concurrent safety lock for grid operations
                    if not hasattr(self.network_base, '_move_lock'):
                        import asyncio
                        self.network_base._move_lock = asyncio.Lock()
                    
                    async with self.network_base._move_lock:
                        success, conflicting_agents = self.network_base._apply_move_concurrent(concurrent_move, exec_ts)
                    
                    # Create response manually  
                    if success:
                        response_status = "OK"
                        reason = "Move successful"
                    else:
                        response_status = "CONFLICT" if conflicting_agents else "REJECT"
                        reason = f"Conflict with agents: {conflicting_agents}" if conflicting_agents else "Move failed"
                    
                    # Send response back to agent
                    response_data = {
                        "type": "MOVE_RESPONSE",
                        "payload": {
                            "agent_id": agent_id,
                            "move_id": move_id,
                            "status": response_status,
                            "reason": reason,
                            "conflicting_agents": conflicting_agents,
                            "suggested_eta_ms": 100 if conflicting_agents else None  # 🔧 FIX: Use relative delay, not absolute timestamp
                        }
                    }
                
                # Send response back to requesting agent via A2A
                status_str = response_data["payload"]["status"]
                if self.output:
                    self.output.progress(f"[NetworkBase] Move request from agent {agent_id}: {status_str}")
                
                try:
                    # Send response back to the agent
                    await self.send_to_agent(agent_id, response_data)
                except Exception as send_error:
                    if self.output:
                        self.output.error(f"Failed to send response to agent {agent_id}: {send_error}")
                
                result = f"[NetworkBase] Processed concurrent move request from agent {agent_id}: {status_str}"
                await safe_enqueue_event(event_queue, new_agent_text_message(result))
                
            except Exception as e:
                error_msg = f"[NetworkBase] Error processing MOVE_REQUEST: {e}"
                if self.output:
                    self.output.error(error_msg)
                
                # 🔧 CRITICAL FIX: Always send MOVE_RESPONSE even when error occurs
                # Extract basic info from the exception context
                try:
                    # Try to extract agent_id and move_id from the original payload
                    agent_id = payload.get("agent_id", 0)
                    move_id = payload.get("move_id", "")
                    
                    # Send error response to prevent Agent from waiting forever
                    error_response = {
                        "type": "MOVE_RESPONSE",
                        "payload": {
                            "agent_id": agent_id,
                            "move_id": move_id,
                            "status": "REJECT",
                            "reason": f"Network error: {str(e)}",
                            "conflicting_agents": [],
                            "suggested_eta_ms": None
                        }
                    }
                    
                    await self.send_to_agent(agent_id, error_response)
                    
                    if self.output:
                        self.output.warning(f"Sent error response to agent {agent_id}")
                        
                except Exception as send_error:
                    if self.output:
                        self.output.error(f"Failed to send error response: {send_error}")
                
                await safe_enqueue_event(event_queue, new_agent_text_message(error_msg))
                
        elif msg_type == "MOVE_CMD":
            # Handle legacy move command (for backward compatibility)
            try:
                move_cmd = MoveCmd(
                    agent_id=payload.get("agent_id"),
                    new_pos=tuple(payload.get("new_pos", [0, 0]))
                )
                
                # Process the move command through NetworkBase
                success = await self.network_base.process_move_command(move_cmd)
                
                result = f"[NetworkBase] Processed legacy move for agent {move_cmd.agent_id}: {'success' if success else 'failed'}"
                if self.output:
                    self.output.info(result)
                    
                await safe_enqueue_event(event_queue, new_agent_text_message(result))
                
            except Exception as e:
                error_msg = f"[NetworkBase] Error processing MOVE_CMD: {e}"
                if self.output:
                    self.output.error(error_msg)
                await safe_enqueue_event(event_queue, new_agent_text_message(error_msg))
                
        elif msg_type == "CONTROL":
            # Handle control messages - drill into payload to get actual cmd
            # The actual data is nested: {"type":"CONTROL","payload":{"cmd":"STEP","agent_id":0,"step_count":4}}
            control_payload = payload if payload else message_data.get("payload", {})
            cmd = control_payload.get("cmd", "unknown")
            agent_id = control_payload.get("agent_id", control_payload.get("sender"))
            step_count = control_payload.get("step_count", 0)
            
            # Handle different control commands properly
            if cmd == "START":
                result = f"[NetworkBase] Agent {agent_id} lifecycle: START"
            elif cmd == "STEP":
                result = f"[NetworkBase] Agent {agent_id} tick: STEP {step_count}"
            elif cmd == "STOP":
                result = f"[NetworkBase] Agent {agent_id} lifecycle: STOP"
            else:
                result = f"[NetworkBase] Agent {agent_id} unknown control: {cmd}"
            
            if self.output:
                self.output.info(result)
                
            await safe_enqueue_event(event_queue, new_agent_text_message(result))
            
        elif msg_type == "CHAT":
            # 🔧 CRITICAL FIX: Handle CHAT messages properly
            try:
                dst = payload.get("dst")
                src = payload.get("src", 0)
                if dst is None:
                    result = f"[NetworkBase] CHAT missing dst: {payload}"
                else:
                    # Forward the complete message_data to the target agent
                    await self.send_to_agent(dst, message_data)
                    result = f"[NetworkBase] Routed CHAT from agent {src} ➜ agent {dst}"
                    
                if self.output:
                    self.output.info(result)
                await safe_enqueue_event(event_queue, new_agent_text_message(result))
                
            except Exception as e:
                error_msg = f"[NetworkBase] Error routing CHAT message: {e}"
                if self.output:
                    self.output.error(error_msg)
                await safe_enqueue_event(event_queue, new_agent_text_message(error_msg))
            
        else:
            # Handle unknown message types
            result = f"[NetworkBase] Received unknown message type: {msg_type}"
            if self.output:
                self.output.warning(result)
                
            await safe_enqueue_event(event_queue, new_agent_text_message(result))

    async def _handle_text_message(self, text: str, event_queue: EventQueue):
        """Handle plain text message"""
        result = f"[NetworkBase] Received text message: {text}"
        if self.output:
            self.output.info(result)
            
        await safe_enqueue_event(event_queue, new_agent_text_message(result))

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Handle cancellation requests"""
        await safe_enqueue_event(event_queue, new_agent_text_message("[NetworkBase] Cancel not supported"))
        raise Exception("cancel not supported")