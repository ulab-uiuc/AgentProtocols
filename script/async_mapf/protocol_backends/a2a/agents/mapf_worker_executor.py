import asyncio
import inspect
import json
from typing import Dict, Any

# A2A SDK imports - these must be available for the system to work
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

# Adjusting import paths for the new structure - use relative imports to avoid conflict with A2A SDK
from ..adapters.a2a_comm_adapter import A2ACommAdapter
from script.async_mapf.core.agent_base import BaseAgent as MAPFCoreAgent

# Safe wrapper for EventQueue.enqueue_event (same as used in network_executor)
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

class MAPFAgentExecutor(AgentExecutor):
    """A2A-SDK compatible executor for a single MAPF robot."""

    def __init__(self, cfg: Dict[str, Any], global_cfg: Dict[str, Any], agent_id: int, router_url: str = None, output=None):
        self.agent_id = agent_id
        self.output = output
        
        # For A2A framework, each agent gets its own network identity
        # The router_url parameter should be the agent network identifier
        if not router_url:
            raise ValueError(f"Agent {agent_id}: A2A network identifier is required")
        
        # Create A2A communication adapter with network identifier
        adapter = A2ACommAdapter(str(agent_id), router_url)
        
        # Start connection task immediately
        asyncio.create_task(adapter.connect())
        if self.output:
            self.output.info(f"Agent {agent_id} initializing A2A communication: {router_url}")
        
        # Create the core agent logic, passing the adapter to it
        # The core agent needs the full config to initialize the LLM
        # and its own specific config for start/goal positions.
        core_agent_config = {
            "agent_config": cfg,
            "model": global_cfg.get("model"),
            "protocol": "a2a"  # 确保protocol信息正确传递
        }
        self.core_agent = MAPFCoreAgent(
            agent_id=agent_id,
            adapter=adapter,
            config=core_agent_config
        )
        
        # Store adapter reference for NetworkBase integration
        self.adapter = adapter
        
        if self.output:
            self.output.info(f"Executor for agent {agent_id} initialized with A2A adapter.")
            
    # ---------- A2A SDK lifecycle method ----------
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        Handles every incoming message from the A2A Router.
        This is the main entry point for any interaction with the agent.
        """
        user_input = context.get_user_input()
        
        # Extract A2A message meta information (similar to shard_qa approach)
        meta = {}
        sender_id = "external"
        try:
            if hasattr(context, 'params') and hasattr(context.params, 'message'):
                message = context.params.message
                if hasattr(message, 'meta') and message.meta:
                    meta = message.meta if isinstance(message.meta, dict) else {}
                elif hasattr(message, 'metadata') and message.metadata:
                    meta = message.metadata if isinstance(message.metadata, dict) else {}
                
                if meta:
                    sender_id = meta.get('sender', sender_id)
                    if self.output and meta.get('type'):
                        self.output.progress(f"[Agent {self.agent_id}] A2A message from {sender_id}: {meta.get('type')}")
        except Exception:
            # Fallback to legacy parsing
            if hasattr(context, 'message'):
                if hasattr(context.message, 'meta'):
                    meta = context.message.meta or {}
                    sender_id = meta.get("sender_id", "external")
                elif hasattr(context.message, 'metadata'):
                    meta = context.message.metadata or {}
                    sender_id = meta.get("sender_id", "external")

        # Debug: log all incoming data for analysis (only if verbose)
        verbose = getattr(self.core_agent, 'verbose', False) if self.core_agent else False
        if self.output and verbose:
            self.output.info(f"[Agent {self.agent_id}] 🔍 Execute called")
            self.output.info(f"   user_input: {user_input}")
            self.output.info(f"   meta type: {type(meta)} -> {meta}")
            
            # 🔧 NEW: Enhanced meta debugging for CONTROL message detection
            if isinstance(meta, dict):
                self.output.info(f"   meta.get('type'): {meta.get('type')}")
                self.output.info(f"   meta.get('cmd'): {meta.get('cmd')}")
                if meta.get('type') == 'CONTROL':
                    self.output.success(f"   🎯 DETECTED CONTROL MESSAGE: {meta}")
            
            if hasattr(context, 'message'):
                self.output.info(f"   context.message: {context.message}")
                if hasattr(context.message, 'metadata'):
                    self.output.info(f"   context.message.metadata: {context.message.metadata}")
                if hasattr(context.message, 'content'):
                    self.output.info(f"   context.message.content: {context.message.content}")
            self.output.info(f"   core_agent exists: {self.core_agent is not None}")
            if self.core_agent:
                self.output.info(f"   core_agent.agent_id: {self.core_agent.agent_id}")
                self.output.info(f"   core_agent.llm: {self.core_agent.llm is not None}")
                self.output.info(f"   autonomous_loop_task exists: {hasattr(self, '_autonomous_loop_task') and not getattr(self, '_autonomous_loop_task', None) is None}")
        elif self.output:
            # Always show basic info
            self.output.info(f"[Agent {self.agent_id}] Execute called")

        # Handle CONTROL messages from NetworkBase (check both meta and parsed user_input)
        control_message = None
        
        if isinstance(meta, dict) and meta.get("type") == "CONTROL":
            control_message = meta
            if self.output:
                self.output.success(f"[Agent {self.agent_id}] 🎯 Found CONTROL in meta: {control_message}")
        elif user_input:
            # Try to parse user_input as JSON for CONTROL messages
            try:
                parsed_input = json.loads(user_input)
                if isinstance(parsed_input, dict) and parsed_input.get("type") == "CONTROL":
                    control_message = parsed_input
                    if self.output:
                        self.output.success(f"[Agent {self.agent_id}] 🎯 Parsed CONTROL from text: {control_message}")
            except (json.JSONDecodeError, TypeError) as e:
                if self.output and verbose:
                    self.output.info(f"[Agent {self.agent_id}] Failed to parse user_input as JSON: {e}")
                pass  # Not JSON, continue with normal processing
        
        if control_message:
            control_cmd = control_message.get("cmd", "unknown")
            if control_cmd == "START":
                result = f"[Agent {self.agent_id}] ✅ Got START - launching autonomous loop"
                if self.output:
                    self.output.success(result)
                
                # Start autonomous loop if not already running
                if not hasattr(self, "_autonomous_loop_task") or self._autonomous_loop_task.done():
                    try:
                        # 详细检查LLM状态 (only if verbose)
                        if self.output and verbose:
                            self.output.info(f"🔍 Agent {self.agent_id} LLM status: {self.core_agent.llm is not None}")
                            if self.core_agent.llm is None:
                                self.output.warning(f"⚠️ Agent {self.agent_id} LLM is None - autonomous loop may not work properly")
                            else:
                                self.output.success(f"✅ Agent {self.agent_id} LLM initialized correctly")
                        elif self.output and self.core_agent.llm is None:
                            # Always warn if LLM is missing
                            self.output.warning(f"⚠️ Agent {self.agent_id} LLM not initialized")
                        
                        self._autonomous_loop_task = asyncio.create_task(self.core_agent.autonomous_loop())
                        if self.output:
                            self.output.success(f"🚀 Agent {self.agent_id} autonomous loop task created successfully!")
                    except Exception as e:
                        if self.output:
                            self.output.error(f"❌ Agent {self.agent_id} failed to start autonomous loop: {e}")
                            import traceback
                            self.output.error(f"Traceback: {traceback.format_exc()}")
                else:
                    if self.output:
                        self.output.info(f"Agent {self.agent_id} autonomous loop already running")
                        
            elif control_cmd == "STEP":
                result = f"[Agent {self.agent_id}] Received STEP signal (legacy mode)"
                if self.output:
                    self.output.progress(result)
                # Legacy STEP handling for backwards compatibility
                if hasattr(self.core_agent, '_execute_planning_step'):
                    asyncio.create_task(self.core_agent._execute_planning_step())
            else:
                result = f"[Agent {self.agent_id}] Unknown control command: {control_cmd}"
                if self.output:
                    self.output.warning(result)
            
            await safe_enqueue_event(event_queue, new_agent_text_message(result))
            return
        
        # Remove duplicate fallback parsing - already handled above

        # Handle CHAT messages
        if meta.get("type") == "CHAT" or getattr(context.message, 'type', None) == "CHAT":
            chat_dict = meta.get("payload", {})
            msg_str = chat_dict.get("msg", "")
            result = await self.core_agent.handle_chat(sender_id, msg_str, chat_dict)
            
            await safe_enqueue_event(event_queue, new_agent_text_message(result))
            return

        # Parse user_input for other message types if meta is empty
        parsed_message = None
        if not meta.get("type") and user_input:
            try:
                parsed_input = json.loads(user_input)
                if isinstance(parsed_input, dict) and parsed_input.get("type"):
                    parsed_message = parsed_input
                    if self.output and verbose:
                        self.output.info(f"[Agent {self.agent_id}] 📨 Parsed {parsed_input.get('type')} from text")
            except (json.JSONDecodeError, TypeError):
                pass  # Not JSON, continue with normal processing
        
        # Use parsed message if available, otherwise use meta
        message_data = parsed_message if parsed_message else meta
        
        # Handle other A2A messages (MOVE_FB, MOVE_RESPONSE, etc.)
        if message_data.get("type") in ["MOVE_FB", "MOVE_CMD", "MOVE_RESPONSE"]:
            # Forward to core agent via its message queue
            payload = message_data.get("payload", {})
            if message_data.get("type") == "MOVE_FB":
                from script.async_mapf.core.types import MoveFeedback
                feedback = MoveFeedback(
                    agent_id=payload.get("agent_id", self.agent_id),
                    success=payload.get("success", False),
                    actual_pos=tuple(payload.get("actual_pos", [0, 0])),
                    reason=payload.get("reason", "")
                )
                await self.core_agent._recv_msgs_queue.put(feedback)
                result = f"[Agent {self.agent_id}] Processed movement feedback: success={feedback.success}"
            elif message_data.get("type") == "MOVE_RESPONSE":
                # Handle concurrent move response from NetworkBase
                await self.core_agent._handle_move_response(payload)
                status = payload.get("status", "UNKNOWN")
                reason = payload.get("reason", "")
                result = f"[Agent {self.agent_id}] Processed move response: {status} - {reason}"
                if self.output:
                    if status == "OK":
                        self.output.success(result)
                    elif status == "REJECT":
                        self.output.error(result)
                    else:
                        self.output.warning(result)
            else:
                result = f"[Agent {self.agent_id}] Received {message_data.get('type')} message"
            
            await safe_enqueue_event(event_queue, new_agent_text_message(result))
            return

        # Handle direct commands
        result = ""
        try:
            # Command to trigger a planning step
            if user_input and user_input.upper().strip() in ["PLAN", "START", "STEP"]:
                result = await self.core_agent.plan_once()
            
            # Default case for other inputs
            else:
                result = f"[Agent {self.agent_id}] Received input: {user_input}"
                if self.output:
                    self.output.info(result)

        except Exception as e:
            result = f"[Agent {self.agent_id}] Error during execution: {e}"
            if self.output:
                self.output.error(result)

        # Send the result of the operation back to the caller via the event queue
        await safe_enqueue_event(event_queue, new_agent_text_message(result))

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Handles cancellation requests from the A2A framework."""
        # Stop autonomous LLM loop if running
        if hasattr(self, 'autonomous_llm_agent') and self.autonomous_llm_agent:
            self.autonomous_llm_agent.stop()
        
        if hasattr(self, 'autonomous_task') and self.autonomous_task:
            self.autonomous_task.cancel()
            
        await safe_enqueue_event(event_queue, new_agent_text_message(f"Agent {self.agent_id} cancelled and stopped autonomous mode"))
        
    async def _send_to_network(self, target: str, message_data: dict):
        """Send message to NetworkBase via A2A adapter"""
        if self.a2a_adapter:
            try:
                # Add receiver_id for Router routing to NetworkBase
                if "receiver_id" not in message_data:
                    message_data["receiver_id"] = "network-base"  # Special NetworkBase ID
                
                await self.a2a_adapter.send(message_data)
                
                if self.output:
                    self.output.progress(f"[Agent {self.agent_id}] 📤 Sent to NetworkBase: {message_data.get('type', 'unknown')}")
                    
            except Exception as e:
                if self.output:
                    self.output.error(f"[Agent {self.agent_id}] ❌ Failed to send to network: {e}")
        else:
            if self.output:
                self.output.error(f"[Agent {self.agent_id}] ❌ No A2A adapter configured") 