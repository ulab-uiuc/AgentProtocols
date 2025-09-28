"""
A2A Protocol Agent Implementation for GAIA Framework.
This agent integrates with A2A Protocol for multi-agent communication.
"""

import asyncio
import time
import os
import threading
import logging
from typing import Dict, Any, Optional
import sys
import httpx
from pathlib import Path

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.agent import MeshAgent
from core.schema import AgentState
from core.schema import Message as GAIAMessage

# Suppress noisy logs
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)  
logging.getLogger("starlette").setLevel(logging.ERROR)

# A2A SDK imports
try:
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.utils import new_agent_text_message
    from a2a.types import MessageSendParams, Message, Role, TextPart
except ImportError as e:
    raise ImportError(f"A2A SDK components required but not available: {e}")

# Core LLM imports
try:
    # Add agent_network/src to path for Core import - use simpler path resolution
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up from protocol_backends/a2a to script/gaia to agent_network to src
    # Current: .../agent_network/script/gaia/protocol_backends/a2a
    # Target:  .../agent_network/src
    src_path = os.path.join(current_dir, '..', '..', '..', '..', 'src')
    src_path = os.path.abspath(src_path)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    # Try direct import first
    try:
        from utils.core import Core
    except ImportError:
        # Fallback: direct module import
        import importlib.util
        core_file = os.path.join(src_path, 'utils', 'core.py')
        spec = importlib.util.spec_from_file_location("core", core_file)
        core_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_module)
        Core = core_module.Core
# print(f"[DEBUG] Core imported successfully from {src_path}")
except ImportError as e:
    raise ImportError(f"Core LLM components required but not available: {e}")

# HTTP server imports
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.requests import Request
from starlette.routing import Route
import uvicorn


class A2AExecutor(AgentExecutor):
    """A2A Executor that integrates with GAIA Core LLM."""
    
    def __init__(self, agent):
        self.agent = agent
        self.core = None

        # Initialize Core LLM
        try:
            llm_config = self._create_llm_config()
            self.core = Core(llm_config)
            print(f"✅ Core LLM initialized: {llm_config.get('model', {}).get('name', 'unknown')}")
        except Exception as e:
            raise RuntimeError(f"Core LLM initialization failed: {e}")
    
    def _create_llm_config(self) -> Dict[str, Any]:
        """Create LLM configuration for Core."""
        return {
            "model": {
                "type": "openai",
                "name": self.agent.config.get('model_name', 'gpt-4o'),
                "openai_api_key": self.agent.config.get('openai_api_key'),
                "openai_base_url": self.agent.config.get('openai_base_url', 'https://api.openai.com/v1'),
                "temperature": self.agent.config.get('temperature', 0.0),
            }
        }
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute A2A request using Core LLM with timeout and token management."""
        try:
            # Extract message from context
            message = None
            if hasattr(context, 'params') and hasattr(context.params, 'message'):
                message = context.params.message
            elif hasattr(context, 'message'):
                message = context.message
            elif hasattr(context, 'request') and hasattr(context.request, 'message'):
                message = context.request.message
            
            if not message:
                raise ValueError("No message found in request context")
            
            # Extract text from message parts
            user_input = self._extract_text_from_message(message)
            
            # Call agent's execute method with timeout
            try:
                # Use asyncio.wait_for to add timeout protection
                response_text = await asyncio.wait_for(
                    self.agent.execute(user_input), 
                    timeout=25.0  # Shorter timeout to prevent hanging
                )
            except asyncio.TimeoutError:
                raise RuntimeError("Agent execution timed out after 25 seconds")
            
            if not response_text or response_text.strip() == "":
                raise RuntimeError("Agent returned empty response")
            
            # Create A2A response message with proper messageId
            import time
            response_message = Message(
                role=Role.agent,
                parts=[TextPart(text=response_text.strip())],
                messageId=str(int(time.time() * 1000))
            )
            await event_queue.enqueue_event(response_message)
            
            print(f"✅ A2A Executor completed successfully (response length: {len(response_text)})")
            
        except Exception as e:
            error_msg = f"Agent execution failed: {e}"
            print(f"❌ A2A Executor error: {error_msg}")
            
            # Send error response with proper messageId
            import time
            error_message = Message(
                role=Role.agent,
                parts=[TextPart(text=f"Error: {error_msg}")],
                messageId=str(int(time.time() * 1000))
            )
            await event_queue.enqueue_event(error_message)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel A2A request execution."""
        import time
        cancel_message = Message(
            role=Role.agent,
            parts=[TextPart(text="Request cancelled")],
            messageId=str(int(time.time() * 1000))
        )
        await event_queue.enqueue_event(cancel_message)
    
    def _extract_text_from_message(self, message) -> str:
        """Extract text content from A2A message."""
        if hasattr(message, 'parts') and message.parts:
            for part in message.parts:
                if hasattr(part, 'text') and part.text:
                    return part.text
        return str(message) if message else ""


class A2AAgent(MeshAgent):
    """
    A2A Protocol Agent that inherits from MeshAgent.
    The A2ANetwork is responsible for spawning an A2A server which calls
    this agent's `execute(message)` coroutine to process requests.
    """
    
    def __init__(self, node_id: int, name: str, tool: str, port: int, 
                 config: Dict[str, Any], task_id: Optional[str] = None):
        super().__init__(node_id, name, tool, port, config, task_id)

        # Runtime flags
        self._connected = False
        self._server = None
        self._server_task = None
        self._executor = None
        self._base_url = f"http://localhost:{port}"
        self._http_client = httpx.AsyncClient(timeout=5.0)

        # A2A specific configuration
        self._model_name = config.get("model_name", "gpt-4o")
        self._temperature = config.get("temperature", 0.0)
        self._openai_api_key = config.get("openai_api_key")
        self._openai_base_url = config.get("openai_base_url", "https://api.openai.com/v1")

        # Pretty initialization output
        print(f"[{name}] Core LLM: {self._model_name} (temp={self._temperature})")
        print(f"[A2AAgent] Initialized with A2A Protocol SDK")

        self._log("A2AAgent initialized (network-managed comms)")

    async def connect(self):
        """Start A2A server and mark agent as connected."""
        if not self._connected:
            await self._start_server()
            self._connected = True
            self._log("A2AAgent connected and server started")

    async def disconnect(self):
        """Stop A2A server and mark agent as disconnected."""
        if self._connected:
            await self._stop_server()
            self._connected = False
            self._log("A2AAgent disconnected and server stopped")

    async def send_msg(self, dst: int, payload: Dict[str, Any]) -> None:
        """No direct send from agent; A2ANetwork delivers via HTTP backend."""
        self._log(f"send_msg called (dst={dst}) - handled by network backend")

    async def recv_msg(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """No inbox polling; requests arrive via A2A server -> execute()."""
        if timeout:
            try:
                await asyncio.sleep(min(timeout, 0.01))
            except Exception:
                pass
        return None

    def get_connection_status(self) -> Dict[str, Any]:
        """Basic connection status for diagnostics."""
        status: Dict[str, Any] = {
            "agent_id": self.id,
            "agent_name": self.name,
            "protocol": "a2a",
            "connected": self._connected,
            "base_url": self._base_url,
            "server_running": self._server_task is not None and not self._server_task.done() if self._server_task else False,
        }
        
        if self._http_client:
            status["client_stats"] = {
                "timeout": str(self._http_client.timeout),
            }
        
        return status

    async def start(self):
        """Start the agent main loop."""
        await self.connect()
        await super().start()

    async def stop(self):
        """Stop the agent and cleanup resources."""
        # Close http client if created
        try:
            if self._http_client:
                await self._http_client.aclose()
        except Exception:
            pass
        await self.disconnect()
        await super().stop()

    # ==================== Execution Entry for A2A Server ====================
    async def execute(self, message: str) -> str:
        """
        Entry point used by A2A server to process a request.
        Fixed to execute only ONE step to align with ACP/Agora behavior.
        Network layer handles the workflow coordination.
        """
        try:
            # Reset state for a fresh request
            self.messages.clear()
            self.memory.messages.clear()
            self.current_step = 0
            self.state = AgentState.IDLE
            
            # Add user message to both internal messages and memory
            user_msg = GAIAMessage.user_message(message)
            self.messages.append(user_msg)
            self.memory.add_message(user_msg)
            
            # Execute ONLY ONE STEP - let network layer handle coordination
            try:
                await self.step()
                self.current_step += 1
                
                # Extract result after one step
                final_result = self._extract_final_result()
                
                # Add assistant response to memory if we have a result
                if final_result:
                    assistant_msg = GAIAMessage.assistant_message(final_result)
                    self.memory.add_message(assistant_msg)
                
                return final_result or "No result generated in this step"
                
            except Exception as e:
                self._log(f"Error in step execution: {e}")
                return f"Error: {e}"
                
        except Exception as e:
            error_msg = f"Error executing request: {e}"
            self._log(error_msg)
            return error_msg

    # ==================== A2A Server Management ====================
    async def _start_server(self):
        """Start A2A server with HTTP endpoints."""
# Debug info (can be enabled for troubleshooting)
        # print(f"[DEBUG] Starting A2A server for {self.name}")
        
        # Create A2A executor
        self._executor = A2AExecutor(self)
        
        async def message_endpoint(request: Request):
            """Handle A2A /message endpoint."""
            try:
                payload = await request.json()
                
                # Create A2A RequestContext
                params = MessageSendParams.model_validate(payload.get("params", {}))
                context = RequestContext(params)
                
                # Create EventQueue
                event_queue = EventQueue()
                
                # Execute request
                await self._executor.execute(context, event_queue)
                
                # Collect events and serialize them
                events = []
                try:
                    while not event_queue.queue.empty():
                        event = await event_queue.dequeue_event()
                        if event:
                            if hasattr(event, 'model_dump'):
                                events.append(event.model_dump(mode='json'))
                            else:
                                events.append(event)
                except:
                    pass
                
                return JSONResponse({"events": events})
                
            except Exception as e:
                print(f"❌ Error in message endpoint: {e}")
                error_event = {
                    "kind": "message",
                    "role": "agent",
                    "parts": [{"kind": "text", "text": f"Error: {e}"}],
                    "messageId": str(time.time_ns())
                }
                return JSONResponse({"events": [error_event]})
        
        async def health_endpoint(_request: Request):
            """Handle A2A /health endpoint."""
            return JSONResponse({
                "ok": True, 
                "agent_id": str(self.id),
                "status": "healthy",
                "timestamp": time.time()
            })
        
        # Create Starlette application
        routes = [
            Route("/message", message_endpoint, methods=["POST"]),
            Route("/health", health_endpoint, methods=["GET"]),
        ]
        app = Starlette(routes=routes)
        
        # Create Uvicorn server with proper logging configuration
        config = uvicorn.Config(
            app=app, 
            host="localhost", 
            port=self.port, 
            log_level="error",  # Reduce log noise
            access_log=False,   # Disable access logs
            server_header=False  # Remove server header
        )
        self._server = uvicorn.Server(config)
        
        # Start server in background
        self._server_task = asyncio.create_task(self._server.serve())
        await asyncio.sleep(0.5)  # Wait for server to start
        
        print(f"✅ A2A server started on {self._base_url}")
    
    async def _stop_server(self):
        """Stop A2A server gracefully."""
        print(f"🛑 Stopping A2A server on {self._base_url}")
        
        # First, signal the server to exit
        if self._server:
            self._server.should_exit = True
            
        # Wait a moment for graceful shutdown
        if self._server_task:
            try:
                # Give the server a chance to shutdown gracefully
                await asyncio.wait_for(self._server_task, timeout=2.0)
            except asyncio.TimeoutError:
                print("⚠️ Server shutdown timed out, forcing cancellation")
                self._server_task.cancel()
                try:
                    await self._server_task
                except asyncio.CancelledError:
                    pass  # Expected when cancelling
            except asyncio.CancelledError:
                pass  # Server was already cancelled
            except Exception as e:
                print(f"⚠️ Error during server shutdown: {e}")
                
        print(f"✅ A2A server stopped on {self._base_url}")

    # ==================== Health Check ====================
    async def health_check(self, agent_id: str = None) -> bool:
        """Check A2A agent health using its HTTP endpoint."""
        # Use self.id if agent_id not provided
        if agent_id is None:
            agent_id = str(self.id)
            
        try:
            client = self._http_client or httpx.AsyncClient(timeout=5.0)
            resp = await client.get(f"{self._base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception as e:
            print(f"[A2AAgent] Health check failed for {agent_id}: {e}")
            return False
