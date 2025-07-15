"""
A2A (Agent-to-Agent) protocol server adapter with SDK native interface support.
"""

import asyncio
import json
import logging
import time
import inspect
from typing import Any, Dict, Tuple
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.requests import Request

from .base_adapter import BaseServerAdapter

logger = logging.getLogger(__name__)

async def _safe_call(handler, *args, **kwargs):
    """Call handler; await only if it's awaitable."""
    try:
        res = handler(*args, **kwargs)
        if inspect.isawaitable(res):
            return await res
        else:
            return res
    except Exception as e:
        logger.exception("Handler error: %s", e)
        raise

async def _safe_enqueue(event_queue, event):
    """Safely enqueue event, handling both sync and async event queues."""
    try:
        res = event_queue.enqueue_event(event)
        if inspect.isawaitable(res):
            return await res
        else:
            return res
    except Exception as e:
        logger.exception("Event queue error: %s", e)
        raise

# A2A SDK imports

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message as _original_new_agent_text_message

def new_agent_text_message(text, role="user"):
    """Wrapper for new_agent_text_message that ensures compatibility"""
    # A2A SDK's new_agent_text_message only takes text parameter
    return _original_new_agent_text_message(text)



class A2AStarletteApplication:
    """A2A server implementation using SDK native executor interface."""
    
    def __init__(self, agent_card: Dict[str, Any], executor: Any):
        self.agent_card = agent_card
        self.executor = executor

        
    def build(self) -> Starlette:
        """Build the Starlette application."""
        routes = [
            Route("/.well-known/agent.json", self.get_agent_card, methods=["GET"]),
            Route("/health", self.health_check, methods=["GET"]),
            Route("/message", self.handle_message, methods=["POST"]),
        ]
        
        return Starlette(routes=routes)
    
    async def get_agent_card(self, request: Request) -> JSONResponse:
        """Return the public agent card."""
        return JSONResponse(self.agent_card)
    
    async def health_check(self, request: Request) -> Response:
        """Health check endpoint."""
        return Response("OK", status_code=200)
    
    async def handle_message(self, request: Request) -> JSONResponse | StreamingResponse:
        """
        Handle incoming messages using SDK native executor interface.
        
        • Convert HTTP JSON → SDK RequestContext  
        • Call executor.execute(ctx, queue)
        • Check Accept header to decide JSON vs SSE response
        """
        try:
            # Parse request body
            body = await request.json()
            
            # A2A SDK expects specific structure for RequestContext
            # We need to create a proper Message object first
            from a2a.types import Message, MessageSendParams
            
            # Extract message from request body
            if 'params' in body and 'message' in body['params']:
                message_data = body['params']['message']
                # Create Message object from the data
                message = Message(**message_data)
                
                # Create MessageSendParams with the message
                params = MessageSendParams(message=message)
                
                # Create SDK RequestContext with proper params
                ctx = RequestContext(params)
            else:
                # Fallback: create a simple text message
                # Use our wrapped function that ensures no Role enum issues
                
                # Extract text from body or use default
                text = body.get('text', str(body))
                # Use our wrapper function that ensures role is string
                message = new_agent_text_message(text, role="user")
                params = MessageSendParams(message=message)
                ctx = RequestContext(params)
            
            # Create EventQueue to collect events
            queue = EventQueue()
            
            # === Call SDK native executor interface ===
            # Use safe_call to handle both sync and async executors
            try:
                await _safe_call(self.executor.execute, ctx, queue)
            except Exception as e:
                logger.error(f"Error in executor.execute: {e}")
                raise
            # ==========================================
            
            # Check if client wants streaming response
            accept_header = request.headers.get("accept", "")
            if "text/event-stream" in accept_header:
                # Return SSE streaming response
                return StreamingResponse(
                    self._sse_generator(queue),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    }
                )
            else:
                # Return unified JSON response with events array
                # Collect events from queue
                events = []
                try:
                    while True:
                        event = await queue.dequeue_event(no_wait=True)
                        events.append(self._event_to_dict(event))
                except asyncio.QueueEmpty:
                    # No more events available
                    pass
                except Exception as e:
                    # Queue closed or other error
                    logger.warning(f"Error collecting events: {e}")
                
                return JSONResponse({
                    "events": events
                })
                
        except Exception as e:
            return JSONResponse(
                {"error": f"Message handling failed: {e}"},
                status_code=500
            )
    
    async def _sse_generator(self, queue: EventQueue):
        """Generate SSE events from EventQueue."""
        try:
            while True:
                try:
                    # Try to get events without waiting first
                    event = await queue.dequeue_event(no_wait=True)
                    event_data = self._event_to_dict(event)
                    yield f"data: {json.dumps(event_data)}\n\n"
                except asyncio.QueueEmpty:
                    # No immediate events, wait for next one
                    try:
                        event = await queue.dequeue_event(no_wait=False)
                        event_data = self._event_to_dict(event)
                        yield f"data: {json.dumps(event_data)}\n\n"
                    except asyncio.QueueEmpty:
                        # Queue is closed and empty
                        break
        except Exception as e:
            logger.warning(f"SSE generator error: {e}")
            # Send error event
            error_data = {"error": str(e), "type": "stream_error"}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    def _event_to_dict(self, event: Any) -> Dict[str, Any]:
        """Convert Event to JSON-serializable dictionary."""
        try:
            # Try pydantic v2 model_dump() first (SDK ≥0.3)
            if hasattr(event, 'model_dump'):
                raw_dict = event.model_dump()
            # Fallback to pydantic v1 dict() method
            elif hasattr(event, 'dict'):
                raw_dict = event.dict()
            # For dict-like objects
            elif isinstance(event, dict):
                raw_dict = event
            # Last resort: manual conversion
            else:
                raw_dict = {
                    "type": getattr(event, "type", event.__class__.__name__),
                    "data": getattr(event, "data", str(event))
                }
            
            # Ensure all values are JSON serializable (convert Role enums to strings)
            return self._sanitize_for_json(raw_dict)
            
        except Exception as e:
            # Fallback for any serialization errors
            return {
                "type": event.__class__.__name__,
                "data": str(event),
                "error": f"Serialization failed: {e}"
            }
    
    def _sanitize_for_json(self, obj: Any) -> Any:
        """Recursively sanitize object to ensure JSON serializability"""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
            # Handle objects with attributes
            return {str(k): self._sanitize_for_json(v) for k, v in obj.__dict__.items()}
        elif hasattr(obj, 'value'):
            # Handle enums (like Role.USER -> "USER")
            return str(obj.value)
        elif hasattr(obj, 'name'):
            # Handle enums without value attribute
            return str(obj.name)
        else:
            # Convert everything else to string
            return str(obj)


class A2AServerAdapter(BaseServerAdapter):
    """Server adapter for A2A (Agent-to-Agent) protocol with SDK native interface."""
    
    protocol_name = "A2A"
    
    def build(
        self,
        host: str,
        port: int,
        agent_id: str,
        executor: Any,
        **kwargs
    ) -> Tuple[uvicorn.Server, Dict[str, Any]]:
        """Build A2A server using SDK native executor interface."""
        
        # Validate executor has SDK native interface
        if not hasattr(executor, 'execute'):
            raise TypeError(
                f"Executor {type(executor)} must implement SDK native interface: "
                "async def execute(context: RequestContext, event_queue: EventQueue)"
            )
        
        # Generate A2A agent card
        agent_card = {
            "name": f"Agent {agent_id}",
            "url": f"http://{host}:{port}/",
            "protocolVersion": "1.0.0",
            "skills": [
                {
                    "id": "agent_execution",
                    "name": "Agent Execution",
                    "description": "Execute agent tasks using A2A SDK native interface"
                }
            ],
            "capabilities": {
                "streaming": True,
                "supportsAuthenticatedExtendedCard": False,
                "nativeSDK": True
            }
        }
        
        # Create A2A Starlette application
        app_builder = A2AStarletteApplication(agent_card, executor)
        app = app_builder.build()
        
        # Configure uvicorn server
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="error"  # Minimize server logs
        )
        server = uvicorn.Server(config)
        
        return server, agent_card