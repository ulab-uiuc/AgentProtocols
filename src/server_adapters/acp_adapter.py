"""
ACP (Agent Communication Protocol) server adapter implementation using ACP SDK.
"""

import json
import logging
import time
from typing import Any, Dict, Tuple, Union, List

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.requests import Request

# ACP SDK imports
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield

from .base_adapter import BaseServerAdapter

logger = logging.getLogger(__name__)


class ACPStarletteApplication:
    """ACP server implementation using ACP SDK native executor interface."""

    def __init__(self, agent_card: Dict[str, Any], executor: Any):
        self.agent_card = agent_card
        self.executor = executor

    def build(self) -> Starlette:
        """Build the Starlette application."""
        routes = [
            Route("/.well-known/agent.json", self.get_agent_card, methods=["GET"]),
            Route("/health", self.health_check, methods=["GET"]),
            Route("/acp/message", self.handle_message, methods=["POST"]),
            Route("/acp/capabilities", self.get_capabilities, methods=["GET"]),
            Route("/acp/status", self.get_status, methods=["GET"]),
        ]

        # Add lifespan handler to prevent startup issues
        async def lifespan(app):
            # Startup
            logger.debug("ACP Starlette app starting up")
            yield
            # Shutdown
            logger.debug("ACP Starlette app shutting down")

        return Starlette(routes=routes, lifespan=lifespan)

    async def get_agent_card(self, request: Request) -> JSONResponse:
        """Return the public agent card."""
        return JSONResponse(self.agent_card)

    async def health_check(self, request: Request) -> Response:
        """Health check endpoint."""
        return Response("OK", status_code=200)

    async def get_capabilities(self, request: Request) -> JSONResponse:
        """Return agent capabilities."""
        return JSONResponse({
            "agent_id": getattr(self.executor, 'agent_id', 'unknown'),
            "capabilities": getattr(self.executor, 'capabilities', []),
            "protocol": "ACP",
            "version": "1.0.0",
            "features": ["streaming", "progress_tracking", "async_generator"]
        })

    async def get_status(self, request: Request) -> JSONResponse:
        """Return agent status."""
        return JSONResponse({
            "agent_id": getattr(self.executor, 'agent_id', 'unknown'),
            "status": "healthy",
            "timestamp": time.time()
        })

    async def handle_message(self, request: Request) -> Union[JSONResponse, StreamingResponse]:
        """
        Handle incoming ACP messages using SDK native executor interface.

        • Convert HTTP JSON → ACP SDK Message objects
        • Create Context
        • Call executor.execute(messages, context) which returns AsyncGenerator
        • Handle RunYield events and collect responses
        """
        try:
            # Parse request body
            body = await request.json()

            # Convert request to ACP SDK Message objects
            messages = []
            if 'messages' in body:
                # Multiple messages - convert each one
                for msg_data in body['messages']:
                    if 'parts' in msg_data:
                        # Already has parts structure
                        message = Message(**msg_data)
                    else:
                        # Convert simple content to parts structure
                        content = msg_data.get('content', '')
                        parts = [MessagePart(content=content)]
                        message = Message(
                            role=msg_data.get('role', 'user'),
                            parts=parts
                        )
                    messages.append(message)
            elif 'message' in body:
                # Single message
                msg_data = body['message']
                if 'parts' in msg_data:
                    message = Message(**msg_data)
                else:
                    content = msg_data.get('content', '')
                    parts = [MessagePart(content=content)]
                    message = Message(
                        role=msg_data.get('role', 'user'),
                        parts=parts
                    )
                messages.append(message)
            else:
                # Fallback: create message from request body content
                # Try different payload structures
                content = (body.get('content') or 
                          body.get('text') or 
                          body.get('payload', {}).get('text') or 
                          body.get('payload', {}).get('content'))
                
                # If still no content, try to extract from nested payload
                if not content:
                    # Handle UTE-style nested payload
                    payload = body.get('payload', {})
                    if isinstance(payload, dict):
                        content = (payload.get('text') or 
                                 payload.get('question') or
                                 payload.get('content'))
                
                # Final fallback
                if not content:
                    content = str(body)
                
                parts = [MessagePart(type="text", text=content)]
                message = Message(role="user", parts=parts)
                messages.append(message)

            # Create ACP SDK Context
            context = Context(
                session=None,
                store=None,
                loader=None,
                executor=self.executor,
                request=request,
                yield_queue=None,
                yield_resume_queue=None
            )

            # Check if client wants streaming response
            accept_header = request.headers.get("accept", "")
            if "text/event-stream" in accept_header:
                # Return SSE streaming response
                return StreamingResponse(
                    self._sse_generator(messages, context),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    }
                )
            else:
                # Collect all results for JSON response
                results = []
                async for result in self.executor(messages, context):
                    results.append(self._yield_to_dict(result))

                return JSONResponse({
                    "results": results,
                    "timestamp": time.time(),
                    "status": "completed"
                })

        except Exception as e:
            logger.error(f"Error handling ACP message: {e}")
            return JSONResponse(
                {"error": f"Message handling failed: {e}"},
                status_code=500
            )

    async def _sse_generator(self, messages: List[Message], context: Context):
        """Generate SSE events from ACP SDK AsyncGenerator."""
        try:
            async for result in self.executor(messages, context):
                result_data = self._yield_to_dict(result)
                yield f"data: {json.dumps(result_data)}\n\n"
        except Exception as e:
            logger.warning(f"SSE generator error: {e}")
            # Send error event
            error_data = {"error": str(e), "type": "stream_error"}
            yield f"data: {json.dumps(error_data)}\n\n"

    def _yield_to_dict(self, result: RunYield) -> Dict[str, Any]:
        """Convert ACP SDK RunYield to JSON-serializable dictionary."""
        import datetime
        
        def _make_json_safe(obj):
            """Recursively make object JSON-serializable."""
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: _make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_make_json_safe(item) for item in obj]
            else:
                return obj
        
        try:
            # Handle string results directly (common from our wrapper)
            if isinstance(result, str):
                return {"type": "text", "content": result, "timestamp": time.time()}
            
            # Handle MessagePart objects (our custom yield objects)
            if hasattr(result, 'text') and hasattr(result, 'type'):
                return {
                    "type": result.type,
                    "text": result.text,
                    "timestamp": time.time()
                }
            # Try pydantic model_dump() first
            elif hasattr(result, 'model_dump'):
                data = result.model_dump()
                return _make_json_safe(data)
            # Fallback to pydantic dict() method
            elif hasattr(result, 'dict'):
                data = result.dict()
                return _make_json_safe(data)
            # For dict-like objects
            elif isinstance(result, dict):
                return _make_json_safe(result)
            # Handle Message objects
            elif hasattr(result, 'role') and hasattr(result, 'parts'):
                return {
                    "type": "message",
                    "role": result.role,
                    "content": result.parts[0].content if result.parts else "",
                    "parts": [{"content": part.content} for part in result.parts] if result.parts else [],
                    "timestamp": time.time()
                }
            # Last resort: manual conversion
            else:
                return {
                    "type": result.__class__.__name__,
                    "data": str(result),
                    "timestamp": time.time()
                }
        except Exception as e:
            # Fallback for any serialization errors
            return {
                "type": "error",
                "data": str(result),
                "error": f"Serialization failed: {e}",
                "timestamp": time.time()
            }


class ACPServerAdapter(BaseServerAdapter):
    """Server adapter for ACP (Agent Communication Protocol) with SDK native interface."""

    protocol_name = "ACP"

    def build(
        self,
        host: str,
        port: int,
        agent_id: str,
        executor: Any,
        **kwargs
    ) -> Tuple[uvicorn.Server, Dict[str, Any]]:
        """Build ACP server using ACP SDK native executor interface."""

        # Validate executor has ACP SDK native interface
        # The executor should be a function that takes (messages: list[Message], context: Context)
        # and returns AsyncGenerator[RunYield, RunYieldResume]
        if not callable(executor):
            raise TypeError(
                f"Executor {type(executor)} must be callable and implement ACP SDK native interface: "
                "async def executor(messages: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]"
            )

        # Generate ACP agent card
        agent_card = {
            "name": f"ACP Agent {agent_id}",
            "url": f"http://{host}:{port}/",
            "protocolVersion": "1.0.0",
            "protocol": "ACP",
            "agent_id": agent_id,
            "capabilities": getattr(executor, 'capabilities', ["text_processing", "async_generation"]),
            "endpoints": {
                "message": f"http://{host}:{port}/acp/message",
                "capabilities": f"http://{host}:{port}/acp/capabilities",
                "status": f"http://{host}:{port}/acp/status",
                "health": f"http://{host}:{port}/health"
            },
            "features": {
                "streaming": True,
                "progress_tracking": True,
                "async_generator": True,
                "sdk_native": True
            }
        }

        # Create ACP Starlette application
        app_builder = ACPStarletteApplication(agent_card, executor)
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
