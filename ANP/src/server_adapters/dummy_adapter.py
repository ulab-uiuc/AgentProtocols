"""
Dummy server adapter for testing purposes.
"""

import json
from typing import Any, Dict, Tuple
import uvicorn

from .base_adapter import BaseServerAdapter


class DummyServerAdapter(BaseServerAdapter):
    """Dummy server adapter for testing purposes."""
    
    protocol_name = "DUMMY"
    
    def build(
        self,
        host: str,
        port: int,
        agent_id: str,
        executor: Any,
        **kwargs
    ) -> Tuple[uvicorn.Server, Dict[str, Any]]:
        """Build a minimal dummy server for testing."""
        
        # Dummy agent card
        agent_card = {
            "name": f"Dummy Agent {agent_id}",
            "url": f"http://{host}:{port}/",
            "protocolVersion": "1.0.0",
            "protocol": "DUMMY"
        }
        
        # Simple ASGI app for testing
        async def dummy_app(scope, receive, send):
            if scope["type"] == "http":
                if scope["path"] == "/health":
                    await send({
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [[b"content-type", b"application/json"]],
                    })
                    await send({
                        "type": "http.response.body",
                        "body": b'{"status": "ok"}',
                    })
                elif scope["path"] == "/.well-known/agent.json":
                    await send({
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [[b"content-type", b"application/json"]],
                    })
                    await send({
                        "type": "http.response.body",
                        "body": json.dumps(agent_card).encode(),
                    })
                else:
                    await send({
                        "type": "http.response.start",
                        "status": 404,
                    })
                    await send({
                        "type": "http.response.body",
                        "body": b"Not Found",
                    })
        
        # Configure uvicorn server
        config = uvicorn.Config(
            dummy_app,
            host=host,
            port=port,
            log_level="error"
        )
        server = uvicorn.Server(config)
        
        return server, agent_card 