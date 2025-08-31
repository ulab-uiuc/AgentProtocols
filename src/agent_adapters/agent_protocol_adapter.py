"""
AgentProtocolAdapter - Agent Protocol 适配器实现
"""

import json
from typing import Any, Dict, Optional, AsyncIterator
from uuid import uuid4

import httpx

# Import base adapter - try relative import first, then absolute
try:
    from .base_adapter import BaseProtocolAdapter
except ImportError:
    # Fall back to absolute import for standalone usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from agent_adapters.base_adapter import BaseProtocolAdapter
from src.core.protocol_converter import DECODE_TABLE


class AgentProtocolAdapter(BaseProtocolAdapter):
    """
    Adapter for Agent Protocol (AP) specification.
    
    Implements the standardized Agent Protocol v1 API for interacting with agents
    that support the Task/Step/Artifact paradigm via HTTP endpoints.
    
    Agent Protocol endpoints:
    - POST /ap/v1/agent/tasks - Create a new task
    - GET /ap/v1/agent/tasks/{task_id} - Get task details
    - POST /ap/v1/agent/tasks/{task_id}/steps - Execute next step
    - GET /ap/v1/agent/tasks/{task_id}/steps - List task steps
    - GET /ap/v1/agent/tasks/{task_id}/steps/{step_id} - Get step details
    - GET /ap/v1/agent/tasks/{task_id}/artifacts - List task artifacts
    - GET /ap/v1/agent/tasks/{task_id}/artifacts/{artifact_id} - Download artifact
    
    Instance granularity: One outbound edge = One Adapter instance.
    Different dst_id / base_url / protocol / Auth do not share instances.
    """

    @property
    def protocol_name(self) -> str:
        return "agentprotocol"

    def __init__(
        self, 
        httpx_client: httpx.AsyncClient, 
        base_url: str,
        auth_headers: Optional[Dict[str, str]] = None,
        agent_card_path: str = "/.well-known/agent.json",
        api_version: str = "v1"
    ):
        """
        Initialize Agent Protocol adapter.
        
        Parameters
        ----------
        httpx_client : httpx.AsyncClient
            Shared HTTP client for connection pooling
        base_url : str
            Base URL of the Agent Protocol agent endpoint
        auth_headers : Optional[Dict[str, str]]
            Authentication headers (e.g., {"Authorization": "Bearer <token>"})
        agent_card_path : str
            Path to agent card endpoint (default: /.well-known/agent.json)
        api_version : str
            Agent Protocol API version (default: v1)
        """
        super().__init__(base_url=base_url, auth_headers=auth_headers or {})
        self.httpx_client = httpx_client
        self.base_url = base_url.rstrip('/')
        self.auth_headers = auth_headers or {}
        self.agent_card_path = agent_card_path
        self.api_version = api_version
        self.agent_card: Dict[str, Any] = {}
        self._api_prefix = f"/ap/{api_version}/agent"

    async def initialize(self) -> None:
        """
        Initialize by fetching the agent card from /.well-known/agent.json.
        
        Raises
        ------
        ConnectionError
            If agent card retrieval fails
        """
        try:
            headers = {}
            headers.update(self.auth_headers)
            
            response = await self.httpx_client.get(
                f"{self.base_url}{self.agent_card_path}",
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            self.agent_card = response.json()
        except httpx.TimeoutException as e:
            raise ConnectionError(f"Timeout fetching agent card from {self.base_url}: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ConnectionError(f"HTTP {e.response.status_code} fetching agent card: {e.response.text}") from e
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Agent Protocol adapter: {e}") from e

    async def send_message(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        """
        Send message via Agent Protocol.
        
        Supports different message types:
        - create_task: Creates a new task
        - execute_step: Executes next step for a task
        - get_task: Retrieves task information
        - get_steps: Retrieves task steps
        - get_artifacts: Retrieves task artifacts
        
        Parameters
        ----------
        dst_id : str
            Destination agent ID (used for routing context)
        payload : Dict[str, Any]
            Message payload containing the operation and data
        
        Returns
        -------
        Any
            Complete JSON response from the agent
            
        Raises
        ------
        TimeoutError
            If request times out (default 30s)
        ConnectionError
            For HTTP 4xx/5xx errors
        RuntimeError
            For other send failures
        """
        message_type = payload.get("type", "create_task")
        request_id = str(uuid4())
        
        try:
            if message_type == "create_task":
                return await self._create_task(payload, request_id)
            elif message_type == "execute_step":
                return await self._execute_step(payload, request_id)
            elif message_type == "get_task":
                return await self._get_task(payload, request_id)
            elif message_type == "get_steps":
                return await self._get_steps(payload, request_id)
            elif message_type == "get_step":
                return await self._get_step(payload, request_id)
            elif message_type == "get_artifacts":
                return await self._get_artifacts(payload, request_id)
            elif message_type == "get_artifact":
                return await self._get_artifact(payload, request_id)
            else:
                # Default to creating a task with the payload as input
                return await self._create_task({
                    "input": payload.get("message", str(payload)),
                    "additional_input": payload.get("additional_input", {})
                }, request_id)
                
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Agent Protocol timeout to {dst_id} (req_id: {request_id})") from e
        except httpx.HTTPStatusError as e:
            raise ConnectionError(f"Agent Protocol HTTP error {e.response.status_code}: {e.response.text} (req_id: {request_id})") from e
        except Exception as e:
            raise RuntimeError(f"Agent Protocol send failed: {e} (req_id: {request_id})") from e

    async def _create_task(self, payload: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Create a new task via Agent Protocol."""
        task_data = {
            "input": payload.get("input", ""),
            "additional_input": payload.get("additional_input", {})
        }
        
        headers = {"Content-Type": "application/json"}
        headers.update(self.auth_headers)
        
        response = await self.httpx_client.post(
            f"{self.base_url}{self._api_prefix}/tasks",
            content=json.dumps(task_data, separators=(',', ':')),
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        
        result = response.json()
        # Add metadata for traceability
        result["_request_id"] = request_id
        result["_operation"] = "create_task"
        return result

    async def _execute_step(self, payload: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Execute next step for a task."""
        task_id = payload.get("task_id")
        if not task_id:
            raise ValueError("task_id is required for execute_step operation")
            
        step_data = {
            "input": payload.get("input", ""),
            "additional_input": payload.get("additional_input", {})
        }
        
        headers = {"Content-Type": "application/json"}
        headers.update(self.auth_headers)
        
        response = await self.httpx_client.post(
            f"{self.base_url}{self._api_prefix}/tasks/{task_id}/steps",
            content=json.dumps(step_data, separators=(',', ':')),
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        
        result = response.json()
        result["_request_id"] = request_id
        result["_operation"] = "execute_step"
        return result

    async def _get_task(self, payload: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Get task details."""
        task_id = payload.get("task_id")
        if not task_id:
            raise ValueError("task_id is required for get_task operation")
            
        headers = {}
        headers.update(self.auth_headers)
        
        response = await self.httpx_client.get(
            f"{self.base_url}{self._api_prefix}/tasks/{task_id}",
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        
        result = response.json()
        result["_request_id"] = request_id
        result["_operation"] = "get_task"
        return result

    async def _get_steps(self, payload: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Get task steps."""
        task_id = payload.get("task_id")
        if not task_id:
            raise ValueError("task_id is required for get_steps operation")
            
        headers = {}
        headers.update(self.auth_headers)
        
        response = await self.httpx_client.get(
            f"{self.base_url}{self._api_prefix}/tasks/{task_id}/steps",
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        
        result = response.json()
        result["_request_id"] = request_id
        result["_operation"] = "get_steps"
        return result

    async def _get_step(self, payload: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Get specific step details."""
        task_id = payload.get("task_id")
        step_id = payload.get("step_id")
        if not task_id or not step_id:
            raise ValueError("task_id and step_id are required for get_step operation")
            
        headers = {}
        headers.update(self.auth_headers)
        
        response = await self.httpx_client.get(
            f"{self.base_url}{self._api_prefix}/tasks/{task_id}/steps/{step_id}",
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        
        result = response.json()
        result["_request_id"] = request_id
        result["_operation"] = "get_step"
        return result

    async def _get_artifacts(self, payload: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Get task artifacts."""
        task_id = payload.get("task_id")
        if not task_id:
            raise ValueError("task_id is required for get_artifacts operation")
            
        headers = {}
        headers.update(self.auth_headers)
        
        response = await self.httpx_client.get(
            f"{self.base_url}{self._api_prefix}/tasks/{task_id}/artifacts",
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        
        result = response.json()
        result["_request_id"] = request_id
        result["_operation"] = "get_artifacts"
        return result

    async def _get_artifact(self, payload: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Download specific artifact."""
        task_id = payload.get("task_id")
        artifact_id = payload.get("artifact_id")
        if not task_id or not artifact_id:
            raise ValueError("task_id and artifact_id are required for get_artifact operation")
            
        headers = {}
        headers.update(self.auth_headers)
        
        response = await self.httpx_client.get(
            f"{self.base_url}{self._api_prefix}/tasks/{task_id}/artifacts/{artifact_id}",
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        
        # Handle binary content
        if response.headers.get("content-type", "").startswith("application/"):
            # Binary artifact
            result = {
                "artifact_id": artifact_id,
                "content_type": response.headers.get("content-type"),
                "content_length": len(response.content),
                "content": response.content.hex(),  # Hex encode for JSON serialization
                "_request_id": request_id,
                "_operation": "get_artifact"
            }
        else:
            # Text artifact
            result = {
                "artifact_id": artifact_id,
                "content_type": response.headers.get("content-type", "text/plain"),
                "content": response.text,
                "_request_id": request_id,
                "_operation": "get_artifact"
            }
        
        return result

    async def send_message_streaming(self, dst_id: str, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Agent Protocol typically doesn't support streaming responses.
        This method falls back to single response and yields it.
        
        Parameters
        ----------
        dst_id : str
            Destination agent ID
        payload : Dict[str, Any]
            Message payload
        
        Yields
        ------
        Dict[str, Any]
            Single response event
        """
        # Agent Protocol is typically request-response, not streaming
        response = await self.send_message(dst_id, payload)
        yield response

    async def receive_message(self) -> Dict[str, Any]:
        """
        Receives messages by polling the task's steps.
        This is a simplified polling mechanism.
        """
        if not self.task_id:
            return {"messages": []}

        try:
            # This is a simplified example. A real implementation would
            # paginate through steps and handle different step states.
            response = await self.httpx_client.get(f"{self.base_url}/agent/tasks/{self.task_id}/steps")
            response.raise_for_status()
            
            raw_steps = response.json().get("steps", [])
            
            # Decode each step into a UTE
            utes = [DECODE_TABLE[self.protocol_name](step) for step in raw_steps]
            
            return {"messages": utes}

        except Exception:
            # In a real scenario, handle exceptions gracefully
            return {"messages": []}

    def get_agent_card(self) -> Dict[str, Any]:
        """Return the cached agent card."""
        return self.agent_card.copy()

    async def health_check(self) -> bool:
        """
        Check if the Agent Protocol agent is responsive.
        
        Uses standard health endpoint or falls back to agent card endpoint.
        
        Returns
        -------
        bool
            True if agent is responsive, False otherwise
        """
        try:
            headers = {}
            headers.update(self.auth_headers)
            
            # Try standard health endpoint first
            try:
                response = await self.httpx_client.get(
                    f"{self.base_url}/health",
                    headers=headers,
                    timeout=5.0
                )
                if response.status_code == 200:
                    return True
            except httpx.HTTPStatusError:
                pass  # Fall back to agent card
            
            # Fall back to agent card endpoint
            response = await self.httpx_client.get(
                f"{self.base_url}{self.agent_card_path}",
                headers=headers,
                timeout=5.0
            )
            return response.status_code == 200
            
        except Exception:
            return False

    async def cleanup(self) -> None:
        """
        Clean up adapter resources.
        
        Note: Does not close httpx_client since it's shared across adapters.
        """
        self.agent_card.clear()

    def get_capabilities(self) -> Dict[str, Any]:
        """Extract capabilities from agent card."""
        return self.agent_card.get("capabilities", {})

    def get_endpoint_info(self) -> Dict[str, Any]:
        """Get endpoint information."""
        return {
            "base_url": self.base_url,
            "protocol": "Agent Protocol",
            "api_version": self.api_version,
            "api_prefix": self._api_prefix,
            "health_endpoint": f"{self.base_url}/health",
            "tasks_endpoint": f"{self.base_url}{self._api_prefix}/tasks",
            "agent_card_endpoint": f"{self.base_url}{self.agent_card_path}",
            "supports_streaming": False,
            "supports_auth": bool(self.auth_headers),
            "supported_operations": [
                "create_task", "execute_step", "get_task", 
                "get_steps", "get_step", "get_artifacts", "get_artifact"
            ]
        }

    def get_protocol_version(self) -> str:
        """Get Agent Protocol version."""
        return self.agent_card.get("protocolVersion", self.api_version)

    def __repr__(self) -> str:
        """Debug representation of AgentProtocolAdapter."""
        return (
            f"AgentProtocolAdapter(base_url='{self.base_url}', "
            f"api_version='{self.api_version}', "
            f"auth={'enabled' if self.auth_headers else 'disabled'}, "
            f"card_loaded={bool(self.agent_card)})"
        )


class AgentProtocolMessageBuilder:
    """
    Helper class for building Agent Protocol messages.
    """
    
    @staticmethod
    def create_task_message(input_text: str, additional_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a create_task message."""
        return {
            "type": "create_task",
            "input": input_text,
            "additional_input": additional_input or {}
        }
    
    @staticmethod
    def execute_step_message(task_id: str, input_text: str = "", additional_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build an execute_step message."""
        return {
            "type": "execute_step",
            "task_id": task_id,
            "input": input_text,
            "additional_input": additional_input or {}
        }
    
    @staticmethod
    def get_task_message(task_id: str) -> Dict[str, Any]:
        """Build a get_task message."""
        return {
            "type": "get_task",
            "task_id": task_id
        }
    
    @staticmethod
    def get_steps_message(task_id: str) -> Dict[str, Any]:
        """Build a get_steps message."""
        return {
            "type": "get_steps",
            "task_id": task_id
        }
    
    @staticmethod
    def get_step_message(task_id: str, step_id: str) -> Dict[str, Any]:
        """Build a get_step message."""
        return {
            "type": "get_step",
            "task_id": task_id,
            "step_id": step_id
        }
    
    @staticmethod
    def get_artifacts_message(task_id: str) -> Dict[str, Any]:
        """Build a get_artifacts message."""
        return {
            "type": "get_artifacts",
            "task_id": task_id
        }
    
    @staticmethod
    def get_artifact_message(task_id: str, artifact_id: str) -> Dict[str, Any]:
        """Build a get_artifact message."""
        return {
            "type": "get_artifact",
            "task_id": task_id,
            "artifact_id": artifact_id
        }
