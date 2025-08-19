"""
Official Agora Protocol Client Adapter for AgentNetwork Framework
Based on analysis of agora-protocol/python official library
"""

import asyncio
from typing import Any, Dict, Optional
import json
import time

# Official Agora imports
import agora

# AgentNetwork Framework imports
try:
    from .base_adapter import BaseProtocolAdapter
    from ..core.protocol_converter import DECODE_TABLE
except ImportError:
    from agent_adapters.base_adapter import BaseProtocolAdapter
    from core.protocol_converter import DECODE_TABLE


class AgoraClientAdapter(BaseProtocolAdapter):
    """
    Client Adapter using official agora-protocol library.
    
    Wraps official Agora Sender to integrate with AgentNetwork's adapter pattern.
    Provides automatic protocol negotiation and efficiency optimization.
    """

    @property
    def protocol_name(self) -> str:
        return "agora"
    
    def __init__(
        self,
        toolformer: agora.Toolformer,
        target_url: str,
        auth_headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize Agora Client Adapter.
        
        Parameters
        ----------
        toolformer : agora.Toolformer
            Official Agora toolformer (LangChain, Camel, OpenAI, etc.)
        target_url : str
            Target agent URL for communication
        auth_headers : Optional[Dict[str, str]]
            Authentication headers (passed through to HTTP requests)
        """
        super().__init__(**kwargs)
        self.toolformer = toolformer
        self.target_url = target_url.rstrip('/')
        self.auth_headers = auth_headers or {}
        
        # Create official Agora sender
        self.sender = agora.Sender.make_default(toolformer)
        
        # Track task usage for metrics
        self.task_usage = {}
        self.total_calls = 0
        self.protocol_negotiations = 0
        
        # Register dynamic communication tasks
        self._register_communication_tasks()
    
    def _register_communication_tasks(self):
        """Register Agora tasks for different communication patterns."""
        
        # Weather communication task
        @self.sender.task()
        def weather_query(
            location: str, 
            date: str = "today",
            format: str = "standard"
        ) -> Dict[str, Any]:
            """
            Query weather information from remote agent.
            
            Parameters:
            location: City or location name
            date: Date for weather query (default: today)
            format: Response format preference
            
            Returns:
            Weather data dictionary with temperature, conditions, etc.
            """
            pass
        
        # Booking/reservation task
        @self.sender.task()
        def booking_request(
            service: str,
            datetime: str,
            details: Dict[str, Any] = None,
            user_info: Dict[str, str] = None
        ) -> Dict[str, Any]:
            """
            Make booking or reservation request.
            
            Parameters:
            service: Type of service (hotel, restaurant, taxi, etc.)
            datetime: Requested date and time
            details: Additional booking requirements
            user_info: User information for booking
            
            Returns:
            Booking confirmation with ID and status
            """
            pass
        
        # Data query task
        @self.sender.task()
        def data_query(
            query_type: str,
            parameters: Dict[str, Any],
            filters: Dict[str, Any] = None
        ) -> Any:
            """
            Generic data query to remote service.
            
            Parameters:
            query_type: Type of data being requested
            parameters: Query parameters
            filters: Optional filters to apply
            
            Returns:
            Query results in appropriate format
            """
            pass
        
        # General message task
        @self.sender.task()
        def send_message(
            message: str,
            message_type: str = "general",
            context: Dict[str, Any] = None
        ) -> Any:
            """
            Send general message to remote agent.
            
            Parameters:
            message: Message content
            message_type: Type of message for context
            context: Additional context information
            
            Returns:
            Response from remote agent
            """
            pass
        
        # Store task references
        self.tasks = {
            'weather': weather_query,
            'booking': booking_request,
            'data': data_query,
            'general': send_message
        }
    
    async def initialize(self) -> None:
        """Initialize Agora client adapter."""
        print(f"🔗 Agora Client Adapter initialized for {self.target_url}")
        print(f"📋 Available tasks: {list(self.tasks.keys())}")
    
    async def send_message(self, dst_id: str, payload: Dict[str, Any]) -> Any:
        """
        Send message using official Agora protocol.
        
        Automatically selects appropriate task based on payload content
        and handles protocol negotiation, routine generation, and efficiency optimization.
        """
        self.total_calls += 1
        start_time = time.time()
        
        try:
            # Determine message type and select appropriate task
            message_type = self._classify_message(payload)
            task_name = self._select_task(message_type, payload)
            
            # Track task usage
            self.task_usage[task_name] = self.task_usage.get(task_name, 0) + 1
            
            # Execute task with official Agora
            result = await self._execute_agora_task(task_name, payload)
            
            # Record metrics
            duration = time.time() - start_time
            print(f"✅ Agora task '{task_name}' completed in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Agora task failed after {duration:.2f}s: {e}")
            raise RuntimeError(f"Agora communication failed: {e}")
    
    def _classify_message(self, payload: Dict[str, Any]) -> str:
        """Classify message type based on payload content."""
        
        # Check explicit type
        if 'type' in payload:
            return payload['type']
        
        # Infer from content
        content_lower = str(payload).lower()
        
        if any(word in content_lower for word in ['weather', 'temperature', 'rain', 'sunny', 'location']):
            return 'weather'
        elif any(word in content_lower for word in ['book', 'reservation', 'service', 'datetime']):
            return 'booking'
        elif any(word in content_lower for word in ['query', 'search', 'data', 'find']):
            return 'data'
        else:
            return 'general'
    
    def _select_task(self, message_type: str, payload: Dict[str, Any]) -> str:
        """Select appropriate Agora task based on message type."""
        
        if message_type in self.tasks:
            return message_type
        else:
            return 'general'  # Fallback to general message task
    
    async def _execute_agora_task(self, task_name: str, payload: Dict[str, Any]) -> Any:
        """Execute Agora task in async context."""
        
        task_func = self.tasks[task_name]
        
        # Prepare task parameters based on task type
        if task_name == 'weather':
            params = {
                'location': payload.get('location', payload.get('city', 'unknown')),
                'date': payload.get('date', 'today'),
                'format': payload.get('format', 'standard'),
                'target': self.target_url
            }
        elif task_name == 'booking':
            params = {
                'service': payload.get('service', 'general'),
                'datetime': payload.get('datetime', payload.get('date', 'now')),
                'details': payload.get('details', {}),
                'user_info': payload.get('user_info', {}),
                'target': self.target_url
            }
        elif task_name == 'data':
            params = {
                'query_type': payload.get('query_type', 'general'),
                'parameters': payload.get('parameters', payload),
                'filters': payload.get('filters', {}),
                'target': self.target_url
            }
        else:  # general
            params = {
                'message': payload.get('message', payload.get('text', str(payload))),
                'message_type': payload.get('type', 'general'),
                'context': payload.get('context', {}),
                'target': self.target_url
            }
        
        # Execute in thread to handle sync/async compatibility
        loop = asyncio.get_event_loop()
        
        def run_task():
            return task_func(**params)
        
        # Run Agora task in executor to handle potential blocking calls
        result = await loop.run_in_executor(None, run_task)
        return result
    
    async def receive_message(self) -> Dict[str, Any]:
        """Receive messages (not applicable for client adapter)."""
        raw_message = {"messages": []} # Placeholder for actual message reception
        ute = DECODE_TABLE[self.protocol_name](raw_message)
        return {"messages": [ute]}
    
    def get_agent_card(self) -> Dict[str, Any]:
        """Get agent card for Agora client adapter."""
        
        efficiency_ratio = 0.0
        if self.total_calls > 0:
            # Estimate efficiency based on task usage patterns
            # Higher usage of specific tasks indicates better protocol optimization
            max_usage = max(self.task_usage.values()) if self.task_usage else 0
            efficiency_ratio = max_usage / self.total_calls if self.total_calls > 0 else 0
        
        return {
            "protocol": "Agora (Official)",
            "version": "1.0.0",
            "library": "agora-protocol",
            "target_url": self.target_url,
            "capabilities": {
                "protocol_negotiation": True,
                "automatic_routines": True,
                "natural_language_fallback": True,
                "multi_framework_support": True,
                "efficiency_optimization": True,
                "cross_platform": True
            },
            "toolformer": type(self.toolformer).__name__,
            "tasks": list(self.tasks.keys()),
            "metrics": {
                "total_calls": self.total_calls,
                "task_usage": self.task_usage,
                "efficiency_ratio": efficiency_ratio,
                "protocol_negotiations": self.protocol_negotiations
            }
        }
    
    async def health_check(self) -> bool:
        """Check health of target agent."""
        try:
            # Use general message task for health check
            result = await self._execute_agora_task('general', {
                'message': 'health_check',
                'type': 'system'
            })
            return True
        except Exception:
            return False
    
    async def cleanup(self) -> None:
        """Clean up Agora client adapter."""
        print(f"🧹 Cleaning up Agora Client Adapter")
        print(f"📊 Final metrics: {self.get_agent_card()['metrics']}")