"""
Official Agora Protocol Adapter for AgentNetwork Framework
Based on analysis of agora-protocol/python official library
"""

import asyncio
import threading
from typing import Any, Dict, Optional, List, Callable
import json
import time
from flask import Flask, jsonify

# Official Agora imports
import agora

# AgentNetwork Framework imports
from agent_adapters.base_adapter import BaseProtocolAdapter
from server_adapters.base_adapter import BaseServerAdapter

# A2A SDK imports for executor bridge
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.types import Role


class AgoraClientAdapter(BaseProtocolAdapter):
    """
    Client Adapter using official agora-protocol library.
    
    Wraps official Agora Sender to integrate with AgentNetwork's adapter pattern.
    Provides automatic protocol negotiation and efficiency optimization.
    """
    
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
        return {"messages": []}
    
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


class AgoraServerAdapter(BaseServerAdapter):
    """
    Server Adapter using official agora-protocol library.
    
    Wraps official Agora Receiver and ReceiverServer to integrate with AgentNetwork.
    Bridges A2A executors to Agora tools.
    """
    
    protocol_name = "Agora (Official)"
    
    def build(
        self,
        host: str,
        port: int,
        agent_id: str,
        executor: Any,
        openai_api_key: Optional[str] = None,
        **kwargs
    ) -> tuple:
        """
        Build Agora server using official library.
        
        Creates Agora Receiver with tools bridged from A2A executor.
        """
        
        # Create toolformer (prefer LangChain, fallback to OpenAI)
        toolformer = self._create_toolformer(openai_api_key, **kwargs)
        
        # Create tools from A2A executor
        tools = self._create_agora_tools(executor, agent_id)
        
        # Create official Agora receiver
        receiver = agora.Receiver.make_default(toolformer, tools=tools)
        
        # Create server wrapper
        server_wrapper = AgoraServerWrapper(
            receiver=receiver,
            host=host,
            port=port,
            agent_id=agent_id,
            executor=executor
        )
        
        # Generate agent card
        agent_card = {
            "name": f"Agora Agent {agent_id}",
            "url": f"http://{host}:{port}/",
            "protocol": "Agora (Official)",
            "version": "1.0.0",
            "description": "Agent using official Agora Protocol library",
            "capabilities": {
                "protocol_negotiation": True,
                "automatic_efficiency": True,
                "multi_framework_support": True,
                "natural_language_processing": True,
                "structured_communication": True,
                "routine_generation": True,
                "cross_platform_interop": True
            },
            "toolformer": type(toolformer).__name__,
            "tools": [tool.__name__ for tool in tools],
            "a2a_executor": type(executor).__name__,
            "endpoints": {
                "health": f"http://{host}:{port}/health",
                "agora_endpoint": f"http://{host}:{port}/",
            }
        }
        
        return server_wrapper, agent_card
    
    def _create_toolformer(self, openai_api_key: Optional[str] = None, **kwargs) -> agora.Toolformer:
        """Create appropriate toolformer based on available dependencies."""
        
        # 获取模型名称，默认是 'gpt-4o-mini'
        model_name = kwargs.get('model', 'gpt-4o-mini')
        print(f"[DEBUG] Trying to create Toolformer with model: {model_name}")

        # 尝试使用 Camel 框架
        try:
            import camel.types
            print("[DEBUG] Using CamelToolformer")
            return agora.toolformers.CamelToolformer(
                camel.types.ModelPlatformType.OPENAI,
                camel.types.ModelType.GPT_4O_MINI
            )
        except ImportError:
            print("[WARN] Camel not available, falling back to LangChainToolformer again...")

        # 最后使用 LangChainToolformer 作为 fallback
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=model_name)
        return agora.toolformers.LangChainToolformer(model)

    def _create_agora_tools(self, executor: Any, agent_id: str) -> List[Callable]:
        """Create Agora-compatible tools from A2A executor."""
        
        tools = []
        
        def weather_service(city: str, date: str = "today"):
            """
            Get weather information for a city.
            
            Args:
                city: Name of the city
                date: Date for weather query
            """
            return self._bridge_to_a2a_executor(
                executor,
                {
                    "location": city,
                    "date": date,
                    "type": "weather",
                    "agent_id": agent_id
                }
            )
        
        def booking_service(service: str, datetime: str, details: str = ""):
            """
            Handle booking and reservation requests.
            
            Args:
                service: Type of service to book
                datetime: Requested date and time
                details: Additional booking details
            """
            import json
            try:
                details_dict = json.loads(details) if details else {}
            except:
                details_dict = {"notes": details}
                
            return self._bridge_to_a2a_executor(
                executor,
                {
                    "service": service,
                    "datetime": datetime,
                    "details": details_dict,
                    "type": "booking",
                    "agent_id": agent_id
                }
            )
        
        def data_service(query_type: str, parameters: str, filters: str = ""):
            """
            Handle data queries and searches.
            
            Args:
                query_type: Type of data query
                parameters: Query parameters as JSON string
                filters: Optional filters as JSON string
            """
            import json
            try:
                parameters_dict = json.loads(parameters) if parameters else {}
            except:
                parameters_dict = {"query": parameters}
                
            try:
                filters_dict = json.loads(filters) if filters else {}
            except:
                filters_dict = {}
                
            return self._bridge_to_a2a_executor(
                executor,
                {
                    "query_type": query_type,
                    "parameters": parameters_dict,
                    "filters": filters_dict,
                    "type": "data",
                    "agent_id": agent_id
                }
            )
        
        def general_service(message: str, context: str = ""):
            """
            Handle general messages and requests.
            
            Args:
                message: Message content
                context: Additional context as JSON string
            """
            import json
            try:
                context_dict = json.loads(context) if context else {}
            except:
                context_dict = {"notes": context}
                
            result = self._bridge_to_a2a_executor(
                executor,
                {
                    "text": message,
                    "context": context_dict,
                    "type": "general",
                    "agent_id": agent_id
                }
            )
            
            # Return string for general service
            if isinstance(result, dict) and "response" in result:
                return result["response"]
            elif isinstance(result, str):
                return result
            else:
                return str(result)
        
        tools.extend([weather_service, booking_service, data_service, general_service])
        return tools
    
    def _bridge_to_a2a_executor(self, executor: Any, payload: Dict[str, Any]) -> Any:
        """Bridge Agora tool call to A2A executor."""
        
        try:
            # Import A2A components
            from a2a.types import Message, MessageSendParams
            from a2a.server.agent_execution import RequestContext
            from a2a.server.events import EventQueue
            from a2a.utils import new_agent_text_message
            from a2a.types import Role
            
            # Convert payload to A2A message format
            if isinstance(payload, dict):
                # For structured data, pass as-is but ensure text representation
                message_text = payload.get("text", json.dumps(payload, indent=2))
            else:
                message_text = str(payload)
            
            # Create A2A message
            message = new_agent_text_message(message_text, role=Role.user)
            params = MessageSendParams(message=message)
            ctx = RequestContext(params)
            
            # Create event queue
            queue = EventQueue()
            
            # Execute A2A executor
            try:
                # Handle async executor
                if asyncio.iscoroutinefunction(executor.execute):
                    # Create new event loop if none exists
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Run in current loop or create new one
                    if loop.is_running():
                        # Create new loop for this execution
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor_pool:
                            future = executor_pool.submit(
                                asyncio.run,
                                executor.execute(ctx, queue)
                            )
                            future.result(timeout=30)  # 30 second timeout
                    else:
                        loop.run_until_complete(executor.execute(ctx, queue))
                else:
                    # Handle sync executor (unlikely but possible)
                    executor.execute(ctx, queue)
                
                # Collect events from queue
                events = []
                response_text = ""
                
                try:
                    while True:
                        event = queue.dequeue_event(no_wait=True)
                        
                        # Convert event to dict
                        if hasattr(event, 'model_dump'):
                            event_dict = event.model_dump()
                        elif hasattr(event, 'dict'):
                            event_dict = event.dict()
                        else:
                            event_dict = {"content": str(event)}
                        
                        events.append(event_dict)
                        
                        # Extract text for response
                        if 'text' in event_dict:
                            response_text += event_dict['text']
                        elif 'content' in event_dict:
                            response_text += str(event_dict['content'])
                
                except Exception:
                    # No more events in queue
                    pass
                
                # Return appropriate format based on payload type
                if payload.get("type") == "general":
                    return response_text or "No response generated"
                else:
                    return {
                        "response": response_text,
                        "events": events,
                        "status": "success",
                        "agent_id": payload.get("agent_id", "unknown")
                    }
                
            except Exception as e:
                return {
                    "error": f"A2A executor failed: {str(e)}",
                    "status": "error",
                    "payload": payload
                }
        
        except Exception as e:
            return f"Bridge error: {str(e)}"


class AgoraServerWrapper:
    """
    Wrapper to make official Agora ReceiverServer compatible with uvicorn.
    
    Bridges the gap between AgentNetwork's uvicorn-based server expectation
    and Agora's built-in server implementation.
    """
    
    def __init__(self, receiver, host: str, port: int, agent_id: str, executor: Any):
        self.receiver = receiver
        self.host = host
        self.port = port
        self.agent_id = agent_id
        self.executor = executor
        self.server_task = None
        self.should_exit_flag = False
        
        # Create official Agora server with health endpoint
        self.agora_server = self._create_enhanced_agora_server(receiver)
    
    def _create_enhanced_agora_server(self, receiver):
        """Create Agora ReceiverServer with additional health endpoint."""
        
        # Get the original Flask app from ReceiverServer
        original_server = agora.ReceiverServer(receiver)
        
        # Add health endpoint to the Flask app
        @original_server.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint for AgentNetwork compatibility."""
            return jsonify({
                "status": "healthy",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }), 200
        
        # Add agent card endpoint for AgentNetwork compatibility
        @original_server.app.route('/.well-known/agent.json', methods=['GET'])
        def agent_card():
            """Agent card endpoint for AgentNetwork compatibility."""
            return jsonify({
                "name": f"Agora Agent {self.agent_id}",
                "url": f"http://{self.host}:{self.port}/",
                "protocol": "Agora (Official)",
                "version": "1.0.0",
                "description": "Agent using official Agora Protocol library",
                "agent_id": self.agent_id,
                "capabilities": {
                    "protocol_negotiation": True,
                    "automatic_efficiency": True,
                    "multi_framework_support": True,
                    "natural_language_processing": True,
                    "structured_communication": True,
                    "routine_generation": True,
                    "cross_platform_interop": True
                },
                "endpoints": {
                    "health": f"http://{self.host}:{self.port}/health",
                    "agora_endpoint": f"http://{self.host}:{self.port}/",
                }
            }), 200
        
        return original_server
    
    async def serve(self):
        """Start the official Agora server in async context."""
        
        print(f"🚀 Starting Agora Server for {self.agent_id} on {self.host}:{self.port}")
        
        try:
            # Run Agora server in background thread
            server_thread = threading.Thread(
                target=self._run_agora_server,
                daemon=True
            )
            server_thread.start()
            
            # Wait a moment for server to start
            await asyncio.sleep(1)
            
            # Keep serving until told to stop
            while not self.should_exit_flag:
                if not server_thread.is_alive():
                    raise RuntimeError("Agora server thread died unexpectedly")
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            print(f"🛑 Agora Server for {self.agent_id} cancelled")
            self.should_exit_flag = True
            raise
        except Exception as e:
            print(f"❌ Agora Server for {self.agent_id} failed: {e}")
            raise
        finally:
            print(f"🔚 Agora Server for {self.agent_id} stopped")
    
    def _run_agora_server(self):
        """Run official Agora server in background thread."""
        try:
            print(f"📡 Agora ReceiverServer starting on {self.host}:{self.port}")
            self.agora_server.run(host=self.host, port=self.port, debug=False)
        except Exception as e:
            print(f"❌ Agora ReceiverServer error: {e}")
            self.should_exit_flag = True
    
    @property
    def should_exit(self):
        """Compatibility property for uvicorn-style shutdown."""
        return self.should_exit_flag
    
    @should_exit.setter
    def should_exit(self, value):
        """Set shutdown flag."""
        self.should_exit_flag = value
        if value:
            print(f"🛑 Shutdown signal received for Agora Server {self.agent_id}")