"""
Official Agora Protocol Server Adapter for AgentNetwork Framework
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
from server_adapters.base_adapter import BaseServerAdapter

# A2A SDK imports for executor bridge
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.types import Role


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

        # 尝试使用 Camel 框架
        try:
            import camel.types
            return agora.toolformers.CamelToolformer(
                camel.types.ModelPlatformType.OPENAI,
                camel.types.ModelType.GPT_4O_MINI
            )
        except ImportError:
            pass  # Fall back to LangChain

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
            import sys
            
            # Check if executor has agora_qa_worker for direct LLM call
            try:
                if hasattr(executor, 'agora_qa_worker') and hasattr(executor.agora_qa_worker, 'answer'):
                    # Create async task for LLM call
                    async def call_agora_llm():
                        return await executor.agora_qa_worker.answer(message)
                    
                    # Run in thread to avoid blocking
                    import concurrent.futures
                    import threading
                    
                    result_container = [None]
                    error_container = [None]
                    
                    def run_async():
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            result = loop.run_until_complete(call_agora_llm())
                            result_container[0] = result
                        except Exception as e:
                            error_container[0] = e
                        finally:
                            loop.close()
                    
                    thread = threading.Thread(target=run_async)
                    thread.start()
                    thread.join(timeout=60)  # Increase timeout for browser_use
                    
                    if error_container[0]:
                        # Return error message instead of None
                        return f"Error: {error_container[0]}"
                    elif result_container[0]:
                        return str(result_container[0])
                    else:
                        return "No result returned from agora_qa_worker"
                
            except Exception as e:
                return f"Error in agora_qa_worker execution: {e}"
            
            # Fallback to original bridge logic
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
        
        # Check if this is our custom wrapper with direct LLM access
        if hasattr(executor, 'agora_qa_worker') and hasattr(executor.agora_qa_worker, 'answer'):
            message_text = payload.get("text", str(payload))
            print(f"[Agora Server] Using wrapper LLM call for: {message_text[:50]}...")
            
            # Create async wrapper for sync bridge call
            async def call_llm():
                return await executor.agora_qa_worker.answer(message_text)
            
            # Run async call in current event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in async context, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, call_llm())
                        result = future.result(timeout=30)
                else:
                    result = loop.run_until_complete(call_llm())
                
                print(f"[Agora Server] LLM result: {len(result)} chars")
                return result
                
            except Exception as e:
                print(f"[Agora Server] LLM call failed: {e}")
                # Fall back to original logic
        
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
        
        # Add message endpoint for direct LLM processing
        @original_server.app.route('/message', methods=['POST'])
        def handle_message():
            """Handle incoming messages for LLM processing."""
            try:
                from flask import request, jsonify
                import sys

                data = request.get_json()

                # Extract message text from nested structure
                text_data = data.get('text', data)
                if isinstance(text_data, dict):
                    # Handle nested structure: {'text': {'message': {'content': '...'}}}
                    message_obj = text_data.get('message', text_data)
                    if isinstance(message_obj, dict):
                        message_text = message_obj.get('content', str(message_obj))
                    else:
                        message_text = str(message_obj)
                else:
                    message_text = str(text_data)

                # Check if executor has agora_qa_worker for direct LLM call
                if hasattr(self.executor, 'agora_qa_worker') and hasattr(self.executor.agora_qa_worker, 'answer'):

                    # Create async function for LLM call
                    async def call_llm():
                        return await self.executor.agora_qa_worker.answer(message_text)

                    # Run async call in new thread with event loop
                    import threading
                    import asyncio

                    result_container = [None]
                    error_container = [None]

                    def run_async():
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            result = loop.run_until_complete(call_llm())
                            result_container[0] = result
                        except Exception as e:
                            error_container[0] = e
                        finally:
                            loop.close()

                    thread = threading.Thread(target=run_async)
                    thread.start()
                    thread.join(timeout=120)  # Increase timeout for complex tasks

                    if error_container[0]:
                        error_msg = str(error_container[0])
                        return jsonify({"error": error_msg}), 500
                    elif result_container[0] is not None:
                        result = result_container[0]

                        # Normalize return types to JSON
                        if isinstance(result, (dict, list)):
                            # Return dict/list directly as JSON
                            return jsonify(result), 200
                        else:
                            # Return stringy results under 'text' key for compatibility
                            return jsonify({"text": str(result), "status": "success"}), 200
                    else:
                        return jsonify({"text": "No result", "status": "timeout"}), 500
                else:
                    # Non-LLM path: return structured JSON for compatibility
                    return jsonify({"text": f"Agora processed: {message_text}", "status": "fallback"}), 200

            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
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