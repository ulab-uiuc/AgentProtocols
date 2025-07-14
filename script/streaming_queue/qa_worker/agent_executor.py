import asyncio
import os
import sys
from pathlib import Path
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message


class QAAgent:
    """QA Agent using Core LLM."""

    def __init__(self, config=None):
        """Initialize QA Agent with Core LLM."""
        self.core = None
        self.use_mock = False
        
        try:
            # Import Core class from utils
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from utils.core import Core
            
            # Use provided config or get default configuration
            if config is None:
                config = self._get_default_config()
            
            self.core = Core(config)
            print(f"[QAAgent] Initialized with Core LLM: {config['model']['name']}")
            
        except Exception as e:
            print(f"[QAAgent] Core LLM initialization failed: {e}")
            print("[QAAgent] Falling back to mock responses")
            self.use_mock = True
    
    def _get_default_config(self) -> dict:
        """Get default configuration for Core LLM."""
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if openai_key and openai_key != "test-key":
            # Use OpenAI configuration
            return {
                "model": {
                    "type": "openai",
                    "name": "gpt-3.5-turbo",
                    "openai_api_key": openai_key,
                    "temperature": 0.3
                }
            }
        else:
            # Use local model configuration
            return {
                "model": {
                    "type": "local",
                    "name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                    "temperature": 0.3
                },
                "base_url": "http://localhost:8000/v1",
                "port": 8000
            }

    async def invoke(self, question: str) -> str:
        """Answer a question using Core LLM."""
        try:
            if self.use_mock or self.core is None:
                # Mock response for testing
                await asyncio.sleep(0.1)
                return f"Mock answer for: {question[:50]}{'...' if len(question) > 50 else ''}"
            
            # Prepare messages for Core
            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant. Provide concise, accurate answers to questions. Keep responses under 150 words."
                },
                {
                    "role": "user", 
                    "content": question
                }
            ]
            
            # Call Core.execute() in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(None, self.core.execute, messages)
            
            return answer.strip() if answer else "Unable to generate response"
            
        except Exception as e:
            print(f"[QAAgent] Error processing question: {e}")
            return f"Error: {str(e)[:100]}..."


class QAAgentExecutor(AgentExecutor):
    """QA Agent Executor Implementation."""

    def __init__(self, config=None):
        self.agent = QAAgent(config)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # Get user input from context
        question = context.get_user_input()
        
        if not question:
            question = "What is artificial intelligence?"  # Default question
        
        # Get answer from QA agent
        result = await self.agent.invoke(question)
        
        # Send simple text message response
        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception('cancel not supported') 