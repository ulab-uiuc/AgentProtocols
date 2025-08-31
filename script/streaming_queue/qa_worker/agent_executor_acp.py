import asyncio
import os
import sys
import json
import time
from pathlib import Path
from typing import List, AsyncGenerator, Dict

# Add ACP SDK imports
try:
    from acp_sdk.models import Message, MessagePart
    from acp_sdk.server import Context
    ACP_AVAILABLE = True
    print("ACP SDK imports successful")
except ImportError as e:
    print(f"Warning: acp-sdk not available: {e}, using mock classes")
    ACP_AVAILABLE = False

    # Mock classes for when acp-sdk is not available
    class Message:
        def __init__(self, parts=None):
            self.parts = parts or []

    class MessagePart:
        def __init__(self, type_=None, text="", **kwargs):
            self.type = type_
            self.text = text

    class Context:
        def __init__(self, session=None, store=None, loader=None, executor=None,
                     request=None, yield_queue=None, yield_resume_queue=None, **kwargs):
            self.session = session
            self.store = store
            self.loader = loader
            self.executor = executor
            self.request = request
            self.yield_queue = yield_queue
            self.yield_resume_queue = yield_resume_queue


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
                    "name": "llama3.1:latest",
                    "base_url": "http://localhost:11434",
                    "temperature": 0.3
                }
            }

    async def answer_question(self, question: str, context: str = "") -> str:
        """Answer a question using Core LLM."""
        if self.use_mock:
            return f"Mock answer for: {question[:50]}..."

        try:
            # Prepare the prompt
            if context:
                prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            else:
                prompt = f"Question: {question}\n\nAnswer:"

            # Use Core LLM to generate response
            messages = [{"role": "user", "content": prompt}]
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.core.execute, messages)

            if response and hasattr(response, 'text'):
                return response.text
            elif isinstance(response, str):
                return response
            else:
                return "No response generated"

        except Exception as e:
            print(f"[QAAgent] Error generating answer: {e}")
            return f"Error: {str(e)}"

    def get_status(self) -> Dict:
        """Get current worker status."""
        return {
            "worker_id": self.worker_id,
            "status": "active",
            "processed_questions": 0,
            "capabilities": self.capabilities
        }

    def process_question(self, question: str, context: str = "") -> str:
        """Process a single question and return an answer."""
        # Simple mock processing - in real implementation, this would use the worker's agent
        return f"Answer to '{question}': This is a mock answer from worker {self.worker_id}"

    async def process_question_batch(self, questions: List[dict]) -> List[dict]:
        """Process a batch of questions."""
        results = []

        for question_data in questions:
            question_id = question_data.get('question_id', 0)
            question = question_data.get('question', '')
            context = question_data.get('context', '')

            try:
                answer = await self.answer_question(question, context)
                results.append({
                    'question_id': question_id,
                    'question': question,
                    'answer': answer,
                    'status': 'completed'
                })
            except Exception as e:
                results.append({
                    'question_id': question_id,
                    'question': question,
                    'answer': f"Error: {str(e)}",
                    'status': 'error'
                })

        return results


class QAAgentExecutorACP:
    """ACP Executor for QA Agent using async generator pattern."""

    def __init__(self, config: dict = None):
        """Initialize the ACP executor."""
        self.config = config or {}
        self.agent = QAAgent(config)
        self.agent_id = "qa-agent-acp"
        self.capabilities = ["question_answering", "batch_processing", "llm_generation"]

    async def execute(self, messages: List[Message], context: Context) -> AsyncGenerator[MessagePart, None]:
        """
        Execute the QA worker logic.

        Args:
            messages: List of messages from the user
            context: ACP context

        Yields:
            MessagePart: Progress updates, results, or errors
        """
        try:
            # Check if we have any messages
            if not messages:
                yield MessagePart(type="error", text="No messages provided")
                return

            # Process the first message
            first_message = messages[0]
            if not first_message.parts:
                yield MessagePart(type="error", text="Message has no parts")
                return

            # Extract command from the first message part
            command_part = first_message.parts[0]
            # Check for text in both 'text' and 'content' fields
            text = ""
            if hasattr(command_part, 'text') and command_part.text:
                text = command_part.text.strip()
            elif hasattr(command_part, 'content') and command_part.content:
                text = command_part.content.strip()
            else:
                text = str(command_part).strip()

            # Check if this is a structured message (JSON)
            if text.startswith('{') and text.endswith('}'):
                try:
                    # Parse structured message
                    message_data = json.loads(text)

                    if "question_id" in message_data:
                        # Single question processing
                        yield MessagePart(type="text", text=f"Processing question {message_data['question_id']}")

                        # Process the question
                        question = message_data.get("question", "")
                        context_info = message_data.get("context", "")

                        # Generate response
                        response = {
                            "question_id": message_data["question_id"],
                            "question": question,
                            "answer": self.process_question(question, context_info),
                            "worker_id": self.worker_id,
                            "timestamp": time.time()
                        }

                        yield MessagePart(type="json", text=json.dumps(response, indent=2))

                    elif "questions" in message_data:
                        # Batch processing
                        questions = message_data["questions"]
                        yield MessagePart(type="text", text=f"Processing batch of {len(questions)} questions")

                        # Process batch
                        yield MessagePart(type="json", text=json.dumps({
                            "batch_id": message_data.get("batch_id", "unknown"),
                            "processed_count": len(questions),
                            "worker_id": self.worker_id,
                            "timestamp": time.time()
                        }, indent=2))

                    else:
                        yield MessagePart(type="error", text="Unknown structured message format")

                except json.JSONDecodeError:
                    # Not valid JSON, treat as plain text
                    pass

            # Handle command-based messages
            parts = text.split()
            if parts:
                command = parts[0].lower()

                if command == "status":
                    status = self.get_status()
                    yield MessagePart(type="json", text=json.dumps(status, indent=2))

                elif command == "help":
                    help_text = """
QA Worker Commands:
- status: Get current worker status
- help: Show this help message
- <question>: Process a question
"""
                    yield MessagePart(type="text", text=help_text)

                else:
                    # Treat as a question to process
                    # Process the question directly with LLM
                    answer = await self.agent.answer_question(text, "")

                    # Return the answer as text (similar to A2A version)
                    yield MessagePart(type="text", text=answer)

        except Exception as e:
            yield MessagePart(type="error", text=f"Execution error: {str(e)}")
            print(f"Error in QAWorkerExecutorACP.execute: {e}")
            import traceback
            traceback.print_exc()

    def set_agent_network(self, agent_network):
        """Set the agent network reference (if needed)."""
        self.agent_network = agent_network