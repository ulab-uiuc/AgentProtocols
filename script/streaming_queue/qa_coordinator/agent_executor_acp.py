print("=== Starting qa_coordinator/agent_executor_acp.py ===")

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, AsyncGenerator

print("Basic imports successful")

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

print("Starting to define classes...")


class QACoordinator:
    """QA Coordinator for dispatching questions to worker agents."""

    def __init__(self, coordinator_id: str = "Coordinator-1", config: dict = None, output=None):
        """Initialize QA Coordinator."""
        print("QACoordinator.__init__ called")
        config = config or {}
        self.coordinator_id = coordinator_id
        coordinator_config = config.get('qa', {}).get('coordinator', {})

        self.batch_size = coordinator_config.get('batch_size', 50)
        self.first_50 = coordinator_config.get('first_50', True)
        self.data_path = coordinator_config.get('data_file', 'data/top1000_simplified.jsonl')
        self.worker_ids: List[str] = []
        self.agent_network = None
        self.output = output

        # Initialize statistics
        self.stats = {
            'total_questions': 0,
            'processed_questions': 0,
            'failed_questions': 0,
            'start_time': None,
            'end_time': None,
            'worker_stats': {}
        }

        # Load questions from data file
        self.questions = self._load_questions()
        print(f"[QACoordinator] Loaded {len(self.questions)} questions from {self.data_path}")

    def _load_questions(self) -> List[Dict]:
        """Load questions from JSONL file."""
        questions = []
        try:
            data_file = Path(self.data_path)
            if not data_file.exists():
                print(f"[QACoordinator] Data file not found: {self.data_path}")
                return []

            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            question_data = json.loads(line)
                            questions.append(question_data)
                        except json.JSONDecodeError as e:
                            print(f"[QACoordinator] Failed to parse line: {line}, error: {e}")

            if self.first_50:
                questions = questions[:50]
                print(f"[QACoordinator] Using first 50 questions")

        except Exception as e:
            print(f"[QACoordinator] Error loading questions: {e}")

        return questions

    def set_agent_network(self, agent_network):
        """Set the agent network reference."""
        self.agent_network = agent_network

    def register_worker(self, worker_id: str):
        """Register a worker agent."""
        if worker_id not in self.worker_ids:
            self.worker_ids.append(worker_id)
            self.stats['worker_stats'][worker_id] = {
                'questions_assigned': 0,
                'questions_completed': 0,
                'avg_response_time': 0
            }
            print(f"[QACoordinator] Registered worker: {worker_id}")

    def unregister_worker(self, worker_id: str):
        """Unregister a worker agent."""
        if worker_id in self.worker_ids:
            self.worker_ids.remove(worker_id)
            print(f"[QACoordinator] Unregistered worker: {worker_id}")

    def get_stats(self):
        """Get coordinator statistics."""
        return {
            "coordinator_id": self.coordinator_id,
            "registered_workers": len(self.worker_ids),
            "worker_ids": self.worker_ids,
            "questions_loaded": len(self.questions),
            "coordinator_status": "active"
        }

    async def dispatch_questions(self) -> AsyncGenerator[MessagePart, None]:
        """Dispatch questions to available workers."""
        if not self.worker_ids:
            yield MessagePart(type="text", text="No workers available for dispatching questions")
            return

        if not self.questions:
            yield MessagePart(type="text", text="No questions to dispatch")
            return

        # Start dispatching
        start_time = time.time()

        yield MessagePart(type="text", text=f"Starting to dispatch {len(self.questions)} questions to {len(self.worker_ids)} workers")

        # Prepare batch assignment
        batch_size = len(self.questions) // len(self.worker_ids)
        remainder = len(self.questions) % len(self.worker_ids)

        # Dispatch questions in batches
        question_index = 0
        for i, worker_id in enumerate(self.worker_ids):
            # Calculate batch size for this worker
            current_batch_size = batch_size + (1 if i < remainder else 0)

            if question_index >= len(self.questions):
                break

            # Assign batch to worker
            batch_questions = self.questions[question_index:question_index + current_batch_size]
            question_index += current_batch_size

            try:
                # Here you would send the batch to the worker
                # For now, we'll simulate processing
                await asyncio.sleep(0.1)  # Simulate dispatch time

                # Update progress
                if (i + 1) % 10 == 0 or i == len(self.worker_ids) - 1:
                    progress = ((i + 1) / len(self.worker_ids)) * 100
                    yield MessagePart(type="progress", text=f"Dispatched {i + 1}/{len(self.questions)} questions ({progress:.1f}%)")

            except Exception as e:
                # Handle dispatch errors
                yield MessagePart(type="error", text=f"Failed to dispatch question {i}: {str(e)}")

        # Final stats
        duration = time.time() - start_time
        yield MessagePart(type="text", text=f"Completed dispatching {self.stats['processed_questions']} questions in {duration:.2f}s")
        yield MessagePart(type="json", text=json.dumps(self.stats, indent=2))

    async def _send_question_to_worker(self, worker_id: str, question: Dict, question_id: int):
        """Send a question to a specific worker."""
        if not self.agent_network:
            raise Exception("Agent network not set")

        # Format the question for the worker
        message_payload = {
            "question_id": question_id,
            "question": question.get("question", ""),
            "context": question.get("context", ""),
            "coordinator_id": self.coordinator_id
        }

        # Send via agent network
        await self.agent_network.send_message(worker_id, message_payload)

    def get_stats(self) -> Dict:
        """Get current statistics."""
        return self.stats.copy()

    async def process_single_question(self, question_data: Dict) -> Dict:
        """Process a single question by sending it to an available worker."""
        try:
            question_id = question_data.get('id', 0)
            question_text = question_data.get('question', question_data.get('q', ''))  # Handle both 'question' and 'q'

            # Check if we have workers available
            if not self.worker_ids:
                return {
                    "question_id": question_id,
                    "question": question_text,
                    "answer": "No workers available",
                    "status": "failed",
                    "processing_time": 0.0
                }

            # Check if agent network is available
            if not self.agent_network:
                # Fall back to mock processing
                await asyncio.sleep(0.1)
                answer = f"Mock answer for question {question_id}: '{question_text[:50]}...'"
                return {
                    "question_id": question_id,
                    "question": question_text,
                    "answer": answer,
                    "status": "completed",
                    "processing_time": 0.1
                }

            # Select worker using round-robin
            worker_id = self.worker_ids[question_id % len(self.worker_ids)]
            print(f"[QACoordinator] Sending question {question_id} to {worker_id}")

            # Prepare ACP message format
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "content": question_text
                            }
                        ]
                    }
                ]
            }

            # Send message to Worker through network
            start_time = time.time()
            try:
                response = await self.agent_network.route_message(self.coordinator_id, worker_id, payload)
                end_time = time.time()

                # Extract answer from ACP response format
                answer = "No answer received"
                if "results" in response and response["results"]:
                    for result in response["results"]:
                        if "text" in result:
                            answer = result["text"]
                            break
                        elif "content" in result:
                            answer = result["content"]
                            break

                # Build result record
                result_record = {
                    "question_id": question_id,
                    "question": question_text,
                    "worker": worker_id,
                    "answer": answer,
                    "response_time": end_time - start_time,
                    "timestamp": time.time(),
                    "status": "completed" if answer != "No answer received" else "failed",
                    "processing_time": end_time - start_time
                }

                print(f"[QACoordinator] Question {question_id} processed by {worker_id}")
                return result_record

            except Exception as e:
                # Build failure record
                result_record = {
                    "question_id": question_id,
                    "question": question_text,
                    "worker": worker_id,
                    "answer": f"Error processing question: {str(e)}",
                    "response_time": 0.0,
                    "timestamp": time.time(),
                    "status": "error",
                    "processing_time": 0.0
                }
                print(f"[QACoordinator] Error processing question {question_id}: {e}")
                return result_record

        except Exception as e:
            return {
                "question_id": question_data.get('id', 0),
                "question": question_data.get('question', ''),
                "answer": f"Error processing question: {str(e)}",
                "status": "error",
                "processing_time": 0.0
            }

    async def dispatch_round(self):
        """Dispatch question round and save results"""
        questions = self._load_questions()
        if not questions:
            return "Error: No questions loaded"

        start_time = time.time()

        # Use dynamic load balancing if we have worker network, otherwise process sequentially
        if self.agent_network and self.worker_ids:
            all_results = await self.dispatch_questions_dynamically(questions)
        else:
            # Process questions sequentially (fallback to mock processing)
            print("[QACoordinator] No agent network available, using sequential processing")
            all_results = []
            for question_data in questions:
                result = await self.process_single_question(question_data)
                all_results.append(result)

        end_time = time.time()

        # Save results to file
        await self.save_results(all_results)

        success_count = len([r for r in all_results if r.get("status") == "completed"])
        failed_count = len([r for r in all_results if r.get("status") in ["error", "failed"]])

        summary = (
            f"Dispatch round completed in {end_time - start_time:.2f} seconds\n"
            f"Total processed: {len(all_results)} questions\n"
            f"Successfully processed: {success_count} questions\n"
            f"Failed: {failed_count} questions\n"
            f"Results saved to file"
        )

        return summary

    async def dispatch_questions_dynamically(self, questions: List[Dict]):
        """Dynamic load balancing question dispatch"""
        if not self.agent_network or not self.worker_ids:
            return []

        print(f"[QACoordinator] Starting dynamic load balancing: {len(questions)} questions, {len(self.worker_ids)} workers")

        # Create question queue
        question_queue = asyncio.Queue()
        for q in questions:
            await question_queue.put(q)

        # Create result queue
        result_queue = asyncio.Queue()

        # Create processing tasks for each Worker
        worker_tasks = []
        for worker_id in self.worker_ids:
            task = asyncio.create_task(
                self.worker_processor(worker_id, question_queue, result_queue)
            )
            worker_tasks.append(task)

        # Result collection task
        result_collector_task = asyncio.create_task(
            self.result_collector(result_queue, len(questions))
        )

        # Wait for all tasks to complete
        results = await result_collector_task

        # Wait for all Worker tasks to complete
        await asyncio.gather(*worker_tasks, return_exceptions=True)

        return results

    async def worker_processor(self, worker_id: str, question_queue: asyncio.Queue, result_queue: asyncio.Queue):
        """Worker processor - continuously get tasks from queue"""
        processed_count = 0

        while True:
            try:
                # Get question from queue (non-blocking, exit if queue is empty)
                try:
                    question_data = question_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                processed_count += 1
                question_id = question_data['id']
                question = question_data.get('question', question_data.get('q', ''))  # Handle both 'question' and 'q'

                print(f"[QACoordinator] {worker_id} starting to process question {question_id}")

                # Prepare ACP message format
                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "content": question
                                }
                            ]
                        }
                    ]
                }

                # Send message to Worker through network
                start_time = time.time()
                try:
                    response = await self.agent_network.route_message(self.coordinator_id, worker_id, payload)
                    end_time = time.time()

                    # Extract answer from ACP response format
                    answer = "No answer received"
                    if "results" in response and response["results"]:
                        for result in response["results"]:
                            if "text" in result:
                                answer = result["text"]
                                break
                            elif "content" in result:
                                answer = result["content"]
                                break

                    # Build result record
                    result_record = {
                        "question_id": question_id,
                        "question": question,
                        "worker": worker_id,
                        "answer": answer,
                        "response_time": end_time - start_time,
                        "timestamp": time.time(),
                        "status": "completed" if answer != "No answer received" else "failed"
                    }

                    # Put result into result queue
                    await result_queue.put(result_record)

                    print(f"[QACoordinator] {worker_id} completed question {question_id}")

                except Exception as e:
                    # Build failure record
                    result_record = {
                        "question_id": question_id,
                        "question": question,
                        "worker": worker_id,
                        "answer": f"Error: {str(e)}",
                        "response_time": 0.0,
                        "timestamp": time.time(),
                        "status": "error"
                    }

                    # Put result into result queue
                    await result_queue.put(result_record)

                    print(f"[QACoordinator] {worker_id} failed to process question {question_id}: {e}")

                # Mark task as done
                question_queue.task_done()

            except Exception as e:
                print(f"[QACoordinator] Error in worker processor {worker_id}: {e}")
                break

        print(f"[QACoordinator] {worker_id} finished processing, total processed: {processed_count}")

    async def result_collector(self, result_queue: asyncio.Queue, expected_count: int):
        """Collect results from result queue"""
        results = []
        collected_count = 0

        while collected_count < expected_count:
            try:
                result = await asyncio.wait_for(result_queue.get(), timeout=30.0)
                results.append(result)
                collected_count += 1
                result_queue.task_done()

                # Progress update
                if collected_count % 10 == 0 or collected_count == expected_count:
                    print(f"[QACoordinator] Collected {collected_count}/{expected_count} results")

            except asyncio.TimeoutError:
                print(f"[QACoordinator] Timeout waiting for result, collected {collected_count}/{expected_count}")
                break

        return results

    async def save_results(self, results):
        """Save results to file"""
        if not results:
            return

        # Create results directory if it doesn't exist
        results_dir = Path(__file__).parent.parent / "data"
        results_dir.mkdir(exist_ok=True)

        # Save results with timestamp
        timestamp = int(time.time())
        results_file = results_dir / f"qa_results_acp_{timestamp}.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "total_questions": len(results),
                "results": results
            }, f, indent=2, ensure_ascii=False)

        print(f"[QACoordinator] Results saved to {results_file}")


class QACoordinatorExecutorACP:
    """ACP Executor for QA Coordinator using async generator pattern."""

    def __init__(self, coordinator_id: str = "Coordinator-1", config: dict = None):
        """Initialize the ACP executor."""
        print("QACoordinatorExecutorACP.__init__ called")
        self.config = config or {}
        self.coordinator = QACoordinator(coordinator_id, config)
        self.agent_id = "qa-coordinator-acp"
        self.coordinator_id = coordinator_id
        self.capabilities = ["qa_coordination", "task_distribution", "progress_tracking"]

    def get_stats(self) -> Dict:
        """Get statistics from the coordinator."""
        return self.coordinator.get_stats()

    async def process_single_question(self, question_data: Dict) -> Dict:
        """Process a single question and return the result."""
        try:
            question_id = question_data.get('id', 0)
            question_text = question_data.get('question', '')

            # For now, we'll simulate processing since we don't have worker communication set up
            # In a real implementation, this would send to a worker and get the response
            await asyncio.sleep(0.1)  # Simulate processing time

            # Mock answer generation
            answer = f"This is a mock answer for question {question_id}: '{question_text[:50]}...'"

            return {
                "question_id": question_id,
                "question": question_text,
                "answer": answer,
                "status": "completed",
                "processing_time": 0.1
            }

        except Exception as e:
            return {
                "question_id": question_data.get('id', 0),
                "question": question_data.get('question', ''),
                "answer": f"Error processing question: {str(e)}",
                "status": "error",
                "processing_time": 0.0
            }

    async def execute(self, messages: List[Message], context: Context) -> AsyncGenerator[MessagePart, None]:
        """
        Execute the QA coordinator logic.

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
            command = ""
            if hasattr(command_part, 'text') and command_part.text:
                command = command_part.text.strip()
            elif hasattr(command_part, 'content') and command_part.content:
                command = command_part.content.strip()
            else:
                command = str(command_part).strip()

            # Check if this is a JSON batch message
            if command.startswith('{') and command.endswith('}'):
                try:
                    batch_data = json.loads(command)

                    if "batch_id" in batch_data and "questions" in batch_data:
                        # Process batch of questions
                        yield MessagePart(type="text", text=f"Processing batch {batch_data['batch_id']} with {len(batch_data['questions'])} questions")

                        # Process each question in the batch
                        batch_results = []
                        for question_data in batch_data["questions"]:
                            result = await self.process_single_question(question_data)
                            batch_results.append(result)

                        # Return batch results
                        response = {
                            "batch_id": batch_data["batch_id"],
                            "total_questions": len(batch_data["questions"]),
                            "results": batch_results
                        }
                        yield MessagePart(type="json", text=json.dumps(response, indent=2))
                        return

                except json.JSONDecodeError:
                    # Not valid JSON, continue with command processing
                    pass

            # Parse command
            parts = command.split()
            if not parts:
                yield MessagePart(type="error", text="Empty command")
                return

            action = parts[0].lower()

            # Handle different commands
            if action == "dispatch" or action == "start_dispatch":
                # Call the existing dispatch_round method
                result = await self.coordinator.dispatch_round()
                yield MessagePart(type="text", text=result)

            elif action == "register":
                if len(parts) >= 3 and parts[1] == "worker":
                    worker_id = parts[2]
                    self.coordinator.worker_ids.append(worker_id)
                    yield MessagePart(type="text", text=f"Worker {worker_id} registered successfully")
                else:
                    yield MessagePart(type="error", text="Usage: register worker <worker_id>")

            elif action == "unregister":
                if len(parts) >= 3 and parts[1] == "worker":
                    worker_id = parts[2]
                    if worker_id in self.coordinator.worker_ids:
                        self.coordinator.worker_ids.remove(worker_id)
                    yield MessagePart(type="text", text=f"Worker {worker_id} unregistered successfully")
                else:
                    yield MessagePart(type="error", text="Usage: unregister worker <worker_id>")

            elif action == "status":
                stats = self.get_stats()
                yield MessagePart(type="json", text=json.dumps(stats, indent=2))

            elif action == "list":
                workers = self.coordinator.worker_ids
                yield MessagePart(type="text", text=f"Registered workers: {', '.join(workers) if workers else 'None'}")

            elif action == "dispatch":
                # Dispatch questions to workers
                async for result in self.coordinator.dispatch_questions():
                    yield result

            elif action == "help":
                help_text = """
QA Coordinator Commands:
- register worker <worker_id>: Register a new worker
- unregister worker <worker_id>: Unregister a worker
- status: Get current statistics
- list: List registered workers
- dispatch: Dispatch questions to workers
- help: Show this help message
"""
                yield MessagePart(type="text", text=help_text)

            else:
                # Invalid command
                yield MessagePart(type="error", text=f"Unknown command: {command}")
                return

        except Exception as e:
            yield MessagePart(type="error", text=f"Error in coordinator: {str(e)}")
            print(f"Error in QACoordinatorExecutorACP.execute: {e}")
            import traceback
            traceback.print_exc()

    def set_agent_network(self, agent_network):
        """Set the agent network reference."""
        self.coordinator.set_agent_network(agent_network)


print("QACoordinatorExecutorACP class defined successfully")


