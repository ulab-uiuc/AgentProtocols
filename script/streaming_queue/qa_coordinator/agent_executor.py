import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add A2A SDK imports
try:
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.utils import new_agent_text_message
    A2A_AVAILABLE = True
except ImportError:
    print("Warning: a2a-sdk not available, using mock classes")
    A2A_AVAILABLE = False
    
    # Mock classes for when a2a-sdk is not available
    class AgentExecutor:
        pass
    
    class RequestContext:
        def get_user_input(self):
            return "Mock input"
    
    class EventQueue:
        def enqueue_event(self, event):
            pass
    
    def new_agent_text_message(text):
        return {"type": "text", "content": text}


class QACoordinator:
    """QA Coordinator for dispatching questions to worker agents."""

    def __init__(self, config: dict = None, output=None):
        """Initialize QA Coordinator."""
        if config is None:
            config = {}
        
        # Get coordinator related parameters from configuration
        coordinator_config = config.get('qa', {}).get('coordinator', {})

        self.batch_size = coordinator_config.get('batch_size', 50)
        self.first_50 = coordinator_config.get('first_50', True)
        self.data_path = coordinator_config.get('data_file', 'script/streaming_queue/data/top1000_simplified.jsonl')
        self.worker_ids: List[str] = []
        self.agent_network = None
        self.coordinator_id = "Coordinator-1"
        self.output = output
        self.config = config

    def set_network(self, network, worker_ids: List[str]):
        """Set the agent network and worker IDs."""
        self.agent_network = network
        self.worker_ids = worker_ids.copy()

    async def load_questions(self):
        """Load questions from data file"""
        questions = []
        file_path = Path(self.data_path)
        
        if not file_path.exists():
            if self.output:
                self.output.error(f"Question file does not exist: {self.data_path}")
            return questions
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    question = item.get('q', '')
                    message_id = item.get('id', str(len(questions) + 1))
                    
                    if question:
                        questions.append({
                            'id': message_id,
                            'question': question
                        })
                    
                    if self.first_50 and len(questions) >= 50:
                        break
                        
                except json.JSONDecodeError as e:
                    if self.output:
                        self.output.error(f"JSON parsing failed: {line[:50]}... Error: {e}")
                    continue
        
        if self.output:
            self.output.system(f"Loaded {len(questions)} questions")
        return questions

    async def dispatch_questions_dynamically(self, questions: List[Dict]):
        """Dynamic load balancing question dispatch"""
        if not self.agent_network or not self.worker_ids:
            return []
        
        if self.output:
            self.output.info(f"Starting dynamic load balancing: {len(questions)} questions, {len(self.worker_ids)} workers")
        
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
                question = question_data['question']
                
                if self.output:
                    self.output.progress(f"{worker_id} starting to process question {question_id}: {question[:50]}...")
                
                # Prepare A2A message format
                payload = {
                    "messageId": str(question_id),
                    "parts": [
                        {
                            "type": "text",
                            "text": question
                        }
                    ],
                    "role": "user"
                }
                
                # Send message to Worker through network
                start_time = time.time()
                try:
                    response = await self.agent_network.route_message(self.coordinator_id, worker_id, payload)
                    end_time = time.time()
                    
                    # Extract answer from response - handle both A2A and Agent Protocol formats
                    answer = "No answer received"
                    
                    # First, try Agent Protocol format (for AP workers)
                    if isinstance(response, dict):
                        # Check for Agent Protocol result format
                        if "result" in response and isinstance(response["result"], dict):
                            result = response["result"]
                            if "output" in result:
                                answer = result["output"]
                            elif "text" in result:
                                answer = result["text"]
                        # Check for Agent Protocol step response format
                        elif "output" in response:
                            answer = response["output"]
                        # Check for Agent Protocol events format (legacy A2A compatibility)
                        elif "events" in response and response["events"]:
                            for event in response["events"]:
                                if isinstance(event, dict):
                                    # Agent Protocol event format
                                    if event.get("type") == "agent_text_message" and "data" in event:
                                        answer = event["data"]
                                        break
                                    # A2A event format (fallback compatibility)
                                    elif event.get("kind") == "message" and "parts" in event:
                                        if event["parts"] and "text" in event["parts"][0]:
                                            answer = event["parts"][0]["text"]
                                            break
                        # Check for direct text response
                        elif "result" in response and isinstance(response["result"], str):
                            answer = response["result"]
                        # Check for simple text field
                        elif "text" in response:
                            answer = response["text"]
                    
                    # If still no answer, try string conversion
                    if answer == "No answer received" and response:
                        answer = str(response)
                    
                    # Build result record
                    result_record = {
                        "question_id": question_id,
                        "question": question,
                        "worker": worker_id,
                        "answer": answer,
                        "response_time": end_time - start_time,
                        "timestamp": time.time(),
                        "status": "success" if answer != "No answer received" else "failed"
                    }
                    
                    # Put result into result queue
                    await result_queue.put(result_record)
                    
                    if self.output:
                        self.output.progress(f"{worker_id} completed question {question_id}, continuing to next...")
                    
                except Exception as e:
                    # Build failure record
                    result_record = {
                        "question_id": question_id,
                        "question": question,
                        "worker": worker_id,
                        "answer": None,
                        "response_time": None,
                        "timestamp": time.time(),
                        "status": "failed",
                        "error": str(e)
                    }
                    await result_queue.put(result_record)
                    if self.output:
                        self.output.error(f"{worker_id} failed to process question {question_id}: {e}")
                
                # Mark task as done
                question_queue.task_done()
                
            except Exception as e:
                if self.output:
                    self.output.error(f"{worker_id} processing exception: {e}")
                try:
                    question_queue.task_done()
                except:
                    pass
                break
        
        if self.output:
            self.output.system(f"{worker_id} completed all tasks, processed {processed_count} questions")

    async def result_collector(self, result_queue: asyncio.Queue, total_questions: int):
        """Result collector"""
        results = []
        collected = 0
        
        while collected < total_questions:
            try:
                result = await result_queue.get()
                results.append(result)
                collected += 1
                
                if collected % 5 == 0 or collected == total_questions:
                    if self.output:
                        self.output.system(f"Collected {collected}/{total_questions} results")
                
                result_queue.task_done()
                
            except Exception as e:
                if self.output:
                    self.output.error(f"Result collection exception: {e}")
                break
        
        if self.output:
            self.output.success(f"Result collection completed, collected {collected} results")
        return results

    async def dispatch_round(self):
        """Dispatch question round and save results"""
        questions = await self.load_questions()
        if not questions:
            return "Error: No questions loaded"
        
        start_time = time.time()
        results = await self.dispatch_questions_dynamically(questions)
        end_time = time.time()
        
        # Save results to file
        await self.save_results(results)
        
        success_count = len([r for r in results if r["status"] == "success"])
        failed_count = len([r for r in results if r["status"] == "failed"])
        
        summary = (
            f"Dispatch round completed in {end_time - start_time:.2f} seconds\n"
            f"Total processed: {len(results)} questions\n"
            f"Successfully processed: {success_count} questions\n"
            f"Failed: {failed_count} questions\n"
            f"Workers used: {len(self.worker_ids)}\n"
            f"Results saved to file"
        )
        
        return summary

    async def save_results(self, results):
        """Save results to file"""
        try:
            result_file = self.config.get('qa', {}).get('coordinator', {}).get('result_file', './data/qa_result.json')
            result_path = Path(result_file)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            
            output_data = {
                "metadata": {
                    "total_questions": len(results),
                    "successful_questions": len([r for r in results if r["status"] == "success"]),
                    "failed_questions": len([r for r in results if r["status"] == "failed"]),
                    "average_response_time": (sum(r["response_time"] for r in results if r["response_time"]) / len([r for r in results if r["response_time"]])) if [r for r in results if r["response_time"]] else 0,
                    "timestamp": time.time(),
                    "network_type": "real_a2a_agent_network"
                },
                "results": results
            }
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            if self.output:
                self.output.success(f"Results saved to: {result_path}")
        except Exception as e:
            if self.output:
                self.output.error(f"Failed to save results: {e}")


class QACoordinatorExecutor(AgentExecutor):
    """QA Coordinator Agent Executor Implementation."""

    def __init__(self, config=None, output=None):
        self.coordinator = QACoordinator(config, output)
        self.config = config or {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute coordinator tasks by calling QACoordinator methods."""
        try:
            # Get message from context
            user_input = context.get_user_input() if hasattr(context, 'get_user_input') else "status"
            
            if isinstance(user_input, dict):
                # Extract text from message parts if it's a structured message
                command = "status"
                if "parts" in user_input:
                    for part in user_input["parts"]:
                        if part.get("kind") == "text" or part.get("type") == "text":
                            command = part.get("text", "status")
                            break
                elif "text" in user_input:
                    command = user_input["text"]
                else:
                    command = str(user_input)
            else:
                command = str(user_input) if user_input else "status"
            
            command = command.lower().strip()
            
            # Route to appropriate coordinator method
            if command == "dispatch" or command == "start_dispatch":
                # Call the existing dispatch_round method
                result = await self.coordinator.dispatch_round()
            elif command == "status":
                # Return coordinator status
                result = await self.get_coordinator_status()
            else:
                result = f"Unknown command: {command}. Available commands: dispatch, status"
            
            # Send response event
            if A2A_AVAILABLE:
                await event_queue.enqueue_event(new_agent_text_message(result))
            else:
                print(f"[QACoordinatorExecutor] {result}")
                
        except Exception as e:
            error_msg = f"QA Coordinator execution failed: {str(e)}"
            print(f"[QACoordinatorExecutor] {error_msg}")
            if A2A_AVAILABLE:
                await event_queue.enqueue_event(new_agent_text_message(error_msg))

    async def get_coordinator_status(self) -> str:
        """Get coordinator status."""
        try:
            network_status = "Connected" if self.coordinator.agent_network else "Not connected"
            worker_count = len(self.coordinator.worker_ids)
            
            status_info = (
                f"QA Coordinator Status:\n"
                f"Configuration: batch_size={self.coordinator.batch_size}, "
                f"first_50={self.coordinator.first_50}\n"
                f"Data path: {self.coordinator.data_path}\n"
                f"Network status: {network_status}\n"
                f"Worker count: {worker_count}\n"
                f"Available commands: dispatch, status"
            )
            
            return status_info
            
        except Exception as e:
            return f"Status check failed: {str(e)}"

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancel coordinator operations."""
        if A2A_AVAILABLE:
            await event_queue.enqueue_event(new_agent_text_message("QA Coordinator operations cancelled."))
        else:
            print("[QACoordinatorExecutor] Operations cancelled.")


