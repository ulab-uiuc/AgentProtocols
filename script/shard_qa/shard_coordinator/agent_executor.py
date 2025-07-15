import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics

# Add A2A SDK imports
try:
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    # Import new_agent_text_message but ensure it uses string role
    from a2a.utils import new_agent_text_message as _original_new_agent_text_message
    
    def new_agent_text_message(text, role="user"):
        """Wrapper for new_agent_text_message that ensures compatibility"""
        # A2A SDK's new_agent_text_message only takes text parameter
        return _original_new_agent_text_message(text)
    
    A2A_AVAILABLE = True
except ImportError:
    print("Warning: a2a-sdk not available, using mock classes")
    A2A_AVAILABLE = False
    
    class AgentExecutor:
        pass
    
    class RequestContext:
        def get_user_input(self):
            return "Mock input"
    
    class EventQueue:
        async def enqueue_event(self, event):
            pass
    
    def new_agent_text_message(text, role="user"):
        return {"type": "text", "content": text, "role": str(role)}

async def safe_enqueue_event(event_queue, event):
    """Safely enqueue event, handling both sync and async event queues."""
    import inspect
    try:
        res = event_queue.enqueue_event(event)
        if inspect.isawaitable(res):
            return await res
        else:
            return res
    except Exception as e:
        print(f"Event queue error: {e}")
        return None

def create_safe_a2a_message(message_id: str, content: str, meta: dict = None) -> dict:
    """Create a safe A2A message that is guaranteed to be JSON serializable"""
    safe_meta = {}
    if meta:
        # Ensure all meta values are JSON serializable
        for key, value in meta.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                safe_meta[str(key)] = value
            elif isinstance(value, list):
                safe_meta[str(key)] = [str(item) for item in value]
            else:
                safe_meta[str(key)] = str(value)
    
    return {
        "messageId": str(message_id),
        "role": "user",  # Always use string, never Role enum
        "parts": [{"type": "text", "text": str(content)}],
        "meta": safe_meta
    }


class ShardCoordinator:
    """Coordinator for 8 Agent Ring Collaborative Retrieval"""

    def __init__(self, config: dict = None, global_config: dict = None, output=None):
        """Initialize Shard Coordinator."""
        self.config = config or {}
        self.global_config = global_config or {}
        self.output = output
        
        # Get coordinator configuration
        coordinator_config = global_config.get('shard_qa', {}).get('coordinator', {})
        
        self.total_groups = coordinator_config.get('total_groups', 200)
        self.result_file = coordinator_config.get('result_file', 'data/v1.1/results.json')
        
        # Worker and network info
        self.worker_ids: List[str] = []
        self.agent_network = None
        self.coordinator_id = "coordinator"
        
        # Message counting for logging control
        self.message_count = 0
        
        # Metrics tracking
        self.metrics = {
            'first_answer_latency': {},  # group_id -> latency
            'avg_hop': {},  # group_id -> avg_hop
            'msg_bytes_total': 0,
            'ttl_exhausted_total': 0,
            'llm_tokens_total': 0
        }
        
        # Task tracking
        self.current_tasks = {}  # group_id -> task_info
        self.results = {}  # group_id -> result_info
        
    def set_network(self, network, worker_ids: List[str]):
        """Set the agent network and worker IDs."""
        self.agent_network = network
        self.worker_ids = worker_ids.copy()
        
        if self.output:
            self.output.system(f"Coordinator configured with {len(worker_ids)} workers: {worker_ids}")

    async def start_benchmark_group(self, group_id: int) -> Dict[str, Any]:
        """Start benchmark for a specific group - each agent processes their own question but can communicate"""
        if self.output:
            self.output.info(f"Starting independent benchmark for group {group_id}")
            self.output.system("Each agent gets their own question but can communicate with neighbors")
        
        # Initialize task tracking
        task_start_time = time.time()
        self.current_tasks[group_id] = {
            'start_time': task_start_time,
            'responses_received': 0,
            'first_answer_time': None,
            'messages': [],
            'completed': False,
            'worker_results': {}  # Track individual worker results
        }
        
        # Send individual tasks to each worker simultaneously - they can communicate
        results = []
        tasks = []  # For concurrent sending
        
        for worker_id in self.worker_ids:
            if self.output:
                self.output.progress(f"Preparing independent task for {worker_id} (group {group_id})")
            
            # Each worker gets their own question but can ask neighbors for help
            individual_message = f"Process GROUP_ID = {group_id} independently. You have your own question to answer, but you can communicate with neighbors if needed."
            
            # 设置初始 TTL - 修复机器控制的TTL
            initial_ttl = self.global_config.get('tool_schema', {}).get('max_ttl', 8)
            
            meta = {
                "sender": "coordinator",
                "group_id": int(group_id),
                "timestamp": float(time.time()),
                "task_type": "independent_with_communication",  # New task type
                "worker_id": worker_id,
                "ttl": initial_ttl,  # ✅ 添加机器控制的初始TTL
                "path": ["coordinator"]  # ✅ 添加初始路径
            }
            message_payload = create_safe_a2a_message(
                f"v1.1-{group_id}-{worker_id}-coordinator",
                individual_message,
                meta
            )
            
            # Create task for concurrent sending
            task = self._send_to_worker(worker_id, message_payload, results)
            tasks.append(task)
        
        # Send all messages simultaneously
        if self.output:
            self.output.info(f"[LAUNCH] Sending {len(tasks)} independent tasks simultaneously...")
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        if self.output:
            self.output.success(f"All {len(self.worker_ids)} independent tasks dispatched!")
        
        # Wait for all workers to complete their individual tasks
        await self._wait_for_completion(group_id, timeout=60.0)
        
        # Compile final results
        return self._compile_group_results(group_id, results)
    
    async def _send_to_worker(self, worker_id: str, message_payload: dict, results: list):
        """Send message to a single worker (for concurrent execution)"""
        try:
            response = await self.agent_network.route_message(
                self.coordinator_id, 
                worker_id, 
                message_payload
            )
            
            results.append({
                'worker_id': worker_id,
                'status': 'sent',
                'response': response
            })
            
        except Exception as e:
            if self.output:
                self.output.error(f"Failed to send to {worker_id}: {e}")
            results.append({
                'worker_id': worker_id,
                'status': 'error',
                'error': str(e)
            })

    async def _wait_for_completion(self, group_id: int, timeout: float = 60.0):
        """Wait for task completion or timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.current_tasks[group_id]['completed']:
                break
            await asyncio.sleep(0.1)
        
        # Mark as completed if timeout
        if not self.current_tasks[group_id]['completed']:
            self.current_tasks[group_id]['completed'] = True
            if self.output:
                self.output.warning(f"Group {group_id} timed out after {timeout} seconds")

    def _compile_group_results(self, group_id: int, broadcast_results: List[Dict]) -> Dict[str, Any]:
        """Compile results for a group"""
        task_info = self.current_tasks.get(group_id, {})
        
        end_time = time.time()
        total_time = end_time - task_info.get('start_time', end_time)
        
        # Calculate metrics
        first_answer_latency = None
        if task_info.get('first_answer_time'):
            first_answer_latency = task_info['first_answer_time'] - task_info['start_time']
            self.metrics['first_answer_latency'][group_id] = first_answer_latency
        
        # Calculate average hop count from messages
        hop_counts = []
        for msg in task_info.get('messages', []):
            # Try to extract hop count from enhanced message format first
            content = msg.get('content', '')
            if content.startswith('[FROM:') and '|HOP:' in content:
                try:
                    import re
                    match = re.search(r'\|HOP:(\d+)\]', content)
                    if match:
                        hop_counts.append(int(match.group(1)))
                        continue
                except:
                    pass
            
            # Try to extract from embedded hop information in content
            if '(hop=' in content:
                try:
                    import re
                    match = re.search(r'\(hop=(\d+)\)', content)
                    if match:
                        hop_counts.append(int(match.group(1)))
                        continue
                except:
                    pass
            
            # Fallback to path-based calculation if available
            path = msg.get('meta', {}).get('path', [])
            if path:
                hop_counts.append(len(path) - 1)
        
        avg_hop = statistics.mean(hop_counts) if hop_counts else 0
        self.metrics['avg_hop'][group_id] = avg_hop
        
        result = {
            'group_id': group_id,
            'total_time': total_time,
            'first_answer_latency': first_answer_latency,
            'avg_hop': avg_hop,
            'responses_received': task_info.get('responses_received', 0),
            'messages_count': len(task_info.get('messages', [])),
            'broadcast_results': broadcast_results,
            'completed': task_info.get('completed', False),
            'timestamp': end_time
        }
        
        self.results[group_id] = result
        
        if self.output:
            self.output.success(f"Group {group_id} results compiled")
            self.output.progress(f"  - Total time: {total_time:.2f}s")
            self.output.progress(f"  - First answer: {first_answer_latency:.2f}s" if first_answer_latency else "  - No answer received")
            self.output.progress(f"  - Avg hops: {avg_hop:.1f}")
            self.output.progress(f"  - Messages: {len(task_info.get('messages', []))}")
        
        return result

    async def handle_worker_response(self, sender: str, content: str, meta: dict = None) -> str:
        """Handle response from shard worker"""
        self.message_count += 1
        
        # Log only every 10 messages or for important messages
        should_log = (self.message_count % 10 == 0) or "ANSWER_FOUND" in content or "TTL_EXHAUSTED" in content
        if self.output and should_log:
            self.output.progress(f"[{self.message_count}] Received from {sender}: {content[:50]}...")
        
        # Update message bytes metric
        self.metrics['msg_bytes_total'] += len(content.encode())
        
        # Extract group_id from meta
        group_id = meta.get('group_id', -1) if meta else -1
        
        if group_id == -1:
            if self.output:
                self.output.warning(f"No group_id in message from {sender}")
            return "Message processed (no group_id)"
        
        # Track message
        if group_id in self.current_tasks:
            task_info = self.current_tasks[group_id]
            
            # Add message to history
            message_info = {
                'sender': sender,
                'content': content,
                'meta': meta or {},
                'timestamp': time.time()
            }
            task_info['messages'].append(message_info)
            
            # Update metrics
            task_info['responses_received'] += 1
            
            # Check for answer (both collaborative and independent)
            if "ANSWER_FOUND:" in content or "INDEPENDENT_ANSWER_FOUND:" in content:
                if not task_info['first_answer_time']:
                    task_info['first_answer_time'] = time.time()
                    if self.output:
                        self.output.success(f"First answer received for group {group_id} from {sender}")
                
                # For independent mode, record individual worker result
                if "INDEPENDENT_ANSWER_FOUND:" in content:
                    answer = content.split("INDEPENDENT_ANSWER_FOUND:")[1].strip()
                    task_info['worker_results'][sender] = {
                        'status': 'found',
                        'answer': answer,
                        'timestamp': time.time()
                    }
                    if self.output:
                        self.output.success(f"[INDEPENDENT] {sender} found answer: {answer}")
                
                # Mark task as completed
                task_info['completed'] = True
                
                return f"Answer received for group {group_id}"
            
            # Check for TTL exhausted
            elif "TTL_EXHAUSTED" in content:
                self.metrics['ttl_exhausted_total'] += 1
                if self.output:
                    self.output.warning(f"TTL exhausted for group {group_id} from {sender}")
                task_info['completed'] = True
                return f"TTL exhausted for group {group_id}"
            
            # Check for LLM token usage
            elif "LLM_TOKENS:" in content:
                try:
                    tokens = int(content.split("LLM_TOKENS:")[1].strip())
                    self.metrics['llm_tokens_total'] += tokens
                    if self.output and self.message_count % 20 == 0:  # Log every 20 token messages
                        self.output.progress(f"LLM tokens: +{tokens} (total: {self.metrics['llm_tokens_total']})")
                except ValueError:
                    pass
                return "Token usage recorded"
            
            # Check for neighbor timeout
            elif "NEIGHBOR_TIMEOUT" in content:
                if self.output:
                    self.output.warning(f"Neighbor timeout for group {group_id} from {sender}")
                task_info['completed'] = True
                return f"Neighbor timeout for group {group_id}"
            
            # Check for search error
            elif "SEARCH_ERROR" in content:
                if self.output:
                    self.output.warning(f"Search error for group {group_id} from {sender}: {content}")
                task_info['completed'] = True
                return f"Search error for group {group_id}"
            
            # Check for other completion conditions
            elif "NO_ANSWER" in content or "INDEPENDENT_NO_ANSWER" in content:
                if self.output:
                    self.output.warning(f"No answer found for group {group_id} from {sender}")
                
                # For independent mode, record individual worker result
                if "INDEPENDENT_NO_ANSWER" in content:
                    task_info['worker_results'][sender] = {
                        'status': 'no_answer',
                        'answer': None,
                        'timestamp': time.time()
                    }
                    if self.output:
                        self.output.warning(f"[INDEPENDENT] {sender} found no answer")
                
                # Check if all workers have responded in independent mode
                if len(task_info['worker_results']) >= len(self.worker_ids):
                    task_info['completed'] = True
                    if self.output:
                        self.output.info(f"All workers completed for group {group_id}")
                
                return f"No answer found for group {group_id}"
        
        return "Message processed"

    async def get_coordinator_status(self) -> str:
        """Get current coordinator status"""
        active_tasks = sum(1 for task in self.current_tasks.values() if not task.get('completed', False))
        completed_tasks = len(self.results)
        
        status = {
            'coordinator_id': self.coordinator_id,
            'worker_count': len(self.worker_ids),
            'active_tasks': active_tasks,
            'completed_tasks': completed_tasks,
            'total_groups': self.total_groups,
            'metrics_summary': {
                'avg_first_answer_latency': statistics.mean(self.metrics['first_answer_latency'].values()) if self.metrics['first_answer_latency'] else 0,
                'avg_hop_count': statistics.mean(self.metrics['avg_hop'].values()) if self.metrics['avg_hop'] else 0,
                'total_messages': self.metrics['msg_bytes_total'],
                'ttl_exhausted': self.metrics['ttl_exhausted_total']
            }
        }
        
        return json.dumps(status, indent=2)

    async def save_results(self):
        """Save results to file"""
        try:
            results_path = Path(__file__).parent.parent / self.result_file
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            output_data = {
                'metadata': {
                    'version': 'v1.1',
                    'coordinator_id': self.coordinator_id,
                    'worker_ids': self.worker_ids,
                    'total_groups_processed': len(self.results),
                    'generation_time': time.time()
                },
                'metrics': self.metrics,
                'results': self.results
            }
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            if self.output:
                self.output.success(f"Results saved to {results_path}")
                
        except Exception as e:
            if self.output:
                self.output.error(f"Failed to save results: {e}")


class ShardCoordinatorExecutor(AgentExecutor):
    """Shard Coordinator A2A Agent Executor"""

    def __init__(self, config=None, global_config=None, output=None):
        self.coordinator = ShardCoordinator(config, global_config, output)
        self.output = output

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # Get user input from context
        user_input = context.get_user_input()
        
        if not user_input:
            await safe_enqueue_event(event_queue, new_agent_text_message("No input received"))
            return
        
        # Extract meta information from A2A message context
        sender = "unknown_worker"
        meta = {}
        
        # Try to extract meta information from message content (enhanced format)
        if user_input.startswith("[FROM:"):
            try:
                # Parse enhanced message format: [FROM:shard0|GROUP:0|HOP:1] CONTENT
                import re
                match = re.match(r'\[FROM:([^|]+)\|GROUP:([^|]+)\|HOP:([^\]]+)\]\s*(.*)', user_input)
                if match:
                    sender = match.group(1)
                    group_id = int(match.group(2))
                    hop = int(match.group(3))
                    user_input = match.group(4)  # Extract the actual content
                    meta = {
                        'sender': sender,
                        'group_id': group_id,
                        'hop': hop,
                        'timestamp': time.time()
                    }
            except Exception as e:
                if self.output:
                    self.output.warning(f"Failed to parse enhanced message format: {e}")
        
        # Fallback: try to extract meta information from context
        if not meta:
            try:
                if hasattr(context, 'params') and hasattr(context.params, 'message'):
                    message = context.params.message
                    if hasattr(message, 'messageId'):
                        # This is an A2A message, try to extract meta
                        if hasattr(message, 'meta'):
                            meta = message.meta if isinstance(message.meta, dict) else {}
                            sender = meta.get('sender', sender)
                        elif hasattr(message, 'parts') and message.parts:
                            # Check if meta is embedded in parts
                            for part in message.parts:
                                if hasattr(part, 'meta'):
                                    meta = part.meta if isinstance(part.meta, dict) else {}
                                    sender = meta.get('sender', sender)
                                    break
            except Exception as e:
                if self.output:
                    self.output.warning(f"Failed to extract meta from context: {e}")
        
        try:
            # Handle different types of commands
            if user_input.lower().startswith("status"):
                # Return coordinator status
                status = await self.coordinator.get_coordinator_status()
                await safe_enqueue_event(event_queue, new_agent_text_message(f"Coordinator Status:\n{status}"))
                
            elif user_input.startswith("Process GROUP_ID ="):
                # Extract group_id and start benchmark
                parts = user_input.split("=")
                if len(parts) >= 2:
                    try:
                        group_id = int(parts[1].strip().split()[0])
                        result = await self.coordinator.start_benchmark_group(group_id)
                        
                        if result:
                            response = f"Benchmark for group {group_id} completed:\n"
                            response += f"- Total time: {result.get('total_time', 0):.2f}s\n"
                            response += f"- First answer: {result.get('first_answer_latency', 'N/A')}\n"
                            response += f"- Avg hops: {result.get('avg_hop', 0):.1f}\n"
                            response += f"- Messages: {result.get('messages_count', 0)}"
                        else:
                            response = f"Benchmark for group {group_id} failed or returned no result"
                        
                        await safe_enqueue_event(event_queue, new_agent_text_message(response))
                        return
                    except ValueError as e:
                        await safe_enqueue_event(event_queue, new_agent_text_message(f"Invalid group_id format: {e}"))
                        return
            
            elif "ANSWER_FOUND:" in user_input or "INDEPENDENT_ANSWER_FOUND:" in user_input or "TTL_EXHAUSTED" in user_input or "NO_ANSWER" in user_input or "INDEPENDENT_NO_ANSWER" in user_input or "LLM_TOKENS:" in user_input or "NEIGHBOR_TIMEOUT" in user_input or "SEARCH_ERROR" in user_input:
                # Handle worker response - these come from A2A messages from workers
                try:
                    result = await self.coordinator.handle_worker_response(sender, user_input, meta)
                    if result:
                        await safe_enqueue_event(event_queue, new_agent_text_message(result))
                except Exception as e:
                    if self.output:
                        self.output.error(f"Error handling worker response: {e}")
                    await safe_enqueue_event(event_queue, new_agent_text_message(f"Error processing worker response: {e}"))
                
            elif user_input.lower() == "save_results":
                # Save current results
                await self.coordinator.save_results()
                await safe_enqueue_event(event_queue, new_agent_text_message("Results saved successfully"))
                
            else:
                # Default response for other messages
                await safe_enqueue_event(event_queue, new_agent_text_message(f"Coordinator received: {user_input[:100]}..."))
                
        except Exception as e:
            error_msg = f"Coordinator error: {str(e)}"
            if self.output:
                self.output.error(error_msg)
            await safe_enqueue_event(event_queue, new_agent_text_message(error_msg))

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception('cancel not supported') 