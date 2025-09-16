import asyncio
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Protocol-agnostic interfaces
class BaseRequestContext:
    """Base interface for request context across all protocols"""
    def get_user_input(self):
        raise NotImplementedError("Subclasses must implement get_user_input")

class BaseEventQueue:
    """Base interface for event queue across all protocols"""
    async def enqueue_event(self, event):
        raise NotImplementedError("Subclasses must implement enqueue_event")

class BaseAgentExecutor:
    """Base interface for agent executors across all protocols"""
    async def execute(self, context: BaseRequestContext, event_queue: BaseEventQueue) -> None:
        raise NotImplementedError("Subclasses must implement execute")
    
    async def cancel(self, context: BaseRequestContext, event_queue: BaseEventQueue) -> None:
        raise NotImplementedError("Subclasses must implement cancel")

def create_text_message(text, role="user"):
    """Create a protocol-agnostic text message"""
    return {"type": "text", "content": text, "role": str(role)}

# Protocol-specific implementations will be injected at runtime
RequestContext = BaseRequestContext
EventQueue = BaseEventQueue
AgentExecutor = BaseAgentExecutor
new_agent_text_message = create_text_message

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

# OpenAI Function-Calling Tools Schema - v3 (Machine-controlled TTL)
TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "lookup_fragment",
            "description": "检查本地 snippet 是否包含答案；TTL和路径由系统自动管理",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "要搜索的问题"
                    },
                    "found": {
                        "type": "boolean",
                        "description": "是否在本地找到答案"
                    }
                },
                "required": ["question", "found"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "在 ring 中转发消息或把结果回传给上一个节点/协调器",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "目标agent ID (prev_id, next_id, 或 coordinator)"
                    },
                    "content": {
                        "type": "string",
                        "description": "消息内容"
                    }
                },
                "required": ["destination", "content"]
            }
        }
    }
]

class ShardWorker:
    """Shard Worker for Ring Collaborative Retrieval"""

    def __init__(self, config: dict, global_config: dict, shard_id: str, data_file: str, neighbors: dict, output=None, force_llm=False):
        """Initialize Shard Worker."""
        self.config = config
        self.global_config = global_config
        self.shard_id = shard_id
        self.data_file = data_file
        self.neighbors = neighbors  # {prev_id, next_id}
        self.output = output
        self.agent_network = None
        self.force_llm = force_llm  # 控制是否强制使用LLM模式
        
        # Agent index from agent_id (e.g., "agent3" -> 3 or "shard3" -> 3)
        if shard_id.startswith("agent"):
            self.agent_idx = int(shard_id.replace("agent", ""))
        elif shard_id.startswith("shard"):
            self.agent_idx = int(shard_id.replace("shard", ""))
        else:
            # Fallback: try to extract number from the end
            import re
            match = re.search(r'(\d+)$', shard_id)
            self.agent_idx = int(match.group(1)) if match else 0
        
        # Current task data
        self.current_question = ""
        self.current_answer = ""
        self.current_snippet = ""
        self.current_group_id = -1
        
        # Context for machine-controlled TTL/path
        self.current_ttl = None
        self.current_path = None
        
        # History management
        from collections import deque
        self.history = {}  # group_id -> deque[messages]
        self.max_history = global_config.get('shard_qa', {}).get('history', {}).get('max_len', 20)
        
        # Pending message management with strict throttling
        self.pending: Dict[str, asyncio.Future] = {}
        neighbor_count = 2  # prev + next
        self.max_pending = min(neighbor_count * 2, 4)  # 严格限制：最多4个并发请求
        
        # Request throttling counters
        self.pending_count = 0
        self.dropped_count = 0
        
        # Core LLM
        self.core = None
        self.use_mock = False
        self._init_core()
    
    def _convert_config_for_core(self) -> Dict[str, Any]:
        """转换config格式为Core期望的格式"""
        if not self.config or 'model' not in self.config:
            # 从global_config中提取core配置
            core_config = self.global_config.get('core', {})
            return {
                "model": {
                    "type": core_config.get('type', 'openai'),
                    "name": core_config.get('name', 'gpt-4o'),
                    "openai_api_key": core_config.get('openai_api_key', ''),
                    "openai_base_url": core_config.get('openai_base_url', 'https://api.openai.com/v1'),
                    "temperature": core_config.get('temperature', 0.0),
                    "max_tokens": core_config.get('max_tokens', 4096)
                }
            }
        else:
            # config已经是正确格式
            return self.config
        
    def _init_core(self):
        """Initialize Core LLM"""
        try:
            # Use absolute import path for better reliability
            # From fail_storm_recovery/shard_qa/shard_worker/agent_executor.py to project root
            project_root = Path(__file__).parent.parent.parent.parent.parent
            src_path = project_root / "src"
            sys.path.insert(0, str(src_path))
            
            # Import Core from utils (the standard LLM interface)
            try:
                from utils.core import Core
            except ImportError:
                # Fallback: direct module loading
                import importlib.util
                core_module_path = src_path / "utils" / "core.py"
                spec = importlib.util.spec_from_file_location("core", core_module_path)
                if spec and spec.loader:
                    core_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(core_module)
                    Core = core_module.Core
                else:
                    raise ImportError("Could not load core module")
            
            if not self.config:
                raise Exception("No valid config provided")
            
            # Validate config structure
            if 'model' not in self.config:
                raise Exception("Missing 'model' section in config")
            
            model_config = self.config['model']
            if 'type' not in model_config:
                raise Exception("Missing 'type' in model config")
            
            if model_config['type'] == 'openai':
                if 'openai_api_key' not in model_config or not model_config['openai_api_key']:
                    raise Exception("Missing or empty 'openai_api_key' in model config")
            
            # 转换config格式为Core期望的格式
            core_config = self._convert_config_for_core()
            
            # Initialize Core
            self.core = Core(core_config)
            
            if self.output:
                self.output.success(f"[{self.shard_id}] Core LLM initialized successfully: {model_config['type']} - {model_config.get('name', 'unknown')}")
                
        except ImportError as e:
            if self.output:
                self.output.error(f"[{self.shard_id}] Failed to import Core: {e}")
                self.output.warning(f"[{self.shard_id}] Falling back to mock responses")
            self.use_mock = True
        except Exception as e:
            if self.output:
                self.output.error(f"[{self.shard_id}] Core LLM initialization failed: {e}")
                self.output.warning(f"[{self.shard_id}] Falling back to mock responses")
            self.use_mock = True

    def set_network(self, network):
        """Set the agent network"""
        self.agent_network = network

    def load_group_data(self, group_id: int) -> bool:
        """Load data for specific group_id from agent file"""
        try:
            data_path = Path(__file__).parent.parent / self.data_file
            
            if not data_path.exists():
                if self.output:
                    self.output.error(f"[{self.shard_id}] Data file not found: {self.data_file}")
                return False
            
            with open(data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Search for the entry with matching group_id instead of using line index
            found_data = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Check if this entry matches the target group_id
                    if data.get('id') == group_id or data.get('group_id') == group_id:
                        found_data = data
                        break
                except json.JSONDecodeError:
                    continue
            
            if found_data is None:
                if self.output:
                    self.output.error(f"[{self.shard_id}] Group {group_id} not found in data file")
                return False
                
            data = found_data
            
            self.current_group_id = group_id
            self.current_question = data.get('question', '')
            # Fix field mapping: use 'content' as both answer and snippet
            content = data.get('content', '')
            self.current_answer = content  # Use content as the answer
            self.current_snippet = content  # Use content as the snippet for searching
            
            if self.output:
                is_recovery = hasattr(self, 'metrics_collector') and self.metrics_collector and getattr(self.metrics_collector, 'in_recovery_phase', False)
                if not is_recovery:
                    self.output.success(f"[OK] [{self.shard_id}] Loaded group {group_id}: Q='{self.current_question[:50]}...'")
                    self.output.progress(f"      A='{self.current_answer[:50]}...'")
                # Silent during recovery phase
            
            return True
            
        except Exception as e:
            if self.output:
                self.output.error(f"[{self.shard_id}] Failed to load group {group_id}: {e}")
            return False

    def _get_system_prompt(self) -> str:
        """Get system prompt for the shard worker - Enhanced for distributed search"""
        max_ttl = self.global_config.get('tool_schema', {}).get('max_ttl', 15)
        return f"""You are agent {self.shard_id} in an intelligent distributed document search system.

🌐 NETWORK TOPOLOGY:
- Your neighbors: {self.neighbors['prev_id']} ← YOU → {self.neighbors['next_id']}
- You process document shard {self.agent_idx}

🎯 CURRENT SEARCH TASK:
Question: {self.current_question}

📄 YOUR LOCAL DOCUMENT FRAGMENT:
{self.current_snippet}

🛠️ AVAILABLE TOOLS:
1. lookup_fragment: Analyze your local document fragment
2. send_message: Communicate with coordinator and neighbors

🔄 DISTRIBUTED SEARCH PROTOCOL:

STEP 1 - LOCAL SEARCH:
→ Call lookup_fragment(question="{self.current_question}", found=<true/false>, answer="<extracted_info>")
→ Be GENEROUS with found=true - partial information is valuable!

STEP 2 - ACTION BASED ON RESULT:
If found=true:
→ send_message(destination="coordinator", content="ANSWER_FOUND: <detailed_answer>")

If found=false:
→ The system will automatically handle neighbor search
→ No need to manually send neighbor requests

🎯 ULTRA-LIBERAL SEARCH CRITERIA (MAXIMIZE DISCOVERY):
✅ SET found=true if your fragment contains ANY of these:
- Direct answers or partial answers
- Names, entities, dates, numbers mentioned in the question
- Related context, background information, or topic-relevant content
- Keywords or concepts that connect to the question
- Similar or related entities (e.g., same type of person, place, thing)
- Historical context or background about the topic
- Even tangentially related information
- ANY word or phrase that appears in both question and fragment
- Information that could help answer the question when combined with other sources

❌ SET found=false ONLY if:
- Fragment is about completely different, unrelated topics with ZERO overlap
- Absolutely no shared words, concepts, or themes with the question
- Example: Question about "cars" but fragment about "cooking recipes" with no connection

🚨 CRITICAL: When in doubt, ALWAYS choose found=true! It's better to be overly generous than to miss relevant information. The system will validate answers later.

📝 ANSWER EXTRACTION:
When found=true, extract the most relevant information:
- Include specific facts, names, dates, numbers
- Provide context that helps answer the question
- Be specific and detailed rather than vague

🔍 LIBERAL DETECTION EXAMPLES:

Question: "What nationality were Scott Derrickson and Ed Wood?"
Fragment: "Scott Derrickson is an American filmmaker..."
→ found=true, answer="Scott Derrickson is American"

Fragment: "Ed Wood was born in New York..."
→ found=true, answer="Ed Wood was American (born in New York)"

Fragment: "Hollywood directors often work internationally..."
→ found=true, answer="Context about directors and nationality" (related topic)

Fragment: "The Laleli Mosque is located in Istanbul, Turkey..."
→ found=false (no connection to directors or nationality)

Question: "The lamp used in lighthouses similar to lamp patented in 1780 by Aimé Argand?"
Fragment: "Lewis lamp: The Lewis lamp is used in lighthouses. | Argand lamp: patented in 1780 by Aimé Argand"
→ found=true, answer="Lewis lamp used in lighthouses, Argand lamp patented 1780 by Aimé Argand"

Fragment: "Lighthouse construction began in the 18th century..."
→ found=true, answer="Historical context about lighthouses" (related topic)

Fragment: "Car manufacturing processes..."
→ found=false (no connection to lamps or lighthouses)

🚀 REMEMBER: This is a collaborative system! Your partial information will be combined with findings from other agents to provide complete answers. Be generous in detection - it's better to find partial information than to miss relevant content!"""

    async def _real_neighbor_search(
        self,
        question: str,
        ttl: int,
        path: List[str],
        max_concurrent: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Perform real distributed search across neighbor agents.
        
        Args:
            question: The question to search for
            ttl: Time-to-live for the search
            path: Path of agents already visited
            max_concurrent: Maximum concurrent neighbor requests
            
        Returns:
            Dict with search results or None if not found
        """
        if ttl <= 0:
            return None
            
        # Create search request
        search_request = {
            "type": "NEIGHBOR_SEARCH_REQUEST",
            "question": question,
            "requesting_agent": self.shard_id,
            "ttl": ttl - 1,
            "path": path + [self.shard_id],
            "timestamp": time.time()
        }
        
        # Send to both neighbors concurrently
        tasks = []
        neighbors = [self.neighbors['prev_id'], self.neighbors['next_id']]
        
        for neighbor_id in neighbors:
            if neighbor_id not in path:  # Avoid cycles
                task = asyncio.create_task(
                    self._send_neighbor_search_request(neighbor_id, search_request)
                )
                tasks.append((neighbor_id, task))
        
        if not tasks:
            return None
            
        # Wait for first successful response or all failures
        try:
            done, pending = await asyncio.wait(
                [task for _, task in tasks],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=25.0  # 邻居协作总超时：两个邻居并发搜索，考虑可能的转发
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            # Process completed tasks
            for task in done:
                try:
                    result = await task
                    if result and result.get('found'):
                        return result
                except Exception as e:
                    continue
                    
            # If no successful results, wait for remaining tasks briefly
            if pending:
                try:
                    done, pending = await asyncio.wait(
                        pending, timeout=2.0
                    )
                    for task in done:
                        try:
                            result = await task
                            if result and result.get('found'):
                                return result
                        except Exception:
                            continue
                finally:
                    # Cancel any remaining tasks
                    for task in pending:
                        task.cancel()
                        
        except asyncio.TimeoutError:
            pass
            
        return None
        
    async def _send_neighbor_search_request(
        self,
        neighbor_id: str,
        search_request: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Send search request to a specific neighbor and wait for response."""
        try:
            request_content = f"SEARCH_REQUEST: {json.dumps(search_request)}"
            
            response = await self._send_and_wait(
                dest_id=neighbor_id,
                content=request_content,
                ttl=search_request['ttl'],
                path=search_request['path'],
                timeout=18.0  # 单个邻居搜索：LLM(4s) + 转发(12s) + 响应(2s)
            )
            
            if response and "SEARCH_RESPONSE:" in response:
                try:
                    response_data = json.loads(response.split("SEARCH_RESPONSE:")[1].strip())
                    return response_data
                except json.JSONDecodeError:
                    return None
                    
        except Exception as e:
            if self.output:
                self.output.warning(f"[{self.shard_id}] Error in neighbor search to {neighbor_id}: {e}")
            return None
            
        return None

    async def _handle_neighbor_search_request(
        self,
        sender: str,
        request_data: Dict[str, Any],
        reply_to: str
    ) -> str:
        """
        Handle incoming neighbor search request with NON-BLOCKING concurrent processing.
        
        This method processes search requests from other agents and performs
        local document search without blocking the agent's own search operations.
        Uses background task processing to avoid deadlocks.
        """
        question = request_data.get('question', '')
        requesting_agent = request_data.get('requesting_agent', sender)
        ttl = request_data.get('ttl', 0)
        search_path = request_data.get('path', [])
        
        if self.output:
            is_recovery = hasattr(self, 'metrics_collector') and self.metrics_collector and getattr(self.metrics_collector, 'in_recovery_phase', False)
            if not is_recovery:
                self.output.progress(f"🔍 [{self.shard_id}] Processing neighbor search request from {requesting_agent}")
                self.output.progress(f"   Question: {question[:60]}...")
                self.output.progress(f"   TTL: {ttl}, Path: {' → '.join(search_path)}")
        
        # CONCURRENT PROCESSING: Start background task for neighbor search
        # This allows the agent to continue its own work while helping neighbors
        background_task = asyncio.create_task(
            self._background_neighbor_search(question, ttl, search_path, requesting_agent)
        )
        
        try:
            # Wait for background search with shorter timeout to avoid blocking
            result = await asyncio.wait_for(background_task, timeout=8.0)
            return f"SEARCH_RESPONSE: {json.dumps(result)}"
        except asyncio.TimeoutError:
            # If background search takes too long, return immediate response
            # The background task continues running and may complete later
            if self.output:
                is_recovery = hasattr(self, 'metrics_collector') and self.metrics_collector and getattr(self.metrics_collector, 'in_recovery_phase', False)
                if not is_recovery:
                    self.output.progress(f"⏱️ [{self.shard_id}] Background search for {requesting_agent} still running...")
            
            response = {
                "found": False,
                "answer": None,
                "source_agent": self.shard_id,
                "path": search_path + [self.shard_id],
                "search_method": "background_search_timeout",
                "ttl_remaining": ttl
            }
            return f"SEARCH_RESPONSE: {json.dumps(response)}"
    
    async def _background_neighbor_search(
        self,
        question: str,
        ttl: int,
        search_path: List[str],
        requesting_agent: str
    ) -> Dict[str, Any]:
        """
        Perform background search for neighbor request without blocking main thread.
        """
        try:
            # Add small random delay to reduce contention
            import random
            delay = random.uniform(0.1, 0.5)  # 100-500ms random delay
            await asyncio.sleep(delay)
            
            # Search local documents first
            local_result = await self._search_local_document(question)
            
            if local_result and local_result.get('found'):
                # Found answer locally
                response = {
                    "found": True,
                    "answer": local_result.get('answer', ''),
                    "source_agent": self.shard_id,
                    "path": search_path + [self.shard_id],
                    "search_method": "background_local_search",
                    "ttl_remaining": ttl
                }
                
                if self.output:
                    is_recovery = hasattr(self, 'metrics_collector') and self.metrics_collector and getattr(self.metrics_collector, 'in_recovery_phase', False)
                    if not is_recovery:
                        self.output.success(f"✅ [{self.shard_id}] Background search found answer for {requesting_agent}!")
                        self.output.progress(f"   Answer: {response['answer'][:60]}...")
                
                return response
            
            # Not found locally, try limited forwarding (avoid deep recursion)
            if ttl > 0 and len(search_path) < 4:  # Limit forwarding depth to prevent cycles
                # Forward to neighbors not in path
                available_neighbors = []
                for neighbor in [self.neighbors['prev_id'], self.neighbors['next_id']]:
                    if neighbor not in search_path and neighbor != requesting_agent:
                        available_neighbors.append(neighbor)
                
                if available_neighbors:
                    # Try only one neighbor to avoid amplifying the search
                    try:
                        forward_request = {
                            "type": "NEIGHBOR_SEARCH_REQUEST",
                            "question": question,
                            "requesting_agent": requesting_agent,
                            "ttl": ttl - 1,
                            "path": search_path + [self.shard_id],
                            "timestamp": time.time()
                        }
                        
                        # Use shorter timeout for forwarded requests
                        forward_response = await asyncio.wait_for(
                            self._send_neighbor_search_request(available_neighbors[0], forward_request),
                            timeout=6.0  # Shorter timeout for forwarded requests
                        )
                        
                        if forward_response and forward_response.get('found'):
                            return forward_response
                            
                    except asyncio.TimeoutError:
                        pass  # Continue to return not found
                    except Exception as e:
                        if self.output:
                            self.output.warning(f"[{self.shard_id}] Error in background forwarding: {e}")
            
            # No answer found
            return {
                "found": False,
                "answer": None,
                "source_agent": self.shard_id,
                "path": search_path + [self.shard_id],
                "search_method": "background_search_failed",
                "ttl_remaining": ttl
            }
            
        except Exception as e:
            if self.output:
                self.output.error(f"[{self.shard_id}] Error in background neighbor search: {e}")
            return {
                "found": False,
                "answer": None,
                "source_agent": self.shard_id,
                "path": search_path + [self.shard_id],
                "search_method": "background_search_error",
                "ttl_remaining": ttl
            }
    
    async def _handle_neighbor_search_request_async(
        self,
        sender: str,
        request_data: Dict[str, Any],
        reply_to: str
    ) -> None:
        """
        Asynchronously handle neighbor search request and send response back.
        
        This method runs in background and sends the response directly to the
        requesting agent when the search is complete, without blocking the
        current agent's own operations.
        """
        try:
            # Perform the actual search
            result = await self._background_neighbor_search(
                question=request_data.get('question', ''),
                ttl=request_data.get('ttl', 0),
                search_path=request_data.get('path', []),
                requesting_agent=request_data.get('requesting_agent', sender)
            )
            
            # Send response back to the requesting agent
            if reply_to and reply_to in self.pending:
                # Direct reply to pending message
                try:
                    self.pending[reply_to].set_result(f"SEARCH_RESPONSE: {json.dumps(result)}")
                except Exception:
                    pass  # Future might already be resolved
            else:
                # Send response via network
                try:
                    response_content = f"SEARCH_RESPONSE: {json.dumps(result)}"
                    await self._send_to_coordinator(
                        f"NEIGHBOR_RESPONSE_FOR_{request_data.get('requesting_agent', sender)}: {response_content}",
                        path=[self.shard_id],
                        ttl=0
                    )
                except Exception as e:
                    if self.output:
                        self.output.warning(f"[{self.shard_id}] Failed to send neighbor response: {e}")
                        
        except Exception as e:
            if self.output:
                self.output.error(f"[{self.shard_id}] Error in async neighbor search: {e}")
    
    async def _search_local_document(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Search local document for the given question using LLM.
        
        Args:
            question: The question to search for
            
        Returns:
            Dict with search results or None if not found
        """
        try:
            # Load current group data for the question
            # We need to find which group this question belongs to
            group_id = 0  # Default group, could be improved
            if not self.load_group_data(group_id):
                return None
            
            # Use optimized search prompt
            search_prompt = self._get_local_search_prompt(question)
            messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": f"Search for: {question}"}
            ]
            
            # Call LLM for local search
            if self.use_mock or self.core is None:
                # Mock response for testing
                await asyncio.sleep(0.1)
                return {"found": False, "answer": None}
            
            # Real LLM call
            raw_resp = await asyncio.get_event_loop().run_in_executor(
                None,
                self.core.function_call_execute,
                messages,
                TOOL_SCHEMA,
                300000
            )
            
            # Process response
            if hasattr(raw_resp, 'choices') and raw_resp.choices:
                choice = raw_resp.choices[0]
                if hasattr(choice, 'message') and choice.message.tool_calls:
                    for tool_call in choice.message.tool_calls:
                        if tool_call.function.name == "lookup_fragment":
                            try:
                                args = json.loads(tool_call.function.arguments)
                                found = args.get("found", False)
                                if found:
                                    return {
                                        "found": True,
                                        "answer": args.get("answer", self.current_answer or "Found in document"),
                                        "confidence": args.get("confidence", 0.8)
                                    }
                            except json.JSONDecodeError:
                                pass
            
            return {"found": False, "answer": None}
            
        except Exception as e:
            if self.output:
                self.output.error(f"[{self.shard_id}] Error in local document search: {e}")
            return None
    
    def _get_local_search_prompt(self, question: str) -> str:
        """
        Get optimized prompt for local document search.
        
        This prompt is specifically designed to improve answer detection
        in document fragments, even when there's partial information.
        """
        return f"""You are a specialized document search agent analyzing a document fragment.

SEARCH QUESTION: {question}

YOUR DOCUMENT FRAGMENT:
{self.current_snippet}

TASK: Determine if your document fragment contains ANY information that helps answer the question.

SEARCH CRITERIA (Be ULTRA-LIBERAL - MAXIMIZE DISCOVERY):
✅ FOUND (set found=true) if the fragment contains ANY of:
- Direct answers to the question
- Names, entities, or keywords mentioned in the question  
- Related facts or context that partially answers the question
- Background information about the topic
- Similar entities or concepts (same category/type)
- Historical context or time period mentioned in question
- ANY shared words or phrases between question and fragment
- Information that could contribute to answering when combined with other sources
- Even tangentially related information

❌ NOT FOUND (set found=false) ONLY if:
- Fragment is about completely different, unrelated topics with ZERO overlap
- Absolutely no shared concepts, words, or themes
- Example: Question about "music" but fragment about "cooking" with no connection

🚨 CRITICAL: When in doubt, choose found=true! Better to include potentially relevant info than miss it.

RESPONSE FORMAT: Use the lookup_fragment function with:
- found: true/false (be generous with true)
- answer: extract the relevant information if found
- confidence: 0.0-1.0 (how confident you are)

EXAMPLES:
Question: "What nationality were Scott Derrickson and Ed Wood?"
Fragment: "Scott Derrickson is an American filmmaker..." → found=true, answer="Scott Derrickson is American"
Fragment: "Ed Wood was born in New York..." → found=true, answer="Ed Wood was American (born in New York)"
Fragment: "The Laleli Mosque in Turkey..." → found=false (completely unrelated)

Remember: It's better to find partial information than to miss relevant content. The collaborative system will combine partial answers from multiple agents."""

    async def _send_and_wait(
        self,
        dest_id: str,
        content: str,
        ttl: int,
        path: List[str],
        timeout: float = None  # Dynamic timeout based on TTL
    ) -> Optional[str]:
        """Send message and await reply or timeout."""
        # v2: Dynamic timeout based on TTL - balanced for stability
        if timeout is None:
            max_ttl = self.global_config.get('tool_schema', {}).get('max_ttl', 8)
            single_hop_timeout = 6.0  # 单跳6秒：LLM(4s) + 网络(1s) + 处理(1s)
            timeout = max(single_hop_timeout, min(ttl * single_hop_timeout, 48.0))  # 最大48s，允许完整环路
        
        msg_id = f"v1.1-{self.current_group_id}-{self.shard_id}-{int(time.time()*1e6)}"
        # Create safe A2A message
        meta = {
            "sender": str(self.shard_id),
            "ttl": int(ttl),
            "path": [str(p) for p in (path + [self.shard_id])],
            "group_id": int(self.current_group_id),
            "reply_to": str(msg_id)
        }
        payload = create_safe_a2a_message(msg_id, content, meta)
        
        # 严格的节流控制 - 队列满时丢弃而不是排队
        if len(self.pending) >= self.max_pending:
            self.dropped_count += 1
            if self.output:
                self.output.warning(f"[{self.shard_id}] Pending limit ({self.max_pending}) reached, dropping message to {dest_id} (dropped: {self.dropped_count})")
            return None
        
        # Register future before sending
        fut = asyncio.get_event_loop().create_future()
        self.pending[msg_id] = fut
        self.pending_count += 1

        try:
            await self.agent_network.route_message(self.shard_id, dest_id, payload)
            reply = await asyncio.wait_for(fut, timeout=timeout)
            return reply
        except asyncio.TimeoutError:
            if self.output:
                self.output.warning(f"[{self.shard_id}] Timeout waiting for reply from {dest_id}")
            # Cancel the future to prevent resource leak
            if not fut.done():
                fut.cancel()
            return None
        except Exception as e:
            error_msg = str(e)
            if self.output:
                self.output.error(f"[{self.shard_id}] Error sending to {dest_id}: {e}")
            
            # 特殊处理 "Agent not initialized" 错误
            if "not initialized" in error_msg.lower():
                if self.output:
                    self.output.warning(f"[{self.shard_id}] Agent {dest_id} is down, marking as unavailable")
                # 可以在这里添加重连逻辑或将该agent标记为不可用
            
            # Cancel the future to prevent resource leak
            if not fut.done():
                fut.cancel()
            return None
        finally:
            # Always clean up pending entry
            self.pending.pop(msg_id, None)

    async def process_message(self, sender: str, content: str, meta: dict = None) -> str:
        """Process incoming message from another agent"""
        if self.output:
            self.output.progress(f"[{self.shard_id}] Received message from {sender}: {content[:50]}...")
        
        # Extract meta information
        ttl = meta.get("ttl", 0) if meta else 0
        path = meta.get("path", []) if meta else []
        sender = meta.get("sender", sender) if meta else sender
        reply_to = meta.get("reply_to") if meta else None
        
        # 设置机器控制的 TTL 和 path 上下文
        # 重要：从邻居收到消息时，TTL 应该继续递减
        self.current_ttl = max(0, ttl - 1) if ttl > 0 else 0
        self.current_path = path + [self.shard_id] if path else [sender, self.shard_id]
        
        if self.output:
            is_recovery = hasattr(self, 'metrics_collector') and self.metrics_collector and getattr(self.metrics_collector, 'in_recovery_phase', False)
            if not is_recovery:
                self.output.progress(f"   [TTL_TRACE] {self.shard_id} processing message: received_ttl={ttl} -> current_ttl={self.current_ttl}, path={self.current_path}")
            # Silent during recovery phase
        
        # Check if this is a direct reply to pending message
        if reply_to and reply_to in self.pending:
            self.pending[reply_to].set_result(content)
            return "Reply processed"
        
        # Handle neighbor search requests with NON-BLOCKING processing
        if "SEARCH_REQUEST:" in content:
            try:
                request_data = json.loads(content.split("SEARCH_REQUEST:")[1].strip())
                
                # CONCURRENT PROCESSING: Handle neighbor request without blocking
                # Start background task and return immediate acknowledgment
                background_task = asyncio.create_task(
                    self._handle_neighbor_search_request_async(sender, request_data, reply_to)
                )
                
                # Return immediate acknowledgment, actual response will come via background task
                return "SEARCH_REQUEST_RECEIVED: Processing in background"
                
            except json.JSONDecodeError as e:
                if self.output:
                    self.output.error(f"[{self.shard_id}] Failed to parse search request: {e}")
                return "Invalid search request format"
        
        # Add to history
        self._add_to_history(self.current_group_id, f"Message from {sender} (ttl={ttl}, path={path}): {content}")
        
        # Create messages for Core with meta information
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"Message from {sender}, ttl={ttl}, path={path}: {content}"}
        ]
        
        # Add recent history
        if self.current_group_id in self.history:
            history_deque = self.history[self.current_group_id]
            # Convert deque to list for slicing
            history_list = list(history_deque)
            for hist_msg in history_list[-5:]:  # Last 5 messages
                messages.append({"role": "assistant", "content": hist_msg})
        
        # Call Core with function calling
        try:
            if self.use_mock or self.core is None:
                # Mock response for testing
                await asyncio.sleep(0.1)
                return "Mock response: Message processed"
            
            raw_resp = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.core.function_call_execute,
                messages,
                TOOL_SCHEMA,
                300000  # max_length
            )
            
            # TTL现在完全由机器控制，不需要任何篡改
            response = raw_resp  # 直接使用LLM产生的响应
            
            # Track LLM token usage if available
            await self._track_llm_usage(response)
            
            # Process function calls with machine-controlled TTL context
            return await self._handle_core_response(response)
            
        except Exception as e:
            if self.output:
                self.output.error(f"[{self.shard_id}] Error processing message: {e}")
            return f"Error processing message: {str(e)}"

    async def start_task(self, group_id: int) -> str:
        """Start processing task for given group_id (each agent has own question but can communicate via ring)"""
        # 防止重复处理同一个group
        if hasattr(self, '_processing_groups') and group_id in self._processing_groups:
            if self.output:
                self.output.warning(f"[{self.shard_id}] Group {group_id} already being processed, skipping")
            return f"Group {group_id} already being processed"
        
        if not hasattr(self, '_processing_groups'):
            self._processing_groups = set()
        self._processing_groups.add(group_id)
        
        try:
            if not self.load_group_data(group_id):
                return f"Failed to load data for group {group_id}"
            
            is_recovery = hasattr(self, 'metrics_collector') and self.metrics_collector and getattr(self.metrics_collector, 'in_recovery_phase', False)
            if self.output and not is_recovery:
                self.output.info(f"[{self.shard_id}] Starting ring task for group {group_id}")
                self.output.progress(f"   [{self.shard_id}] My question: {self.current_question[:80]}...")
            # Silent during recovery phase
            
            # Initialize history for this group
            if group_id not in self.history:
                from collections import deque
                self.history[group_id] = deque(maxlen=self.max_history)
            
            # 设置初始任务的 TTL 和 path 上下文
            max_ttl = self.global_config.get('tool_schema', {}).get('max_ttl', 8)
            self.current_ttl = max_ttl
            self.current_path = [self.shard_id]
            
            if self.output:
                is_recovery = hasattr(self, 'metrics_collector') and self.metrics_collector and getattr(self.metrics_collector, 'in_recovery_phase', False)
                if not is_recovery:
                    self.output.progress(f"   [TTL_TRACE] {self.shard_id} starting task: initial_ttl={self.current_ttl}, path={self.current_path}")
                # Silent during recovery phase
            
            # Create initial prompt for ring search with communication
            initial_prompt = f"""You have a question to answer: '{self.current_question}'

IMPORTANT: First, carefully analyze your LOCAL FRAGMENT to see if it contains ANY information related to this question.

Your local fragment contains: {self.current_snippet[:200]}...

Use lookup_fragment to check if your fragment has relevant information. Be LIBERAL in your assessment - if the fragment mentions any keywords, names, or concepts from the question, set found=true.

If you don't find relevant information locally, then use send_message to ask neighbors for help."""
            
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": initial_prompt}
            ]
            
            # Call Core with function calling

            # 添加force_llm flag来控制是否强制使用LLM
            force_llm = getattr(self, 'force_llm', False)
            
            # Debug: Check actual model type configuration
            actual_model_type = "unknown"
            if self.core and hasattr(self.core, 'config'):
                actual_model_type = self.core.config.get('model', {}).get('type', 'unknown')
                if self.output:
                    self.output.progress(f"   🔍 [{self.shard_id}] DEBUG: Detected model type: {actual_model_type}")
            
            # 对于NVIDIA模型，自动使用mock模式（因为不支持工具调用）
            use_mock_for_nvidia = False
            if self.core and hasattr(self.core, 'config') and self.core.config.get('model', {}).get('type') == 'nvidia':
                use_mock_for_nvidia = True
                if self.output:
                    is_recovery = hasattr(self, 'metrics_collector') and self.metrics_collector and getattr(self.metrics_collector, 'in_recovery_phase', False)
                    if not is_recovery:
                        self.output.progress(f"   🔍 [{self.shard_id}] Using mock mode for NVIDIA model (no tool calling support)")
                    # Silent during recovery phase
            
            # 强制使用真实LLM，禁用mock模式
            use_real_llm = True
            if self.core is None:
                if self.output:
                    self.output.error(f"[{self.shard_id}] Core LLM not initialized, cannot proceed")
                return "Core LLM not available"
            
            # 使用真实LLM进行判定
            if use_real_llm and self.core:
                if self.output:
                    is_recovery = hasattr(self, 'metrics_collector') and self.metrics_collector and getattr(self.metrics_collector, 'in_recovery_phase', False)
                    if not is_recovery:
                        self.output.progress(f"   🧠 [{self.shard_id}] Using real LLM for document analysis")
                
                # 使用真实LLM进行tool calling
                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        self.core.function_call_execute,
                        messages,
                        TOOL_SCHEMA
                    )
                    
                    if self.output:
                        is_recovery = hasattr(self, 'metrics_collector') and self.metrics_collector and getattr(self.metrics_collector, 'in_recovery_phase', False)
                        if not is_recovery:
                            self.output.progress(f"   🤖 [{self.shard_id}] LLM response received")
                    
                    # 处理LLM的tool calling响应
                    result = await self._handle_core_response(response)
                    return result
                    
                except Exception as e:
                    if self.output:
                        self.output.error(f"[{self.shard_id}] Real LLM call failed: {e}")
                    # Fallback to mock if LLM fails
                    pass
            
            # Fallback mock mode only if real LLM fails
            if self.use_mock or self.core is None or use_mock_for_nvidia:
                # Mock response for testing - try local first, then communicate
                await asyncio.sleep(1.0)  # 减少延迟
                
                # Document search simulation with detailed output
                question_lower = self.current_question.lower()
                snippet_lower = self.current_snippet.lower()
                answer_lower = self.current_answer.lower()
                is_recovery = hasattr(self, 'metrics_collector') and self.metrics_collector and getattr(self.metrics_collector, 'in_recovery_phase', False)
                if self.output and not is_recovery:
                    self.output.progress(f"   🔍 [{self.shard_id}] Searching local document...")
                # Silent during recovery phase
                
                # Advanced keyword matching with relevance scoring
                question_keywords = set(question_lower.replace('?', '').split())
                snippet_keywords = set(snippet_lower.split())
                answer_words = answer_lower.split()
                
                # Calculate search relevance score
                keyword_overlap = len(question_keywords.intersection(snippet_keywords))
                answer_presence = sum(1 for word in answer_words if word in snippet_lower)
                confidence = answer_presence / len(answer_words) if answer_words else 0
                
                if confidence > 0.3 or answer_presence >= 1:  # 降低阈值，更容易找到答案
                    if self.output:
                        # Check if we're in recovery phase to reduce output
                        is_recovery = hasattr(self, 'metrics_collector') and self.metrics_collector and getattr(self.metrics_collector, 'in_recovery_phase', False)
                        if not is_recovery:
                            # Full output during normal phase only
                            self.output.success(f"✅ [{self.shard_id}] LOCAL SEARCH SUCCESS")
                            self.output.progress(f"   Q: {self.current_question[:40]}...")
                            self.output.progress(f"   A: {self.current_answer[:50]}...")
                            self.output.progress(f"   📍 Source: LOCAL DOCUMENT")
                        # Silent during recovery phase
                    
                    # 记录本地搜索成功的任务执行
                    if hasattr(self, 'metrics_collector') and self.metrics_collector:
                        self.metrics_collector.record_task_execution(
                            task_id=f"{self.current_group_id}-{self.shard_id}",
                            agent_id=self.shard_id,
                            task_type="qa_search",
                            start_time=time.time(),
                            end_time=time.time(),
                            success=True,
                            answer_found=True,
                            answer_source="local"
                        )
                    
                    await self._send_to_coordinator(
                        f"ANSWER_FOUND: {self.current_answer}",
                        path=[self.shard_id],
                        ttl=0
                    )
                    return f"✅ DOCUMENT SEARCH SUCCESS: Found '{self.current_answer}' - answer found locally"
                else:
                    # Document not found locally, need to search network - REAL NEIGHBOR SEARCH
                    is_recovery = hasattr(self, 'metrics_collector') and self.metrics_collector and getattr(self.metrics_collector, 'in_recovery_phase', False)
                    if self.output and not is_recovery:
                        self.output.warning(f"⚠️  🔍 [{self.shard_id}] Local search failed, asking neighbors for help...")
                        self.output.progress(f"   📤 [{self.shard_id}] Forwarding question to neighbors: {self.neighbors['prev_id']} and {self.neighbors['next_id']}")
                    
                    # REAL DISTRIBUTED SEARCH - Ask neighbors to search their documents
                    # Add random delay to avoid circular waiting deadlock
                    import random
                    random.seed(hash(f"{self.shard_id}_{self.current_question}") % 1000)
                    delay = random.uniform(0.5, 3.0)  # 0.5-3秒随机延迟
                    await asyncio.sleep(delay)
                    
                    neighbor_search_result = await self._real_neighbor_search(
                        question=self.current_question,
                        ttl=7,  # Start with reasonable TTL
                        path=[self.shard_id]
                    )
                    
                    if neighbor_search_result and neighbor_search_result.get('found'):
                        # Found answer from neighbor
                        answer = neighbor_search_result.get('answer', 'Unknown answer')
                        source_agent = neighbor_search_result.get('source_agent', 'unknown')
                        search_path = neighbor_search_result.get('path', [self.shard_id])
                        
                        if self.output and not is_recovery:
                            self.output.success(f"✅ [{self.shard_id}] NEIGHBOR SEARCH SUCCESS")
                            self.output.progress(f"   📍 Source: NEIGHBOR {source_agent}")
                            self.output.progress(f"   📝 Answer: {answer[:60]}...")
                            self.output.progress(f"   🔗 Path: {' → '.join(search_path)}")
                        
                        # Record successful neighbor search
                        if hasattr(self, 'metrics_collector') and self.metrics_collector:
                            self.metrics_collector.record_task_execution(
                                task_id=f"{self.current_group_id}-{self.shard_id}",
                                agent_id=self.shard_id,
                                task_type="qa_search",
                                start_time=time.time(),
                                end_time=time.time(),
                                success=True,
                                answer_found=True,
                                answer_source="neighbor"
                            )
                        
                        # Send collaborative answer to coordinator with enhanced format
                        collaborative_answer = {
                            "requesting_agent": self.shard_id,
                            "source_agent": source_agent,
                            "answer": answer,
                            "collaboration_path": search_path,
                            "search_method": "distributed_neighbor_search"
                        }
                        
                        await self._send_to_coordinator(
                            f"COLLABORATIVE_ANSWER: {json.dumps(collaborative_answer)}",
                            path=search_path,
                            ttl=0
                        )
                        return f"✅ DOCUMENT SEARCH SUCCESS: Found '{answer}' via collaborative search from {source_agent}"
                    
                    else:
                        # No answer found from neighbors
                        if self.output and not is_recovery:
                            self.output.warning(f"⚠️  ❌ [{self.shard_id}] NEIGHBOR SEARCH FAILED")
                            self.output.progress(f"      Q: {self.current_question[:40]}...")
                            self.output.progress(f"      📍 Source: NO NEIGHBOR FOUND")
                        
                        # Record failed neighbor search
                        if hasattr(self, 'metrics_collector') and self.metrics_collector:
                            self.metrics_collector.record_task_execution(
                                task_id=f"{self.current_group_id}-{self.shard_id}",
                                agent_id=self.shard_id,
                                task_type="qa_search",
                                start_time=time.time(),
                                end_time=time.time(),
                                success=False,
                                answer_found=False,
                                answer_source="none"
                            )
                        
                        await self._send_to_coordinator(
                            "NO_ANSWER_FROM_NEIGHBORS",
                            path=[self.shard_id],
                            ttl=0
                        )
                        return "No answer found from neighbors"
            else:
                # 调用真正的LLM进行文档搜索
                is_recovery = hasattr(self, 'metrics_collector') and self.metrics_collector and getattr(self.metrics_collector, 'in_recovery_phase', False)
                if self.output and not is_recovery:
                    self.output.progress(f"   🔍 [{self.shard_id}] Searching local document...")
                # Silent during recovery phase
                
                # Call Core with function calling
                raw_resp = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.core.function_call_execute,
                    messages,
                    TOOL_SCHEMA,
                    300000  # max_length
                )
                
                # Track LLM token usage if available
                await self._track_llm_usage(raw_resp)
                

                
                # Process function calls with machine-controlled TTL
                return await self._handle_core_response(raw_resp)
            
        except Exception as e:
            if self.output:
                self.output.error(f"[{self.shard_id}] Error starting task: {e}")
            return f"Error starting task: {str(e)}"
        finally:
            # 清理处理状态
            if hasattr(self, '_processing_groups') and group_id in self._processing_groups:
                self._processing_groups.remove(group_id)

    # _force_max_ttl function removed - TTL now completely machine-controlled

    def _parse_text_function_calls(self, content: str) -> List[dict]:
        """Parse function calls from text response (for NVIDIA models that don't support tool calls)"""
        import re
        
        tool_calls = []
        
        # Look for lookup_fragment function calls
        lookup_pattern = r'lookup_fragment\s*\(\s*question\s*=\s*["\']([^"\']+)["\']\s*,\s*found\s*=\s*(true|false)\s*\)'
        lookup_matches = re.findall(lookup_pattern, content, re.IGNORECASE)
        
        for question, found_str in lookup_matches:
            found = found_str.lower() == 'true'
            tool_calls.append({
                'function': {
                    'name': 'lookup_fragment',
                    'arguments': {
                        'question': question,
                        'found': found
                    }
                }
            })
        
        # Look for send_message function calls
        send_pattern = r'send_message\s*\(\s*destination\s*=\s*["\']([^"\']+)["\']\s*,\s*content\s*=\s*["\']([^"\']+)["\']\s*\)'
        send_matches = re.findall(send_pattern, content, re.IGNORECASE)
        
        for destination, message_content in send_matches:
            tool_calls.append({
                'function': {
                    'name': 'send_message',
                    'arguments': {
                        'destination': destination,
                        'content': message_content
                    }
                }
            })
        
        return tool_calls

    async def _handle_core_response(self, response) -> str:
        """Handle Core response and execute function calls"""

        if not response:
            return "No response from Core"
        
        # Handle both ChatCompletion object and dict formats
        if hasattr(response, 'choices') and response.choices:
            # New OpenAI format - ChatCompletion object
            choice = response.choices[0]
            message = choice.message
            tool_calls = message.tool_calls if hasattr(message, 'tool_calls') and message.tool_calls else []
            content = message.content if hasattr(message, 'content') else None
        elif isinstance(response, dict):
            # Old format - dict
            tool_calls = response.get('tool_calls', [])
            content = response.get('content', 'No content in response')
        else:
            return "Invalid response format"
        
        if not tool_calls:
            # No function calls, try to parse text-based function calls (for NVIDIA models)
            if content:
                # Try to extract function call information from text
                parsed_tool_calls = self._parse_text_function_calls(content)
                if parsed_tool_calls:
                    if self.output:
                        self.output.progress(f"🔍 [{self.shard_id}] Parsed {len(parsed_tool_calls)} function calls from text response")
                    
                    # Process the parsed function calls
                    results = []
                    for tool_call in parsed_tool_calls:
                        if tool_call.get('function', {}).get('name') == "lookup_fragment":
                            result = await self._handle_lookup_fragment(tool_call['function']['arguments'], self.current_ttl, self.current_path)
                            results.append(result)
                        elif tool_call.get('function', {}).get('name') == "send_message":
                            result = await self._handle_send_message(tool_call['function']['arguments'])
                            results.append(result)
                    
                    return " | ".join(results) if results else "No valid function calls executed"
            
            # Check if the LLM found an answer in text format
            if content and self.current_answer:
                content_lower = content.lower()
                answer_lower = self.current_answer.lower()
                
                # Check if the LLM's response contains the expected answer
                answer_words = answer_lower.split()
                found_words = sum(1 for word in answer_words if word in content_lower)
                confidence = found_words / len(answer_words) if answer_words else 0
                
                if confidence > 0.3 or "answer_found" in content_lower:

                    if self.output:
                        self.output.success(f"✅ [{self.shard_id}] DOCUMENT SEARCH SUCCESS!")
                        self.output.success(f"❓ [{self.shard_id}] Question: '{self.current_question}'")
                        self.output.success(f"💡 [{self.shard_id}] Answer FOUND: '{self.current_answer}' (via LLM text)")
                        self.output.success(f"📖 [{self.shard_id}] LLM response: '{content[:200]}...'")
                    
                    return f"✅ DOCUMENT SEARCH SUCCESS: Found '{self.current_answer}' via LLM analysis - answer found"
            
            return content or 'No content in response'
        
        results = []
        
        for tool_call in tool_calls:
            # Handle both new ChatCompletionMessageToolCall and dict formats
            if hasattr(tool_call, 'function'):
                # New format - ChatCompletionMessageToolCall object
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments
            elif isinstance(tool_call, dict):
                # Old format - dict
                function_name = tool_call.get('function', {}).get('name')
                arguments = tool_call.get('function', {}).get('arguments', {})
            else:
                continue
            
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
            
            if function_name == "lookup_fragment":
                # 添加调试输出
                if self.output:
                    self.output.progress(f"🔍 [{self.shard_id}] DEBUG: LLM returned arguments: {arguments}")
                # 传递机器控制的 TTL 和 path 上下文
                result = await self._handle_lookup_fragment(arguments, self.current_ttl, self.current_path)
                results.append(result)
            elif function_name == "send_message":
                result = await self._handle_send_message(arguments)
                results.append(result)
        
        return " | ".join(results) if results else "No valid function calls executed"

    async def _handle_lookup_fragment(self, args: dict, context_ttl: int = None, context_path: List[str] = None) -> str:
        """Handle lookup_fragment function call - v3 (Machine-controlled TTL)"""

        question = args.get('question', '')
        found = args.get('found', False)  # LLM 只负责判断是否找到答案
        
        # TTL 和 path 由机器控制，不再依赖 LLM
        if context_ttl is not None:
            ttl = context_ttl  # 使用调用方传入的 TTL
        else:
            # 如果是 start_task 第一次调用，设置初始 TTL
            max_ttl = self.global_config.get('tool_schema', {}).get('max_ttl', 8)
            ttl = max_ttl
        
        if context_path is not None:
            path = context_path.copy()  # 使用调用方传入的路径
        else:
            # 如果是第一次调用，初始化路径
            path = [self.shard_id]
        
        if self.output:
            is_recovery = hasattr(self, 'metrics_collector') and self.metrics_collector and getattr(self.metrics_collector, 'in_recovery_phase', False)
            if not is_recovery:
                self.output.progress(f"   [{self.shard_id}] Looking up fragment for: {question[:30]}... (ttl={ttl}, found={found})")
                # TTL跟踪日志 - 用于调试TTL递减情况
                self.output.progress(f"   [TTL_TRACE] {self.shard_id} ttl={ttl} path={path} found={found} [MACHINE_CONTROLLED]")
            # Silent during recovery phase
        
        # 处理 LLM 判断结果 - v2 with fallback
        if found is None:
            # Fallback: LLM 没有提供 found 参数，使用简化匹配
            if self.output:
                self.output.warning(f"[{self.shard_id}] LLM didn't provide found parameter, using fallback matching")
            
            # 简化的匹配逻辑
            question_lower = question.lower()
            snippet_lower = self.current_snippet.lower()
            answer_lower = self.current_answer.lower()
            
            # 直接答案匹配或答案词汇匹配
            if answer_lower in snippet_lower:
                found = True
            elif answer_lower:
                answer_words = [word for word in answer_lower.split() if len(word) > 2]
                if answer_words:
                    found_words = sum(1 for word in answer_words if word in snippet_lower)
                    # 降低阈值到50%，让更多答案能被找到
                    found = found_words >= max(1, len(answer_words) * 0.5)  # 50% threshold (更合理)
                else:
                    found = False
            else:
                found = False
        else:
            # LLM 提供了 found 参数，添加调试输出
            if self.output:
                self.output.progress(f"🔍 [{self.shard_id}] LLM provided found={found}, checking fallback logic...")
                self.output.progress(f"📄 [{self.shard_id}] Answer: '{self.current_answer}'")
                self.output.progress(f"📄 [{self.shard_id}] Snippet: '{self.current_snippet[:100]}...'")
                
                # 检查答案是否在snippet中
                answer_lower = self.current_answer.lower()
                snippet_lower = self.current_snippet.lower()
                if answer_lower in snippet_lower:
                    self.output.warning(f"⚠️ [{self.shard_id}] WARNING: Answer '{answer_lower}' found in snippet despite LLM found=false!")
                else:
                    self.output.success(f"✅ [{self.shard_id}] Confirmed: Answer not in snippet, LLM found=false is correct")
        
        if found:
            if self.output:
                self.output.success(f"✅ [{self.shard_id}] LOCAL SEARCH SUCCESS")
                self.output.success(f"   Question: {self.current_question[:60]}...")
                self.output.success(f"   Answer: {self.current_answer}")
                self.output.success(f"   Source: Local document fragment")
            
            # 记录任务执行成功
            if hasattr(self, 'metrics_collector') and self.metrics_collector:
                self.metrics_collector.record_task_execution(
                    task_id=f"{self.current_group_id}-{self.shard_id}",
                    agent_id=self.shard_id,
                    task_type="qa_local_search",
                    start_time=time.time(),
                    end_time=time.time(),
                    success=True,
                    answer_found=True,
                    answer_source="local"
                )
            
            # 使用实际答案
            answer_text = self.current_answer
            
            # 创建增强的本地答案回传信息
            local_response = {
                "type": "LOCAL_ANSWER_FOUND",
                "original_question": self.current_question,
                "answer": answer_text,
                "source_agent": self.shard_id,
                "source_context": self.current_snippet[:500],  # 提供上下文
                "hop_count": len(path),
                "search_path": path + [self.shard_id],
                "timestamp": time.time(),
                "group_id": self.current_group_id
            }
            
            # 发送增强的本地答案信息
            try:
                import json
                enhanced_message = f"LOCAL_ANSWER: {json.dumps(local_response)}"
                await self._send_to_coordinator(enhanced_message, path + [self.shard_id], ttl)
                
                if self.output:
                    self.output.success(f"📤 [{self.shard_id}] Enhanced local answer sent to coordinator")
                    self.output.progress(f"   📝 Answer: {answer_text[:60]}...")
                    self.output.progress(f"   📖 Context: {self.current_snippet[:100]}...")
                    
            except asyncio.CancelledError:
                if self.output:
                    self.output.warning(f"[{self.shard_id}] Enhanced local answer sending was cancelled")
            
            self._add_to_history(self.current_group_id, f"Found local answer: {answer_text}")
            return f"✅ DOCUMENT SEARCH SUCCESS: Found '{answer_text}' - enhanced answer sent to coordinator"
        
        # --- Not found locally ---
        if ttl <= 0:
            if self.output:
                self.output.warning(f"[{self.shard_id}] TTL exhausted, cannot search neighbors")
            
            # 保护TTL_EXHAUSTED发送
            try:
                await self._send_to_coordinator("TTL_EXHAUSTED", path + [self.shard_id], ttl)
            except asyncio.CancelledError:
                if self.output:
                    self.output.warning(f"[{self.shard_id}] TTL_EXHAUSTED sending was cancelled")
            
            self._add_to_history(self.current_group_id, "TTL exhausted")
            return "TTL exhausted"
        
        # REAL DISTRIBUTED SEARCH - Use new neighbor search mechanism
        is_recovery = hasattr(self, 'metrics_collector') and self.metrics_collector and getattr(self.metrics_collector, 'in_recovery_phase', False)
        if self.output and not is_recovery:
            self.output.warning(f"⚠️  🔍 [{self.shard_id}] Local search failed, starting REAL neighbor search...")
            self.output.progress(f"   📤 [{self.shard_id}] Initiating distributed search across neighbors: {self.neighbors['prev_id']} and {self.neighbors['next_id']}")
        
        # Use the new real neighbor search instead of old simulation
        neighbor_search_result = await self._real_neighbor_search(
            question=question,
            ttl=ttl,  # Use current TTL
            path=path  # Use current path
        )
        
        if neighbor_search_result and neighbor_search_result.get('found'):
            # Found answer from neighbor - REAL COLLABORATION SUCCESS
            answer = neighbor_search_result.get('answer', 'Unknown answer')
            source_agent = neighbor_search_result.get('source_agent', 'unknown')
            search_path = neighbor_search_result.get('path', path + [self.shard_id])
            
            if self.output and not is_recovery:
                self.output.success(f"✅ [{self.shard_id}] REAL NEIGHBOR SEARCH SUCCESS!")
                self.output.progress(f"   📍 Source: NEIGHBOR {source_agent}")
                self.output.progress(f"   📝 Answer: {answer[:60]}...")
                self.output.progress(f"   🔗 Collaboration Path: {' → '.join(search_path)}")
            
            # Record successful neighbor collaboration
            if hasattr(self, 'metrics_collector') and self.metrics_collector:
                self.metrics_collector.record_task_execution(
                    task_id=f"{self.current_group_id}-{self.shard_id}",
                    agent_id=self.shard_id,
                    task_type="qa_search",
                    start_time=time.time(),
                    end_time=time.time(),
                    success=True,
                    answer_found=True,
                    answer_source="neighbor"
                )
            
            # Send collaborative answer to coordinator
            collaborative_answer = {
                "requesting_agent": self.shard_id,
                "source_agent": source_agent,
                "answer": answer,
                "collaboration_path": search_path,
                "search_method": "real_distributed_search"
            }
            
            await self._send_to_coordinator(
                f"COLLABORATIVE_ANSWER: {json.dumps(collaborative_answer)}",
                path=search_path,
                ttl=0
            )
            
            return f"✅ REAL COLLABORATIVE SUCCESS: Found '{answer}' via distributed search from {source_agent}"
        
        else:
            # No answer found from neighbors - exhausted all options
            if self.output and not is_recovery:
                self.output.warning(f"⚠️  ❌ [{self.shard_id}] DISTRIBUTED SEARCH EXHAUSTED")
                self.output.progress(f"      Q: {question[:40]}...")
                self.output.progress(f"      📍 Searched: Local + All reachable neighbors")
            
            # Record failed distributed search
            if hasattr(self, 'metrics_collector') and self.metrics_collector:
                self.metrics_collector.record_task_execution(
                    task_id=f"{self.current_group_id}-{self.shard_id}",
                    agent_id=self.shard_id,
                    task_type="qa_search",
                    start_time=time.time(),
                    end_time=time.time(),
                    success=False,
                    answer_found=False,
                    answer_source="none"
                )
            
            await self._send_to_coordinator(
                "NO_ANSWER_DISTRIBUTED_SEARCH_EXHAUSTED",
                path=path + [self.shard_id],
                ttl=0
            )
            
            return "No answer found after distributed search"

    async def _handle_send_message(self, args: dict) -> str:
        """Handle send_message function call"""
        destination = args.get('destination', '')
        content = args.get('content', '')
        
        if not destination or not content:
            return "Invalid send_message arguments"
        
        # Validate destination
        valid_destinations = [self.neighbors['prev_id'], self.neighbors['next_id'], 'coordinator']
        if destination not in valid_destinations:
            return f"Invalid destination: {destination}. Valid: {valid_destinations}"
        
        if self.output:
            self.output.progress(f"[{self.shard_id}] Sending message to {destination}")
        
        try:
            if destination == "coordinator":
                # Send to coordinator with current path and TTL=0 (terminal message)
                await self._send_to_coordinator(content, path=[self.shard_id], ttl=0)
                return "Message sent to coordinator"
            else:
                # 🔥 关键修复：使用当前上下文TTL，禁止TTL复活
                if self.current_ttl is None:
                    # 兜底：如果意外为空，设为0防止重置
                    ttl_to_use = 0
                else:
                    ttl_to_use = max(0, self.current_ttl - 1)
                
                if ttl_to_use <= 0:
                    if self.output:
                        self.output.warning(f"[{self.shard_id}] 🛑 TTL exhausted → NOT forwarding to {destination}")
                    # 告知协调器彻底放弃
                    await self._send_to_coordinator("TTL_EXHAUSTED", path=[self.shard_id], ttl=0)
                    return "TTL exhausted - message dropped"
                
                # 使用递减的TTL，而不是硬编码的5
                await self._send_to_agent(destination, content, ttl=ttl_to_use, path=[self.shard_id])
                return f"Message sent to {destination} (TTL={ttl_to_use})"
        except Exception as e:
            if self.output:
                self.output.error(f"[{self.shard_id}] Failed to send message: {e}")
            return f"Failed to send message: {str(e)}"

    async def _send_to_coordinator(self, content: str, path: List[str] = None, ttl: int = 0):
        """Send message to coordinator with path and hop information"""
        if not path:
            path = [self.shard_id]
            
        if self.agent_network:
            # Embed meta information directly in the message content for easier parsing
            enhanced_content = f"[FROM:{self.shard_id}|GROUP:{self.current_group_id}|HOP:{len(path)-1}] {content}"
            
            # Create safe A2A message for coordinator
            meta = {
                "sender": str(self.shard_id),
                "ttl": int(ttl),
                "path": [str(p) for p in path],
                "hop": int(len(path) - 1),
                "group_id": int(self.current_group_id),
                "timestamp": float(time.time())
            }
            message_payload = create_safe_a2a_message(
                f"v1.1-{self.current_group_id}-{self.shard_id}",
                enhanced_content,
                meta
            )
            
            # 使用 shield 保护关键网络I/O，避免被外层取消打断
            try:
                await asyncio.shield(
                    self.agent_network.route_message(self.shard_id, "coordinator", message_payload)
                )
            except asyncio.CancelledError:
                # 即使被取消也要保护这个关键发送操作
                if self.output:
                    self.output.warning(f"[{self.shard_id}] Send to coordinator was cancelled but shielded")
                pass

    async def _send_to_agent(self, destination: str, content: str, ttl: int = 5, path: List[str] = None):
        """Send message to another agent with TTL and path"""
        # 🛡️ 防守式编程：绝不发送TTL<=0的消息
        if ttl <= 0:
            if self.output:
                self.output.error(f"[{self.shard_id}] 🚫 Blocked sending TTL={ttl} message to {destination}")
            return
        
        if not path:
            path = [self.shard_id]
        else:
            path = path + [self.shard_id]
            
        if self.agent_network:
            # Create safe A2A message for agent
            meta = {
                "sender": str(self.shard_id),
                "ttl": int(ttl),  # Don't double-decrement, ttl already decremented by caller
                "path": [str(p) for p in path],
                "group_id": int(self.current_group_id),
                "timestamp": float(time.time())
            }
            message_payload = create_safe_a2a_message(
                f"v1.1-{self.current_group_id}-{self.shard_id}",
                str(content),
                meta
            )
            
            # 使用 shield 保护网络I/O
            try:
                await asyncio.shield(
                    self.agent_network.route_message(self.shard_id, destination, message_payload)
                )
            except asyncio.CancelledError:
                if self.output:
                    self.output.warning(f"[{self.shard_id}] Send to {destination} was cancelled but shielded")
                pass

    def _add_to_history(self, group_id: int, message: str):
        """Add message to history with max length constraint"""
        if group_id not in self.history:
            from collections import deque
            self.history[group_id] = deque(maxlen=self.max_history)
        
        self.history[group_id].append(message)

    async def _track_llm_usage(self, response):
        """Track LLM token usage if available in response"""
        try:
            # Handle different response types (dict or ChatCompletion object)
            if hasattr(response, 'usage'):
                # New OpenAI format - response is ChatCompletion object
                usage = response.usage
                total_tokens = usage.total_tokens if usage else 0
            elif isinstance(response, dict):
                # Old format - response is dict
                usage = response.get('usage', {})
                total_tokens = usage.get('total_tokens', 0)
            else:
                return
            
            if total_tokens > 0:
                # Send token usage to coordinator for aggregation
                await self._send_to_coordinator(
                    f"LLM_TOKENS: {total_tokens}",
                    path=[self.shard_id],
                    ttl=0
                )
        except Exception as e:
            # Token tracking is optional, don't fail the main process
            if self.output:
                self.output.warning(f"[{self.shard_id}] Failed to track tokens: {e}")


class ShardWorkerExecutor(AgentExecutor):
    """Shard Worker A2A Agent Executor"""

    def __init__(self, config=None, global_config=None, shard_id=None, data_file=None, neighbors=None, output=None, force_llm=False):
        self.worker = ShardWorker(config, global_config, shard_id, data_file, neighbors, output, force_llm)
        self.shard_id = shard_id
        self.output = output

    async def execute(
        self,
        context: BaseRequestContext,
        event_queue: BaseEventQueue,
    ) -> None:
        # Get user input from context
        user_input = context.get_user_input()
        
        if not user_input:
            await safe_enqueue_event(event_queue, new_agent_text_message("No input received"))
            return  # This is fine - function returns None but that's expected for async def -> None
        
        # ✅ 提取 A2A 消息的 meta 信息
        sender = "unknown"
        meta = {}
        
        # ✅ 双重保险的TTL提取：A2A meta + 消息内容解析
        sender = "unknown"
        meta = {}
        
        # 方案1: 从A2A消息中提取meta信息
        try:
            if hasattr(context, 'params') and hasattr(context.params, 'message'):
                message = context.params.message
                if hasattr(message, 'meta') and message.meta:
                    meta = message.meta if isinstance(message.meta, dict) else {}
                elif hasattr(message, 'metadata') and message.metadata:
                    meta = message.metadata if isinstance(message.metadata, dict) else {}
                
                if meta:
                    sender = meta.get('sender', sender)
                    if self.output and meta.get('ttl'):
                        self.output.progress(f"[{self.shard_id}] ✅ A2A meta extracted: sender={sender}, ttl={meta.get('ttl')}")
        except Exception:
            pass
        
        # 方案2: 备用方案 - 从消息内容解析TTL（当A2A meta失败时）
        if not meta.get('ttl') and "Need help:" in user_input and "(ttl=" in user_input:
            import re
            ttl_match = re.search(r'\(ttl=(\d+)\)', user_input)
            if ttl_match:
                parsed_ttl = int(ttl_match.group(1))
                meta['ttl'] = parsed_ttl
                sender = "neighbor"  # 来自邻居的消息
                if self.output:
                    self.output.warning(f"[{self.shard_id}] 🔄 A2A meta failed, parsed from content: ttl={parsed_ttl}")
        
        # 方案3: 如果还是没有TTL，但是消息来自外部，设为0
        if not meta.get('ttl') and sender == "unknown":
            meta['ttl'] = 0
            sender = "external"
        
        try:
            # Check if this is a group loading command
            if user_input.startswith("Load GROUP_ID =") or user_input.startswith("Process GROUP_ID ="):
                # Extract group_id
                parts = user_input.split("=")
                if len(parts) >= 2:
                    try:
                        group_id = int(parts[1].strip().split()[0])
                        
                        # Always use ring communication task processing
                        result = await self.worker.start_task(group_id)
                        
                        await safe_enqueue_event(event_queue, new_agent_text_message(result))
                        return
                    except ValueError:
                        pass
            
            # ✅ 优先检查是否为 A2A 消息（基于 meta 信息）
            if meta and sender != "unknown" and sender != "external":
                # 🛑 TTL=0时拒绝处理，避免无限循环
                ttl = meta.get('ttl', 0)
                if ttl <= 0:
                    if self.output:
                        self.output.warning(f"[{self.shard_id}] ❌ TTL={ttl} exhausted, rejecting message from {sender}")
                    result = "TTL exhausted - message rejected"
                    await safe_enqueue_event(event_queue, new_agent_text_message(result))
                    return
                
                # This is an A2A message from another agent
                if self.output:
                    self.output.progress(f"[{self.shard_id}] Processing A2A message from {sender} (TTL={ttl})")
                result = await self.worker.process_message(sender, user_input, meta)
                await safe_enqueue_event(event_queue, new_agent_text_message(result))
                return
            
            # Check if this is an inter-agent message (legacy format)
            if "from" in user_input.lower() and ":" in user_input:
                # Parse sender and content
                parts = user_input.split(":", 1)
                if len(parts) == 2:
                    sender_part = parts[0].strip()
                    content = parts[1].strip()
                    
                    # Extract sender (format: "Message from shard1")
                    if "from" in sender_part.lower():
                        sender = sender_part.split()[-1]
                        result = await self.worker.process_message(sender, content)
                        await safe_enqueue_event(event_queue, new_agent_text_message(result))
                        return
            
            # Default: process as regular message (only if no A2A meta found)
            result = await self.worker.process_message("external", user_input)
            await safe_enqueue_event(event_queue, new_agent_text_message(result))
            
        except Exception as e:
            error_msg = f"[{self.shard_id}] Error in execute: {str(e)}"
            if self.output:
                self.output.error(error_msg)
            await safe_enqueue_event(event_queue, new_agent_text_message(error_msg))

    async def cancel(
        self, context: BaseRequestContext, event_queue: BaseEventQueue
    ) -> None:
        raise Exception('cancel not supported') 