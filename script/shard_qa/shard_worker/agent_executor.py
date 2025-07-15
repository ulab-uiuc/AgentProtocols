import asyncio
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

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

    def __init__(self, config: dict, global_config: dict, shard_id: str, data_file: str, neighbors: dict, output=None):
        """Initialize Shard Worker."""
        self.config = config
        self.global_config = global_config
        self.shard_id = shard_id
        self.data_file = data_file
        self.neighbors = neighbors  # {prev_id, next_id}
        self.output = output
        self.agent_network = None
        
        # Agent index from shard_id (e.g., "shard3" -> 3)
        self.agent_idx = int(shard_id.replace("shard", ""))
        
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
        
    def _init_core(self):
        """Initialize Core LLM"""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
            from utils.core import Core
            
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
            
            # Initialize Core
            self.core = Core(self.config)
            
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
                
            if group_id >= len(lines):
                if self.output:
                    self.output.error(f"[{self.shard_id}] Group {group_id} not found in data file")
                return False
            
            line = lines[group_id].strip()
            if not line:
                return False
                
            data = json.loads(line)
            
            # Verify group_id matches
            if data.get('group_id') != group_id:
                if self.output:
                    self.output.warning(f"[{self.shard_id}] Group ID mismatch: expected {group_id}, got {data.get('group_id')}")
            
            self.current_group_id = group_id
            self.current_question = data.get('question', '')
            self.current_answer = data.get('answer', '')
            self.current_snippet = data.get('snippet', '')
            
            if self.output:
                self.output.success(f"[{self.shard_id}] Loaded group {group_id}: Q='{self.current_question[:50]}...'")
            
            return True
            
        except Exception as e:
            if self.output:
                self.output.error(f"[{self.shard_id}] Failed to load group {group_id}: {e}")
            return False

    def _get_system_prompt(self) -> str:
        """Get system prompt for the shard worker - v2"""
        max_ttl = self.global_config.get('tool_schema', {}).get('max_ttl', 15)
        return f"""You are {self.shard_id} in an 8-node ring topology network (shard0-shard7).

Your neighbors are:
- Previous: {self.neighbors['prev_id']}
- Next: {self.neighbors['next_id']}

YOUR QUESTION: {self.current_question}

YOUR LOCAL FRAGMENT:
{self.current_snippet}

Available tools:
1. lookup_fragment: 检查本地snippet是否包含答案，必须设置found=true/false
2. send_message: 向其他agent发送消息

PROTOCOL:
1. 总是先调用 lookup_fragment(question="{self.current_question}", found=<true/false>) 检查本地片段
2. 如果found=true，立即 send_message(destination="coordinator", content="ANSWER_FOUND: <答案>")
3. 如果found=false且当前TTL>0，向邻居求助：
   - send_message(destination="{self.neighbors['prev_id']}", content="Need help: {self.current_question}")
   - send_message(destination="{self.neighbors['next_id']}", content="Need help: {self.current_question}")
   ⚠️ 重要：如果TTL已耗尽(≤0)，不要调用send_message，直接停止

Example successful flow:
1. lookup_fragment(question="{self.current_question}", found=true)
   → found locally
   → send_message(destination="coordinator", content="ANSWER_FOUND: <答案>")

2. lookup_fragment(question="{self.current_question}", found=false)
   → not found locally
   → send_message(destination="{self.neighbors['prev_id']}", content="Need help: {self.current_question}")
   → send_message(destination="{self.neighbors['next_id']}", content="Need help: {self.current_question}")
   → 等待邻居回复

注意：TTL和path参数由系统自动管理，你只需要专注于判断found=true/false。

CRITICAL: You must analyze the LOCAL FRAGMENT above to determine if it contains the answer to YOUR QUESTION. Set found=true only if you can extract a clear answer from the fragment."""

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
            single_hop_timeout = 4.0  # 增加到4s，给HTTP请求更多时间
            timeout = max(single_hop_timeout, min(ttl * single_hop_timeout, 16.0))  # 最大16s
        
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
            self.output.progress(f"[TTL_TRACE] {self.shard_id} processing message: received_ttl={ttl} -> current_ttl={self.current_ttl}, path={self.current_path}")
        
        # Check if this is a direct reply to pending message
        if reply_to and reply_to in self.pending:
            self.pending[reply_to].set_result(content)
            return "Reply processed"
        
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
            
            if self.output:
                self.output.info(f"[{self.shard_id}] Starting ring task for group {group_id}")
                self.output.progress(f"[{self.shard_id}] My question: {self.current_question[:80]}...")
            
            # Initialize history for this group
            if group_id not in self.history:
                from collections import deque
                self.history[group_id] = deque(maxlen=self.max_history)
            
            # 设置初始任务的 TTL 和 path 上下文
            max_ttl = self.global_config.get('tool_schema', {}).get('max_ttl', 8)
            self.current_ttl = max_ttl
            self.current_path = [self.shard_id]
            
            if self.output:
                self.output.progress(f"[TTL_TRACE] {self.shard_id} starting task: initial_ttl={self.current_ttl}, path={self.current_path}")
            
            # Create initial prompt for ring search with communication
            initial_prompt = f"You have your own question to answer: '{self.current_question}'. Start by searching your local fragment with lookup_fragment. If you don't find the answer, you can ask neighbors for help using send_message."
            
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": initial_prompt}
            ]
            
            # Call Core with function calling
            if self.use_mock or self.core is None:
                # Mock response for testing - try local first, then communicate
                await asyncio.sleep(0.2)
                
                # Simple keyword matching to determine if answer is found locally
                question_lower = self.current_question.lower()
                snippet_lower = self.current_snippet.lower()
                answer_lower = self.current_answer.lower()
                
                # Check if this fragment contains the answer
                answer_words = answer_lower.split()
                found_words = sum(1 for word in answer_words if word in snippet_lower)
                confidence = found_words / len(answer_words) if answer_words else 0
                
                if confidence > 0.3:  # Found locally
                    if self.output:
                        self.output.success(f"[{self.shard_id}] Found answer locally: {self.current_answer}")
                    
                    await self._send_to_coordinator(
                        f"INDEPENDENT_ANSWER_FOUND: {self.current_answer}",
                        path=[self.shard_id],
                        ttl=0
                    )
                    return f"Independent answer found locally: {self.current_answer}"
                else:
                    # Try asking neighbors (simulate communication)
                    if self.output:
                        self.output.progress(f"[{self.shard_id}] No local answer, asking neighbors...")
                    
                    # Simulate asking neighbors and potentially finding answer
                    import random
                    if random.random() > 0.7:  # 30% chance neighbors have answer
                        simulated_answer = "neighbor_provided_answer"
                        if self.output:
                            self.output.success(f"[{self.shard_id}] Found answer from neighbor: {simulated_answer}")
                        
                        await self._send_to_coordinator(
                            f"INDEPENDENT_ANSWER_FOUND: {simulated_answer}",
                            path=[self.shard_id, "neighbor"],
                            ttl=0
                        )
                        return f"Independent answer found from neighbor: {simulated_answer}"
                    else:
                        if self.output:
                            self.output.warning(f"[{self.shard_id}] No answer found even with neighbors")
                        
                        await self._send_to_coordinator(
                            "INDEPENDENT_NO_ANSWER",
                            path=[self.shard_id],
                            ttl=0
                        )
                        return "No answer found even with communication"
            else:
                # Call Core - TTL now controlled by machine, not LLM
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
            # No function calls, return text response
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
            self.output.progress(f"[{self.shard_id}] Looking up fragment for: {question[:30]}... (ttl={ttl}, found={found})")
            # TTL跟踪日志 - 用于调试TTL递减情况
            self.output.progress(f"[TTL_TRACE] {self.shard_id} ttl={ttl} path={path} found={found} [MACHINE_CONTROLLED]")
        
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
                    found = found_words >= max(1, len(answer_words) * 0.5)  # 50% threshold
                else:
                    found = False
            else:
                found = False
        
        if found:
            if self.output:
                self.output.success(f"[{self.shard_id}] Found answer in local fragment!")
            
            # 使用实际答案
            answer_text = self.current_answer
            
            # 保护最重要的ANSWER_FOUND消息发送
            try:
                await self._send_to_coordinator(
                    f"ANSWER_FOUND: {answer_text} (hop={len(path)})",
                    path + [self.shard_id],
                    ttl
                )
            except asyncio.CancelledError:
                if self.output:
                    self.output.warning(f"[{self.shard_id}] ANSWER_FOUND sending was cancelled")
            
            self._add_to_history(self.current_group_id, f"Found answer: {answer_text}")
            return f"Found answer locally: {answer_text}"
        
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
        
        if self.output:
            self.output.progress(f"[{self.shard_id}] Not found locally, forwarding to neighbors...")
        
        # Ask both neighbors concurrently with machine-controlled TTL - v3
        next_ttl = ttl - 1
        # 双重保险：TTL信息同时放在消息内容和meta中
        ask = f"Need help: {question} (ttl={next_ttl})"
        
        if self.output:
            # TTL递减跟踪日志
            self.output.progress(f"[TTL_TRACE] {self.shard_id} forwarding to neighbors: {ttl} -> {next_ttl} [MACHINE_CONTROLLED]")
        
        # 创建任务列表，用于智能错误处理
        tasks = []
        task_names = []
        
        # 尝试发送给上一个邻居 - 使用机器控制的 TTL
        fut_prev = asyncio.create_task(
            self._send_and_wait(self.neighbors["prev_id"], ask, next_ttl, path)
        )
        tasks.append(fut_prev)
        task_names.append(f"prev({self.neighbors['prev_id']})")
        
        # 尝试发送给下一个邻居 - 使用机器控制的 TTL
        fut_next = asyncio.create_task(
            self._send_and_wait(self.neighbors["next_id"], ask, next_ttl, path)
        )
        tasks.append(fut_next)
        task_names.append(f"next({self.neighbors['next_id']})")
        
        try:
            # v2: FIRST_COMPLETED 策略，收到第一个非空答案就终止
            max_single_hop_timeout = 4.0  # 单跳4s，给HTTP请求足够时间
            total_timeout = min(16.0, max_single_hop_timeout * 4)  # 给两邻居和回路足够时间
            
            done, pending = await asyncio.wait(
                tasks, 
                return_when=asyncio.FIRST_COMPLETED,
                timeout=total_timeout
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            if done:
                # 处理第一个完成的任务
                completed_task = list(done)[0]
                reply = completed_task.result()
                
                # 找出是哪个邻居回复的
                completed_index = tasks.index(completed_task)
                neighbor_name = task_names[completed_index]
                
                if reply and reply != "No answer found in neighbours" and "ANSWER_FOUND" in reply:
                    if self.output:
                        self.output.success(f"[{self.shard_id}] Received answer from {neighbor_name}")
                    
                    # 保护邻居答案转发
                    try:
                        await self._send_to_coordinator(reply, path + [self.shard_id], ttl)
                    except asyncio.CancelledError:
                        if self.output:
                            self.output.warning(f"[{self.shard_id}] Answer forwarding was cancelled")
                    
                    self._add_to_history(self.current_group_id, f"Forwarded answer from {neighbor_name}: {reply}")
                    return "Answer forwarded to coordinator"
                else:
                    if self.output:
                        self.output.warning(f"[{self.shard_id}] No valid answer from {neighbor_name}")
                    
                    # 检查是否已被取消再发送NO_ANSWER
                    try:
                        await self._send_to_coordinator("NO_ANSWER", path + [self.shard_id], ttl)
                    except asyncio.CancelledError:
                        if self.output:
                            self.output.warning(f"[{self.shard_id}] NO_ANSWER sending was cancelled")
                    
                    self._add_to_history(self.current_group_id, f"No answer from {neighbor_name}")
                    return "No answer found in neighbors"
            else:
                if self.output:
                    self.output.warning(f"[{self.shard_id}] Timeout waiting for neighbors: {task_names}")
                
                # 检查是否已被取消再发送NEIGHBOR_TIMEOUT
                try:
                    await self._send_to_coordinator("NEIGHBOR_TIMEOUT", path + [self.shard_id], ttl)
                except asyncio.CancelledError:
                    if self.output:
                        self.output.warning(f"[{self.shard_id}] NEIGHBOR_TIMEOUT sending was cancelled")
                
                return "Neighbor timeout"
                
        except Exception as e:
            if self.output:
                self.output.error(f"[{self.shard_id}] Error during neighbor search: {e}")
            
            # 检查当前任务是否已被取消，避免连环CancelledError
            try:
                current_task = asyncio.current_task()
                if current_task and current_task.cancelled():
                    if self.output:
                        self.output.warning(f"[{self.shard_id}] Task already cancelled, skipping error report")
                    return f"Search cancelled: {str(e)}"
                
                await self._send_to_coordinator("SEARCH_ERROR", path + [self.shard_id], ttl)
            except asyncio.CancelledError:
                if self.output:
                    self.output.warning(f"[{self.shard_id}] Error reporting was cancelled")
                return f"Search cancelled: {str(e)}"
            
            return f"Search error: {str(e)}"

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

    def __init__(self, config=None, global_config=None, shard_id=None, data_file=None, neighbors=None, output=None):
        self.worker = ShardWorker(config, global_config, shard_id, data_file, neighbors, output)
        self.shard_id = shard_id
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
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception('cancel not supported') 