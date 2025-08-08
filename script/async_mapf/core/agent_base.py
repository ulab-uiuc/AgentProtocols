"""
BaseAgent implementation for MAPF agents.

Provides:
① LLM context building: Base Prompt + Info + Chat History + Tool Schema
② Tool invocation handling: move() & send_msg() tools
③ Memory management: conversation tracking with other agents
④ Communication interface: adapter abstraction for network interaction

Design Note:
This is a foundation for both local and distributed agent implementations,
with protocol-agnostic communication through adapters.
"""

import asyncio
import collections
import json
import time
import uuid
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field

from .comm import AbstractCommAdapter
from .types import MoveCmd, MoveFeedback, Coord
from .concurrent_types import ConcurrentMoveCmd, MoveResponse, MoveStatus

# Import logging utilities
try:
    from ..utils.log_utils import get_log_manager, setup_colored_logging, log_agent_action
    setup_colored_logging()  # 初始化彩色日志
except ImportError:
    # Fallback if log_utils not available
    get_log_manager = lambda: None
    log_agent_action = lambda *args: None

# Import Core with error handling to avoid circular imports
try:
    import sys
    import os
    # Add project root to Python path if not already there
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.utils.core import Core
except ImportError as e:
    # Fallback when running in isolation
    try:
        # Direct file import as fallback
        import importlib.util
        core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src/utils/core.py'))
        if os.path.exists(core_path):
            spec = importlib.util.spec_from_file_location("core", core_path)
            core_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(core_module)
            Core = core_module.Core
        else:
            Core = None
    except Exception as fallback_error:
        Core = None


class AgentState(Enum):
    """Agent状态机状态"""
    IDLE = "idle"
    PLANNING = "planning"
    WAITING_MOVE_RESP = "waiting_move_resp"
    CONFLICT_RETRY = "conflict_retry"
    WAITING_MSG = "waiting_msg"


@dataclass
class PositionInfo:
    """位置信息缓存"""
    pos: Coord
    timestamp: float
    ttl_sec: float = 1.0
    
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl_sec


@dataclass
class PendingMessage:
    """待回复的消息"""
    msg_id: str
    target_agent: int
    content: str
    sent_time: float
    timeout_sec: float = 8.0
    
    def is_timeout(self) -> bool:
        return time.time() - self.sent_time > self.timeout_sec


# Base tool schemas for LLM interaction (keeping original format)
MOVE_SCHEMA = {
    "name": "move",
    "description": "Move the robot on the grid.",
    "parameters": {
        "type": "object",
        "properties": {
            "mode":  {"type": "string", "enum": ["immediate", "scheduled"]},
            "dir":   {"type": "string", "enum": ["U","D","L","R","S"]},
            "exec_ts": {"type": "integer", "description": "optional timestamp"}
        },
        "required": ["mode", "dir"]
    }
}

COMM_SCHEMA = {
    "name": "send_msg",
    "description": "Send a P2P text message to another agent (async, fire-and-forget).",
    "parameters": {
        "type": "object",
        "properties": {
            "dst":  {"type": "integer", "description": "Target agent ID"},
            "msg":  {"type": "string", "description": "Message content"}
        },
        "required": ["dst", "msg"]
    }
}

TOOLS = [MOVE_SCHEMA, COMM_SCHEMA]

# Base prompt for agent instruction
BASE_PROMPT = """
You are a robot agent on a 19×19 grid. Your tasks:

1. Reach your goal while avoiding collisions.
2. Communicate only via the provided tools.
3. Never hallucinate positions you do not know.

Always reply in JSON with a single tool call.
"""


class BaseAgent:
    """
    Base agent implementation for MAPF scenarios.
    
    Integrates LLM planning, communication, and movement capabilities.
    Uses an adapter to interact with the network layer.
    """

    # === Tuning Parameters ===
    FAILURE_FALLBACK = 6          # 连续 REJECT/CONFLICT 次数触发 BFS
    BFS_EMERGENCY_THRESHOLD = 40.0  # 振荡惩罚 ≥ 40 立刻触发 BFS (下调)
    OSC_BAN_THRESHOLD = 3         # 连续多少次 A-B-A-B 才封格子 (提高门槛)
    OSC_TEMP_BAN_SECONDS = 60     # 临时封禁时长 (秒)
    OSC_PERMA_BAN_COUNT = 2       # 同一格被封多少次才永久
    MAX_BFS_EXPAND = 400          # BFS 最大展开节点数 (安全阈)


    def __init__(self, agent_id: int, adapter: AbstractCommAdapter,
             config: Dict[str, Any] = None):
        """
        Initialize BaseAgent with core components and autonomous capabilities.
        
        Args:
            agent_id: Unique identifier for this agent
            adapter: Communication adapter for network interaction
            config: Optional configuration parameters, including model config
        """
        # Agent identity
        self.id = agent_id
        self.agent_id = agent_id  # Alias for compatibility
        self.start_pos = None  # Will be set by registration
        self.goal_pos = None   # Will be set by configuration
        self.current_pos = None  # Current position
        
        # Communication components
        self.adapter = adapter
        self.tools = TOOLS
        
        # Enhanced state management for autonomous operation
        self.state = AgentState.IDLE
        self.is_running = False
        self.main_task = None  # Track the main loop task to prevent multiple loops
        self._wait_reply_event = asyncio.Event()  # 🔧 等待MOVE_RESPONSE的事件
        
        # Configuration parameters
        self.planning_interval_ms = random.randint(500, 1500)
        self.base_eta_ms = 200
        self.time_window_ms = 80
        self.max_retries = 3
        self.ttl_sec = 1.0
        self.backoff_factor = 1.5
        
        # Enhanced internal state
        self.coord_cache: Dict[int, PositionInfo] = {}  # agent_id -> 位置信息
        self.pending_moves: Dict[str, ConcurrentMoveCmd] = {}  # move_id -> 移动请求
        self.pending_messages: Dict[str, PendingMessage] = {}  # msg_id -> 待回复消息
        self.retry_count = 0
        self.target_pos: Optional[Coord] = None
        
        # === 新增振荡控制变量 ===
        self._aba_counter = 0               # 连续A-B-A-B振荡计数
        self.temp_ban: Dict[Tuple[int, int], float] = {}  # pos -> expiry_ts 临时封禁
        self.perma_ban: Set[Tuple[int, int]] = set()      # 永久封禁格子
        self._temp_ban_counts: Dict[Tuple[int, int], int] = {}  # pos -> 封禁次数
        
        # 🔧 CRITICAL FIX: Add move reply timeout for fail-safe
        self.move_reply_deadline: float = 0.0  # Absolute timestamp in ms
        
        # 🔧 CRITICAL FIX: Add temporary avoidance for coordination
        self.avoidance_zones: Dict[Tuple[int, int], float] = {}  # position -> expire_time
        
        # 🔧 CRITICAL FIX: Add conflict resolution strategy
        self.coordination_attempts: Dict[int, int] = {}  # agent_id -> attempt_count
        self.last_coordination_time: Dict[int, float] = {}  # agent_id -> last_attempt_time
        
        # 🔧 NEW: Enhanced retry and conflict tracking  
        self.consecutive_failures = 0  # Track consecutive REJECT/CONFLICT count
        self.last_failure_time = 0.0  # Time of last failure
        
        # 🔧 NEW: Reply tracking for LLM-generated responses
        self._current_reply_to = None  # Track msg_id when replying
        
        # 🆕 NEW: IDLE state management
        self.idle_needs_coordination = False  # Flag to indicate IDLE agent needs to move for coordination
        
        # 🆕 NEW: Movement history tracking for anti-oscillation
        self.movement_history: List[Tuple[int, int, float]] = []  # (x, y, timestamp)
        self.max_history_length = 10  # Keep last 10 positions
        self.oscillation_penalty = 0.0  # Penalty for repeated back-and-forth movement
        self.oscillation_threshold = 3  # Number of times visiting same area to trigger penalty
        self.oscillation_decay = 0.95   # Decay factor for penalty over time
        
        # 🚀 NEW: Permanent position banning for severe oscillation
        self.banned_positions: Set[Tuple[int, int]] = set()  # Permanently banned positions
        self.cooling_positions: Dict[Tuple[int, int], float] = {}  # Position -> cooldown_end_time
        self.step_count = 0  # Track total steps for periodic replanning
        self.last_replan_step = 0  # Last step when global replanning was triggered
        self._force_replan = False  # Flag to force emergency replanning
        
        # Memory management (backwards compatible)
        # Format: {agent_id: [(role, content, timestamp), ...]}
        # where role is 'you' or 'other'
        self.chat_memory = collections.defaultdict(list) 
        
        # Synchronous messaging state (legacy support)
        self.blocked = False       # If waiting for any sync reply
        self.waiting_for = set()   # Set of agent IDs we're waiting for
        self.max_parallel_waits = 1  # Limit concurrent waits
        
        # 🔥 NEW: Track who we still owe an ACK to (prevents deadlock)
        self.pending_acks: Dict[int, str] = {}  # agent_id -> msg_id
        
        # 🚀 NEW: Conversation tracking to prevent message storms
        self.conv_seq: Dict[Tuple[int, int], int] = collections.defaultdict(int)  # (src,dst) -> sequence number
        self.last_sync_attempt: Dict[int, float] = {}  # agent_id -> last sync attempt time
        self.sync_retry_count: Dict[int, int] = collections.defaultdict(int)  # agent_id -> retry count
        
        # 🚀 NEW: Conflict resolution and priority system
        self.conflict_counter: Dict[int, int] = collections.defaultdict(int)  # other_agent_id -> conflict count
        self.last_conflict_time: Dict[int, float] = {}  # agent_id -> last conflict time
        self.yielding_to: Set[int] = set()  # Set of agents we're currently yielding to
        
        # Movement tracking (legacy support)
        self.collisions = 0
        self.next_move = None  # (timestamp, direction) of next scheduled move
        
        # Timestamps
        self.start_time = time.time() * 1000  # Convert to ms
        self.now = 0
        
        # Configuration and LLM Core initialization
        self.config = config or {}
        if Core is not None and self.config.get("model"):
            try:
                self.llm = Core(config=self.config)
                # Note: logger not yet initialized, will log this later
            except Exception as e:
                # Note: logger not yet initialized, will log this later
                self.llm = None
                self._llm_init_error = str(e)
        else:
            self.llm = None
        
        # Initialize logging system with debug configuration
        debug_config = self.config.get("debug", {})
        self.log_manager = get_log_manager(debug_config)
        # 获取agent的protocol信息，如果没有就使用默认值
        protocol = self.config.get("protocol", "local")
        self.verbose = debug_config.get("verbose", False)
        
        if self.log_manager:
            self.logger = self.log_manager.get_agent_logger(self.agent_id, protocol)
        else:
            # Fallback logger if log manager not available
            import logging
            self.logger = logging.getLogger(f"Agent-{self.agent_id}")
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # 从配置中提取位置信息
        if self.config.get("agent_config"):
            agent_cfg = self.config["agent_config"]
            self.start_pos = tuple(agent_cfg.get("start", (0, 0)))
            self.goal_pos = tuple(agent_cfg.get("goal", (0, 0)))
            self.current_pos = self.start_pos
        
        # Extract grid size if available
        self.grid_size = self.config.get("grid_size", (19, 19))
        
        # 🔧 Debug: Log the agent's configuration (now that logger is ready)
        if hasattr(self, 'start_pos') and hasattr(self, 'goal_pos'):
            self.logger.info(f"🎯 Agent {self.agent_id} configured: start={self.start_pos}, goal={self.goal_pos}, current={self.current_pos}")
        
        # Log LLM initialization status (now that logger is ready)
        if self.llm is not None:
            self.logger.info(f"✅ Agent {self.agent_id} LLM initialized successfully")
        elif hasattr(self, '_llm_init_error'):
            self.logger.error(f"❌ Agent {self.agent_id} LLM initialization failed: {self._llm_init_error}")
        else:
            self.logger.info(f"ℹ️ Agent {self.agent_id} LLM not configured")
    
    def _get_logical_time(self) -> int:
        """Get current logical time in milliseconds."""
        if self.now > 0:
            return self.now
        
        # Fall back to wall time if logical time not set
        return int(time.time() * 1000) - int(self.start_time)
    
    def update_state(self, feedback: MoveFeedback) -> None:
        """
        Update agent state based on movement feedback.
        
        Args:
            feedback: Movement feedback from network
        """
        # Record previous position before updating
        old_pos = self.current_pos
        self.current_pos = feedback.actual_pos
        self.now = feedback.step_ts
        
        # 🆕 Update movement history and check for oscillation
        if self.current_pos and self.current_pos != old_pos:
            self._record_movement(self.current_pos)
        
        if feedback.collision:
            self.collisions += 1
            
        # Clear next_move if this was the move we were waiting for
        if self.next_move and self.next_move[0] <= self.now:
            self.next_move = None
    
    def _build_info_block(self) -> str:
        """Build INFO block for LLM context."""
        lines = [
            "[INFO]",
            f"➤ id: {self.id}",
        ]
        
        if self.start_pos:
            lines.append(f"➤ start: {self.start_pos}")
            
        if self.goal_pos:
            lines.append(f"➤ goal : {self.goal_pos}")
            
        lines.append(f"➤ now  : {self._get_logical_time()} ms")
        
        if self.current_pos:
            lines.append(f"➤ pos  : {self.current_pos}")
            
        if self.next_move:
            ts, direction = self.next_move
            lines.append(f"➤ next : at {ts} ms → {direction}")
            
        lines.append(f"➤ collisions: {self.collisions}")
        
        return "\n".join(lines)
    
    def _build_chat_history(self) -> str:
        """
        Build formatted chat history string for LLM context.

        chat_memory item structure is either:
            (role, content, ts)
            (role, content, ts, msg_id)      # since sync-msg support
        We slice the first three fields to stay backward compatible.
        """
        blocks = []
        
        # Get all agent IDs we've communicated with
        agent_ids = sorted(self.chat_memory.keys())
        
        for aid in agent_ids:
            chat = self.chat_memory[aid]
            
            if not chat:
                blocks.append(f"[CHAT with Agent-{aid}]\nEMPTY")
                continue
                
            lines = [f"[CHAT with Agent-{aid}]"]
            
            for entry in chat:
                role, content, ts = entry[:3]  # ignore extra fields safely
                
                if role == "you":
                    if isinstance(content, dict) and content.get("status") == "PENDING":
                        lines.append(f"You ➜ {content['msg']}        (ts-{ts})")
                        lines.append(f"Agent-{aid} ➜ (PENDING)                (waiting)")
                    else:
                        lines.append(f"You ➜ {content}        (ts-{ts})")
                elif role == "ack":
                    lines.append(f"You ➜ [ACK] {content}        (ts-{ts})")
                else:  # role == "other"
                    lines.append(f"Agent-{aid} ➜ {content}              (ts-{ts})")
            
            blocks.append("\n".join(lines))
        
        return "\n\n".join(blocks)
    
    def _build_tools_description(self) -> str:
        """Build tool description block for LLM context."""
        return """
Available tools:
1. move: Change position on grid
   - immediate mode: Execute now
   - scheduled mode: Execute at timestamp

2. send_msg: Message other agents
   - wait=true: Block until reply
   - wait=false: Continue immediately
"""
    
    def _build_context(self) -> List[Dict[str, str]]:
        """
        Build complete context for LLM call.
        
        Returns:
            List of message dictionaries for LLM API
        """
        # Core info blocks
        info_block = self._build_info_block()
        chat_history = self._build_chat_history()
        tools_description = self._build_tools_description()
        
        # Build messages array
        messages = [
            {"role": "system", "content": BASE_PROMPT},
            {"role": "system", "content": info_block},
            {"role": "system", "content": tools_description},
        ]
        
        # Add chat history if not empty
        if chat_history.strip():
            messages.append({"role": "system", "content": chat_history})
        
        # Add empty user message to trigger LLM response
        messages.append({"role": "user", "content": ""})
        
        return messages
    
    async def _handle_tool(self, response: Dict[str, Any]) -> None:
        """
        Handle tool call from LLM response.
        
        Args:
            response: Parsed response with tool call
        """
        # Extract tool call from response
        try:
            name, arguments = self._extract_tool_call(response)
        except Exception as e:
            self.logger.error(f"Error parsing tool call: {e}")
            return
        
        # Handle different tool types
        if name == "move":
            await self._handle_move_tool(arguments)
        elif name == "send_msg":
            await self._handle_send_msg_tool(arguments)
        else:
            self.logger.warning(f"Unknown tool: {name}")
    
    async def _handle_move_tool(self, args: Dict[str, Any]) -> None:
        """
        Handle move tool call with enhanced autonomous capabilities.
        
        Args:
            args: Tool arguments
        """
        protocol = self.config.get("protocol", "local")

        mode = args.get("mode")
        direction = args.get("dir")
        exec_ts = args.get("exec_ts")
        
        # Validate required arguments
        if not mode or not direction:
            self.logger.warning("Invalid move command: missing required arguments")
            return
        
        # Validate direction
        if direction not in "UDLRS":
            self.logger.error(f"🚨 INVALID DIRECTION: '{direction}' (type: {type(direction)}), must be one of U/D/L/R/S")
            self.logger.error(f"🚨 LLM response args: {args}")
            return
        
        # 🔧 CRITICAL FIX: Prohibit staying in place to prevent infinite loops
        if direction == "S":
            self.logger.info("LLM proposed Stay, ignoring. Agent must move toward goal or use send_msg to coordinate.")
            return
        
        # Calculate target position based on direction using effective position
        effective_pos = self._effective_pos()
        target_pos = self._calculate_target_position(effective_pos, direction)
        
        # 🔧 CRITICAL DEBUG: Log movement calculation
        self.logger.info(f"🎯 LLM chose direction '{direction}': {effective_pos} → {target_pos}")
        
        # 🔧 CRITICAL FIX: Check for same-position movement (original cause of infinite loops)
        if target_pos == effective_pos:
            self.logger.warning(f"⚠️ LLM requested move to same position {target_pos}. This indicates direction='{direction}' or calculation error.")
            return
        
        # 🔧 CRITICAL FIX: 用pending_moves去重，防止重复MOVE_REQUEST
        if any(m.new_pos == target_pos for m in self.pending_moves.values()):
            self.logger.info(f"⏩ 已有同位置的 pending move to {target_pos}，忽略本次请求")
            return
        
        # 🔧 CRITICAL FIX: Add local bounds checking to prevent repeated invalid requests
        if not (0 <= target_pos[0] < self.grid_size[0] and 0 <= target_pos[1] < self.grid_size[1]):
            error_msg = f"ERROR: Direction '{direction}' leads to out-of-bounds position {target_pos}. VALID DIRECTIONS: {self._valid_dirs()}. Choose ONLY from valid directions!"
            self.logger.warning(f"⚠️ Invalid target {target_pos} from direction {direction} - out of bounds. Skipping move.")
            
            # 🎯 CRITICAL FIX: Write bounds error to chat_memory so LLM can learn
            self._add_to_chat_memory(self.agent_id, "system", error_msg)
            
            return  # Don't force "S", just skip this move entirely
        
        # 🔧 CRITICAL FIX: Check for temporary avoidance zones
        if self._is_position_avoided(target_pos):
            avoidance_msg = f"COORDINATION: Position {target_pos} is temporarily avoided due to coordination with other agents. Choose a different direction."
            self.logger.info(f"🚫 Avoiding coordinated position {target_pos}, skipping move")
            self._add_to_chat_memory(self.agent_id, "system", avoidance_msg)
            return
        
        # For autonomous mode, create enhanced move command if supported
        if hasattr(self, 'state') and self.state != AgentState.IDLE:
            # Create concurrent move command for autonomous operation
            move_cmd = ConcurrentMoveCmd(
                agent_id=self.agent_id,
                new_pos=target_pos,
                eta_ms=self.base_eta_ms + random.randint(-20, 20),
                time_window_ms=self.time_window_ms,
                move_id=str(uuid.uuid4()),
                priority=3
            )
            await self._send_move_request(move_cmd)
        else:
            # Legacy mode - create standard move command
            cmd = MoveCmd(
                agent_id=self.id,
                action=direction,
                client_ts=self._get_logical_time(),
                exec_ts=exec_ts if mode == "scheduled" else None
            )
            
            # Store next move info
            if mode == "scheduled" and exec_ts:
                self.next_move = (exec_ts, direction)
                
            # Send to network
            await self.adapter.send(cmd)
            # Log agent move command
            if self.log_manager and hasattr(self.log_manager, 'log_agent_action'):
                self.log_manager.log_agent_action(self.agent_id, protocol, "MOVE_CMD", cmd.__dict__)
    
    def _calculate_target_position(self, current_pos: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """Calculate target position based on current position and direction."""
        x, y = current_pos
        
        if direction == "U":
            return (x, y - 1)
        elif direction == "D":
            return (x, y + 1)
        elif direction == "L":
            return (x - 1, y)
        elif direction == "R":
            return (x + 1, y)
        elif direction == "S":
            return (x, y)  # Stay in place
        else:
            # This should never happen if direction validation works properly
            self.logger.error(f"🚨 CRITICAL: _calculate_target_position received invalid direction '{direction}'")
            raise ValueError(f"Invalid direction: {direction}")
    
    async def _handle_send_msg_tool(self, args: Dict[str, Any]) -> None:
        """
        Handle send_msg tool call.
        
        Args:
            args: Tool arguments
        """
        dst = args.get("dst")
        msg = args.get("msg")
        wait = False
        
        if dst is None or msg is None:
            self.logger.warning("Invalid send_msg command: missing required arguments")
            return
        
        # 🔥 NEW: If we owe this agent an ACK, merge it into current message
        owed_id = self.pending_acks.pop(dst, None)
        reply_to_field = None
        if owed_id:
            reply_to_field = owed_id
            wait = False  # Reply cannot lock ourselves
            self.logger.info(f"🔓 Merging ACK (reply_to={owed_id[:8]}...) into LLM message → Agent {dst}")
            self._unblock(dst)  # 立刻解锁，保持状态一致
        
        # 🔥 FIX: Prevent second wait=true message to same agent (deadlock prevention)
        if dst in self.waiting_for and wait:
            self.logger.warning(f"🚨 Already waiting ACK from Agent {dst}, forcing async to prevent deadlock")
            wait = False
            self._unblock(dst)  # 放弃等待改异步，保持状态一致
        
        # 🚀 NEW: Conversation tracking and message storm prevention
        conv_key = (self.agent_id, dst)
        current_time = time.time()
        
        # Update conversation sequence
        self.conv_seq[conv_key] += 1
        current_seq = self.conv_seq[conv_key]
        
        # Check for message storm patterns
        if wait:  # Only track sync message attempts
            if dst in self.last_sync_attempt:
                time_since_last = current_time - self.last_sync_attempt[dst]
                if time_since_last < 5.0:  # Less than 5 seconds since last sync attempt
                    self.sync_retry_count[dst] += 1
                    if self.sync_retry_count[dst] >= 3:
                        # After 3 rapid sync attempts, force async mode
                        self.logger.warning(f"🌪️ MESSAGE STORM detected with Agent {dst} ({self.sync_retry_count[dst]} sync attempts in {time_since_last:.1f}s), forcing async mode")
                        wait = False
                        msg += f" [Auto-async: too many sync retries]"
                else:
                    self.sync_retry_count[dst] = 1  # Reset if enough time passed
            else:
                self.sync_retry_count[dst] = 1
            
            self.last_sync_attempt[dst] = current_time
        
        # Current timestamp
        now = self._get_logical_time()
        
        # Create message packet - unified CHAT format
        packet = {
            "type": "CHAT",
            "payload": {
                "src": self.id,
                "dst": dst,
                "msg": msg,
                "ts": now,
                "conv_seq": current_seq  # 🚀 Add conversation sequence number
            }
        }
        
        # 🔧 Add reply_to field (priority: owed ACK > _current_reply_to)
        if reply_to_field:
            packet["payload"]["reply_to"] = reply_to_field
        elif hasattr(self, '_current_reply_to') and self._current_reply_to:
            packet["payload"]["reply_to"] = self._current_reply_to
            self.logger.info(f"📩 Adding reply_to={self._current_reply_to[:8]}... to unlock sender")
        
        # 🔧 Let LLM decide wait behavior completely, no auto-detection
            current_time = time.time()
            
            # Track coordination attempts
            if dst not in self.coordination_attempts:
                self.coordination_attempts[dst] = 0
            
            # Check if we've been coordinating too frequently
            if dst in self.last_coordination_time:
                time_since_last = current_time - self.last_coordination_time[dst]
                if time_since_last < 10.0:  # Less than 10 seconds since last attempt
                    self.coordination_attempts[dst] += 1
                else:
                    self.coordination_attempts[dst] = 1  # Reset if enough time passed
            else:
                self.coordination_attempts[dst] = 1
            
            self.last_coordination_time[dst] = current_time
            
            # Apply conflict resolution strategy
            attempt_count = self.coordination_attempts[dst]
            should_yield = self._should_yield_to_agent(dst, attempt_count)
            
            # 🔧 CRITICAL FIX: Hard limit on coordination attempts to prevent infinite loops
            # But allow LLM-generated replies to use wait=true for proper conversation flow
            if attempt_count > 20 and not msg.startswith("ACK:"):  # Increased threshold and exclude ACKs
                self.logger.warning(f"🚨 COORDINATION OVERLOAD: {attempt_count} attempts with agent {dst}, FORCING FALLBACK to wait=false")
                wait = False  # Force async mode to break the cycle
                msg += f" [FALLBACK: Too many coordination attempts ({attempt_count}), switching to async mode]"
            elif should_yield and attempt_count > 2:
                # After 2+ attempts, consider yielding by finding alternative path
                self.logger.info(f"🔄 High coordination attempts ({attempt_count}) with agent {dst}, considering yielding strategy")
                
                # Add extended avoidance to force finding alternative paths
                if self.current_pos:
                    # Avoid areas around the conflict zone
                    conflict_area = self._get_conflict_area_with_agent(dst)
                    for pos in conflict_area:
                        self._add_temporary_avoidance(pos, duration_sec=15.0)
                    
                    msg += f" [After {attempt_count} attempts, I'll find an alternative path to avoid this area]"
                    self.logger.info(f"🛤️ Adding alternative path strategy due to repeated conflicts with agent {dst}")
        
        # Update memory
        if wait:
            # Check if we're already at max parallel waits
            # Allow replies to bypass this restriction to maintain conversation flow
            if len(self.waiting_for) >= self.max_parallel_waits and not msg.startswith("ACK:"):
                self.logger.warning(f"Already waiting for {self.waiting_for}, "
                                   f"can't wait for {dst} simultaneously (but allowing replies)")
                # Fall back to async mode for non-reply messages
                wait = False
            else:
                # 🔧 CRITICAL FIX: Add timeout mechanism for synchronous messages
                msg_id = str(uuid.uuid4())
                pend = PendingMessage(msg_id, dst, msg, time.time(), timeout_sec=8.0)  # 3秒超时
                self.pending_messages[msg_id] = pend
                
                # For synchronous messages, mark as pending
                self.chat_memory[dst].append(("you", {"msg": msg, "status": "PENDING"}, now, msg_id))
                self.blocked = True
                self.waiting_for.add(dst)
                
                # 🔧 CRITICAL FIX: Set proper state for sync message waiting
                self.state = AgentState.WAITING_MSG
                
                # Add to packet for tracking
                packet["payload"]["msg_id"] = msg_id
                packet["payload"]["need_ack"] = True  # 请求对方回复确认
                
                self.logger.info(f"📤 Sending sync message to agent {dst}, msg_id={msg_id[:8]}..., entering WAITING_MSG state")
        else:
            # For async messages, just record the message
            self.chat_memory[dst].append(("you", msg, now, None))
        
        # 🆕 Enhanced message logging before sending
        is_reply = bool(packet["payload"].get("reply_to"))
        wait_status = "sync" if wait else "async"
        msg_preview = msg[:60] + "..." if len(msg) > 60 else msg
        
        if is_reply:
            self.logger.info(f"📩 Agent {self.agent_id} → Agent {dst} ({wait_status} REPLY): '{msg_preview}'")
        else:
            self.logger.info(f"📤 Agent {self.agent_id} → Agent {dst} ({wait_status}): '{msg_preview}'")
        
        # Send message
        await self.adapter.send(packet)
        # Log agent message
        if self.log_manager and hasattr(self.log_manager, 'log_agent_action'):
            protocol = self.config.get("protocol", "local")
            self.log_manager.log_agent_action(self.agent_id, protocol, "SEND_MSG", packet)
    
    async def _recv_msgs(self) -> None:
        """Receive and process incoming messages."""
        try:
            while True:
                try:
                    # Use non-blocking receive
                    pkt = self.adapter.recv_nowait()
                    
                    # Process message based on type
                    if isinstance(pkt, MoveFeedback):
                        # Movement feedback (legacy support)
                        self.update_state(pkt)
                    elif isinstance(pkt, dict):
                        # Handle new message formats first
                        msg_type = pkt.get("type")
                        
                        if msg_type in ["MOVE_RESPONSE", "CHAT", "POS_REQUEST", "POS_REPLY"]:
                            # New autonomous agent message types
                            await self.handle_network_packet(pkt)
                        elif msg_type == "CONTROL":
                            # Control signal from NetworkBase
                            cmd = pkt.get("cmd", "unknown")
                            if cmd == "START":
                                self.logger.info(f"Received START signal from NetworkBase")
                                self.blocked = False  # Ensure we can start planning
                            elif cmd == "STEP":
                                # Step signal from NetworkBase - trigger one planning step
                                if not self.blocked and self.llm:
                                    self.logger.info(f"Received STEP signal, executing planning...")
                                    await self._execute_planning_step()
                                else:
                                    self.logger.info(f"Received STEP but blocked={self.blocked}, llm={self.llm is not None}")
                            else:
                                self.logger.info(f"Received unknown control command: {cmd}")
                        # Legacy msg format removed - all messages now use unified CHAT format
                        elif msg_type == "msg":
                            # DEPRECATED: Legacy format, redirect to CHAT handler
                            self.logger.warning(f"⚠️ Received legacy 'msg' format, converting to CHAT")
                            # Convert legacy format to new CHAT format
                            converted_pkt = {
                                "type": "CHAT",
                                "payload": {
                                    "src": pkt.get("src"),
                                    "dst": pkt.get("dst"),
                                    "msg": pkt.get("msg"),
                                    "ts": pkt.get("ts", self._get_logical_time()),
                                    "msg_id": pkt.get("msg_id"),
                                    "need_ack": pkt.get("need_ack", False),
                                    "reply_to": pkt.get("reply_to")
                                }
                            }
                            await self.handle_network_packet(converted_pkt)
                            
                            # Record in memory with msg_id for reply tracking
                            self._add_to_chat_memory(src, "other", msg, msg_id)
                            self.logger.info(f"📝 Recorded message from Agent {src} to chat_memory: '{msg[:50]}...' (need_ack={need_ack}, msg_id={msg_id[:8] if msg_id else None})")
                            
                            # 🔧 All messages (including need_ack) will be handled by LLM in planning phase
                            # No immediate auto-reply - let unified reply mechanism handle it
                            
                            # 🔥 CRITICAL FIX: 恢复立即ACK机制，解除"互相等待"死锁
                            if need_ack and msg_id:
                                # 使用统一的立即ACK方法
                                await self._send_immediate_ack(src, msg_id)
                                self.logger.debug(f"⚡ Sent immediate ACK for legacy msg {msg_id[:8]}..., LLM will handle detailed reply later")
                            
                            # 🔧 CRITICAL FIX: Handle ACK replies properly
                            reply_to = pkt.get("reply_to")
                            if reply_to and reply_to in self.pending_messages:
                                # This is an ACK reply to one of our synchronous messages
                                pending_msg = self.pending_messages[reply_to]
                                target_agent = pending_msg.target_agent
                                
                                self.logger.info(f"✅ Received ACK from agent {src} for message {reply_to[:8]}...")
                                
                                # Clean up the pending message
                                del self.pending_messages[reply_to]
                                if target_agent in self.waiting_for:
                                    self.waiting_for.remove(target_agent)
                                
                                # If no more pending messages, unblock
                                if not self.waiting_for:
                                    self.blocked = False
                                    if self.state == AgentState.WAITING_MSG:
                                        self.state = AgentState.PLANNING
                                        self.logger.info(f"✅ All sync messages acknowledged, returning to PLANNING")
                            
                            # Legacy: If we were blocked waiting for this agent, unblock
                            elif src in self.waiting_for:
                                self.waiting_for.remove(src)
                                # Clear corresponding pending message
                                if msg_id:
                                    self.pending_messages.pop(msg_id, None)
                                if not self.waiting_for:
                                    self.blocked = False
                                    if self.state == AgentState.WAITING_MSG:
                                        self.state = AgentState.PLANNING
                                        self.logger.info(f"✅ Received expected reply from agent {src}, returning to PLANNING")
                        else:
                            self.logger.debug(f"Unhandled message type: {msg_type}")
                    else:
                        self.logger.warning(f"Unknown message type: {type(pkt)}")
                except Exception as e:
                    # Handle both asyncio.QueueEmpty and queue.Empty
                    if "Empty" in str(type(e).__name__):
                        # This is a queue empty exception, break the inner loop
                        break
                    else:
                        # Log unexpected exceptions but don't crash the loop
                        self.logger.warning(f"Unexpected exception in _recv_msgs: {e}")
                        break  # Exit inner loop to prevent infinite exceptions
        except Exception as e:
            self.logger.error(f"Error receiving messages: {e}")

    async def _execute_planning_step(self) -> None:
        """执行一步规划"""
        try:
            # 1. Build context for LLM
            msgs = self._build_context()
            
            # 记录LLM输入
            if self.log_manager:
                protocol = self.config.get("protocol", "local")
                self.log_manager.log_llm_interaction(
                    self.agent_id, 
                    protocol, 
                    {"messages": msgs, "functions": self.tools}, 
                    None  # 输出稍后记录
                )
            
            # 2. Call LLM with timeout
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.llm.function_call_execute,
                        messages=msgs,
                        functions=self.tools
                    ),
                    timeout=30.0  # 30秒超时
                )
            except asyncio.TimeoutError:
                self.logger.error(f"LLM call timed out after 30 seconds for Agent {self.agent_id}")
                return
            except Exception as e:
                self.logger.error(f"LLM call failed for Agent {self.agent_id}: {e}")
                return
            
            # 记录LLM输出
            if self.log_manager:
                protocol = self.config.get("protocol", "local")
                self.log_manager.log_llm_interaction(
                    self.agent_id, 
                    protocol, 
                    None,  # 输入已记录
                    response
                )
            
            # 3. Handle tool call
            await self._handle_tool(response)
            
        except Exception as e:
            self.logger.error(f"Planning step failed: {e}")
            # Reset state to allow next planning attempt
            self.state = AgentState.PLANNING
            # Add a small delay to prevent immediate retry loops  
            await asyncio.sleep(0.5)
    
    async def autonomous_loop(self) -> None:
        """
        主要的自驱动循环，基于状态机运行
        """
        # 🔧 防止多个循环同时运行
        if self.main_task and not self.main_task.done():
            self.logger.warning(f"⚠️ Autonomous loop already running for Agent {self.agent_id}")
            return
        
        # 🔧 强制检查：确保没有其他循环在运行
        if self.is_running:
            self.logger.error(f"❌ Agent {self.agent_id} is_running=True but main_task is done. Possible legacy loop conflict!")
            return
        
        # 🔧 调试标识：确认只有一条循环启动
        self.logger.debug(f"♻️ Autonomous loop started for Agent {self.agent_id}, id={id(self)}, task={id(asyncio.current_task())}")
        self.logger.info(f"🚀 Starting autonomous loop for Agent {self.agent_id}")
        
        self.is_running = True
        self.state = AgentState.PLANNING
        self.main_task = asyncio.current_task()
        
        # 随机初始延迟，避免同步
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        tick_count = 0
        while self.is_running:
            try:
                tick_count += 1
                # Debug: 确认循环继续执行
                if tick_count % 10 == 1:  # 每10次循环打印一次
                    self.logger.debug(f"🔄 Agent {self.agent_id} autonomous loop tick {tick_count}, state={self.state.value}")
                
                # 🔧 CRITICAL DEBUG: Log every 20 ticks to catch hanging
                if tick_count % 20 == 0:
                    avoidance_info = f", avoidance_zones={len(self.avoidance_zones)}" if self.avoidance_zones else ""
                    coordination_info = ""
                    if self.coordination_attempts:
                        total_attempts = sum(self.coordination_attempts.values())
                        coordination_info = f", coordination_attempts={total_attempts}"
                    self.logger.info(f"🔄 HEARTBEAT: Agent {self.agent_id} tick {tick_count}, state={self.state.value}, is_running={self.is_running}, current_pos={self.current_pos}, blocked={self.blocked}, waiting_for={self.waiting_for}, pending_msgs={len(self.pending_messages)}{avoidance_info}{coordination_info}")
                
                # 🔧 Check if reached FINAL goal - enter IDLE but keep polling for coordination
                if self.goal_pos and self.current_pos == self.goal_pos:
                    if self.state != AgentState.IDLE:
                        self.logger.info(f"🎉 Agent {self.agent_id} reached final goal {self.goal_pos}, entering IDLE mode but keeping responsive for coordination.")
                        self.state = AgentState.IDLE
                    # Continue running to process messages and coordination requests
                    await asyncio.sleep(0.5)
                    continue
                
                # 1. Poll for incoming messages
                await self._recv_msgs()
                
                # 2. 清理过期数据 (重要：包括同步消息超时检查)
                self._cleanup_expired_data()
                
                # 🔧 DEBUG: Log blocked state if persistent  
                if self.blocked and tick_count % 10 == 0:
                    pending_details = []
                    for msg_id, pending_msg in self.pending_messages.items():
                        remaining = pending_msg.timeout_sec - (time.time() - pending_msg.sent_time)
                        pending_details.append(f"{msg_id[:8]}->agent{pending_msg.target_agent}({remaining:.1f}s)")
                    self.logger.warning(f"⏸️ Agent {self.agent_id} blocked for {len(self.waiting_for)} agents: {self.waiting_for}, pending: {pending_details}")
                
                # 3. 状态机处理
                planning_executed = False
                if self.state == AgentState.PLANNING:
                    if not self.blocked and self.llm:
                        # 🔧 NEW: Check for unanswered messages first, reply before planning
                        unanswered_msgs = self._get_unanswered_msgs()
                        self.logger.debug(f"🔍 Agent {self.agent_id} checking unanswered messages: found {len(unanswered_msgs)}")
                        if unanswered_msgs:
                            self.logger.debug(f"💬 Agent {self.agent_id} found {len(unanswered_msgs)} unanswered messages, replying first...")
                            await self._reply_pending_messages(unanswered_msgs)
                            # After replying, continue to normal planning in next iteration
                            planning_executed = True  # Mark as executed to prevent double-planning
                        else:
                            # No pending messages, proceed with normal planning
                            self.logger.debug(f"🎯 Agent {self.agent_id} executing planning (tick {tick_count})")
                            await self._execute_autonomous_planning_step()
                            planning_executed = True
                    else:
                        self.logger.debug(f"⏸️ Agent {self.agent_id} planning blocked: blocked={self.blocked}, llm={self.llm is not None}")
                elif self.state == AgentState.WAITING_MOVE_RESP:
                    # 🔧 CRITICAL FIX: Check timeout first - fail-safe recovery
                    current_time = self._get_logical_time()
                    if current_time > self.move_reply_deadline:
                        self.logger.warning(f"🚨 Agent {self.agent_id} MOVE_RESPONSE timeout! deadline={self.move_reply_deadline}ms, now={current_time}ms → fallback to PLANNING")
                        self.state = AgentState.PLANNING
                        self.pending_moves.clear()  # Clear stuck pending moves
                        self._wait_reply_event.set()  # Wake up any waiters
                        continue
                    
                    # 🔧 额外防护：检查是否有孤儿状态（无pending但还在等待）
                    if not self.pending_moves:
                        self.logger.warning(f"⚠️ Agent {self.agent_id} in WAITING_MOVE_RESP but no pending moves! Auto-recovering to PLANNING.")
                        self.state = AgentState.PLANNING
                        self._wait_reply_event.set()  # 清除可能的阻塞
                        continue
                    
                    self.logger.debug(f"⏳ Agent {self.agent_id} waiting for move response (tick {tick_count}, deadline in {self.move_reply_deadline - current_time}ms)")
                    
                    # Use timeout-based wait instead of indefinite wait
                    timeout = max(0.05, (self.move_reply_deadline - current_time) / 1000.0)
                    try:
                        await asyncio.wait_for(self._wait_reply_event.wait(), timeout=timeout)
                        self._wait_reply_event.clear()       # 准备下一次使用
                    except asyncio.TimeoutError:
                        # This will be caught by the timeout check at the top of next iteration
                        pass
                    continue                             # 回到循环顶端重新检查状态
                elif self.state == AgentState.CONFLICT_RETRY:
                    self.logger.debug(f"🔄 Agent {self.agent_id} handling conflict retry")
                    await self._handle_conflict_retry()
                elif self.state == AgentState.WAITING_MSG:
                    # 🔧 CRITICAL FIX: Implement proper WAITING_MSG timeout handling
                    self.logger.debug(f"💬 Agent {self.agent_id} waiting for message reply (tick {tick_count})")
                    
                    # 🔥 FIX: Allow ACK processing even when blocked to prevent deadlock
                    unanswered = self._get_unanswered_msgs()
                    if unanswered:
                        self.logger.debug(f"🔄 Agent {self.agent_id} blocked but have {len(unanswered)} unanswered msgs, sending ACKs to prevent deadlock...")
                        await self._reply_pending_messages(unanswered)
                    
                    # Check for timeouts and auto-unblock
                    current_time = time.time()
                    timeout_detected = False
                    
                    for msg_id, pend_msg in list(self.pending_messages.items()):
                        if pend_msg.is_timeout():
                            target_agent = pend_msg.target_agent
                            self.logger.warning(f"⏰ Message timeout! Agent {target_agent} didn't reply to msg {msg_id[:8]}... within 3s")
                            
                            # Clean up timeout message
                            del self.pending_messages[msg_id]
                            if target_agent in self.waiting_for:
                                self.waiting_for.remove(target_agent)
                            
                            # Add timeout message to chat memory
                            timeout_msg = f"[TIMEOUT] No reply from agent {target_agent} after 3s"
                            self.chat_memory[target_agent].append(("system", timeout_msg, time.time(), None))
                            timeout_detected = True
                    
                    # If no more pending messages, unblock and return to planning
                    if not self.pending_messages or timeout_detected:
                        self.blocked = False
                        self.waiting_for.clear()
                        self.state = AgentState.PLANNING
                        self.logger.info(f"✅ Message waiting complete, returning to PLANNING state")
                        continue
                    
                    await asyncio.sleep(0.1)  # 等待消息回复
                elif self.state == AgentState.IDLE:
                    # 🔧 IDLE状态：仍然响应消息和位置请求，但不主动规划
                    # 🔧 NEW: Check for unanswered messages even in IDLE state
                    if self.llm:
                        unanswered_msgs = self._get_unanswered_msgs()
                        self.logger.debug(f"🔍 Agent {self.agent_id} (IDLE) checking unanswered messages: found {len(unanswered_msgs)}")
                        if unanswered_msgs:
                            self.logger.info(f"💬 Agent {self.agent_id} (IDLE) found {len(unanswered_msgs)} unanswered messages, replying...")
                            await self._reply_pending_messages(unanswered_msgs)
                            # 🆕 NEW: After replying to messages, mark for coordination and enter PLANNING
                            self.idle_needs_coordination = True
                            self.state = AgentState.PLANNING
                            self.logger.info(f"🚶 Agent {self.agent_id} entering PLANNING from IDLE for coordination after message reply")
                            continue  # Skip sleep and continue to planning immediately
                    
                    # 🆕 Check if we need to perform coordination movement
                    if self.idle_needs_coordination:
                        self.state = AgentState.PLANNING
                        self.logger.info(f"🚶 Agent {self.agent_id} entering PLANNING from IDLE for pending coordination")
                        continue  # Skip sleep and continue to planning immediately
                    
                    self.logger.debug(f"😴 Agent {self.agent_id} is IDLE at goal {self.goal_pos}, processing messages only (tick {tick_count})")
                    await asyncio.sleep(0.5)  # IDLE状态检查消息间隔
                
                # 4. 🔧 CRITICAL FIX: 只在执行了规划步骤后才延迟，避免卡死
                if planning_executed and self.state == AgentState.WAITING_MOVE_RESP:
                    # 发送了移动请求，短暂等待后继续循环
                    await asyncio.sleep(0.05)
                elif self.state == AgentState.PLANNING and not planning_executed:
                    # 没有执行规划（可能被阻塞），短暂等待
                    await asyncio.sleep(0.1)
                elif planning_executed and self.state == AgentState.PLANNING:
                    # 规划完成但没有发送移动请求，用规划间隔
                    jitter = random.uniform(0.8, 1.2)
                    await asyncio.sleep(min(self.planning_interval_ms * jitter / 1000.0, 1.0))  # 最大1秒
                else:
                    await asyncio.sleep(0.1)  # 默认短暂等待
                
            except asyncio.CancelledError:
                self.logger.info(f"Agent {self.agent_id} autonomous loop cancelled")
                self.is_running = False
                break
            except Exception as e:
                self.logger.error(f"Error in autonomous loop: {e}")
                # Reset state to prevent hanging
                self.state = AgentState.PLANNING
                await asyncio.sleep(1.0)  # 错误恢复
        
        # 🔧 清理main_task引用
        self.main_task = None
        self.logger.info(f"🏁 Agent {self.agent_id} autonomous loop finished")

    async def loop(self) -> None:
        """Legacy main agent loop - DISABLED to prevent dual-loop conflicts."""
        # 🔧 CRITICAL: Disable legacy loop completely to prevent dual-loop conflicts
        self.logger.error(f"❌ Legacy loop() is DISABLED for Agent {self.agent_id}")
        self.logger.error(f"   Reason: Dual-loop conflicts cause move duplication and queue race conditions")
        self.logger.error(f"   Solution: Use autonomous_loop() only. Legacy loop disabled for safety.")
        return
        
        # 🔧 Legacy code completely disabled to prevent dual-loop conflicts
        # ALL CODE BELOW IS COMMENTED OUT TO PREVENT RACE CONDITIONS
        #
        # if self.main_task and not self.main_task.done():
        #     self.logger.warning(f"⚠️ Main autonomous loop already running, skipping legacy loop")
        #     return
        # 
        # self.logger.info(f"🔄 Starting legacy loop for Agent {self.agent_id}")
        # 
        # while True:
        #     try:
        #         # Update logical clock as fallback
        #         current_wall_time = int(time.time() * 1000) - int(self.start_time)
        #         self.now = max(self.now, current_wall_time)
        #         
        #         # 1. Poll for incoming messages
        #         await self._recv_msgs()
        #     except asyncio.CancelledError:
        #         self.logger.info(f"Agent {self.agent_id} legacy loop cancelled")
        #         break
        #         
        #     try:
        #         # 2. Skip planning if waiting for synchronous reply or LLM is not available
        #         if self.blocked:
        #             await asyncio.sleep(0.01)
        #             continue
        #         
        #         if not self.llm:
        #             self.logger.error("LLM Core not initialized, skipping planning.")
        #             await asyncio.sleep(1) # prevent busy loop
        #             continue
        #         
        #         # 3. Build context for LLM
        #         msgs = self._build_context()
        #         
        #         # 记录LLM输入
        #         if self.log_manager:
        #             protocol = self.config.get("protocol", "local")
        #             self.log_manager.log_llm_interaction(
        #                 self.agent_id, 
        #                 protocol, 
        #                 {"messages": msgs, "functions": self.tools}, 
        #                 None
        #             )
        #         
        #         # 4. Call LLM with timeout
        #         try:
        #             response = await asyncio.wait_for(
        #                 asyncio.to_thread(
        #                     self.llm.function_call_execute,
        #                     messages=msgs,
        #                     functions=self.tools
        #                 ),
        #                 timeout=30.0  # 30秒超时
        #             )
        #         except asyncio.TimeoutError:
        #             self.logger.error(f"Legacy LLM call timed out after 30 seconds for Agent {self.agent_id}")
        #             continue  # 继续下一次循环
        #         except Exception as e:
        #             self.logger.error(f"Legacy LLM call failed for Agent {self.agent_id}: {e}")
        #             continue  # 继续下一次循环
        #         
        #         # 记录LLM输出
        #         if self.log_manager:
        #             protocol = self.config.get("protocol", "local")
        #             self.log_manager.log_llm_interaction(
        #                 self.agent_id, 
        #                 protocol, 
        #                 None,
        #                 response
        #             )
        #         
        #         # 5. Handle tool call
        #         await self._handle_tool(response)
        #         
        #     except asyncio.CancelledError:
        #         self.logger.info(f"Agent {self.agent_id} legacy loop cancelled during processing")
        #         break 

    def _extract_tool_call(self, response: Any) -> Tuple[str, Dict[str, Any]]:
        """
        Safely extract tool call information from LLM response, compatible with multiple API formats.
        
        Args:
            response: LLM response object
            
        Returns:
            Tuple of (tool_name, arguments_dict)
            
        Raises:
            ValueError: If tool call cannot be parsed
        """
        # 尝试新版OpenAI格式
        try:
            if hasattr(response, "choices") and response.choices:
                message = response.choices[0].message
                if hasattr(message, "tool_calls") and message.tool_calls:
                    tool_call = message.tool_calls[0]
                    if hasattr(tool_call, "function"):
                        return (tool_call.function.name, 
                               json.loads(tool_call.function.arguments))
        except Exception:
            pass
        
        # 尝试直接格式
        try:
            if hasattr(response, "name") and hasattr(response, "arguments"):
                return response.name, response.arguments
            elif isinstance(response, dict):
                return response.get("name"), response.get("arguments")
        except Exception:
            pass
        
        raise ValueError("无法解析LLM响应中的工具调用")
    
    # ===== Autonomous Agent Methods =====
    
    async def _execute_autonomous_planning_step(self):
        """
        执行一次自主LLM规划步骤
        """
        self.logger.debug(f"🧠 Agent {self.agent_id} starting planning step")

        # >>>>> 高阈值 Fallback 入口 <<<<<
        if (self.consecutive_failures >= self.FAILURE_FALLBACK or
            self.oscillation_penalty >= self.BFS_EMERGENCY_THRESHOLD):
            self.logger.error(
                f"🚨 BFS-fallback triggered! failures={self.consecutive_failures}, "
                f"osc_penalty={self.oscillation_penalty:.1f}")
            await self._fallback_bfs_two_steps()
            # 等待 MOVE_RESPONSE 由主循环处理
            return

        try:
            # 构建系统信息
            system_info = self._build_system_info()
            
            # 构建对话历史
            conversation = self._build_conversation_context()
            
            # Debug: Print conversation context to verify feedback is included
            if conversation != "No recent conversations":
                self.logger.info(f"💭 LLM Context includes: {conversation}")
            
            # 🔧 Add prominent REJECT warning to prompt
            reject_alert = ""
            if hasattr(self, 'last_reject_reason') and self.last_reject_reason:
                reject_alert = f"""
🚨 CRITICAL ALERT: Your last move was REJECTED! 
Reason: {self.last_reject_reason}
You MUST choose a different direction to avoid repeating this error!
"""
            
            # 🆕 Add anti-oscillation warning
            oscillation_warning = self._get_anti_oscillation_warning()
            
            # 组装增强的prompt  
            prompt = f"""You are Agent {self.agent_id} on a {self.grid_size[0]}×{self.grid_size[1]} grid.
{reject_alert}{oscillation_warning}
🎯 Final goal: {self.goal_pos or "Not set"}
📍 Current position: {self.current_pos}

Current Status:
{system_info}

Recent Conversations:
{conversation}

Available Tools: move (with directions U/D/L/R), send_msg

⚠️ IMPORTANT RULES:
- Never choose direction "S" (Stay). You MUST move one step toward your goal.
- 🚫 CRITICAL: NEVER move to positions marked as "AVOID POSITIONS" in your status info.
- 🔄 ANTI-OSCILLATION: Avoid moving back and forth between the same positions. Check your MOVEMENT HISTORY and avoid repeating patterns.
- If direct movement would cause collision, use send_msg to coordinate with other agents.
- 📤 CRITICAL MESSAGE RULES:
  * When asking another agent to move/let you pass/coordinate: ALWAYS set wait=true to get confirmation
  * When just informing about your position/status: set wait=false
  * Examples: "Can you move?" → wait=true; "I'm at position X" → wait=false
- Coordinate System & Directions:
  * U = Move Up (y-1, toward top edge y=0)
  * D = Move Down (y+1, toward bottom edge y={self.grid_size[1]-1})  
  * L = Move Left (x-1, toward left edge x=0)
  * R = Move Right (x+1, toward right edge x={self.grid_size[0]-1})
- 🏠 IDLE STATE COORDINATION:
  * If you're IDLE at goal and receive coordination requests: move briefly to help, then return to goal
  * If you're helping from IDLE: make minimal moves, then return to your goal position
  * Priority: Help others when possible, but return to your assigned goal position
- 🛤️ DETOUR STRATEGY:
  * If your direct path is blocked, consider taking a detour route
  * Example: Position (16,0) → Goal (18,0), but (17,0) occupied
    → Solution: (16,0) → (16,1) → (17,1) → (18,1) → (18,0)
  * Choose detour directions that avoid known agent positions and move around obstacles
  * Prefer shorter detours when possible, but don't hesitate to take longer paths to avoid conflicts
- 💬 SYNCHRONOUS MESSAGING:
  * When replying to a synchronous request (providing ACK via reply_to) use wait=false
  * Only the first request in a dialogue should use wait=true
  * If you need to coordinate, send ONE sync message and wait for their reply before continuing
- 🗣️ COORDINATION COMMUNICATION:
  * Be SPECIFIC when requesting coordination - don't just say "move aside"
  * Include details: WHERE to move, for HOW LONG, WHEN to return
  * 🚨 CRITICAL: Only send ONE sync message per conversation topic, then switch to wait=false
  * If you already sent a request and got an ACK, don't repeat the same request with wait=true
  * Example: "Could you move to (17,9) for 3 seconds? I need to pass through (18,8) to reach my goal. You can return after I pass."
  * Example: "I'm at (16,7) heading to (18,7). Could you temporarily move up to (18,6) until I reach (17,7)? Should take 2 seconds."
  * Clear requests get faster responses and better cooperation
- 🚦 CONFLICT RESOLUTION & PRIORITY SYSTEM:
  * When facing movement conflicts, follow priority rules: lower agent_id has priority
  * If you have higher agent_id AND conflict count ≥2 with same agent, consider YIELDING:
    - Step aside temporarily (move to adjacent empty cell)
    - Find alternative path around conflict zone
    - Wait 1-2 turns then retry original path
  * If you have priority, proceed with direct path but be considerate
  * Example yielding moves: if blocked at (10,0) → step to (10,1) or (9,0) temporarily
- 🌪️ MESSAGE DISCIPLINE:
  * CRITICAL: With same agent, only send ONE sync message (wait=true) per conversation topic
  * After receiving ACK, ALL follow-up messages must use wait=false
  * If agent doesn't reply in 3 seconds, try ONE async retry, then move to alternative path
  * NEVER repeat the same sync request multiple times - this causes message storms
  * If 3+ sync attempts with same agent, switch to alternative strategy (detour/yield)
- 🎯 GLOBAL NAVIGATION PRIORITY:
  * ALWAYS prioritize making progress toward your final goal over local conflict avoidance
  * If you're stuck in oscillation (A-B-A-B pattern) or small cycles, BREAK OUT immediately:
    - Calculate direct Manhattan distance to goal and choose direction that minimizes it
    - Ignore local "optimal" moves if they keep you in same small area
    - Take longer detours if necessary to escape oscillation zones
  * If positions are BANNED or COOLING DOWN, you MUST find alternative paths
  * Manhattan distance to goal should be your PRIMARY cost function, conflicts secondary
  * Example: Goal (18,18), Current (17,7) → ALWAYS prefer moves toward (18,18) even if suboptimal locally

Your goal is to navigate efficiently to your target while avoiding conflicts with other agents.
Plan your next action:"""
            
            # 构建消息
            messages = [
                {"role": "system", "content": BASE_PROMPT},
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Plan your next action"}
            ]
            
            # 记录LLM输入
            if self.log_manager:
                protocol = self.config.get("protocol", "local")
                self.log_manager.log_llm_interaction(
                    self.agent_id, 
                    protocol, 
                    {"messages": messages, "functions": self.tools}, 
                    None
                )
            
            # 调用LLM进行决策 with timeout
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.llm.function_call_execute,
                        messages=messages,
                        functions=self.tools
                    ),
                    timeout=30.0  # 30秒超时
                )
            except asyncio.TimeoutError:
                self.logger.error(f"Autonomous LLM call timed out after 30 seconds for Agent {self.agent_id}")
                # Add delay to prevent rapid retries that could cause resource exhaustion
                await asyncio.sleep(2.0)
                return
            except Exception as e:
                self.logger.error(f"Autonomous LLM call failed for Agent {self.agent_id}: {e}")
                # Add delay to prevent rapid retries that could cause resource exhaustion
                await asyncio.sleep(1.0)
                return
            
            # 记录LLM输出
            if self.log_manager:
                protocol = self.config.get("protocol", "local")
                self.log_manager.log_llm_interaction(
                    self.agent_id, 
                    protocol, 
                    None,
                    response
                )
            
            # 处理工具调用
            await self._handle_tool(response)
            self.logger.debug(f"🧠 Agent {self.agent_id} completed planning step successfully")
                
        except Exception as e:
            self.logger.error(f"Autonomous planning error: {e}")
            # Reset state to allow next planning attempt
            self.state = AgentState.PLANNING
            # Add a small delay to prevent immediate retry loops  
            await asyncio.sleep(0.5)
    
    async def _fallback_bfs_two_steps(self) -> None:
        """ ...docstring 省略... """
        import time, uuid, asyncio
        from collections import deque
        from typing import Tuple

        if self.current_pos is None or self.goal_pos is None:
            self.logger.error("BFS fallback aborted: position/goal unknown")
            return

        start: Tuple[int, int] = self.current_pos
        goal:  Tuple[int, int] = self.goal_pos
        now_wall = time.time()

        # ---------- 障碍集合 ----------
        blocked: set[Tuple[int, int]] = set(getattr(self, "static_walls", set()))
        # ② 其它agent的实时占位缓存
        blocked.update({i.pos for i in self.coord_cache.values() if not i.is_expired()})
        # ③ 临时规避区 (旧的avoidance_zones)
        blocked.update({p for p, exp in self.avoidance_zones.items() if exp > now_wall})
        # ④ 新的临时封禁
        blocked.update({pos for pos, exp in self.temp_ban.items() if exp > now_wall})
        # ⑤ 永久封禁
        blocked.update(self.perma_ban)

        # ---------- BFS ----------
        DIRS = [(0, -1, "U"), (0, 1, "D"), (-1, 0, "L"), (1, 0, "R")]
        q = deque([start]); pre = {}; visited = {start}
        found = False; expand_cnt = 0
        while q and expand_cnt < self.MAX_BFS_EXPAND:
            expand_cnt += 1
            x, y = q.popleft()
            if (x, y) == goal:
                found = True; break
            for dx, dy, d in DIRS:
                nx, ny = x + dx, y + dy
                nxt = (nx, ny)
                if ( 0 <= nx < self.grid_size[0]
                    and 0 <= ny < self.grid_size[1]
                    and nxt not in visited
                    and nxt not in blocked ):
                    visited.add(nxt); pre[nxt] = ((x, y), d); q.append(nxt)

        if not found:
            self.logger.error("BFS fallback failed: no path found, yield 1 tick")
            await asyncio.sleep(self.tick_interval)          # 让出执行权
            self._force_replan = True
            self.consecutive_failures += 1
            return

        # ---------- 回溯前两步 ----------
        path_dirs = []
        cur = goal
        while cur != start:
            prev, d = pre[cur]
            path_dirs.append(d); cur = prev
        path_dirs = list(reversed(path_dirs))[:2]
        if not path_dirs:
            self.logger.warning("BFS fallback produced empty path")
            return

        # ---------- 串行发送 ----------
        next_pos = self._effective_pos()
        for idx, d in enumerate(path_dirs, 1):
            target = self._calculate_target_position(next_pos, d)
            await self._send_move_request( ConcurrentMoveCmd(
                agent_id = self.agent_id,
                new_pos   = target,
                eta_ms    = self._get_logical_time() + 40,
                time_window_ms = self.time_window_ms,
                move_id   = str(uuid.uuid4()),
                priority  = 2
            ))
            self.logger.info(f"🏃 BFS-fallback {idx}/2: {d} → {target}")

            try:
                await asyncio.wait_for(self._wait_reply_event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self.logger.error("BFS fallback aborted: no response")
                self.consecutive_failures += 1
                return
            finally:
                self._wait_reply_event.clear()

            if self.state != AgentState.PLANNING:       # 步骤失败
                self.consecutive_failures += 1
                return
            next_pos = target                           # 下一步起点

        # ---------- 复位 ----------
        self.consecutive_failures = 0
        self.oscillation_penalty  = 0.0
        self._force_replan        = False

    def _effective_pos(self) -> Tuple[int, int]:
        """
        返回“下一刻真正所在的位置”——
        * 优先使用 **仍然有效** 的 pending_move 里的 new_pos
        * 若没有有效 pending_move，就用 current_pos
        """
        now_logic = self._get_logical_time()  # ms

        # 找到最早即将执行、且仍在时间窗口内的 pending move
        valid_moves = [mv for mv in self.pending_moves.values()
                    if mv.eta_ms + mv.time_window_ms >= now_logic]
        if valid_moves:
            # 取 ETA 最小的那条
            next_mv = min(valid_moves, key=lambda m: m.eta_ms)
            return next_mv.new_pos

        # 否则使用已确认的位置
        return self.current_pos or (0, 0)
    
    def _valid_dirs(self) -> List[str]:
        """计算当前位置的可行移动方向"""
        dirs = []
        x, y = self._effective_pos()  # 🔧 使用有效位置而不是current_pos
        max_x, max_y = self.grid_size
        
        if y > 0:        dirs.append("U")
        if y < max_y-1:  dirs.append("D") 
        if x > 0:        dirs.append("L")
        if x < max_x-1:  dirs.append("R")
        
        return dirs
    
    def _build_system_info(self) -> str:
        """构建系统状态信息块"""
        known_positions = []
        for agent_id, pos_info in self.coord_cache.items():
            status = "fresh" if not pos_info.is_expired() else "stale"
            known_positions.append(f"Agent {agent_id}: {pos_info.pos} ({status})")
        
        # 🔧 CRITICAL FIX: Add prominent REJECT warning for LLM
        reject_warning = ""
        if hasattr(self, 'last_reject_reason') and self.last_reject_reason:
            reject_warning = f"- ⚠️ LAST MOVE REJECTED: {self.last_reject_reason}\n"
        
        # 🎯 Dynamic valid directions for current position
        valid_directions = self._valid_dirs()
        
        # 🔧 Add avoidance zones info for LLM decision making
        avoidance_info = ""
        if self.avoidance_zones:
            current_time = time.time()
            active_avoidances = []
            for pos, expire_time in self.avoidance_zones.items():
                remaining = expire_time - current_time
                if remaining > 0:
                    active_avoidances.append(f"{pos}({remaining:.1f}s)")
            if active_avoidances:
                avoidance_info = f"\n- 🚫 AVOID POSITIONS: {active_avoidances} (coordination with other agents)"

        # 🆕 Add movement history and anti-oscillation info
        movement_summary = self._get_movement_history_summary()
        
        # 🆕 Add IDLE state context for coordination
        idle_info = ""
        if self.state == AgentState.IDLE:
            idle_info = f"\n- 🏠 STATUS: At goal position, available for coordination"
        elif self.idle_needs_coordination:
            idle_info = f"\n- 🚶 STATUS: Temporarily moved from goal for coordination, should return after helping"
        
        # 🆕 Add path planning analysis
        path_analysis = self._get_path_analysis()
        
        # 🚀 Add banned positions and cooling info
        banned_info = ""
        cooling_info = ""
        if self.banned_positions:
            banned_list = list(self.banned_positions)[:5]  # Show first 5
            banned_info = f"\n- 🚫 BANNED POSITIONS: {banned_list} (permanent - find alternative paths!)"
        
        if self.cooling_positions:
            current_time = time.time()
            active_cooling = []
            for pos, end_time in self.cooling_positions.items():
                remaining = end_time - current_time
                if remaining > 0:
                    active_cooling.append(f"{pos}({remaining:.0f}s)")
            if active_cooling:
                cooling_info = f"\n- ❄️ COOLING DOWN: {active_cooling[:3]} (avoid these temporarily)"
        
        # Emergency replanning flag
        replan_info = ""
        if getattr(self, '_force_replan', False):
            replan_info = f"\n- 🚨 EMERGENCY REPLAN REQUIRED: Severe oscillation detected - find completely new path!"
        
        # 🚀 Add conflict analysis and yielding suggestions
        conflict_info = ""
        yielding_info = ""
        if self.conflict_counter:
            conflicts = [f"Agent {aid}({count})" for aid, count in self.conflict_counter.items()]
            conflict_info = f"\n- ⚡ CONFLICT COUNTS: {', '.join(conflicts)}"
        
        if self.yielding_to:
            yielding_agents = list(self.yielding_to)
            if self.current_pos and self.goal_pos:
                # Get suggested yielding moves
                failing_pos = getattr(self, '_last_conflict_pos', self.goal_pos)
                yielding_moves = self._get_yielding_moves(failing_pos)
                if yielding_moves:
                    yielding_info = f"\n- 🚶‍♂️ YIELDING to Agents {yielding_agents}: consider temporary moves to {yielding_moves[:3]}"
                else:
                    yielding_info = f"\n- 🚶‍♂️ YIELDING to Agents {yielding_agents}: find alternative path"
        
        return f"""
- My Position: {self.current_pos}
- Grid Size: {self.grid_size}  
- State: {self.state.value}
{reject_warning}- Known Positions: {known_positions or "None"}
- Pending Moves: {len(self.pending_moves)}
- Final Goal: {self.goal_pos or "None"}
- 🔄 VALID DIRECTIONS: {valid_directions} (only choose from these!){avoidance_info}{idle_info}{conflict_info}{yielding_info}{banned_info}{cooling_info}{replan_info}
- 🛤️ PATH ANALYSIS: {path_analysis}
- 📍 MOVEMENT HISTORY: {movement_summary}
"""
    
    def _build_conversation_context(self) -> str:
        """
        Build recent conversation snippet for prompt.

        Same tuple-length issue as _build_chat_history, so slice.
        """
        recent_messages = []
        now = time.time()

        for agent_id, hist in self.chat_memory.items():
            for entry in hist[-3:]:
                role, content, ts = entry[:3]
                time_ago = int(now - ts)
                recent_messages.append(
                    f"[{time_ago}s ago] Agent {agent_id} ({role}): {content}"
                )

        return "\n".join(recent_messages) if recent_messages else "No recent conversations"
    
    async def handle_network_packet(self, packet: Any):
        """
        处理来自网络的数据包
        """
        if isinstance(packet, dict):
            msg_type = packet.get("type")
            payload = packet.get("payload", {})
            
            if msg_type == "MOVE_RESPONSE":
                await self._handle_move_response(payload)
            elif msg_type == "CHAT":
                await self._handle_chat_message(payload)
            elif msg_type == "POS_REQUEST":
                await self._handle_position_request(payload)
            elif msg_type == "POS_REPLY":
                await self._handle_position_reply(payload)
            else:
                self.logger.debug(f"Unhandled packet type: {msg_type}")
    
    def _get_yielding_moves(self, target_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Generate possible yielding moves - adjacent empty positions to step aside temporarily
        
        Returns:
            List of (x, y) positions that could be used for yielding
        """
        if not self.current_pos:
            return []
            
        x, y = self.current_pos
        potential_moves = [
            (x, y + 1),  # Up
            (x, y - 1),  # Down  
            (x - 1, y),  # Left
            (x + 1, y),  # Right
        ]
        
        valid_moves = []
        for px, py in potential_moves:
            # Check grid bounds
            if 0 <= px < self.grid_size[0] and 0 <= py < self.grid_size[1]:
                # Avoid the target position that caused conflict
                if (px, py) != target_pos:
                    valid_moves.append((px, py))
        
        return valid_moves

    async def _handle_move_response(self, payload: Dict):
        """处理移动响应"""
        move_id = payload.get("move_id")
        status = payload.get("status")
        
        if move_id not in self.pending_moves:
            self.logger.warning(f"Received response for unknown move {move_id}")
            return
        
        original_move = self.pending_moves[move_id]
        
        if status == "OK":
            # 移动成功
            old_pos = self.current_pos
            self.current_pos = original_move.new_pos
            self.retry_count = 0
            
            # 🆕 Record movement in history
            self._record_movement(self.current_pos)
            
            # 🔧 CRITICAL FIX: Handle soft collisions (movement succeeds but collision is noted)
            collision = payload.get("collision", False)
            if collision:
                self.collisions += 1
                self.logger.warning(f"⚠️ Move successful but collision detected at {self.current_pos} (total collisions: {self.collisions})")
            else:
                self.logger.info(f"✅ Successfully moved to {self.current_pos}")
            
            # 🔧 Clear previous reject reason on successful move
            if hasattr(self, 'last_reject_reason'):
                self.last_reject_reason = None
            
            # 🔧 清理CONFLICT事件（方案B）- 移动成功后清理相关冲突事件
            if hasattr(self, 'event_history'):
                before_count = len(self.event_history)
                self.event_history = [
                    event for event in self.event_history
                    if 'CONFLICT' not in event.get('message', '')
                ]
                cleaned_conflicts = before_count - len(self.event_history)
                if cleaned_conflicts > 0:
                    self.logger.debug(f"🧹 Cleared {cleaned_conflicts} conflict events after successful move")
            
            # 🔧 额外防护：检查当前状态是否正确
            if self.state != AgentState.WAITING_MOVE_RESP:
                self.logger.debug(f"Late OK response after state changed from {self.state.value}; proceeding anyway.")
            
            # 🔧 NEW: Reset failure counter on successful move
            if self.consecutive_failures > 0:
                self.logger.info(f"✅ Successful move after {self.consecutive_failures} failures, resetting counter")
                self.consecutive_failures = 0
            
            # 🆕 NEW: Smart state transition after successful move
            if self.idle_needs_coordination:
                # We were doing coordination movement from IDLE state
                self.idle_needs_coordination = False
                if self.goal_pos and self.current_pos == self.goal_pos:
                    # Still at goal after coordination move, return to IDLE
                    self.state = AgentState.IDLE
                    self.logger.info(f"🏠 Agent {self.agent_id} completed coordination move, returning to IDLE at goal {self.goal_pos}")
                else:
                    # Coordination moved us away from goal, continue planning to return
                    self.state = AgentState.PLANNING
                    self.logger.info(f"🏃 Agent {self.agent_id} completed coordination move, continuing PLANNING to return to goal {self.goal_pos}")
            else:
                # Normal movement, continue planning
                self.state = AgentState.PLANNING
            
            self._wait_reply_event.set()  # 🔧 通知autonomous_loop可以继续
            
        elif status == "CONFLICT":
            # 处理冲突
            conflicting_agents = payload.get("conflicting_agents", [])
            suggested_eta = payload.get("suggested_eta_ms")
            
            # 🔧 NEW: Track consecutive failures for escalation
            self.consecutive_failures += 1
            self.last_failure_time = time.time()
            
            # 🚀 NEW: Conflict priority system and yielding mechanism
            current_time = time.time()
            should_yield = False
            priority_analysis = []
            
            for other_agent in conflicting_agents:
                # Update conflict counter
                self.conflict_counter[other_agent] += 1
                self.last_conflict_time[other_agent] = current_time
                
                # Priority rules: lower agent_id has priority (simple rule)
                # Alternative rules can be added: y-coordinate, distance to goal, etc.
                has_priority = self.agent_id < other_agent
                conflict_count = self.conflict_counter[other_agent]
                
                if not has_priority and conflict_count >= 2:
                    should_yield = True
                    self.yielding_to.add(other_agent)
                    priority_analysis.append(f"Agent {other_agent} (priority={has_priority}, conflicts={conflict_count}) → YIELD")
                else:
                    priority_analysis.append(f"Agent {other_agent} (priority={has_priority}, conflicts={conflict_count}) → COMPETE")
            
            # Record conflict position for yielding suggestions
            self._last_conflict_pos = original_move.new_pos
            
            self.logger.warning(f"⚡ Move conflict with agents {conflicting_agents} (failure #{self.consecutive_failures})")
            self.logger.info(f"🏆 Priority analysis: {', '.join(priority_analysis)}")
            
            # 🔧 Enhanced conflict message with priority info
            conflict_msg = f"CONFLICT (#{self.consecutive_failures}): Your move to {original_move.new_pos} failed due to conflict with agents {conflicting_agents}."
            if should_yield:
                conflict_msg += f" PRIORITY DECISION: You should YIELD to agents {list(self.yielding_to)} - consider stepping aside or finding alternative path."
                self.logger.info(f"🚶‍♂️ Agent {self.agent_id} yielding to agents {list(self.yielding_to)}")
            else:
                conflict_msg += " You have priority - try a direct path or coordinate with other agents."
            
            self._add_to_chat_memory(self.agent_id, "system", conflict_msg)
            
            if self.retry_count < self.max_retries and suggested_eta:
                self._prepare_conflict_retry(original_move, suggested_eta)
            else:
                self.logger.info("🚫 Max retries reached or no suggestion, giving up move")
                self.state = AgentState.PLANNING
                # 重置重试计数器，为下次规划做准备
                self.retry_count = 0
                self._wait_reply_event.set()  # 🔧 通知可以继续规划
                
        elif status == "REJECT":
            # 移动被拒绝
            reason = payload.get("reason", "Unknown")
            
            # 🔧 NEW: Track consecutive failures for escalation
            self.consecutive_failures += 1
            self.last_failure_time = time.time()
            
            self.logger.warning(f"❌ Move rejected: {reason} (failure #{self.consecutive_failures})")
            
            # 🔧 CRITICAL FIX: Store last reject reason for immediate LLM visibility
            self.last_reject_reason = reason
            
            # Add clear system feedback to chat memory so LLM can learn from it
            feedback_msg = f"IMPORTANT (#{self.consecutive_failures}): Your previous move was REJECTED. {reason}. You must choose a different action to avoid this error."
            
            self._add_to_chat_memory(self.agent_id, "system", feedback_msg)
            
            self.logger.info(f"💭 Added feedback to memory: {feedback_msg}")
            self.logger.info(f"🚨 Stored last reject reason: {reason}")
            
            self.state = AgentState.PLANNING
        
        # 清理已处理的移动
        if move_id in self.pending_moves:
            del self.pending_moves[move_id]
            self.logger.info(f"🧹 Cleaned up pending move {move_id}, state now: {self.state.value}, pending: {len(self.pending_moves)}")
    
    def _prepare_conflict_retry(self, original_move: ConcurrentMoveCmd, suggested_eta: int):
        """准备冲突重试"""
        self.retry_count += 1
        
        # 🔧 CRITICAL FIX: Convert relative suggested_eta to absolute timestamp with safe range
        now_ms = self._get_logical_time()
        # Ensure suggested_eta is reasonable (50ms to 3000ms range)
        safe_delay = max(50, min(suggested_eta, 3000))  # 50ms minimum, 3s maximum
        absolute_eta = now_ms + safe_delay
        
        # 创建重试移动命令
        retry_move = ConcurrentMoveCmd(
            agent_id=original_move.agent_id,
            new_pos=original_move.new_pos,
            eta_ms=absolute_eta,  # Use computed absolute time
            time_window_ms=original_move.time_window_ms,
            move_id=str(uuid.uuid4()),
            priority=max(original_move.priority - 1, 1)  # 降低优先级
        )
        
        self.pending_moves[retry_move.move_id] = retry_move
        self.state = AgentState.CONFLICT_RETRY
        
        self.logger.info(f"🔄 Preparing retry #{self.retry_count}: suggested={suggested_eta}ms, safe={safe_delay}ms, now={now_ms}ms → eta={absolute_eta}ms")
    
    async def _handle_conflict_retry(self):
        """处理冲突重试状态"""
        # 等待一小段时间后重试
        backoff_time = self.backoff_factor ** self.retry_count * 0.1
        await asyncio.sleep(backoff_time)
        
        # 发送重试请求
        for move_id, move_cmd in list(self.pending_moves.items()):
            await self._send_move_request(move_cmd)
            break  # 一次只处理一个重试
    
    async def _handle_chat_message(self, payload: dict) -> None:
        """
        Process incoming CHAT packet from another agent.

        Payload schema (legacy & new):
        {
            "src" | "from_agent": int,
            "msg" | "message"   : str,
            "msg_id"            : str,     # optional
            "need_ack"          : bool     # optional
            ...
        }
        """
        # --- 1. Parse fields (兼容旧/新键名) ---
        from_agent = payload.get("src") or payload.get("from_agent")
        message    = payload.get("msg") or payload.get("message")
        msg_id     = payload.get("msg_id")
        need_ack   = payload.get("need_ack", False)
        reply_to   = payload.get("reply_to")            # ACK from peer

        if from_agent is None or message is None:
            self.logger.warning(f"CHAT payload missing src/msg: {payload}")
            return

        # --- 2. 记录到 chat_memory，正确标记为 "other" 并保存 msg_id ---
        self._add_to_chat_memory(from_agent, "other", message, msg_id)
        
        # 🆕 Enhanced receive logging
        msg_preview = message[:60] + "..." if len(message) > 60 else message
        msg_type = "sync" if need_ack else "async"
        reply_status = " (ACK)" if reply_to else ""
        self.logger.info(f"📨 Agent {self.agent_id} ← Agent {from_agent} ({msg_type}{reply_status}): '{msg_preview}'")

        # --- 3. 如果这是对我同步消息的 ACK，则解锁等待 ---
        if reply_to and reply_to in self.pending_messages:
            pend = self.pending_messages.pop(reply_to)
            self._unblock(from_agent)  # 使用集中式解锁函数保持状态一致
            self.logger.info(f"✅ Received ACK from agent {from_agent} for msg {reply_to[:8]}…")
            return  # ACK 不需要再回复

        # --- 4. 非 ACK 消息 → 立即发送ACK，然后由LLM统一回复详细内容 ---
        if need_ack and msg_id:
            # 🔥 NEW: 立即发送简单ACK解锁对方，避免阻塞
            await self._send_immediate_ack(from_agent, msg_id)
            # 注意：不再记录pending_acks，因为已经立即回复了
            self.logger.debug(f"⚡ Sent immediate ACK for msg {msg_id[:8]}..., LLM will handle detailed reply later")
        
        # 保持逻辑统一，让 _get_unanswered_msgs + _reply_pending_messages 处理详细回复内容
    
    async def _handle_position_request(self, payload: Dict):
        """处理位置请求"""
        src_agent = payload.get("src")
        if src_agent is not None:
            # 立即回复当前位置
            pos_reply = {
                "type": "POS_REPLY",
                "payload": {
                    "src": self.agent_id,
                    "dst": src_agent,
                    "pos": list(self.current_pos) if self.current_pos else [0, 0],
                    "client_ts": int(time.time() * 1000)
                },
                "receiver_id": src_agent
            }
            
            await self.adapter.send(pos_reply)
            self.logger.info(f"📍 Replied position to Agent {src_agent}: {self.current_pos}")
    
    async def _handle_position_reply(self, payload: Dict):
        """处理位置回复"""
        src_agent = payload.get("src")
        pos = payload.get("pos")
        
        if src_agent is not None and pos:
            # 更新位置缓存
            self.coord_cache[src_agent] = PositionInfo(
                pos=tuple(pos),
                timestamp=time.time(),
                ttl_sec=self.ttl_sec
            )
            
            self.logger.info(f"📍 Updated position cache for Agent {src_agent}: {pos}")
    
    def _cleanup_expired_data(self):
        """清理过期的 coord_cache / avoidance / pending_messages / pending_moves 并同步 blocked 状态"""
        now_wall   = time.time()              # 秒
        now_logic  = self._get_logical_time() # ms

        # ---------- 1. 位置缓存 ----------
        self.coord_cache = {aid:info for aid,info in self.coord_cache.items()
                            if not info.is_expired()}

        # ---------- 2. 临时避让 ----------
        self.avoidance_zones = {pos:exp for pos,exp in self.avoidance_zones.items()
                                if exp > now_wall}
        
        # ---------- 2.5. 临时封禁清理 ----------
        self.temp_ban = {pos:exp for pos,exp in self.temp_ban.items() if exp > now_wall}

        # ---------- 3. 同步消息 ----------
        for mid, pend in list(self.pending_messages.items()):
            if pend.is_timeout():
                self.logger.warning(f"⏰ Message to agent {pend.target_agent} timeout, drop {mid[:8]}")
                self.waiting_for.discard(pend.target_agent)
                self.pending_messages.pop(mid, None)

        # ---------- 4. 过期 pending-move（★关键修补） ----------
        for mid, mv in list(self.pending_moves.items()):
            if mv.eta_ms + mv.time_window_ms < now_logic:
                self.logger.debug(f"🗑️ Expired pending-move {mid[:8]} -> {mv.new_pos}")
                self.pending_moves.pop(mid, None)

        # ---------- 4.5. 清理过期的事件历史 ----------
        EVENT_TTL = 30.0  # 30秒过期
        if hasattr(self, 'event_history'):
            original_count = len(self.event_history)
            self.event_history = [
                event for event in self.event_history
                if now_wall - event.get('timestamp', 0) < EVENT_TTL
            ]
            cleaned_count = original_count - len(self.event_history)
            if cleaned_count > 0:
                self.logger.debug(f"🗑️ Cleaned {cleaned_count} expired events from history")

        # ---------- 5. blocked / state 同步 ----------
        self.blocked = bool(self.waiting_for)
        if not self.blocked and self.state == AgentState.WAITING_MSG:
            self.state = AgentState.PLANNING
    
    def _add_to_chat_memory(self,
                            agent_id: int,
                            role: str,
                            content: str,
                            msg_id: str | None = None) -> None:
        """
        Append one chat record to memory.

        Args
        ----
        agent_id : int
            Peer agent ID.
        role     : str
            'you' | 'other' | 'system'.
        content  : str
            Text content.
        msg_id   : str | None
            Optional message‑id for ACK tracking.
        """
        if agent_id not in self.chat_memory:
            self.chat_memory[agent_id] = []

        # Use logical time instead of wall clock for stable prompts
        timestamp = self._get_logical_time()

        # Store 4‑tuple to keep msg_id for later reply_to
        self.chat_memory[agent_id].append((role, content, timestamp, msg_id))

        # Trim to last 20 entries
        if len(self.chat_memory[agent_id]) > 20:
            self.chat_memory[agent_id] = self.chat_memory[agent_id][-20:]
    
    def _get_unanswered_msgs(self) -> List[Tuple[int, str, float, Optional[str]]]:
        """
        检查chat_memory中未回复的消息
        
        Returns:
            List of (agent_id, message_text, timestamp) for messages needing reply
        """
        unanswered = []
        for agent_id, history in self.chat_memory.items():
            if not history:
                continue
            
            # 🔥 CRITICAL FIX: 从后往前找最近的"other"消息，忽略立即ACK
            # 这样即使插入了立即ACK，仍能发现需要详细回复的消息
            for entry in reversed(history):
                if len(entry) >= 3:
                    role, content, timestamp = entry[:3]
                    msg_id = entry[3] if len(entry) > 3 else None
                    
                    if role == "other":
                        # 找到最近的对方消息，检查是否需要详细回复
                        # Skip automatic ACK messages and system notifications
                        if not (content.lower().startswith("ack") or 
                               content.lower().startswith("[timeout]") or
                               content.lower().startswith("[system]")):
                            unanswered.append((agent_id, content, timestamp, msg_id))
                        break  # 找到最近的other消息就停止
                    elif role in ("you", "ack"):
                        # 如果是自己的消息或立即ACK，继续回溯寻找
                        continue
        
        return unanswered
    
    async def _reply_pending_messages(self, unanswered_msgs: List[Tuple[int, str, float, Optional[str]]]):
        """
        统一回复待处理的消息
        
        Args:
            unanswered_msgs: List of (agent_id, message_text, timestamp, msg_id) to reply to
        """
        for agent_id, message_text, timestamp, original_msg_id in unanswered_msgs:
            try:
                # Build context for LLM reply (including full context but focused on reply)
                context_messages = self._build_context()
                
                # Add specific instruction for replying to this message
                reply_instruction = {
                    "role": "user", 
                    "content": f"""You need to provide a detailed reply to Agent {agent_id} who sent you this message: "{message_text}"

Generate a helpful, detailed response using the send_msg tool. IMPORTANT:
1. Set wait=false since this is a follow-up detailed reply (the sender has already been unlocked by immediate ACK)
2. Be polite and provide comprehensive coordination details
3. If they are asking for coordination, provide specific plans, timings, and positions  
4. If they are informing you, acknowledge and provide relevant status updates
5. This is your main substantive reply after the initial ACK

Use ONLY one tool call: send_msg with wait=false"""
                }
                context_messages.append(reply_instruction)
                
                # Call LLM with timeout for reply
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.llm.function_call_execute,
                        messages=context_messages,
                        functions=self.tools
                    ),
                    timeout=15.0  # Shorter timeout for replies
                )
                
                # Log the reply interaction
                if self.log_manager:
                    protocol = self.config.get("protocol", "local")
                    self.log_manager.log_llm_interaction(
                        self.agent_id, 
                        protocol, 
                        {"messages": context_messages, "functions": self.tools}, 
                        response
                    )
                
                # Handle the tool call - add reply_to for synchronous messages
                try:
                    name, arguments = self._extract_tool_call(response)
                    if name == "send_msg" and original_msg_id:
                        # This is a detailed reply to a synchronous message (ACK already sent)
                        # 🔥 CRITICAL FIX: 强制详细回复为异步，因为ACK已经发送过了
                        arguments["wait"] = False  # 强制异步，因为immediate ACK已经解锁了对方
                        # 不需要设置reply_to，因为immediate ACK已经处理了解锁
                        await self._handle_send_msg_tool(arguments)
                    else:
                        await self._handle_tool(response)
                except Exception as e:
                    self.logger.error(f"Error handling reply tool call: {e}")
                    # Fallback to direct tool handling
                    await self._handle_tool(response)
                
                # Extract the actual reply content for logging
                try:
                    name, arguments = self._extract_tool_call(response)
                    if name == "send_msg":
                        reply_content = arguments.get("msg", "unknown")
                        self.logger.info(f"📤 Agent {self.agent_id} replied to Agent {agent_id}: '{reply_content[:100]}{'...' if len(reply_content) > 100 else ''}'")
                    else:
                        self.logger.debug(f"📤 Agent {self.agent_id} replied to Agent {agent_id} with {name}")
                except:
                    self.logger.debug(f"📤 Agent {self.agent_id} replied to Agent {agent_id}")
                
            except asyncio.TimeoutError:
                self.logger.warning(f"⏰ Reply timeout for message from Agent {agent_id}")
                # Send a simple timeout acknowledgment
                try:
                    await self._handle_send_msg_tool({
                        "dst": agent_id,
                        "msg": "Sorry, processing delay. Acknowledged.",
                        "wait": False
                    })
                except Exception as e:
                    self.logger.error(f"Failed to send timeout acknowledgment: {e}")
            except Exception as e:
                self.logger.error(f"❌ Failed to reply to Agent {agent_id}: {e}")
                # Try to send a simple error acknowledgment
                try:
                    await self._handle_send_msg_tool({
                        "dst": agent_id,
                        "msg": "Acknowledged (processing error).",
                        "wait": False
                    })
                except Exception as inner_e:
                    self.logger.error(f"Failed to send error acknowledgment: {inner_e}")
    
    async def _send_move_request(self, move_cmd: ConcurrentMoveCmd):
        """发送移动请求到NetworkBase (race-condition safe with timeout)"""
        # 0. 先记录待处理的移动
        self.pending_moves[move_cmd.move_id] = move_cmd
        
        # 1. 🔧 CRITICAL FIX: 在发送之前就切换到等待状态，避免竞态条件
        self.state = AgentState.WAITING_MOVE_RESP
        self.target_pos = move_cmd.new_pos
        self._wait_reply_event.clear()  # 准备等待即将到来的响应
        
        # 🔧 CRITICAL FIX: Set timeout deadline for fail-safe recovery
        self.move_reply_deadline = self._get_logical_time() + 2000  # 最多等2秒
        
        # 2. 构建移动请求消息
        move_request = {
            "type": "MOVE_REQUEST",
            "payload": {
                "agent_id": move_cmd.agent_id,
                "new_pos": list(move_cmd.new_pos),
                "eta_ms": move_cmd.eta_ms,
                "time_window_ms": move_cmd.time_window_ms,
                "move_id": move_cmd.move_id,
                "priority": move_cmd.priority
            },
            "receiver_id": "network-base"
        }
        
        # 3. 现在发送 (响应可能在此await期间到达，但状态已正确设置)
        await self.adapter.send(move_request)
        
        # 4. 仅做日志和记录，不再改动 state 或事件
        move_id_short = move_cmd.move_id[:8] if move_cmd.move_id else "None"
        self.logger.info(f"🎯 >> MOVE_REQUEST {move_id_short}... to {move_cmd.new_pos} (eta: {move_cmd.eta_ms}ms, priority: {move_cmd.priority}, deadline: {self.move_reply_deadline}ms)")
        
        # Log agent move request
        if self.log_manager and hasattr(self.log_manager, 'log_agent_action'):
            protocol = self.config.get("protocol", "local")
            self.log_manager.log_agent_action(self.agent_id, protocol, "MOVE_REQUEST", move_request)
    
    def stop(self):
        """停止自主循环"""
        self.is_running = False
        self.state = AgentState.IDLE
        
        # 🔧 取消main_task if running
        if self.main_task and not self.main_task.done():
            self.main_task.cancel()
            self.main_task = None
            
        self.logger.info(f"🛑 Agent {self.agent_id} stopped")
    
    def _add_temporary_avoidance(self, pos: Tuple[int, int], duration_sec: float = 30.0, area_type: str = "single"):
        """
        Add position(s) to temporary avoidance list for coordination.
        
        Args:
            pos: Center position to avoid
            duration_sec: How long to avoid (default 30s, extended from 5s)
            area_type: "single" for just the position, "nine_grid" for 3x3 area around it
        """
        expire_time = time.time() + duration_sec
        positions_to_avoid = []
        
        if area_type == "nine_grid":
            # Add 3x3 grid around the position (including the center)
            x, y = pos
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    new_pos = (x + dx, y + dy)
                    # Only add valid positions (within grid bounds)
                    if 0 <= new_pos[0] < 19 and 0 <= new_pos[1] < 19:
                        positions_to_avoid.append(new_pos)
        else:
            # Single position avoidance (default)
            positions_to_avoid.append(pos)
        
        # Add all positions to avoidance zones
        for avoid_pos in positions_to_avoid:
            self.avoidance_zones[avoid_pos] = expire_time
        
        self.logger.info(f"🚫 Added temporary avoidance for {len(positions_to_avoid)} positions around {pos} "
                        f"({area_type}, expires in {duration_sec}s). Total avoidance zones: {len(self.avoidance_zones)}")
    
    def _is_position_avoided(self, pos: Tuple[int, int]) -> bool:
        """Check if a position should be avoided due to coordination."""
        current_time = time.time()
        if pos in self.avoidance_zones:
            if current_time < self.avoidance_zones[pos]:
                remaining_time = self.avoidance_zones[pos] - current_time
                self.logger.debug(f"🚫 Position {pos} is avoided (remaining: {remaining_time:.1f}s)")
                return True
            else:
                # Clean up expired avoidance
                del self.avoidance_zones[pos]
                self.logger.debug(f"🔓 Avoidance for position {pos} expired and removed")
        return False
    
    def _cleanup_expired_avoidance(self):
        """Clean up expired avoidance zones."""
        current_time = time.time()
        expired_zones = []
        for pos, expire_time in self.avoidance_zones.items():
            if current_time >= expire_time:
                expired_zones.append(pos)
        
        for pos in expired_zones:
            del self.avoidance_zones[pos]
            self.logger.debug(f"🔓 Cleared expired avoidance for position {pos}")
    
    def _should_yield_to_agent(self, other_agent_id: int, attempt_count: int) -> bool:
        """Determine if this agent should yield to another agent based on priority rules."""
        # Priority rule 1: Higher ID yields to lower ID (simple but effective)
        # This prevents deadlocks by establishing clear hierarchy
        if self.agent_id > other_agent_id:
            return True
        
        # Priority rule 2: After many attempts, lower ID agent also considers yielding
        if attempt_count > 4:
            return True
            
        return False
    
    def _get_conflict_area_with_agent(self, other_agent_id: int) -> List[Tuple[int, int]]:
        """Get positions that are likely conflict points with another agent."""
        conflict_positions = []
        
        if not self.current_pos:
            return conflict_positions
        
        # Add current position
        conflict_positions.append(self.current_pos)
        
        # Add positions around current location (potential conflict zone)
        x, y = self.current_pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.grid_size[0] and 
                    0 <= new_y < self.grid_size[1]):
                    conflict_positions.append((new_x, new_y))
        
        # If we know the other agent's position, add that area too
        if other_agent_id in self.coord_cache:
            other_pos = self.coord_cache[other_agent_id].pos
            conflict_positions.append(other_pos)
            
            # Add positions around the other agent
            ox, oy = other_pos
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    new_x, new_y = ox + dx, oy + dy
                    if (0 <= new_x < self.grid_size[0] and 
                        0 <= new_y < self.grid_size[1]):
                        conflict_positions.append((new_x, new_y))
        
        return list(set(conflict_positions))  # Remove duplicates
    
    async def _send_immediate_ack(self, to_agent: int, original_msg_id: str) -> None:
        """
        立即发送简单ACK解锁对方，无需LLM参  与
        
        Args:
            to_agent: 目标agent ID
            original_msg_id: 原始消息ID用于reply_to
        """
        ack_packet = {
            "type": "CHAT",
            "payload": {
                "src": self.agent_id,
                "dst": to_agent,
                "msg": "ACK",                 # 简明一点即可
                "ts":  self._get_logical_time(),
                "reply_to": original_msg_id,
                "need_ack": False
            }
        }
        
        await self.adapter.send(ack_packet)
        self.logger.info(f"⚡ Sent immediate ACK to Agent {to_agent} for msg {original_msg_id[:8]}...")
        
        # 记录到chat_memory中，使用特殊role标识立即ACK
        self.chat_memory[to_agent].append(("ack", "收到消息，稍后详细回复", self._get_logical_time(), original_msg_id))
    
    def _unblock(self, agent_id: int = None) -> None:
        """
        集中式状态同步函数：保证 waiting_for、blocked 与 pending_messages 保持一致
        
        Args:
            agent_id: 特定要解锁的agent ID，None表示重新计算全部状态
        """
        if agent_id is not None:
            self.waiting_for.discard(agent_id)
            self.logger.debug(f"🔓 Unblocked Agent {agent_id}, remaining waiting_for: {self.waiting_for}")
        
        # 若仍有人欠 ACK，则保持 blocked
        if self.waiting_for:
            self.blocked = True
        else:
            self.blocked = False
            if self.state == AgentState.WAITING_MSG:
                self.state = AgentState.PLANNING
                self.logger.debug(f"🔄 No more pending ACKs, returning to PLANNING")
    
    async def handle_chat(self, sender_id: int, msg_str: str, chat_dict: dict) -> str:
        """
        Handle CHAT messages from A2A executor.
        
        This method is called by MAPFAgentExecutor when receiving CHAT messages.
        It forwards the message to internal chat handling - NO AUTO-REPLY.
        
        Args:
            sender_id: ID of the sending agent
            msg_str: Message content string  
            chat_dict: Complete chat message dictionary
            
        Returns:
            String result for the A2A executor
        """
        try:
            # 🔧 CRITICAL FIX: Extract from correct payload level
            raw = chat_dict["payload"] if "payload" in chat_dict else chat_dict           # Get nested payload
            
            # Create proper payload for internal chat handling
            payload = {
                "src": sender_id,
                "dst": self.agent_id,
                "msg": msg_str,
                "msg_id": raw.get("msg_id"),              # ✅ Get from payload level
                "need_ack": raw.get("need_ack", False),   # ✅ Get from payload level  
                "reply_to": raw.get("reply_to")           # ✅ Get from payload level
            }
            
            # Forward to internal chat handler (LLM will handle replies)
            await self._handle_chat_message(payload)
            
            # Log successful routing
            ack_status = " (ACK required)" if payload["need_ack"] else ""
            self.logger.info(f"✅ Routed CHAT from Agent {sender_id}: '{msg_str[:50]}{'...' if len(msg_str) > 50 else ''}'{ack_status}")
            
            return f"[Agent {self.agent_id}] Routed CHAT from Agent {sender_id}"
            
        except Exception as e:
            error_msg = f"[Agent {self.agent_id}] Error routing CHAT from Agent {sender_id}: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    # ===== 🆕 Movement History and Anti-Oscillation Methods =====
    
    def _record_movement(self, new_pos: Tuple[int, int]) -> None:
        """
        记录移动历史并更新振荡惩罚
        
        Args:
            new_pos: 新位置
        """
        current_time = time.time()
        
        # 🚀 Update step counter and check for periodic replanning
        self.step_count += 1
        
        # 添加到历史记录
        self.movement_history.append((new_pos[0], new_pos[1], current_time))
        
        # 保持历史记录长度限制
        if len(self.movement_history) > self.max_history_length:
            self.movement_history = self.movement_history[-self.max_history_length:]
        
        # 更新振荡惩罚
        self._update_oscillation_penalty()
        
        # 🚀 Trigger periodic replanning every 20 steps OR when forced
        if (self.step_count - self.last_replan_step >= 20) or getattr(self, '_force_replan', False):
            if getattr(self, '_force_replan', False):
                self.logger.error(f"💥 EMERGENCY REPLANNING triggered at step {self.step_count} due to severe oscillation!")
                self._force_replan = False  # Reset flag
            else:
                self.logger.info(f"🔄 Periodic replanning triggered at step {self.step_count}")
            
            self.last_replan_step = self.step_count
            # The actual replanning will be handled by LLM with enhanced global guidance
        
        self.logger.debug(f"📍 Recorded movement to {new_pos}, history length: {len(self.movement_history)}, penalty: {self.oscillation_penalty:.2f}")
    
    def _update_oscillation_penalty(self) -> None:
        """更新振荡惩罚分数并判定是否需要封禁/紧急BFS"""
        if len(self.movement_history) < 4:
            return

        # 衰减现有惩罚
        self.oscillation_penalty *= self.oscillation_decay
        
        # 提取最近4个位置
        p1, p2, p3, p4 = [tuple(p[:2]) for p in self.movement_history[-4:]]

        # ----------- A-B-A-B 检测 ----------
        if p1 == p3 and p2 == p4 and p1 != p2:
            self._aba_counter += 1            # 连续振荡计数
            self.oscillation_penalty += 5.0   # 适度增加惩罚
        else:
            self._aba_counter = 0             # 断开即清零

        # ----------- 先看 BFS 是否该触发 ----------
        if self.oscillation_penalty >= self.BFS_EMERGENCY_THRESHOLD:
            self.logger.error(f"🚨 penalty≥{self.oscillation_penalty:.1f} - emergency BFS")
            self._force_replan = True         # 外层 loop 里看到就会走 _fallback_bfs_two_steps()
            # 不 return，让后续逻辑仍可累积计数

        # ----------- 再看要不要封格子 ----------
        if self._aba_counter >= self.OSC_BAN_THRESHOLD:
            current_time = time.time()
            # 记录封禁次数
            for pos in (p1, p2):
                self._temp_ban_counts[pos] = self._temp_ban_counts.get(pos, 0) + 1

                # 临时封禁
                expiry = current_time + self.OSC_TEMP_BAN_SECONDS
                self.temp_ban[pos] = expiry

                # 达到次数阈值时永久封禁
                if self._temp_ban_counts[pos] >= self.OSC_PERMA_BAN_COUNT:
                    self.perma_ban.add(pos)
                    self.logger.error(f"🚫 PERMANENT BAN: {pos} (banned {self._temp_ban_counts[pos]} times)")

            self.logger.error(
                f"🚫 Oscillation {self._aba_counter}× -> temp-ban {p1},{p2} for {self.OSC_TEMP_BAN_SECONDS}s; "
                f"penalty={self.oscillation_penalty:.1f}"
            )
            self._aba_counter = 0       # 重置，避免立刻再次封
        
        # 🆕 检测循环模式 (包括四个格子转圈)
        self._detect_circular_patterns()
        
        # 检测在小区域内的频繁访问
        position_counts = {}
        for x, y, _ in self.movement_history[-6:]:  # 检查最近6步
            pos = (x, y)
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        # 如果任何位置访问次数超过阈值，增加惩罚
        for pos, count in position_counts.items():
            if count >= self.oscillation_threshold:
                penalty_increase = (count - self.oscillation_threshold + 1) * 0.5
                self.oscillation_penalty += penalty_increase
                self.logger.warning(f"🔄 Position {pos} visited {count} times in recent history, penalty increased by {penalty_increase:.2f}")
    
    def _get_movement_history_summary(self) -> str:
        """获取移动历史摘要用于LLM context"""
        if not self.movement_history:
            return "No movement history"
        
        # 生成路径字符串
        path_str = ""
        for i, (x, y, timestamp) in enumerate(self.movement_history[-8:]):  # 显示最近8步
            if i == 0:
                path_str = f"({x},{y})"
            else:
                path_str += f" → ({x},{y})"
        
        # 检测模式
        patterns = []
        
        # A-B-A-B oscillation detection
        if len(self.movement_history) >= 4:
            recent_4 = [(x, y) for x, y, _ in self.movement_history[-4:]]
            if len(set(recent_4)) == 2 and recent_4[0] == recent_4[2] and recent_4[1] == recent_4[3]:
                patterns.append("⚠️ A-B-A-B oscillation detected")
        
        # 🆕 Circular pattern detection
        recent_positions = [(x, y) for x, y, _ in self.movement_history[-8:]]
        for cycle_length in [3, 4, 5]:
            if self._has_repeating_cycle(recent_positions, cycle_length):
                cycle_pattern = recent_positions[-cycle_length:]
                pattern_str = "→".join([f"({x},{y})" for x, y in cycle_pattern])
                patterns.append(f"🔄 {cycle_length}-position cycle: {pattern_str}")
        
        # 统计最近访问的位置
        recent_positions = [f"({x},{y})" for x, y, _ in self.movement_history[-6:]]
        position_freq = {}
        for pos in recent_positions:
            position_freq[pos] = position_freq.get(pos, 0) + 1
        
        frequent_visits = [f"{pos}×{count}" for pos, count in position_freq.items() if count >= 2]
        if frequent_visits:
            patterns.append(f"Frequent visits: {', '.join(frequent_visits)}")
        
        summary = f"Recent path ({len(self.movement_history)} steps): {path_str}"
        if patterns:
            summary += f"\n🚨 Patterns: {'; '.join(patterns)}"
        if self.oscillation_penalty > 0.1:
            summary += f"\n⚡ Oscillation penalty: {self.oscillation_penalty:.2f}"
        
        return summary
    
    def _get_anti_oscillation_warning(self) -> str:
        """获取反振荡警告信息"""
        if self.oscillation_penalty < 0.5:
            return ""
        
        warning = f"""
🚨 ANTI-OSCILLATION WARNING (Penalty: {self.oscillation_penalty:.2f}):
You are showing signs of inefficient movement patterns. Avoid:
- Moving back and forth between the same 2 positions (A-B-A-B pattern)
- Staying in a small area without making progress toward your goal
- Visiting the same position multiple times within a few steps

Focus on:
- Making steady progress toward your goal at {self.goal_pos}
- Finding alternative paths if blocked
- Coordinating with other agents rather than repeatedly trying the same move
"""
        return warning
    
    def _get_path_analysis(self) -> str:
        """
        Analyze path from current position to goal and suggest detour options.
        
        Returns:
            String describing direct path status and detour suggestions
        """
        if not self.goal_pos or not self.current_pos:
            return "No goal set or position unknown"
        
        if self.current_pos == self.goal_pos:
            return "Already at goal position"
        
        # Calculate direct path
        start_x, start_y = self.current_pos
        goal_x, goal_y = self.goal_pos
        
        # Determine primary directions needed
        dx = goal_x - start_x
        dy = goal_y - start_y
        distance = abs(dx) + abs(dy)  # Manhattan distance
        
        primary_dirs = []
        if dx > 0:
            primary_dirs.append("R (East)")
        elif dx < 0:
            primary_dirs.append("L (West)")
        
        if dy > 0:
            primary_dirs.append("D (South)")
        elif dy < 0:
            primary_dirs.append("U (North)")
        
        # Check for known obstacles on direct path
        obstacles = []
        blocked_positions = set()
        
        # Add known agent positions as obstacles
        for agent_id, pos_info in self.coord_cache.items():
            if not pos_info.is_expired():
                blocked_positions.add(pos_info.pos)
        
        # Add positions to avoid
        current_time = time.time()
        for pos, expire_time in self.avoidance_zones.items():
            if current_time < expire_time:
                blocked_positions.add(pos)
        
        # Check next step on direct path
        direct_blocked = False
        next_direct_pos = None
        
        if dx != 0:
            # Try horizontal movement first
            next_x = start_x + (1 if dx > 0 else -1)
            next_direct_pos = (next_x, start_y)
            if next_direct_pos in blocked_positions:
                direct_blocked = True
                obstacles.append(f"Direct path blocked at {next_direct_pos}")
        elif dy != 0:
            # Try vertical movement
            next_y = start_y + (1 if dy > 0 else -1)
            next_direct_pos = (start_x, next_y)
            if next_direct_pos in blocked_positions:
                direct_blocked = True
                obstacles.append(f"Direct path blocked at {next_direct_pos}")
        
        # Generate detour suggestions if direct path is blocked
        detour_suggestions = []
        if direct_blocked and next_direct_pos:
            # Suggest perpendicular detours
            detour_options = []
            
            # If we were going horizontally, try vertical detours
            if dx != 0:
                # Try going up first
                detour_up = (start_x, start_y - 1)
                detour_down = (start_x, start_y + 1)
                
                if (0 <= detour_up[1] < self.grid_size[1] and 
                    detour_up not in blocked_positions):
                    detour_options.append(f"U to {detour_up}, then continue toward goal")
                
                if (0 <= detour_down[1] < self.grid_size[1] and 
                    detour_down not in blocked_positions):
                    detour_options.append(f"D to {detour_down}, then continue toward goal")
            
            # If we were going vertically, try horizontal detours
            elif dy != 0:
                # Try going left/right first
                detour_left = (start_x - 1, start_y)
                detour_right = (start_x + 1, start_y)
                
                if (0 <= detour_left[0] < self.grid_size[0] and 
                    detour_left not in blocked_positions):
                    detour_options.append(f"L to {detour_left}, then continue toward goal")
                
                if (0 <= detour_right[0] < self.grid_size[0] and 
                    detour_right not in blocked_positions):
                    detour_options.append(f"R to {detour_right}, then continue toward goal")
            
            if detour_options:
                detour_suggestions = [f"Detour options: {'; '.join(detour_options)}"]
        
        # Build analysis summary
        analysis_parts = [f"Distance to goal: {distance} steps"]
        
        if primary_dirs:
            analysis_parts.append(f"Primary directions needed: {', '.join(primary_dirs)}")
        
        if obstacles:
            analysis_parts.extend(obstacles)
        
        if detour_suggestions:
            analysis_parts.extend(detour_suggestions)
        elif not direct_blocked:
            analysis_parts.append("Direct path appears clear")
        
        return "; ".join(analysis_parts)
    
    def _detect_circular_patterns(self) -> None:
        """
        Detect circular movement patterns (e.g., moving in a 3x3 or 4x4 loop).
        
        检测循环移动模式，包括四个格子转圈等情况。
        """
        if len(self.movement_history) < 6:  # Need at least 6 positions for meaningful cycle detection
            return
        
        # Extract recent positions for pattern analysis
        recent_positions = [(x, y) for x, y, _ in self.movement_history[-8:]]  # Check last 8 steps
        
        # 🔍 Detect cycles of different lengths (3, 4, 5 positions)
        for cycle_length in [3, 4, 5]:
            if len(recent_positions) >= cycle_length * 2:  # Need at least 2 full cycles
                # Check if we have a repeating pattern
                if self._has_repeating_cycle(recent_positions, cycle_length):
                    # Extract the cycle pattern
                    cycle_pattern = recent_positions[-cycle_length:]
                    pattern_str = " → ".join([f"({x},{y})" for x, y in cycle_pattern])
                    
                    # 🚀 CRITICAL FIX: Ban all positions in severe cycles
                    if cycle_length == 4:  # 4-position cycles are particularly problematic
                        penalty_increase = 50.0  # Severe penalty
                        self.oscillation_penalty += penalty_increase
                        
                        # Ban all positions in the cycle
                        current_time = time.time()
                        for pos in cycle_pattern:
                            self.banned_positions.add(pos)
                            self.cooling_positions[pos] = current_time + 20.0  # 20秒冷却
                        
                        self.logger.error(f"🚨 SEVERE 4-position cycle detected: {pattern_str}")
                        self.logger.error(f"🚫 BANNING all cycle positions: {cycle_pattern}")
                        self.logger.error(f"💥 Emergency penalty: +{penalty_increase}, total: {self.oscillation_penalty:.2f}")
                        self._force_replan = True
                    else:
                        penalty_increase = cycle_length * 1.5  # Normal penalty but higher
                        self.oscillation_penalty += penalty_increase
                        self.logger.warning(f"🔄 Detected {cycle_length}-position cycle: {pattern_str}, penalty increased by {penalty_increase:.1f}")
                    
                    break  # Only penalize once per detection
    
    def _has_repeating_cycle(self, positions: List[Tuple[int, int]], cycle_length: int) -> bool:
        """
        Check if the recent positions contain a repeating cycle of given length.
        
        Args:
            positions: List of (x, y) positions
            cycle_length: Length of cycle to check for
            
        Returns:
            True if a repeating cycle is detected
        """
        if len(positions) < cycle_length * 2:
            return False
        
        # Extract the last cycle_length positions as the pattern
        pattern = positions[-cycle_length:]
        
        # Check if this pattern repeats in the previous cycle_length positions
        previous_pattern = positions[-cycle_length*2:-cycle_length]
        
        # Also check if we're starting to repeat again (partial match)
        if len(positions) >= cycle_length * 2 + 1:
            next_pos = positions[-cycle_length*2-1:-cycle_length*2] if len(positions) >= cycle_length * 2 + 1 else []
            if next_pos and next_pos[0] == pattern[0]:
                # We're starting the cycle again
                return pattern == previous_pattern
        
        return pattern == previous_pattern 