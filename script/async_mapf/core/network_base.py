"""
NetworkBase implementation for local single-process MAPF coordination.

Provides:
① Real-time grid state maintenance (world_state)
② Agent action request processing → validation → position updates  
③ Action completion feedback generation
④ Complete simulation history recording
⑤ Abstract communication layer that shields specific protocol details

Design Note: 
In single-process mode, agents call network.send_move(cmd) directly for simplicity.
For distributed protocols, extend with _poll_adapters() to recv from agent adapters.
"""

import asyncio

import time
import copy
import csv
import json
import logging
import collections
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from .types import Coord, MoveCmd, StepRecord, MoveFeedback
from .comm import AbstractCommAdapter, LocalQueueAdapter
import queue

from ..utils.log_utils import get_log_manager, setup_colored_logging, log_network_event
setup_colored_logging()

# Import NetworkBaseExtensions for concurrent mode features
try:
    from .network_base_extensions import NetworkBaseExtensions
except ImportError:
    # Fallback if extensions not available
    class NetworkBaseExtensions:
        pass

class NetworkBase(NetworkBaseExtensions):
    """
    Global coordinator with abstract protocol I/O hooks.
    
    Implements:
    - Real-time grid state management
    - Event-driven action processing
    - Collision detection and conflict arbitration
    - Performance monitoring and history recording
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize network coordinator.
        
        Args:
            config: Configuration dict with grid_size, tick_ms, agents
        """
        # Setup logging with unique instance ID to avoid duplication
        import uuid
        self.instance_id = str(uuid.uuid4())[:8]
        self.log_manager = get_log_manager()
        if self.log_manager:
            self.logger = self.log_manager.get_network_logger()
        else:
            # Fallback logger configuration only if log_manager is not available
            self.logger = logging.getLogger(f"NetworkBase-{self.instance_id}")
            if not self.logger.handlers:  # Only add handler if none exists
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
                self.logger.propagate = False
        
        # Configuration
        self.grid_size: int = config.get("grid_size", 19)
        self.concurrent_mode: bool = config.get("mode", "concurrent") == "concurrent"
        self._init_termination_controls(config)
        # Time management
        self.now: int = 0  # logical clock in milliseconds
        self.is_running: bool = False
        self.start_time: float = 0.0
        
        # World state: agent_id -> current position
        self.world_state: Dict[int, Coord] = {}
        
        # Communication adapters: agent_id -> adapter
        self.adapters: Dict[int, AbstractCommAdapter] = {}
        
        # History recording
        self.history: List[StepRecord] = []
        
        # A2A communication callback (set by NetworkBaseExecutor)
        self._a2a_send_callback = None
        
        # Register agents from config
        self.register_agents(config.get("agents", []))

    def _init_termination_controls(self, config: Dict[str, Any]) -> None:
        """Initialize termination-related fields."""
        # Whether to stop the simulation automatically when all goals are reached
        self.stop_on_all_goals: bool = config.get("stop_on_all_goals", True)
        # Agent goals mapping: agent_id -> Coord or None
        self.goals: Dict[int, Optional[Coord]] = {}
        # Optional callback to notify external runner when simulation completes
        self._on_simulation_complete_cb = None
        # Guard to avoid scheduling finalize multiple times
        self._completion_scheduled: bool = False
        self.hard_exit_on_complete: bool = config.get("hard_exit_on_complete", False)
            
    def register_agents(self, agent_configs: List[Dict[str, Any]]) -> None:
        """
        Register agents from configuration.
        
        Args:
            agent_configs: List of agent configuration dicts
        """
        for agent_cfg in agent_configs:
            agent_id = agent_cfg["id"]
            start_pos = tuple(agent_cfg["start"])  # [x, y] -> (x, y)
            
            # Adapter is now expected to be created and set by the runner.
            # We create a default LocalQueueAdapter only if one isn't already there.
            if agent_id not in self.adapters:
                adapter = LocalQueueAdapter()
                self.adapters[agent_id] = adapter
            
            self.world_state[agent_id] = start_pos
            goal_raw = (
                agent_cfg.get("goal")
                or agent_cfg.get("target")
                or agent_cfg.get("dest")
                or agent_cfg.get("destination")
                or agent_cfg.get("end")
            )
            self.goals[agent_id] = tuple(goal_raw) if goal_raw is not None else None
            
            self.logger.info(f"Registered agent {agent_id} at {start_pos}")
            # Log network event
            log_network_event("REGISTER_AGENT", {"agent_id": agent_id, "start_pos": start_pos})
    
    async def send_move(self, cmd: MoveCmd) -> None:
        """
        Process incoming move command from agent (immediate execution).
        
        Args:
            cmd: Move command from agent
        """
        # Validate agent exists
        if cmd.agent_id not in self.adapters:
            self.logger.warning(f"Move from unregistered agent {cmd.agent_id}")
            return
        
        # Immediate execution
        exec_ts = int(time.time() * 1000)
        # Log network move command
        log_network_event("MOVE_CMD", {"agent_id": cmd.agent_id, "action": cmd.action})
        ok, collision = self._apply_move(cmd, exec_ts)
        
        # Send feedback immediately
        feedback = MoveFeedback(
            agent_id=cmd.agent_id,
            success=ok,
            actual_pos=self.world_state[cmd.agent_id],
            collision=collision,
            latency_ms=exec_ts - cmd.client_ts,
            step_ts=exec_ts
        )
        
        await self.adapters[cmd.agent_id].send(feedback)
        self.logger.debug(f"Processed move for agent {cmd.agent_id}: {cmd.action} -> {'OK' if ok else 'COLLISION'}")
    
    def set_a2a_send_callback(self, callback):
        """Set the A2A send callback function"""
        self._a2a_send_callback = callback
        self.logger.info("A2A send callback configured")

    def set_on_simulation_complete(self, callback) -> None:
        """
        Register a callback to be invoked when the simulation completes
        (i.e., all agents with defined goals have reached their goals).
        The callback can be sync or async; it receives no arguments.
        """
        self._on_simulation_complete_cb = callback
        self.logger.info("on_simulation_complete callback registered")  

    def _all_agents_at_goals(self) -> bool:
        """
        Return True if every agent that has a defined goal is at its goal.
        Agents without a goal are ignored for the termination check.
        """
        any_goal = False
        for aid, goal in self.goals.items():
            if goal is None:
                continue  # ignore agents with no goal configured
            any_goal = True
            if self.world_state.get(aid) != goal:
                return False
        # If no agent has a goal at all, we consider "not complete".
        return any_goal

    async def _finalize_simulation(self, exec_ts: int) -> None:
        """
        Broadcast STOP, stop the loop, and invoke completion callback.
        """
        if not self.is_running:
            return

        # 1) Broadcast CONTROL STOP to agents (A2A or local adapters)
        try:
            stop_msg = {"type": "CONTROL", "cmd": "STOP", "ts": exec_ts}
            if self._a2a_send_callback:
                for aid in self.world_state.keys():
                    try:
                        await self._a2a_send_callback(aid, stop_msg)
                    except Exception as e:
                        self.logger.warning(f"Failed to send STOP to agent {aid}: {e}")
            else:
                for aid, adapter in self.adapters.items():
                    try:
                        await adapter.send(stop_msg)
                    except Exception as e:
                        self.logger.warning(f"Failed to send STOP via adapter to agent {aid}: {e}")
        except Exception as e:
            self.logger.error(f"Error broadcasting STOP: {e}")

        # 2) Stop network loop
        self.is_running = False
        self.logger.info("All agents reached goals. Stopping NetworkBase...")
        try:
            log_network_event("SIMULATION_COMPLETE", {
                "ts": exec_ts,
                "total_steps": len(self.history),
                "metrics": self.get_performance_metrics()
            })
        except Exception:
            pass

        # 3) Invoke optional runner callback (can be sync or async)
        cb = self._on_simulation_complete_cb
        if cb:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb()
                else:
                    cb()
            except Exception as e:
                self.logger.warning(f"on_simulation_complete callback error: {e}")

    def _maybe_trigger_completion(self, exec_ts: int) -> None:
        """
        If termination conditions are met, schedule finalization task once.
        """
        if not self.stop_on_all_goals:
            return
        if self._completion_scheduled:
            return
        if self._all_agents_at_goals():
            self._completion_scheduled = True
            asyncio.create_task(self._finalize_simulation(exec_ts))

    async def process_move_command(self, cmd) -> None:
        """
        Process move command and send immediate response.
        
        Args:
            cmd: Move command (MoveCmd or ConcurrentMoveCmd) from agent
        """
        try:
            # Convert ConcurrentMoveCmd to direction if needed
            if hasattr(cmd, 'new_pos'):
                # This is a ConcurrentMoveCmd from autonomous agent
                agent_id = cmd.agent_id
                exec_ts = int(time.time() * 1000)
                ok, conflicting_agents = self._apply_move_concurrent(cmd, exec_ts)
                
                # Send MOVE_RESPONSE back
                resp = {
                    "type": "MOVE_RESPONSE",
                    "payload": {
                        "move_id": cmd.move_id,
                        "status": "OK" if ok else "CONFLICT",
                        "conflicting_agents": conflicting_agents or [],
                        "suggested_eta_ms": exec_ts + 100 if conflicting_agents else None
                    },
                    "receiver_id": agent_id
                }
                await self._send(resp)
                self.logger.debug(f"Processed concurrent move for agent {agent_id}: {cmd.new_pos} -> {'OK' if ok else 'CONFLICT'}")
            else:
                # Legacy MoveCmd
                await self.send_move(cmd)
            
        except Exception as e:
            self.logger.error(f"Error processing move command: {e}")
    
    async def _send(self, message):
        """Send message using appropriate adapter or callback."""
        receiver_id = message.get("receiver_id")
        if self._a2a_send_callback:
            await self._a2a_send_callback(receiver_id, message)
        elif receiver_id in self.adapters:
            await self.adapters[receiver_id].send(message)
        else:
            self.logger.warning(f"No way to send message to {receiver_id}")
    
    def _apply_move_concurrent(self, cmd, exec_ts):
        """
        Apply concurrent move command with immediate conflict detection.

        Returns:
            (success: bool, conflicting_agents: List[int])
        """
        agent_id = cmd.agent_id
        target_pos = cmd.new_pos
        current_pos = self.world_state.get(agent_id, (0, 0))

        # Bounds check
        if not (0 <= target_pos[0] < self.grid_size and 0 <= target_pos[1] < self.grid_size):
            # record as failed (stay)
            self.history.append(StepRecord(
                step_ts=exec_ts,
                agent_id=agent_id,
                from_pos=current_pos,
                to_pos=current_pos,
                latency_ms=exec_ts - (cmd.eta_ms or 0),
                collision=False
            ))
            self._maybe_trigger_completion(exec_ts)
            return False, []

        # Conflict detection
        conflicting_agents = []
        for other_agent_id, other_pos in self.world_state.items():
            if other_agent_id != agent_id and other_pos == target_pos:
                conflicting_agents.append(other_agent_id)

        if conflicting_agents:
            # Collision: stay in place
            self.history.append(StepRecord(
                step_ts=exec_ts,
                agent_id=agent_id,
                from_pos=current_pos,
                to_pos=current_pos,
                latency_ms=exec_ts - (cmd.eta_ms or 0),
                collision=True
            ))
            self._maybe_trigger_completion(exec_ts)
            return False, conflicting_agents

        # Move successful
        self.world_state[agent_id] = target_pos
        self.history.append(StepRecord(
            step_ts=exec_ts,
            agent_id=agent_id,
            from_pos=current_pos,
            to_pos=target_pos,
            latency_ms=exec_ts - (cmd.eta_ms or 0),
            collision=False
        ))

        # After state update, maybe stop the simulation
        self._maybe_trigger_completion(exec_ts)

        return True, []

    
    def _calculate_target_position(self, current_pos: Coord, action: str) -> Coord:
        """
        Calculate target position based on action.
        
        Args:
            current_pos: Current (x, y) position
            action: Action string ('U', 'D', 'L', 'R', 'S')
            
        Returns:
            Target (x, y) position
        """
        x, y = current_pos
        
        if action == 'U':
            return (x, y - 1)
        elif action == 'D':
            return (x, y + 1)
        elif action == 'L':
            return (x - 1, y)
        elif action == 'R':
            return (x + 1, y)
        elif action == 'S':
            return (x, y)  # Stay
        else:
            # Invalid action, stay in place
            self.logger.warning(f"Invalid action '{action}', treating as Stay")
            return (x, y)
    
    def _is_valid_position(self, pos: Coord) -> bool:
        """
        Check if position is within grid boundaries.
        
        Args:
            pos: Position to check
            
        Returns:
            True if position is valid
        """
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
    
    def _detect_collision(self, agent_id: int, target_pos: Coord) -> bool:
        """
        Check if target position would cause collision.
        
        Args:
            agent_id: Moving agent ID
            target_pos: Target position
            
        Returns:
            True if collision would occur
        """
        # Check if any other agent is already at target position
        for other_id, other_pos in self.world_state.items():
            if other_id != agent_id and other_pos == target_pos:
                return True
        return False
    

    
    def _apply_move(self, cmd: MoveCmd, exec_ts: int) -> tuple[bool, bool]:
        """
        Apply a single move command immediately.

        Returns:
            (success, collision)
        """
        agent_id = cmd.agent_id
        current_pos = self.world_state.get(agent_id, (0, 0))
        target_pos = self._calculate_target_position(current_pos, cmd.action)

        # Bounds check
        if not self._is_valid_position(target_pos):
            # Record failed move
            self.history.append(StepRecord(
                step_ts=exec_ts,
                agent_id=agent_id,
                from_pos=current_pos,
                to_pos=current_pos,
                latency_ms=exec_ts - cmd.client_ts,
                collision=False
            ))
            # Check completion anyway (in case everyone started at goal)
            self._maybe_trigger_completion(exec_ts)
            return False, False

        # Collision check
        collision = self._detect_collision(agent_id, target_pos)
        final_pos = current_pos if collision else target_pos

        # Update world state
        if not collision:
            self.world_state[agent_id] = target_pos

        # Record step
        self.history.append(StepRecord(
            step_ts=exec_ts,
            agent_id=agent_id,
            from_pos=current_pos,
            to_pos=final_pos,
            latency_ms=exec_ts - cmd.client_ts,
            collision=collision
        ))

        # After state update, maybe stop the simulation
        self._maybe_trigger_completion(exec_ts)

        return (not collision), collision


    
    def get_state(self) -> Dict[int, Coord]:
        """
        Get current world state (deep copy for UI access).
        
        Returns:
            Copy of current world state
        """
        return copy.deepcopy(self.world_state)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics and statistics.
        
        Returns:
            Dictionary of metrics
        """
        if not self.history:
            return {
                "total_steps": 0,
                "avg_latency_ms": 0.0,
                "collision_count": 0,
                "collision_rate": 0.0,
                "elapsed_time_s": (time.time() - self.start_time) if self.start_time > 0 else 0.0
            }
        
        total_steps = len(self.history)
        avg_latency = sum(record.latency_ms for record in self.history) / total_steps
        collision_count = sum(1 for record in self.history if record.collision)
        collision_rate = collision_count / total_steps
        
        return {
            "total_steps": total_steps,
            "avg_latency_ms": avg_latency,
            "collision_count": collision_count,
            "collision_rate": collision_rate,
            "elapsed_time_s": (time.time() - self.start_time) if self.start_time > 0 else 0.0
        }
    
    def export_csv(self, path: str) -> None:
        """
        Export history to CSV file.
        
        Args:
            path: Output CSV file path
        """
        with open(path, 'w', newline='') as csvfile:
            if not self.history:
                return
            
            fieldnames = list(asdict(self.history[0]).keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for record in self.history:
                writer.writerow(asdict(record))
        
        self.logger.info(f"Exported {len(self.history)} records to {path}")
    
    def export_json(self, path: str) -> None:
        """
        Export history to JSON file.
        
        Args:
            path: Output JSON file path
        """
        data = {
            "metadata": {
                "grid_size": self.grid_size,
                "mode": "concurrent",
                "total_steps": len(self.history),
                "metrics": self.get_performance_metrics()
            },
            "history": [asdict(record) for record in self.history]
        }
        
        with open(path, 'w') as jsonfile:
            json.dump(data, jsonfile, indent=2)
        
        self.logger.info(f"Exported {len(self.history)} records to {path}")
    
    def create_shared_queue_for_agent(self, agent_id: int) -> asyncio.Queue:
        """为agent创建共享队列，用于NetworkBase和Agent之间的通信"""
        queue = asyncio.Queue()
        self._agent_queues = getattr(self, '_agent_queues', {})
        self._agent_queues[agent_id] = queue
        return queue

    async def start_agents(self) -> None:
        """启动所有注册的 agent，触发它们开始规划循环"""
        self.logger.info("Starting all registered agents...")
        
        # Use A2A communication if available, otherwise fall back to adapters
        agent_ids = list(self.world_state.keys()) if self._a2a_send_callback else list(self.adapters.keys())
        
        for agent_id in agent_ids:
            try:
                # 发送控制信号给 agent
                start_signal = {"type": "CONTROL", "cmd": "START", "agent_id": agent_id}
                
                if self._a2a_send_callback:
                    # Use A2A communication
                    await self._a2a_send_callback(agent_id, start_signal)
                elif agent_id in self.adapters:
                    # Fall back to direct adapter
                    adapter = self.adapters[agent_id]
                    await adapter.send(start_signal)
                    
                self.logger.info(f"✅ Sent start signal to agent {agent_id}")
            except Exception as e:
                self.logger.error(f"Failed to start agent {agent_id}: {e}")
        
        self.logger.info("All agents have been signaled to start")

    async def process_move_requests(self) -> None:
        """Process incoming move requests as they arrive (concurrent mode)"""
        self.logger.info("NetworkBase running in concurrent mode - no centralized ticks")
        self.logger.info("Agents can submit move requests anytime, conflicts resolved on-demand")
        
        # Initialize reservation tracking
        self.reservations = {}  # cell -> [(start_time, end_time, agent_id, move_id), ...]
        self.pending_moves = {}  # move_id -> move_request
        
        while self.is_running:
            # In concurrent mode, NetworkBase is passive - just waits and responds
            # All processing happens in network_executor when MOVE_REQUEST messages arrive
            
            # Periodic status reporting
            if hasattr(self, 'conflict_manager'):
                summary = self.conflict_manager.get_conflict_summary()
                if summary["active_reservations"] > 0:
                    self.logger.info(f"Conflict manager status: {summary}")
            
            await asyncio.sleep(5.0)  # Status heartbeat every 5 seconds

    async def _delayed_start_agents(self):
        """延迟启动 agent，确保所有组件都已准备就绪"""
        self.logger.info("Waiting 10 seconds for all agent services to fully start...")
        await asyncio.sleep(10)  # 等待10秒确保所有服务器都启动完成（包括LLM初始化）
        self.logger.info("Now sending START signals to all agents...")
        await self.start_agents()
    
    async def _poll_adapters(self):
        """Poll all agent adapters for incoming messages."""
        while self.is_running:
            for agent_id, adapter in self.adapters.items():
                try:
                    # Non-blocking read from adapter's queue
                    msg = adapter.recv_nowait()
                    
                    if isinstance(msg, MoveCmd):
                        # Route MoveCmd to be processed by the network
                        await self.send_move(msg)
                    elif isinstance(msg, MoveFeedback):
                        # MoveFeedback is for agents, not for network processing
                        self.logger.debug(f"MoveFeedback from agent {agent_id} (agent-internal)")
                    elif isinstance(msg, dict) and msg.get("type") in ["msg", "CHAT"]:
                        # Route P2P message to destination agent's adapter
                        if msg.get("type") == "CHAT":
                            # New unified CHAT format
                            payload = msg.get("payload", {})
                            dst_id = payload.get("dst")
                        else:
                            # Legacy msg format
                            dst_id = msg.get("dst")
                            
                        if dst_id in self.adapters:
                            await self.adapters[dst_id].send(msg)
                        else:
                            self.logger.warning(
                                f"Message from {agent_id} to unknown agent {dst_id}")
                    elif isinstance(msg, dict) and msg.get("type") == "CONTROL":
                        # Handle control messages (just log them, they are for agents)
                        cmd = msg.get("cmd", "unknown")
                        self.logger.debug(f"Control message '{cmd}' from/to agent {agent_id}")
                    else:
                        self.logger.warning(f"Unknown message type from adapter: {type(msg)}")

                except (asyncio.QueueEmpty, queue.Empty):
                    # No message from this adapter, continue to the next
                    continue
            
            # Small sleep to prevent a busy-wait loop
            await asyncio.sleep(0.001)

    async def run(self) -> None:
        """
        Main network coordinator - runs in concurrent mode.
        """
        await self._run_concurrent()
    
    async def _run_concurrent(self):
        """Fully event-driven main loop (no ticks)."""
        self.start_time = time.time()
        self.is_running = True
        self.now = int(time.time() * 1000)  # Use wall clock time

        self.logger.info(f"Starting NetworkBase with {len(self.adapters)} agents")
        self.logger.info(f"Grid size: {self.grid_size}x{self.grid_size}, Mode: concurrent")
        self.logger.info("NetworkBase running in *concurrent* mode (no ticks)")

        # Kick off polling task
        poll_task = asyncio.create_task(self._poll_adapters())

        # Delay-start agents to ensure all services are ready
        self.logger.info("Scheduling delayed start for agents (waiting for services to be ready)...")
        start_task = asyncio.create_task(self._delayed_start_agents())

        # In case all agents already start at their goals, check immediately
        try:
            self._maybe_trigger_completion(self.now)
        except Exception as e:
            self.logger.warning(f"Initial completion check failed: {e}")

        try:
            while self.is_running:
                await asyncio.sleep(5)
                metrics = self.get_performance_metrics()
                self.logger.debug(f"NetworkBase metrics: {metrics}")
        finally:
            # Cancel background tasks
            poll_task.cancel()
            start_task.cancel()
            await asyncio.gather(poll_task, start_task, return_exceptions=True)

            # Final metrics
            final_metrics = self.get_performance_metrics()
            self.logger.info("NetworkBase completed")
            self.logger.info(f"Final metrics: {final_metrics}")
            self.logger.info("NetworkBase stopped")

            # OPTIONAL HARD EXIT: stop event loop if requested by config
            if getattr(self, "hard_exit_on_complete", False):
                try:
                    loop = asyncio.get_running_loop()
                    self.logger.info("hard_exit_on_complete=True; requesting event loop stop...")
                    # Use call_soon to allow current finally-block logs to flush
                    loop.call_soon(loop.stop)
                except Exception as e:
                    self.logger.warning(f"Failed to stop event loop: {e}")


        
    def stop(self) -> None:
        """Stop the network coordinator."""
        self.is_running = False
        self.logger.info("NetworkBase stop requested") 