#!/usr/bin/env python3
"""
Fail-Storm Recovery Scenario Runner

This module implements the main runner for the Fail-Storm Recovery scenario.
It orchestrates all components including MeshNetwork, Gaia agents, fault injection,
and metrics collection to provide a comprehensive test of system resilience.

Timeline:
- t=0s:   Startup and Gaia document broadcast
- t=30s:  Normal Gaia workflow execution  
- t=60s:  Fault injection (kill 30% agents)
- t=120s: Evaluation and results output
"""

import asyncio
import json
import time
import signal
import subprocess
import sys
import os
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import tempfile
import shutil

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))
from src.core.base_agent import BaseAgent
from core.mesh_network import MeshNetwork
from core.failstorm_metrics import FailStormMetricsCollector
# Note: 直接使用 shard_qa，不再需要 gaia_shard_adapter
# from gaia_agents.gaia_shard_adapter import create_gaia_shard_executor

# Import colorama for output
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    class Fore:
        RED = GREEN = YELLOW = BLUE = CYAN = WHITE = ""
    class Style:
        BRIGHT = RESET_ALL = ""


class ColoredOutput:
    """Helper class for colored console output."""
    
    def info(self, message: str) -> None:
        print(f"{Fore.BLUE}{Style.BRIGHT}[INFO]  {message}{Style.RESET_ALL}")
    
    def success(self, message: str) -> None:
        print(f"{Fore.GREEN}{Style.BRIGHT}[OK] {message}{Style.RESET_ALL}")
    
    def warning(self, message: str) -> None:
        print(f"{Fore.YELLOW}{Style.BRIGHT}⚠️  {message}{Style.RESET_ALL}")
    
    def error(self, message: str) -> None:
        print(f"{Fore.RED}{Style.BRIGHT}[ERROR] {message}{Style.RESET_ALL}")
    
    def progress(self, message: str) -> None:
        print(f"{Fore.WHITE}   {message}{Style.RESET_ALL}")


class FailStormRunner:
    """
    Main runner for the Fail-Storm Recovery scenario.
    
    This class orchestrates the entire fail-storm test including:
    - Agent network setup with chosen protocol
    - Gaia document processing workflow
    - Controlled fault injection
    - Recovery monitoring and metrics collection
    - Results analysis and reporting
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the Fail-Storm runner.
        
        Parameters
        ----------
        config_path : str
            Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.output = ColoredOutput()
        
        # Core components
        self.mesh_network: Optional[MeshNetwork] = None
        self.metrics_collector: Optional[FailStormMetricsCollector] = None
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_pids: List[int] = []
        
        # Scenario state
        self.scenario_start_time: float = 0.0
        self.gaia_document: Dict[str, Any] = {}
        self.workspace_dir: Path = Path("workspaces")
        self.results_dir: Path = Path("results")
        
        # Fault injection
        self.fault_injection_process: Optional[subprocess.Popen] = None
        self.killed_agents: Set[str] = set()
        
        # Timing control
        self.phase_timers: Dict[str, float] = {}
        
        # Global shutdown handler
        self.shutdown_event = asyncio.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file."""
        config_file = Path(__file__).parent / config_path
        
        # Default configuration if file doesn't exist
        default_config = {
            "scenario": {
                "protocol": "simple_json",
                "agent_count": 8,
                "kill_fraction": 0.3,
                "fault_injection_time": 60.0,
                "total_runtime": 120.0,
                "heartbeat_interval": 3.0,
                "heartbeat_timeout": 30.0
            },
            "gaia": {
                "document_path": "docs/gaia_document.txt",
                "tools": ["search", "extract", "triple", "reason"],
                "workflow_cycles": 5
            },
            "agents": {
                "base_port": 9000,
                "host": "127.0.0.1"
            },
            "llm": {
                "type": "nvidia",
                "model": "nvdev/nvidia/llama-3.1-nemotron-70b-instruct",
                "nvidia_api_key": "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",
                "nvidia_base_url": "https://integrate.api.nvidia.com/v1",
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 8192,
                "name": "nvdev/nvidia/llama-3.1-nemotron-70b-instruct",
                "openai_api_key": "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"
            },
            "output": {
                "results_file": "failstorm_metrics.json",
                "logs_dir": "logs",
                "artifacts_dir": "artifacts"
            }
        }
        
        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                
                # Merge with defaults
                def merge_configs(default, loaded):
                    for key, value in loaded.items():
                        if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                            merge_configs(default[key], value)
                        else:
                            default[key] = value
                    return default
                
                return merge_configs(default_config, loaded_config)
                
            except Exception as e:
                print(f"Warning: Failed to load config {config_path}: {e}")
                return default_config
        else:
            print(f"Warning: Config file {config_path} not found, using defaults")
            return default_config

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.output.warning("Received shutdown signal, initiating graceful cleanup...")
        self.shutdown_event.set()

    async def run_scenario(self) -> Dict[str, Any]:
        """
        Execute the complete Fail-Storm recovery scenario.
        
        Returns
        -------
        Dict[str, Any]
            Complete scenario results including metrics
        """
        try:
            self.output.info("🚀 Starting Fail-Storm Recovery Scenario")
            self.scenario_start_time = time.time()
            
            # Phase 0: Setup (t=0s)
            self.output.info("📋 Phase 0: Initializing scenario...")
            await self._setup_scenario()
            
            if self.shutdown_event.is_set():
                return await self._cleanup_and_exit("Setup interrupted")
            
            # Phase 1: Normal Operation (t=0s - t=60s)
            self.output.info("⚡ Phase 1: Normal Gaia workflow execution...")
            await self._execute_normal_phase()
            
            if self.shutdown_event.is_set():
                return await self._cleanup_and_exit("Normal phase interrupted")
            
            # Phase 2: Fault Injection (t=60s)
            self.output.info("💥 Phase 2: Fault injection...")
            await self._execute_fault_injection()
            
            if self.shutdown_event.is_set():
                return await self._cleanup_and_exit("Fault injection interrupted")
            
            # Phase 3: Recovery Monitoring (t=60s - t=120s)
            self.output.info("🔄 Phase 3: Recovery monitoring...")
            await self._monitor_recovery()
            
            # Phase 4: Evaluation and Results (t=120s)
            self.output.info("📊 Phase 4: Evaluation and results...")
            results = await self._finalize_scenario()
            
            self.output.success("✅ Fail-Storm scenario completed successfully!")
            return results
            
        except Exception as e:
            self.output.error(f"Scenario execution failed: {e}")
            return await self._cleanup_and_exit(f"Error: {e}")
        
        finally:
            await self._cleanup_resources()

    async def _setup_scenario(self) -> None:
        """Setup the scenario components."""
        self.output.progress("Setting up workspace directories...")
        self._setup_directories()
        
        self.output.progress("Loading Gaia document...")
        self.gaia_document = self._load_gaia_document()
        
        self.output.progress("Initializing metrics collector...")
        protocol_name = self.config["scenario"]["protocol"]
        self.metrics_collector = FailStormMetricsCollector(protocol_name)
        
        self.output.progress("Creating mesh network...")
        heartbeat_interval = self.config["scenario"]["heartbeat_interval"]
        self.mesh_network = MeshNetwork(heartbeat_interval=heartbeat_interval)
        
        self.output.progress("Setting up Gaia agents...")
        await self._setup_gaia_agents()
        
        self.output.progress("Establishing mesh topology...")
        await self._setup_mesh_topology()
        
        self.output.progress("Broadcasting Gaia document...")
        await self._broadcast_gaia_document()
        
        self.phase_timers["setup_completed"] = time.time()
        self.output.success(f"Setup completed in {self.phase_timers['setup_completed'] - self.scenario_start_time:.2f}s")

    def _setup_directories(self) -> None:
        """Setup workspace and results directories."""
        self.workspace_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create individual agent workspaces
        agent_count = self.config["scenario"]["agent_count"]
        for i in range(agent_count):
            agent_workspace = self.workspace_dir / f"agent_{i}"
            agent_workspace.mkdir(exist_ok=True)

    def _load_gaia_document(self) -> Dict[str, Any]:
        """Load the Gaia document for processing."""
        doc_path = Path(__file__).parent / self.config["gaia"]["document_path"]
        
        if doc_path.exists():
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                document = {
                    "title": "Gaia Test Document",
                    "content": content,
                    "metadata": {
                        "source": str(doc_path),
                        "size": len(content),
                        "load_time": time.time()
                    }
                }
                print(f"[DEBUG] Loaded Gaia document: title='{document['title']}', content_length={len(content)}")
                return document
            except Exception as e:
                self.output.warning(f"Failed to load Gaia document: {e}")
        
        # No fallback - the document must exist
        self.output.error(f"Gaia document not found at {doc_path}")
        raise FileNotFoundError(f"Required Gaia document not found: {doc_path}")

    async def _setup_gaia_agents(self) -> None:
        """Setup the four types of Gaia agents."""
        agent_count = self.config["scenario"]["agent_count"]
        base_port = self.config["agents"]["base_port"]
        host = self.config["agents"]["host"]
        protocol = self.config["scenario"]["protocol"]
        
        # Prepare LLM configuration with proper field mapping
        llm_base_config = self.config["llm"].copy()
        
        # Handle API key resolution based on LLM type
        import os
        llm_type = llm_base_config.get("type", "nvidia")
        
        if llm_type == "nvidia":
            # Handle NVIDIA API key
            api_key = llm_base_config.get("nvidia_api_key", "")
            if api_key.startswith("${") and api_key.endswith("}"):
                # Environment variable reference
                env_var = api_key[2:-1]
                resolved_api_key = os.getenv(env_var, "")
            else:
                resolved_api_key = api_key
            
            # Map for compatibility with agent_executor
            llm_base_config["openai_api_key"] = resolved_api_key
            
            # Check if valid NVIDIA API key
            if not resolved_api_key or resolved_api_key == "your_nvidia_api_key_here":
                self.output.warning("No valid NVIDIA API key found, using mock responses")
                llm_base_config["type"] = "mock"
        else:
            # Handle other API types (OpenAI, etc.)
            api_key = llm_base_config.get("api_key", "")
            if api_key.startswith("${") and api_key.endswith("}"):
                # Environment variable reference
                env_var = api_key[2:-1]
                resolved_api_key = os.getenv(env_var, "")
            else:
                resolved_api_key = api_key
            
            # Map api_key to openai_api_key for compatibility with agent_executor
            llm_base_config["openai_api_key"] = resolved_api_key
            
            # If no valid API key, enable mock mode
            if not resolved_api_key or resolved_api_key == "your_openai_api_key_here":
                self.output.warning("No valid API key found, using mock responses")
                llm_base_config["type"] = "mock"
        
        # Add name field if missing (Core class expects this)
        if "name" not in llm_base_config:
            default_model = "nvdev/nvidia/llama-3.1-nemotron-70b-instruct" if llm_type == "nvidia" else "gpt-4o"
            llm_base_config["name"] = llm_base_config.get("model", default_model)
        
        llm_config = {
            "model": llm_base_config
        }
        
        # Create agents with different tools
        tool_types = self.config["gaia"]["tools"]
        
        for i in range(agent_count):
            agent_id = f"gaia_agent_{i}"
            port = base_port + i
            tool_type = tool_types[i % len(tool_types)]
            
            # Create workspace for agent
            agent_workspace = self.workspace_dir / f"agent_{i}"
            
            # Use real ShardWorkerExecutor instead of dummy
            # Import from shard_qa
            import sys
            from pathlib import Path
            shard_qa_path = Path(__file__).parent.parent / "shard_qa"
            sys.path.insert(0, str(shard_qa_path))
            
            from shard_worker.agent_executor import ShardWorkerExecutor
            
            # Create proper config for the worker with correct structure
            worker_config = llm_config.copy()  # This contains {"model": llm_base_config}
            worker_config.update({
                "tool_type": tool_type,
                "agent_id": agent_id
            })
            
            # Create fake data file for testing
            import tempfile
            import json
            test_data = [{
                "group_id": i,
                "question": f"Test question {i} for {tool_type}",
                "answer": f"Test answer {i}",
                "snippet": f"This is test data for {tool_type} tool processing."
            } for i in range(3)]
            
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
            for item in test_data:
                temp_file.write(json.dumps(item) + '\n')
            temp_file.flush()
            temp_file.close()
            
            # Define neighbors (for ring topology simulation)
            neighbors = {
                "prev_id": f"gaia_agent_{(i-1) % agent_count}",
                "next_id": f"gaia_agent_{(i+1) % agent_count}"
            }
            
            executor = ShardWorkerExecutor(
                config=worker_config,
                global_config=self.config,
                shard_id=f"shard{i}",
                data_file=temp_file.name,
                neighbors=neighbors,
                output=self.output
            )
            
            # Create agent based on protocol
            if protocol.lower() == "a2a":
                agent = await BaseAgent.create_a2a(
                    agent_id=agent_id,
                    host=host,
                    port=port,
                    executor=executor
                )
            elif protocol.lower() == "anp":
                agent = await BaseAgent.create_anp(
                    agent_id=agent_id,
                    host=host,
                    port=port,
                    executor=executor
                )
            elif protocol.lower() == "acp":
                agent = await BaseAgent.create_acp(
                    agent_id=agent_id,
                    host=host,
                    port=port,
                    executor=executor
                )
            elif protocol.lower() == "simple_json":
                agent = await BaseAgent.create_simple_json(
                    agent_id=agent_id,
                    host=host,
                    port=port,
                    executor=executor
                )
            else:
                raise ValueError(f"Unsupported protocol: {protocol}")
            
            # Register with network
            await self.mesh_network.register_agent(agent)
            self.agents[agent_id] = agent
            
            # Track PID for fault injection
            self.agent_pids.append(os.getpid())  # In real implementation, track actual agent PIDs
            
            self.output.progress(f"Created {tool_type} agent: {agent_id} on port {port}")
        
        self.output.success(f"Created {len(self.agents)} Gaia agents with {protocol.upper()} protocol")

    async def _setup_mesh_topology(self) -> None:
        """Setup full mesh topology between agents."""
        await self.mesh_network.setup_mesh_topology()
        
        # Wait for topology to stabilize
        await asyncio.sleep(2.0)
        
        # Verify connectivity
        topology = self.mesh_network.get_topology()
        expected_connections = len(self.agents) * (len(self.agents) - 1)
        actual_connections = sum(len(edges) for edges in topology.values())
        
        self.output.success(f"Mesh topology established: {actual_connections}/{expected_connections} connections")

    async def _broadcast_gaia_document(self) -> None:
        """Broadcast the Gaia document to all agents."""
        if not self.agents:
            raise RuntimeError("No agents available for broadcast")
        
        # Use first agent as broadcaster
        broadcaster_id = list(self.agents.keys())[0]
        
        results = await self.mesh_network.broadcast_init(self.gaia_document, broadcaster_id)
        
        successful_deliveries = sum(1 for result in results.values() if "error" not in str(result))
        total_targets = len(results)
        
        self.output.success(f"Gaia document broadcast: {successful_deliveries}/{total_targets} deliveries successful")

    async def _execute_normal_phase(self) -> None:
        """Execute normal Gaia workflow for the first phase."""
        normal_phase_duration = self.config["scenario"]["fault_injection_time"]
        workflow_cycles = self.config["gaia"]["workflow_cycles"]
        
        self.output.progress(f"Running normal workflow for {normal_phase_duration}s...")
        
        # Start workflow cycles
        cycle_duration = normal_phase_duration / workflow_cycles
        
        for cycle in range(workflow_cycles):
            if self.shutdown_event.is_set():
                break
            
            self.output.progress(f"Workflow cycle {cycle + 1}/{workflow_cycles}")
            
            # Execute workflow: search -> extract -> triple -> reason
            await self._execute_workflow_cycle(cycle)
            
            # Wait between cycles
            if cycle < workflow_cycles - 1:
                await asyncio.sleep(cycle_duration)
        
        self.phase_timers["normal_phase_completed"] = time.time()
        elapsed = self.phase_timers["normal_phase_completed"] - self.scenario_start_time
        self.output.success(f"Normal phase completed in {elapsed:.2f}s")

    async def _execute_workflow_cycle(self, cycle: int) -> None:
        """Execute one cycle of the Gaia workflow."""
        # Simulate inter-agent collaboration for Gaia tasks
        agent_ids = list(self.agents.keys())
        
        # Each tool type processes the document
        for i, tool_type in enumerate(["search", "extract", "triple", "reason"]):
            # Find agents with this tool type
            tool_agents = [
                agent_id for agent_id in agent_ids 
                if f"_{i % len(self.config['gaia']['tools'])}" in agent_id or tool_type in agent_id
            ]
            
            if tool_agents:
                # Send task to one of the tool agents
                target_agent = tool_agents[0]
                source_agent = agent_ids[0] if agent_ids else target_agent
                
                # Create appropriate message format based on protocol
                task_content = f"Process Gaia document with {tool_type} tool (cycle {cycle})"
                protocol = self.config["scenario"]["protocol"].lower()
                
                if protocol == "simple_json":
                    # Simple JSON format
                    task_payload = {
                        "type": "task_request",
                        "tool": tool_type,
                        "cycle": cycle,
                        "text": task_content,
                        "timestamp": time.time(),
                        "from": source_agent
                    }
                else:
                    # A2A format for compatibility
                    import uuid
                    task_payload = {
                        "message": {
                            "messageId": str(uuid.uuid4()),
                            "role": "user",
                            "parts": [
                                {
                                    "type": "text",
                                    "text": task_content
                                }
                            ]
                        },
                        "context": {
                            "tool_type": tool_type,
                            "cycle": cycle,
                            "timestamp": time.time()
                        },
                        "source": source_agent
                    }
                
                try:
                    # Record task start
                    task_id = f"cycle_{cycle}_{tool_type}"
                    if self.metrics_collector:
                        self.metrics_collector.start_task_execution(task_id, target_agent, tool_type)
                    
                    # Handle self-execution vs message sending
                    if source_agent == target_agent:
                        # Direct local execution for self-tasks
                        result = await self._execute_local_task(target_agent, task_payload)
                    else:
                        # Send message to different agent
                        result = await self.mesh_network.route_message(source_agent, target_agent, task_payload)
                    
                    # Record completion
                    if self.metrics_collector:
                        success = "error" not in str(result).lower()
                        result_size = len(str(result).encode('utf-8'))
                        self.metrics_collector.complete_task_execution(task_id, success, result_size)
                    
                except Exception as e:
                    self.output.warning(f"Workflow cycle {cycle} {tool_type} failed: {e}")
                    
                    if self.metrics_collector:
                        self.metrics_collector.complete_task_execution(task_id, False, error=str(e))

    async def _execute_local_task(self, agent_id: str, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task locally on the target agent without network routing."""
        try:
            # Get the agent instance
            if agent_id not in self.agents:
                raise KeyError(f"Agent {agent_id} not found")
            
            agent = self.agents[agent_id]
            
            # Use the agent's send method to itself (which should handle local processing)
            # Create a local endpoint URL for direct HTTP calls
            agent_url = f"http://{agent.host}:{agent.port}/message"
            
            # Send HTTP request directly to the agent
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    agent_url,
                    json=task_payload,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {"success": True, "result": result}
                else:
                    return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                    
        except Exception as e:
            return {"success": False, "error": f"Local execution failed: {e}"}

    async def _execute_fault_injection(self) -> None:
        """Execute the fault injection at t=60s."""
        fault_time = self.config["scenario"]["fault_injection_time"]
        elapsed = time.time() - self.scenario_start_time
        
        if elapsed < fault_time:
            wait_time = fault_time - elapsed
            self.output.progress(f"Waiting {wait_time:.1f}s until fault injection time...")
            await asyncio.sleep(wait_time)
        
        # Record fault injection time
        fault_injection_time = time.time()
        if self.metrics_collector:
            self.metrics_collector.set_fault_injection_time(fault_injection_time)
        
        if self.mesh_network:
            self.mesh_network.set_fault_injection_time(fault_injection_time)
        
        # Execute fault injection
        kill_fraction = self.config["scenario"]["kill_fraction"]
        await self._inject_faults(kill_fraction)
        
        self.phase_timers["fault_injection_completed"] = time.time()
        self.output.warning(f"Fault injection completed at t={fault_injection_time - self.scenario_start_time:.1f}s")

    async def _inject_faults(self, kill_fraction: float) -> None:
        """Inject faults by killing random agents."""
        agent_ids = list(self.agents.keys())
        num_victims = max(1, int(len(agent_ids) * kill_fraction))
        
        import random
        victims = random.sample(agent_ids, num_victims)
        
        self.output.warning(f"Killing {len(victims)} agents: {', '.join(victims)}")
        
        # Kill agents
        killed_agents = set()
        for agent_id in victims:
            try:
                # Stop the agent
                agent = self.agents[agent_id]
                await agent.stop()
                
                # Remove from network
                await self.mesh_network.unregister_agent(agent_id)
                
                # Remove from local tracking
                del self.agents[agent_id]
                killed_agents.add(agent_id)
                
                # Update agent state in metrics
                if self.metrics_collector:
                    self.metrics_collector.update_agent_state(agent_id, "failed")
                
                self.output.progress(f"✗ Killed agent: {agent_id}")
                
            except Exception as e:
                self.output.error(f"Failed to kill agent {agent_id}: {e}")
        
        self.killed_agents = killed_agents
        
        # Trigger network recovery
        if self.mesh_network:
            for agent_id in killed_agents:
                await self.mesh_network._handle_node_failure(agent_id)

    async def _monitor_recovery(self) -> None:
        """Monitor the recovery process after fault injection."""
        total_runtime = self.config["scenario"]["total_runtime"]
        fault_time = self.config["scenario"]["fault_injection_time"]
        recovery_duration = total_runtime - fault_time
        
        recovery_start = time.time()
        recovery_timeout = recovery_start + recovery_duration
        
        self.output.progress(f"Monitoring recovery for {recovery_duration}s...")
        
        # Monitor for recovery signs
        first_recovery_detected = False
        
        while time.time() < recovery_timeout and not self.shutdown_event.is_set():
            # Check for recovery indicators
            if not first_recovery_detected:
                # Check if any failed agents are showing signs of recovery
                # or if remaining agents are re-establishing connections
                
                topology_health = self.mesh_network.get_topology_health() if self.mesh_network else {}
                alive_agents = topology_health.get("alive_agents", [])
                
                if len(alive_agents) > 0:
                    # Check connectivity
                    avg_connectivity = sum(
                        status.get("connectivity_ratio", 0) 
                        for status in topology_health.get("connectivity_status", {}).values()
                    ) / max(len(alive_agents), 1)
                    
                    if avg_connectivity > 0.7:  # 70% connectivity restored
                        if self.metrics_collector and not first_recovery_detected:
                            self.metrics_collector.set_first_recovery_time()
                            first_recovery_detected = True
                            self.output.success("🔄 First recovery signs detected!")
            
            # Continue workflow with remaining agents
            if len(self.agents) > 0:
                try:
                    await self._execute_recovery_workflow()
                except Exception as e:
                    self.output.warning(f"Recovery workflow error: {e}")
            
            await asyncio.sleep(2.0)  # Check every 2 seconds
        
        # Wait for steady state
        if self.mesh_network:
            try:
                steady_time = await self.mesh_network.wait_for_steady_state(min_stability_time=5.0)
                if self.metrics_collector:
                    self.metrics_collector.set_steady_state_time()
                self.output.success(f"🎯 Steady state reached in {steady_time:.2f}s")
            except asyncio.TimeoutError:
                self.output.warning("Steady state not reached within timeout")
        
        self.phase_timers["recovery_completed"] = time.time()

    async def _execute_recovery_workflow(self) -> None:
        """Execute simplified workflow with remaining agents during recovery."""
        if not self.agents:
            return
        
        # Try to execute a simple task with remaining agents
        agent_ids = list(self.agents.keys())
        if len(agent_ids) >= 2:
            source = agent_ids[0]
            target = agent_ids[1]
            
            recovery_task = {
                "type": "recovery_test",
                "message": "Testing connectivity during recovery"
            }
            
            try:
                await self.mesh_network.route_message(source, target, recovery_task)
                
                # Record recovery activity
                if self.metrics_collector:
                    self.metrics_collector.record_network_event(
                        event_type="recovery_message",
                        source_agent=source,
                        target_agent=target,
                        bytes_transferred=len(str(recovery_task).encode('utf-8'))
                    )
                    
            except Exception:
                pass  # Expected during recovery

    async def _finalize_scenario(self) -> Dict[str, Any]:
        """Finalize the scenario and generate results."""
        end_time = time.time()
        total_runtime = end_time - self.scenario_start_time
        
        self.output.progress("Collecting final metrics...")
        
        # Generate comprehensive results
        results = {
            "metadata": {
                "scenario": "fail_storm_recovery",
                "protocol": self.config["scenario"]["protocol"],
                "start_time": self.scenario_start_time,
                "end_time": end_time,
                "total_runtime": total_runtime,
                "config": self.config
            },
            "agent_summary": {
                "initial_count": self.config["scenario"]["agent_count"],
                "killed_count": len(self.killed_agents),
                "surviving_count": len(self.agents),
                "killed_agents": list(self.killed_agents),
                "surviving_agents": list(self.agents.keys())
            },
            "timing": {
                "setup_time": self.phase_timers.get("setup_completed", 0) - self.scenario_start_time,
                "normal_phase_time": self.phase_timers.get("normal_phase_completed", 0) - self.scenario_start_time,
                "fault_injection_time": self.phase_timers.get("fault_injection_completed", 0) - self.scenario_start_time,
                "recovery_time": self.phase_timers.get("recovery_completed", 0) - self.scenario_start_time
            },
            "network_metrics": {},
            "failstorm_metrics": {}
        }
        
        # Collect network metrics
        if self.mesh_network:
            results["network_metrics"] = self.mesh_network.get_failure_metrics()
        
        # Collect failstorm metrics
        if self.metrics_collector:
            results["failstorm_metrics"] = self.metrics_collector.calculate_recovery_metrics()
            results["performance_analysis"] = self.metrics_collector.get_performance_summary()
        
        # Save results
        await self._save_results(results)
        
        return results

    async def _save_results(self, results: Dict[str, Any]) -> None:
        """Save scenario results to files."""
        # Save main results file
        results_file = self.results_dir / self.config["output"]["results_file"]
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.output.success(f"📁 Results saved to: {results_file}")
            
        except Exception as e:
            self.output.error(f"Failed to save results: {e}")
        
        # Save detailed metrics if available
        if self.metrics_collector:
            detailed_metrics_file = self.results_dir / "detailed_failstorm_metrics.json"
            try:
                self.metrics_collector.export_to_json(str(detailed_metrics_file))
            except Exception as e:
                self.output.error(f"Failed to save detailed metrics: {e}")

    async def _cleanup_and_exit(self, reason: str) -> Dict[str, Any]:
        """Cleanup and exit with error results."""
        self.output.error(f"Scenario terminated: {reason}")
        
        # Generate minimal results
        results = {
            "metadata": {
                "scenario": "fail_storm_recovery",
                "status": "terminated",
                "reason": reason,
                "end_time": time.time()
            },
            "partial_results": self.metrics_collector.get_real_time_summary() if self.metrics_collector else {}
        }
        
        await self._save_results(results)
        return results

    async def _cleanup_resources(self) -> None:
        """Clean up all resources."""
        self.output.progress("Cleaning up resources...")
        
        # Stop fault injection process
        if self.fault_injection_process:
            try:
                self.fault_injection_process.terminate()
                self.fault_injection_process.wait(timeout=5)
            except:
                pass
        
        # Stop all agents
        cleanup_tasks = []
        for agent in self.agents.values():
            cleanup_tasks.append(agent.stop())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Cleanup network
        if self.mesh_network:
            await self.mesh_network.cleanup()
        
        # Clear state
        self.agents.clear()
        self.agent_pids.clear()
        
        self.output.success("Cleanup completed")


async def main():
    """Main entry point for the Fail-Storm scenario."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fail-Storm Recovery Scenario Runner")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path")
    parser.add_argument("--protocol", "-p", choices=["a2a", "anp", "acp", "simple_json"], help="Protocol to test")
    parser.add_argument("--agents", "-n", type=int, help="Number of agents")
    parser.add_argument("--kill-fraction", "-k", type=float, help="Fraction of agents to kill")
    parser.add_argument("--runtime", "-r", type=float, help="Total runtime in seconds")
    parser.add_argument("--output", "-o", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create runner
    runner = FailStormRunner(args.config)
    
    # Override config with command line arguments
    if args.protocol:
        runner.config["scenario"]["protocol"] = args.protocol
    if args.agents:
        runner.config["scenario"]["agent_count"] = args.agents
    if args.kill_fraction:
        runner.config["scenario"]["kill_fraction"] = args.kill_fraction
    if args.runtime:
        runner.config["scenario"]["total_runtime"] = args.runtime
    if args.output:
        runner.results_dir = Path(args.output)
        runner.results_dir.mkdir(exist_ok=True)
    
    # Validate and adjust timing parameters
    total_runtime = runner.config["scenario"]["total_runtime"]
    fault_injection_time = runner.config["scenario"]["fault_injection_time"]
    
    if fault_injection_time >= total_runtime:
        # Auto-adjust fault injection time to be 60% of total runtime
        adjusted_fault_time = total_runtime * 0.6
        runner.config["scenario"]["fault_injection_time"] = adjusted_fault_time
        print(f"⚠️  Auto-adjusted fault injection time: {fault_injection_time}s → {adjusted_fault_time}s (60% of total runtime)")
    
    # Run scenario
    try:
        results = await runner.run_scenario()
        
        # Print summary
        print("\n" + "="*60)
        print("FAIL-STORM SCENARIO SUMMARY")
        print("="*60)
        
        if "failstorm_metrics" in results:
            metrics = results["failstorm_metrics"]
            print(f"Recovery Time: {metrics.get('recovery_ms', 'N/A')} ms")
            print(f"Steady State: {metrics.get('steady_state_ms', 'N/A')} ms")
            print(f"Success Rate Drop: {metrics.get('success_rate_drop', 'N/A'):.1%}")
            print(f"Duplicate Work: {metrics.get('duplicate_work_ratio', 'N/A'):.1%}")
            print(f"Reconnect Bytes: {metrics.get('bytes_reconnect', 'N/A')}")
        
        if "agent_summary" in results:
            summary = results["agent_summary"]
            print(f"Agents Killed: {summary['killed_count']}/{summary['initial_count']}")
            print(f"Survivors: {summary['surviving_count']}")
        
        print(f"Protocol: {results['metadata']['protocol'].upper()}")
        print(f"Total Runtime: {results['metadata']['total_runtime']:.1f}s")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nScenario interrupted by user")
        return 1
    except Exception as e:
        print(f"\nScenario failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)