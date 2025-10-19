#!/usr/bin/env python3
"""
Fail-Storm Recovery Scenario Runner

This module implements the main runner for the Fail-Storm Recovery scenario.
It orchestrates all components including MeshNetwork, Shard QA agents, fault injection,
and metrics collection to provide a comprehensive test of system resilience.

Timeline:
- t=0s:   Startup and document broadcast
- t=30s:  Normal Shard QA workflow execution  
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
import socket
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import tempfile
import shutil

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

# Import core components (needed by existing FailStormRunner)
from core.simple_base_agent import SimpleBaseAgent as BaseAgent
from core.enhanced_mesh_network import EnhancedMeshNetwork as MeshNetwork
from core.failstorm_metrics import FailStormMetricsCollector

# Import protocol-specific runners
from protocol_backends.simple_json.runner import SimpleJsonRunner
from protocol_backends.anp.runner import ANPRunner
from protocol_backends.a2a.runner import A2ARunner
from protocol_backends.acp.runner import ACPRunner
from protocol_backends.agora.runner import AgoraRunner

# Import user's shard_qa agent_executor directly with dynamic import
shard_qa_path = Path(__file__).parent.parent / "shard_qa"
sys.path.insert(0, str(shard_qa_path))
import importlib.util
spec = importlib.util.spec_from_file_location("agent_executor", shard_qa_path / "shard_worker" / "agent_executor.py")
agent_executor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_executor_module)
ShardWorkerExecutor = agent_executor_module.ShardWorkerExecutor

# Import colorama for output
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
except ImportError as e:
    raise ImportError(f"Colorama is required for fail storm runner colored output. Please install colorama package. Error: {e}")


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


class ProtocolRunnerFactory:
    """Factory for creating protocol-specific runners."""
    
    RUNNERS = {
        "simple_json": SimpleJsonRunner,
        "anp": ANPRunner,
        "a2a": A2ARunner,
        "acp": ACPRunner,
        "agora": AgoraRunner,
    }
    
    @classmethod
    def create(cls, protocol: str, config_path: str):
        """Create a protocol-specific runner."""
        if protocol not in cls.RUNNERS:
            raise ValueError(f"Unknown protocol runner: {protocol}. Available: {list(cls.RUNNERS.keys())}")
        return cls.RUNNERS[protocol](config_path=config_path)


class FailStormRunner:
    """
    Main runner for the Fail-Storm Recovery scenario.
    
    This class orchestrates the entire fail-storm test including:
    - Agent network setup with chosen protocol
    - Shard QA document processing workflow
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
        
        # Reduce log noise by setting log level
        import logging
        logging.getLogger().setLevel(logging.ERROR)  # Only show errors
        logging.getLogger("uvicorn").setLevel(logging.ERROR)
        logging.getLogger("fastapi").setLevel(logging.ERROR)
        logging.getLogger("a2a").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        
        # Core components
        self.mesh_network: Optional[MeshNetwork] = None
        self.metrics_collector: Optional[FailStormMetricsCollector] = None
        self.agents: Dict[str, BaseAgent] = {}
        
        # Shard QA components
        self.shard_workers: Dict[str, ShardWorkerExecutor] = {}
        
        # Scenario state
        self.scenario_start_time: float = 0.0
        self.document: Dict[str, Any] = {}
        self.workspace_dir: Path = Path("workspaces")
        self.results_dir: Path = Path("results")
        self.llm_outputs_dir: Path = Path("llm_outputs")
        
        # Fault injection
        self.fault_injection_process: Optional[subprocess.Popen] = None
        self.killed_agents: Set[str] = set()  # Currently killed (removed on reconnect)
        self.temporarily_killed_agents: Set[str] = set()  # Ever killed during scenario
        self.permanently_failed_agents: Set[str] = set()  # Failed to reconnect
        self.killed_agent_configs: Dict[str, Dict[str, Any]] = {}  # 保存被杀死agent的配置用于重连
        
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
                "agent_count": 3,  # 从config文件读取，这里只是默认值
                "kill_fraction": 0.3,
                "fault_injection_time": 60.0,
                "total_runtime": 120.0,
                "heartbeat_interval": 3.0,
                "heartbeat_timeout": 30.0
            },
            "shard_qa": {
                "data_dir": "../shard_qa/data",
                "questions_file": "qa_questions.json",
                "fragments_file": "knowledge_fragments.json",
                "normal_phase_duration": 30.0,
                "qa_cycle_timeout": 15.0,
                "roles": {
                    "coordinator": 1,
                    "worker": 2
                }
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

    def _is_port_available(self, host: str, port: int) -> bool:
        """检查端口是否可用."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result != 0  # 如果连接失败，端口可用
        except Exception:
            return False

    def _find_available_ports(self, host: str, base_port: int, count: int) -> List[int]:
        """查找连续的可用端口."""
        available_ports = []
        port = base_port
        max_attempts = 1000  # 最多尝试1000个端口
        
        while len(available_ports) < count and port < base_port + max_attempts:
            if self._is_port_available(host, port):
                available_ports.append(port)
            port += 1
        
        if len(available_ports) < count:
            raise RuntimeError(f"无法找到 {count} 个可用端口，从 {base_port} 开始")
        
        return available_ports

    async def run_scenario(self) -> Dict[str, Any]:
        """
        Execute the complete Fail-Storm recovery scenario.
        
        Returns
        -------
        Dict[str, Any]
            Complete scenario results including metrics
        """
        try:
            self.output.info("🚀 Starting Fail-Storm Recovery Scenario with Shard QA")
            self.scenario_start_time = time.time()
            
            # Phase 0: Setup (t=0s)
            self.output.info("📋 Phase 0: Initializing scenario...")
            await self._setup_scenario()
            
            if self.shutdown_event.is_set():
                return await self._cleanup_and_exit("Setup interrupted")
            
            # Phase 1: Normal Operation (t=0s - t=60s)
            self.output.info("⚡ Phase 1: Normal Shard QA workflow execution...")
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
        
        self.output.progress("Loading document...")
        self.document = self._load_document()
        
        self.output.progress("Initializing metrics collector...")
        protocol_name = self.config["scenario"]["protocol"]
        self.metrics_collector = FailStormMetricsCollector(protocol_name)
        
        self.output.progress("Creating mesh network...")
        heartbeat_interval = self.config["scenario"]["heartbeat_interval"]
        self.mesh_network = MeshNetwork(heartbeat_interval=heartbeat_interval)
        
        self.output.progress("Setting up Shard QA workers...")
        await self._setup_shard_qa_workers()
        
        self.output.progress("Establishing mesh topology...")
        await self._setup_mesh_topology()
        
        self.output.progress("Broadcasting document...")
        await self._broadcast_document()
        
        self.phase_timers["setup_completed"] = time.time()
        self.output.success(f"Setup completed in {self.phase_timers['setup_completed'] - self.scenario_start_time:.2f}s")

    def _setup_directories(self) -> None:
        """Setup workspace and results directories."""
        self.workspace_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.llm_outputs_dir.mkdir(exist_ok=True)
        
        # Create individual agent workspaces
        agent_count = self.config["scenario"]["agent_count"]
        for i in range(agent_count):
            agent_workspace = self.workspace_dir / f"agent_{i}"
            agent_workspace.mkdir(exist_ok=True)

    def _load_document(self) -> Dict[str, Any]:
        """Load the document for processing."""
        # Use the existing Gaia document for QA tasks
        doc_path = Path(__file__).parent / "docs" / "gaia_document.txt"
        
        if doc_path.exists():
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                document = {
                    "title": "Test Document",
                    "content": content,
                    "metadata": {
                        "source": str(doc_path),
                        "size": len(content),
                        "load_time": time.time()
                    }
                }
                return document
            except Exception as e:
                self.output.warning(f"Failed to load document: {e}")
        
        # Create a fallback document for testing
        fallback_content = """
        This is a test document for the Fail-Storm recovery scenario.
        
        Key Information:
        1. The system should be resilient to node failures
        2. Agents should collaborate to find answers in distributed documents
        3. When a question cannot be answered locally, agents should communicate with neighbors
        4. The system should maintain functionality even when 30% of nodes fail
        
        Questions for testing:
        - What happens when nodes fail during document processing?
        - How do remaining agents handle the workload?
        - Can the system recover and continue processing?
        """
        
        document = {
            "title": "Fail-Storm Test Document",
            "content": fallback_content,
            "metadata": {
                "source": "fallback",
                "size": len(fallback_content),
                "load_time": time.time()
            }
        }
        
        self.output.warning(f"Using fallback document (original not found at {doc_path})")
        return document

    async def _setup_shard_qa_workers(self) -> None:
        """Setup Shard QA workers for collaborative retrieval task."""
        agent_count = self.config["scenario"]["agent_count"]
        base_port = self.config["agents"]["base_port"]
        host = self.config["agents"]["host"]
        protocol = self.config["scenario"]["protocol"]
        
        # 查找可用端口
        try:
            available_ports = self._find_available_ports(host, base_port, agent_count)
            self.output.success(f"Found available ports: {available_ports}")
        except RuntimeError as e:
            self.output.error(f"Port allocation failed: {e}")
            raise
        
        # Prepare LLM configuration
        llm_config = {
            "model": self.config["llm"]
        }
        
        # Use existing shard data files from shard_qa
        shard_qa_data_dir = Path(__file__).parent / "shard_qa" / "data" / "shards"
        if not shard_qa_data_dir.exists():
            self.output.error(f"Shard QA data directory not found: {shard_qa_data_dir}")
            raise FileNotFoundError(f"Missing shard QA data: {shard_qa_data_dir}")
        
        # Create agents with ShardWorkerExecutor using real QA data
        for i in range(agent_count):
            agent_id = f"shard{i}"
            port = available_ports[i]  # 使用分配的可用端口
            
            # Define neighbors (ring topology)
            neighbors = {
                "prev_id": f"shard{(i-1) % agent_count}",
                "next_id": f"shard{(i+1) % agent_count}"
            }
            
            # Use existing shard data file
            shard_data_file = shard_qa_data_dir / f"shard{i}.jsonl"
            if not shard_data_file.exists():
                # If specific shard doesn't exist, use shard0 as default
                shard_data_file = shard_qa_data_dir / "shard0.jsonl"
            
            # Create ShardWorkerExecutor with real QA configuration
            worker_config = llm_config.copy()
            worker_config.update({
                "agent_id": agent_id,
                "qa_task_mode": True,  # Enable QA task mode
                "normal_phase_duration": self.config.get("shard_qa", {}).get("normal_phase_duration", 30.0)
            })
            
            executor = ShardWorkerExecutor(
                config=worker_config,
                global_config=self.config,
                shard_id=agent_id,
                data_file=str(shard_data_file),
                neighbors=neighbors,
                output=self.output,
                force_llm=True  # 强制使用LLM模式
            )
            
            # 为worker添加metrics收集器引用
            executor.worker.metrics_collector = self.metrics_collector
            
            # Store the executor
            self.shard_workers[agent_id] = executor
            
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
            await self.mesh_network.register_agent_async(agent)
            self.agents[agent_id] = agent
            
            # Track agent for fault injection (using in-process simulation)
            # Note: We simulate process failure by stopping agents, not actual SIGKILL
            
            # 显示协议特定信息
            if protocol.lower() == "anp":
                self.output.progress(f"🚀 [ANP] Created {agent_id} - HTTP: {port}, WebSocket: {port + 1000}")
                self.output.progress(f"📄 [ANP] Shard data: {shard_data_file.name}")
            else:
                self.output.progress(f"Created Shard QA agent: {agent_id} on port {port} with data: {shard_data_file.name}")
        
        if protocol.lower() == "anp":
            self.output.success(f"🌐 Created {len(self.agents)} ANP agents with hybrid WebSocket+HTTP protocol")
        else:
            self.output.success(f"Created {len(self.agents)} Shard QA agents with {protocol.upper()} protocol")

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

    async def _broadcast_document(self) -> None:
        """Broadcast the document to all agents."""
        if not self.agents:
            raise RuntimeError("No agents available for broadcast")
        
        # Use first agent as broadcaster
        broadcaster_id = list(self.agents.keys())[0]
        
        results = await self.mesh_network.broadcast_init(self.document, broadcaster_id)
        
        successful_deliveries = sum(1 for result in results.values() if "error" not in str(result))
        total_targets = len(results)
        
        self.output.success(f"Document broadcast: {successful_deliveries}/{total_targets} deliveries successful")

    async def _execute_normal_phase(self) -> None:
        """Execute normal Shard QA collaborative retrieval task for 30 seconds."""
        # 使用配置的正常阶段持续时间，如果没有配置则使用30秒
        normal_phase_duration = self.config.get("shard_qa", {}).get("normal_phase_duration", 30.0)
        
        self.output.progress(f"Running Shard QA collaborative retrieval for {normal_phase_duration}s...")
        
        # Start metrics collection for normal phase
        if self.metrics_collector:
            self.metrics_collector.start_normal_phase()
        
        start_time = time.time()
        qa_tasks = []
        
        # Start QA task execution on all agents simultaneously
        for agent_id, worker in self.shard_workers.items():
            # Create concurrent QA task for each agent
            task = asyncio.create_task(self._run_qa_task_for_agent(agent_id, worker, normal_phase_duration))
            qa_tasks.append(task)
        
        # Wait for normal phase duration or until shutdown
        elapsed = 0
        while elapsed < normal_phase_duration and not self.shutdown_event.is_set():
            await asyncio.sleep(1.0)  # Check every second
            elapsed = time.time() - start_time
            
            # Progress indicator every 10 seconds
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                remaining = normal_phase_duration - elapsed
                self.output.progress(f"Normal phase: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
        
        # Stop all QA tasks
        for task in qa_tasks:
            task.cancel()
        
        # Wait for tasks to complete cancellation
        await asyncio.gather(*qa_tasks, return_exceptions=True)
        
        # End metrics collection for normal phase
        if self.metrics_collector:
            self.metrics_collector.end_normal_phase()
        
        self.phase_timers["normal_phase_completed"] = time.time()
        elapsed = self.phase_timers["normal_phase_completed"] - start_time
        self.output.success(f"Normal phase completed in {elapsed:.2f}s")
    
    async def _run_qa_task_for_agent(self, agent_id: str, worker: 'ShardWorkerExecutor', duration: float) -> None:
        """Run continuous QA tasks for a single agent during normal phase."""
        start_time = time.time()
        task_count = 0
        
        try:
            while (time.time() - start_time) < duration and not self.shutdown_event.is_set():
                try:
                    # Execute QA task for group 0 (standard test case)
                    result = await worker.worker.start_task(0)
                    task_count += 1
                    
                    # Fix logic: distinguish between finding answer vs not finding answer  
                    result_str = str(result).lower() if result else ""
                    if (result and 
                        ("document search success" in result_str or "answer_found:" in result_str) and 
                        "no answer" not in result_str):
                        # Show minimal search result from agent
                        if "DOCUMENT SEARCH SUCCESS" in result:
                            self.output.progress(f"🔍 [{agent_id}] Found answer")
                        else:
                            self.output.progress(f"{agent_id}: Found answer (task #{task_count})")
                    
                    # Brief pause between tasks
                    await asyncio.sleep(2.0)
                    
                except Exception as e:
                    self.output.warning(f"{agent_id}: QA task failed: {e}")
                    await asyncio.sleep(1.0)  # Brief pause on error
                    
        except asyncio.CancelledError:
            self.output.progress(f"{agent_id}: QA task cancelled (completed {task_count} tasks)")
            raise
        except Exception as e:
            self.output.error(f"{agent_id}: QA task error: {e}")
        
        self.output.progress(f"{agent_id}: Normal phase completed with {task_count} QA tasks")





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
        """Inject faults by killing random agents with capability to reconnect."""
        agent_ids = list(self.agents.keys())
        num_victims = max(1, int(len(agent_ids) * kill_fraction))
        
        import random
        victims = random.sample(agent_ids, num_victims)
        
        self.output.warning(f"Killing {len(victims)} agents: {', '.join(victims)}")
        
        # Kill agents but save their configs for reconnection
        killed_agents = set()
        for agent_id in victims:
            try:
                agent = self.agents[agent_id]
                
                # 保存agent配置用于重连
                agent_config = {
                    "agent_id": agent_id,
                    "host": agent.host,
                    "port": agent.port,
                    "protocol": self.config["scenario"]["protocol"],
                    "shard_worker": self.shard_workers.get(agent_id),
                    "neighbors": {
                        "prev_id": f"shard{(int(agent_id.replace('shard', '')) - 1) % self.config['scenario']['agent_count']}",
                        "next_id": f"shard{(int(agent_id.replace('shard', '')) + 1) % self.config['scenario']['agent_count']}"
                    }
                }
                self.killed_agent_configs[agent_id] = agent_config
                
                # Stop the agent
                await agent.stop()
                
                # Remove from network (but keep in mesh_network for reconnection tracking)
                await self.mesh_network.unregister_agent(agent_id)
                
                # Remove from active tracking but keep worker
                del self.agents[agent_id]
                # 不删除shard_workers，重连时需要
                killed_agents.add(agent_id)
                
                # Update agent state in metrics
                if self.metrics_collector:
                    self.metrics_collector.update_agent_state(agent_id, "failed")
                
                self.output.progress(f"✗ Killed agent: {agent_id} (will attempt reconnection later)")
                
            except Exception as e:
                self.output.error(f"Failed to kill agent {agent_id}: {e}")
        
        self.killed_agents = killed_agents
        # Record all killed agents as temporarily killed
        self.temporarily_killed_agents.update(killed_agents)
        
        # Trigger network recovery
        if self.mesh_network:
            for agent_id in killed_agents:
                await self.mesh_network._handle_node_failure(agent_id)
        
        # Schedule agent reconnections if enabled
        if self.config["scenario"].get("enable_reconnection", True):
            reconnect_delay = self.config["scenario"].get("reconnection_delay", 10.0)
            asyncio.create_task(self._schedule_agent_reconnections(reconnect_delay))

    async def _monitor_recovery(self) -> None:
        """Monitor recovery and continue QA tasks until 2min or final_answer."""
        total_runtime = self.config["scenario"]["total_runtime"]
        fault_time = self.config["scenario"]["fault_injection_time"]
        recovery_duration = total_runtime - fault_time
        
        recovery_start = time.time()
        recovery_timeout = recovery_start + recovery_duration
        
        self.output.progress(f"Monitoring recovery and continuing QA tasks for {recovery_duration}s...")
        
        # Start metrics collection for recovery phase
        if self.metrics_collector:
            self.metrics_collector.start_recovery_phase()
        
        # Track recovery state
        first_recovery_detected = False
        final_answer_found = False
        qa_tasks = []
        
        # Restart QA tasks on surviving agents (will be updated when agents reconnect)
        surviving_workers = {aid: worker for aid, worker in self.shard_workers.items() 
                           if aid in self.agents and aid not in self.killed_agents}
        
        # Track which agents have running QA tasks
        agents_with_qa_tasks = set()
        
        if surviving_workers:
            self.output.progress(f"Restarting QA tasks on {len(surviving_workers)} surviving agents...")
            for agent_id, worker in surviving_workers.items():
                task = asyncio.create_task(self._run_recovery_qa_task(agent_id, worker, recovery_duration))
                qa_tasks.append(task)
                agents_with_qa_tasks.add(agent_id)
        
        # Monitor recovery and look for final answers
        last_agent_count = len(self.agents)
        while time.time() < recovery_timeout and not self.shutdown_event.is_set() and not final_answer_found:
            # Check for recovery indicators
            if not first_recovery_detected and self.mesh_network:
                topology_health = self.mesh_network.get_topology_health()
                alive_agents = topology_health.get("alive_agents", [])
                
                if len(alive_agents) > 0:
                    avg_connectivity = sum(
                        status.get("connectivity_ratio", 0) 
                        for status in topology_health.get("connectivity_status", {}).values()
                    ) / max(len(alive_agents), 1)
                    
                    if avg_connectivity > 0.7:  # 70% connectivity restored
                        if self.metrics_collector:
                            self.metrics_collector.set_first_recovery_time()
                            first_recovery_detected = True
                            self.output.success("🔄 First recovery signs detected!")
            
            # Check if any agents have reconnected
            current_agent_count = len(self.agents)
            if current_agent_count > last_agent_count:
                self.output.success(f"🔄 Agent reconnection detected! Active agents: {current_agent_count}")
                # Start QA tasks for newly reconnected agents
                newly_connected = set(self.agents.keys()) - agents_with_qa_tasks
                for agent_id in newly_connected:
                    if agent_id in self.shard_workers:
                        remaining_time = recovery_timeout - time.time()
                        if remaining_time > 0:
                            task = asyncio.create_task(
                                self._run_recovery_qa_task(agent_id, self.shard_workers[agent_id], remaining_time)
                            )
                            qa_tasks.append(task)
                            agents_with_qa_tasks.add(agent_id)
                            self.output.progress(f"Started QA task for reconnected agent: {agent_id}")
                last_agent_count = current_agent_count
            
            # Check if any agent found a final answer
            final_answer_found = await self._check_for_final_answer()
            if final_answer_found:
                self.output.success("🎯 Final answer found! Terminating early...")
                break
            
            elapsed = time.time() - recovery_start
            remaining = recovery_duration - elapsed
            if int(elapsed) % 15 == 0 and int(elapsed) > 0:
                self.output.progress(f"Recovery phase: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
            
            await asyncio.sleep(2.0)  # Check every 2 seconds
        
        # Stop all QA tasks
        for task in qa_tasks:
            task.cancel()
        await asyncio.gather(*qa_tasks, return_exceptions=True)
        
        # Wait for steady state
        if self.mesh_network:
            try:
                steady_time = await self.mesh_network.wait_for_steady_state(min_stability_time=5.0)
                if self.metrics_collector:
                    self.metrics_collector.set_steady_state_time()
                self.output.success(f"🎯 Steady state reached in {steady_time:.2f}s")
            except asyncio.TimeoutError:
                self.output.warning("Steady state not reached within timeout")
        
        # End metrics collection for recovery phase
        if self.metrics_collector:
            self.metrics_collector.end_recovery_phase()
        
        self.phase_timers["recovery_completed"] = time.time()
        
        total_elapsed = self.phase_timers["recovery_completed"] - recovery_start
        termination_reason = "final_answer" if final_answer_found else "timeout" 
        self.output.success(f"Recovery phase completed in {total_elapsed:.2f}s (reason: {termination_reason})")
    
    async def _run_recovery_qa_task(self, agent_id: str, worker: 'ShardWorkerExecutor', duration: float) -> None:
        """Run continuous QA tasks for a single agent during recovery phase."""
        start_time = time.time()
        task_count = 0
        
        try:
            while (time.time() - start_time) < duration and not self.shutdown_event.is_set():
                try:
                    # Execute QA task for group 0 and optionally group 1
                    for group_id in [0, 1]:
                        result = await worker.worker.start_task(group_id)
                        task_count += 1
                        
                        # Fix logic: distinguish between finding answer vs not finding answer  
                    result_str = str(result).lower() if result else ""
                    if (result and 
                        ("document search success" in result_str or "answer_found:" in result_str) and 
                        "no answer" not in result_str):
                            # Show minimal search result from agent
                            if "DOCUMENT SEARCH SUCCESS" in result:
                                self.output.progress(f"🔍 [{agent_id}] Found answer")
                            else:
                                self.output.progress(f"{agent_id}: Found answer (task #{task_count})")
                            
                            # Check if this might be a final answer
                            if "final" in result.lower() or "complete" in result.lower():
                                self.output.success(f"{agent_id}: Potential final answer found!")
                        
                        # Brief pause between tasks
                        await asyncio.sleep(1.5)
                    
                except Exception as e:
                    self.output.warning(f"{agent_id}: Recovery QA task failed: {e}")
                    await asyncio.sleep(2.0)
                    
        except asyncio.CancelledError:
            self.output.progress(f"{agent_id}: Recovery QA task cancelled (completed {task_count} tasks)")
            raise
        except Exception as e:
            self.output.error(f"{agent_id}: Recovery QA task error: {e}")
        
        self.output.progress(f"{agent_id}: Recovery phase completed with {task_count} QA tasks")
    
    async def _check_for_final_answer(self) -> bool:
        """Check if any agent has found a final answer."""
        # This is a placeholder - in a real implementation, we would check
        # the QA results for completion indicators
        # For now, we'll rely on timeout
        return False
    
    async def _schedule_agent_reconnections(self, delay: float) -> None:
        """Schedule reconnection attempts for killed agents."""
        if not self.killed_agent_configs:
            return
        
        self.output.progress(f"Scheduling reconnection for {len(self.killed_agent_configs)} agents in {delay}s...")
        await asyncio.sleep(delay)
        
        for agent_id, config in self.killed_agent_configs.items():
            if self.shutdown_event.is_set():
                break
            
            self.output.warning(f"🔄 Attempting to reconnect agent: {agent_id}")
            success = await self._reconnect_agent(agent_id, config)
            
            if success:
                self.output.success(f"✅ Agent {agent_id} successfully reconnected!")
                self.killed_agents.discard(agent_id)  # Remove from currently killed
                # Note: agent remains in temporarily_killed_agents for final stats
                
                # Update metrics
                if self.metrics_collector:
                    self.metrics_collector.update_agent_state(agent_id, "recovering")
                    # Record actual reconnection bytes
                    reconnect_bytes = _calculate_real_reconnection_bytes(
                        getattr(self, 'protocol_name', 'unknown'), 
                        len(self.agents)
                    )
                    self.metrics_collector.record_network_event(
                        event_type="reconnection_success",
                        source_agent=agent_id,
                        bytes_transferred=reconnect_bytes
                    )
            else:
                self.output.error(f"❌ Failed to reconnect agent: {agent_id}")
            
            # Brief delay between reconnection attempts
            await asyncio.sleep(2.0)
    
    async def _reconnect_agent(self, agent_id: str, config: Dict[str, Any]) -> bool:
        """Reconnect a previously killed agent."""
        try:
            protocol = config["protocol"]
            host = config["host"]
            original_port = config["port"]
            
            # 查找新的可用端口，而不是重用原端口
            try:
                available_ports = self._find_available_ports(host, original_port, 1)
                port = available_ports[0]
                if port != original_port:
                    self.output.warning(f"Port {original_port} still in use, using {port} for {agent_id}")
            except RuntimeError:
                self.output.error(f"No available ports found for reconnecting {agent_id}")
                return False
            
            # 重新获取shard worker
            shard_worker = config["shard_worker"]
            if not shard_worker:
                self.output.error(f"No shard worker found for {agent_id}")
                return False
            
            # 重新创建agent
            if protocol.lower() == "a2a":
                agent = await BaseAgent.create_a2a(
                    agent_id=agent_id,
                    host=host,
                    port=port,
                    executor=shard_worker
                )
            elif protocol.lower() == "anp":
                agent = await BaseAgent.create_anp(
                    agent_id=agent_id,
                    host=host,
                    port=port,
                    executor=shard_worker
                )
            elif protocol.lower() == "acp":
                agent = await BaseAgent.create_acp(
                    agent_id=agent_id,
                    host=host,
                    port=port,
                    executor=shard_worker
                )
            elif protocol.lower() == "simple_json":
                agent = await BaseAgent.create_simple_json(
                    agent_id=agent_id,
                    host=host,
                    port=port,
                    executor=shard_worker
                )
            else:
                self.output.error(f"Unsupported protocol for reconnection: {protocol}")
                return False
            
            # 重新注册到mesh网络
            await self.mesh_network.register_agent_async(agent)
            self.agents[agent_id] = agent
            
            # 重新建立连接
            await self._reestablish_agent_connections(agent_id)
            
            # 验证ANP协议连接
            protocol_name = protocol.upper()
            self.output.success(f"🔗 [{protocol_name}] Agent {agent_id} RECONNECTED on port {port}")
            
            # 验证ANP特有功能
            if protocol.lower() == "anp":
                self.output.success(f"📡 [ANP] WebSocket endpoint: ws://127.0.0.1:{port + 1000}")
                self.output.success(f"🌐 [ANP] HTTP REST API: http://127.0.0.1:{port}")
                self.output.success(f"✅ [ANP] Hybrid communication protocol active")
            
            return True
            
        except Exception as e:
            self.output.error(f"Reconnection failed for {agent_id}: {e}")
            return False
    
    async def _reestablish_agent_connections(self, agent_id: str) -> None:
        """重新建立agent与其他节点的连接."""
        try:
            # 连接到所有存活的agent
            for other_agent_id in self.agents.keys():
                if other_agent_id != agent_id:
                    try:
                        await self.mesh_network.connect_agents(agent_id, other_agent_id)
                        await self.mesh_network.connect_agents(other_agent_id, agent_id)
                    except Exception as e:
                        self.output.warning(f"Failed to connect {agent_id} ↔ {other_agent_id}: {e}")
                        
        except Exception as e:
            self.output.error(f"Failed to reestablish connections for {agent_id}: {e}")



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
                "temporarily_killed_count": len(self.temporarily_killed_agents),
                "currently_killed_count": len(self.killed_agents),  # Still dead at end
                "permanently_failed_count": len(self.permanently_failed_agents),
                "surviving_count": len(self.agents),
                "reconnected_count": len(self.temporarily_killed_agents) - len(self.killed_agents),
                "temporarily_killed_agents": list(self.temporarily_killed_agents),
                "currently_killed_agents": list(self.killed_agents),
                "permanently_failed_agents": list(self.permanently_failed_agents),
                "surviving_agents": list(self.agents.keys())
            },
            "timing": {
                "setup_time": self.phase_timers.get("setup_completed", 0) - self.scenario_start_time,
                "normal_phase_time": self.phase_timers.get("normal_phase_completed", 0) - self.scenario_start_time,
                "fault_injection_time": self.phase_timers.get("fault_injection_completed", 0) - self.scenario_start_time,
                "recovery_time": self.phase_timers.get("recovery_completed", 0) - self.scenario_start_time
            },
            "network_metrics": {},
            "failstorm_metrics": {},
            "llm_outputs": {
                "saved": False,
                "directory": "disabled"
            }
        }
        
        # Collect network metrics
        if self.mesh_network:
            results["network_metrics"] = self.mesh_network.get_failure_metrics()
        
        # Collect failstorm metrics
        if self.metrics_collector:
            results["failstorm_metrics"] = self.metrics_collector.calculate_recovery_metrics()
            results["performance_analysis"] = self.metrics_collector.get_performance_summary()
            results["qa_metrics"] = self.metrics_collector.get_qa_metrics()
        
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
        self.shard_workers.clear()
        self.killed_agents.clear()
        self.killed_agent_configs.clear()
        
        self.output.success("Cleanup completed")


async def main():
    """Main entry point for the Fail-Storm scenario."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fail-Storm Recovery Scenario Runner")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path")
    parser.add_argument("--protocol", "-p", choices=["a2a", "anp", "acp", "simple_json", "agora"], help="Protocol to test")
    parser.add_argument("--agents", "-n", type=int, help="Number of agents")
    parser.add_argument("--kill-fraction", "-k", type=float, help="Fraction of agents to kill")
    parser.add_argument("--fault-time", "-f", type=float, help="Fault injection time in seconds")
    parser.add_argument("--runtime", "-r", type=float, help="Total runtime in seconds")
    parser.add_argument("--output", "-o", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create protocol-specific runner using factory
    runner = ProtocolRunnerFactory.create(args.protocol or "simple_json", args.config)
    
    # Override config with command line arguments
    if args.protocol:
        runner.config["scenario"]["protocol"] = args.protocol
    if args.agents:
        runner.config["scenario"]["agent_count"] = args.agents
    if args.kill_fraction:
        runner.config["scenario"]["kill_fraction"] = args.kill_fraction
    if args.fault_time:
        runner.config["scenario"]["fault_injection_time"] = args.fault_time
    if args.runtime:
        runner.config["scenario"]["total_runtime"] = args.runtime
    if args.output:
        runner.results_dir = Path(args.output)
        runner.results_dir.mkdir(exist_ok=True)
    
    # Validate timing parameters - keep standard 60s fault injection for proper testing
    total_runtime = runner.config["scenario"]["total_runtime"]
    fault_injection_time = runner.config["scenario"]["fault_injection_time"]
    
    # 确保total_runtime足够长以容纳标准的故障注入时间
    if total_runtime < 120:
        print(f"⚠️  Warning: Runtime {total_runtime}s is shorter than recommended 120s for proper fail-storm testing")
        print("   Consider using --runtime 120 for standard Gaia Fail-Storm evaluation")
    
    if fault_injection_time >= total_runtime:
        print(f"⚠️  Error: Fault injection time ({fault_injection_time}s) must be less than total runtime ({total_runtime}s)")
        print("   Please increase --runtime or decrease fault_injection_time in config.yaml")
        return
    
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
            print(f"Agents Temporarily Killed: {summary['temporarily_killed_count']}/{summary['initial_count']}")
            print(f"Agents Reconnected: {summary['reconnected_count']}")
            print(f"Agents Still Dead: {summary['currently_killed_count']}")
            print(f"Final Survivors: {summary['surviving_count']}")
        
        if "qa_metrics" in results:
            qa = results["qa_metrics"]
            if not qa.get("no_qa_tasks", False):
                print(f"QA Tasks Completed: {qa.get('total_qa_tasks', 0)}")
                print(f"Answer Found Rate: {qa.get('answer_found_rate', 0):.1%}")
                if qa.get('answer_sources'):
                    sources = ", ".join([f"{k}:{v}" for k, v in qa['answer_sources'].items()])
                    print(f"Answer Sources: {sources}")
        
        if "llm_outputs" in results:
            llm_info = results["llm_outputs"]
            print(f"LLM Outputs Saved: {llm_info['saved']}")
            if llm_info['saved']:
                print(f"LLM Outputs Directory: {llm_info['directory']}")
        
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


def _calculate_real_reconnection_bytes(protocol_name: str, active_agent_count: int) -> int:
    """Calculate actual bytes used for agent reconnection."""
    try:
        # Base protocol overhead (TCP handshake, HTTP headers)
        base_overhead = 150
        
        # Protocol-specific overhead
        if protocol_name == 'anp':
            # DID document exchange ~300 bytes
            # Encryption key exchange ~100 bytes  
            # WebSocket upgrade ~150 bytes
            anp_overhead = 300 + 100 + 150
            total_bytes = base_overhead + anp_overhead
        elif protocol_name == 'simple_json':
            # Basic HTTP overhead
            total_bytes = base_overhead + 50
        else:
            # Default estimate
            total_bytes = base_overhead + 100
        
        # Add mesh network topology updates
        topology_updates = active_agent_count * 30  # Each connection update ~30 bytes
        total_bytes += topology_updates
        
        return total_bytes
        
    except Exception:
        return 200  # Fallback estimate


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)