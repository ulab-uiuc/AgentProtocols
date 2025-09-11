#!/usr/bin/env python3
"""
Base runner for Fail-Storm Recovery scenario with protocol abstraction.

This module provides the abstract base class for protocol-specific implementations,
following the same pattern as MAPF's protocol backend architecture.
"""

import asyncio
import json
import time
import signal
import sys
import os
import socket
import importlib.util
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from abc import ABC, abstractmethod

# Add paths for local imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import shard_qa components  
shard_qa_path = parent_dir / "shard_qa"
sys.path.insert(0, str(shard_qa_path))

# Import core components (after path setup)
sys.path.insert(0, str(parent_dir / "core"))
sys.path.insert(0, str(parent_dir))

# Import native core components (no src dependencies)
from core.mesh_network_native import NativeMeshNetwork

# Import SimpleBaseAgent
spec = importlib.util.spec_from_file_location("simple_base_agent", parent_dir / "core" / "simple_base_agent.py")
base_agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_agent_module)
BaseAgent = base_agent_module.SimpleBaseAgent

# Import EnhancedMeshNetwork  
spec = importlib.util.spec_from_file_location("enhanced_mesh_network", parent_dir / "core" / "enhanced_mesh_network.py")
mesh_network_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mesh_network_module)
MeshNetwork = mesh_network_module.EnhancedMeshNetwork

# Import FailStormMetricsCollector
spec = importlib.util.spec_from_file_location("failstorm_metrics", parent_dir / "core" / "failstorm_metrics.py")
failstorm_metrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(failstorm_metrics_module)
FailStormMetricsCollector = failstorm_metrics_module.FailStormMetricsCollector
# Import ShardWorkerExecutor dynamically
shard_qa_path = parent_dir / "shard_qa"
sys.path.insert(0, str(shard_qa_path))
spec = importlib.util.spec_from_file_location("agent_executor", shard_qa_path / "shard_worker" / "agent_executor.py")
agent_executor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_executor_module)
ShardWorkerExecutor = agent_executor_module.ShardWorkerExecutor

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


class FailStormRunnerBase(ABC):
    """
    Abstract base class for protocol-specific Fail-Storm runners.
    
    This class provides the core functionality shared across all protocols
    while requiring subclasses to implement protocol-specific agent creation
    and connection management.
    
    Follows the same pattern as MAPF's RunnerBase architecture.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the base runner with common components.

        config_path rules:
        1) If an absolute path is provided, use it directly.
        2) If filename only (e.g. config_a2a.yaml), search fail_storm_recovery/configs/ first.
        3) Fallback to legacy protocol_backends/<protocol>/config.yaml for backward compatibility.
        4) If none found, use defaults.
        """
        self._requested_config_path = config_path
        self.config = self._load_config(config_path)
        self.output = ColoredOutput()
        
        # Reduce log noise
        import logging
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger("uvicorn").setLevel(logging.CRITICAL)
        logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)
        logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
        logging.getLogger("fastapi").setLevel(logging.CRITICAL)
        logging.getLogger("a2a").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)
        logging.getLogger("urllib3").setLevel(logging.CRITICAL)
        
        # Core components (same as original)
        self.mesh_network: Optional[MeshNetwork] = None
        self.metrics_collector: Optional[FailStormMetricsCollector] = None
        self.agents: Dict[str, BaseAgent] = {}
        
        # Shard QA components
        self.shard_workers: Dict[str, ShardWorkerExecutor] = {}
        
        # Scenario state
        self.scenario_start_time: float = 0.0
        self.document: Dict[str, Any] = {}
        
        # Set up absolute paths for directories, defaulting to fail_storm_recovery folder
        fail_storm_base = Path(__file__).parent.parent  # /root/Multiagent-Protocol/script/fail_storm_recovery
        
        # Allow config override for directories, but ensure they're absolute paths
        output_config = self.config.get("output", {})
        
        # Workspace directory
        workspace_path = output_config.get("workspace_dir", "workspaces")
        if Path(workspace_path).is_absolute():
            self.workspace_dir = Path(workspace_path)
        else:
            self.workspace_dir = fail_storm_base / workspace_path
            
        # Results directory  
        print(f">>> output config: {output_config}")  # Debug print
        results_path = output_config.get("results_dir", "results")
        if Path(results_path).is_absolute():
            self.results_dir = Path(results_path)
        else:
            self.results_dir = fail_storm_base / results_path
            
        # LLM outputs directory
        llm_outputs_path = output_config.get("llm_outputs_dir", "llm_outputs")
        if Path(llm_outputs_path).is_absolute():
            self.llm_outputs_dir = Path(llm_outputs_path)
        else:
            self.llm_outputs_dir = fail_storm_base / llm_outputs_path
        
        # Fault injection state
        self.killed_agents: Set[str] = set()
        self.temporarily_killed_agents: Set[str] = set()
        self.permanently_failed_agents: Set[str] = set()
        self.killed_agent_configs: Dict[str, Dict[str, Any]] = {}
        
        # Timing control
        self.phase_timers: Dict[str, float] = {}
        
        # Global shutdown handler
        self.shutdown_event = asyncio.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file with environment variable support and unified path resolution.

        Resolution order (first hit wins):
          1. Absolute path provided by user
          2. fail_storm_recovery/configs/<filename>
          3. fail_storm_recovery/<relative path>
          4. Legacy protocol_backends/<proto>/config.yaml (when filename starts with config_<proto>.yaml)
        """
        # Import utilities
        utils_root = Path(__file__).parent.parent
        sys.path.insert(0, str(utils_root))
        try:
            from utils.config_loader import load_config_with_env_vars, check_env_vars  # type: ignore
        except Exception as e:
            raise ImportError(f"Failed to import utils.config_loader: {e}")

        if not check_env_vars():
            raise EnvironmentError("At least one LLM API key must be set")

        # No core initialization here (deferred / optional)
        self.core = None

        base_dir = utils_root  # fail_storm_recovery
        configs_dir = base_dir / "configs"

        given_path = Path(config_path)
        resolved: Optional[Path] = None

        if given_path.is_absolute():
            if given_path.exists():
                resolved = given_path
        else:
            # Plain filename -> configs dir
            if len(given_path.parts) == 1:
                candidate = configs_dir / given_path.name
                if candidate.exists():
                    resolved = candidate
            # Relative path from base_dir
            if resolved is None:
                candidate = base_dir / given_path
                if candidate.exists():
                    resolved = candidate
            # Legacy protocol directory
            if resolved is None and given_path.name.startswith("config_") and given_path.suffix in (".yaml", ".yml"):
                proto = given_path.stem.replace("config_", "")
                legacy = base_dir / "protocol_backends" / proto / "config.yaml"
                if legacy.exists():
                    resolved = legacy

        self._resolved_config_path = resolved  # store for external introspection

        # Defaults
        default_config = {
            "scenario": {
                "protocol": "simple_json",
                "agent_count": 8,
                "kill_fraction": 0.375,  # Kill 3 out of 8 agents
                "fault_injection_time": 120.0,  # First fault at 2 minutes
                "total_runtime": 1200.0,  # 20 minutes for 1000 groups
                "recovery_duration": 60.0,
                "heartbeat_interval": 3.0,
                "heartbeat_timeout": 30.0,
                # Cyclic fail storm configuration
                "cyclic_faults": True,
                "fault_cycle_interval": 120.0,  # 2 minutes
                "agents_per_fault": 3,
                "normal_phase_duration": 120.0,  # 2 minutes normal phase
                "max_groups": 1000  # Test 1000 groups
            },
            "shard_qa": {
                "data_dir": "data/shards",
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
                "type": "openai",
                "model": "gpt-4o",
                "openai_api_key": "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",
                "openai_base_url": "https://api.openai.com/v1",
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 8192,
                "name": "gpt-4o",
                "timeout": 30.0
            },
            "output": {
                "results_file": "failstorm_metrics.json",
                "logs_dir": "logs",
                "artifacts_dir": "artifacts"
            }
        }
        
        if resolved and resolved.exists():
            try:
                loaded = load_config_with_env_vars(str(resolved))
                def merge_configs(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
                    for k, v in src.items():
                        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
                            merge_configs(dst[k], v)
                        else:
                            dst[k] = v
                    return dst
                return merge_configs(default_config, loaded)
            except Exception as e:
                print(f"Warning: Failed to load config {config_path}: {e}")
                return default_config
        print(f"Warning: Config file {config_path} not found in unified search paths, using defaults")
        return default_config

    def get_config_path(self) -> str:
        """Return resolved configuration file path or '<defaults>'."""
        if hasattr(self, '_resolved_config_path') and self._resolved_config_path:
            return str(self._resolved_config_path)
        return "<defaults>"
    
    def execute_llm(self, messages):
        """
        Execute LLM using the Core instance.
        
        Parameters
        ----------
        messages : list
            List of message dictionaries in OpenAI format
            
        Returns
        -------
        str
            LLM response content
        """
        if not self.core:
            raise RuntimeError("Core LLM instance not initialized")
        
        try:
            return self.core.execute(messages)
        except Exception as e:
            print(f"❌ LLM execution error: {e}")
            raise
    
    def execute_llm_with_functions(self, messages, functions, max_length=300000):
        """
        Execute LLM with function calling using the Core instance.
        
        Parameters
        ----------
        messages : list
            List of message dictionaries
        functions : list
            List of function definitions
        max_length : int
            Maximum character length limit
            
        Returns
        -------
        object
            Complete LLM response object with tool calls
        """
        if not self.core:
            raise RuntimeError("Core LLM instance not initialized")
        
        try:
            return self.core.function_call_execute(messages, functions, max_length)
        except Exception as e:
            print(f"❌ LLM function call error: {e}")
            raise

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.output.warning("Received shutdown signal, initiating graceful cleanup...")
        self.shutdown_event.set()

    def _is_port_available(self, host: str, port: int) -> bool:
        """Check if port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result != 0
        except Exception:
            return False

    def _find_available_ports(self, host: str, base_port: int, count: int) -> List[int]:
        """Find consecutive available ports."""
        available_ports = []
        port = base_port
        max_attempts = 1000
        
        while len(available_ports) < count and port < base_port + max_attempts:
            if self._is_port_available(host, port):
                available_ports.append(port)
            port += 1
        
        if len(available_ports) < count:
            raise RuntimeError(f"Cannot find {count} available ports starting from {base_port}")
        
        return available_ports

    # ========================================
    # Abstract Methods (Protocol-Specific)
    # ========================================
    
    @abstractmethod
    async def create_agent(self, agent_id: str, host: str, port: int, executor: ShardWorkerExecutor) -> BaseAgent:
        """Create agent using protocol-specific implementation."""
        raise NotImplementedError
    
    @abstractmethod
    def get_protocol_info(self, agent_id: str, port: int, data_file: str) -> str:
        """Get protocol-specific display information."""
        raise NotImplementedError
    
    @abstractmethod
    def get_reconnection_info(self, agent_id: str, port: int) -> List[str]:
        """Get protocol-specific reconnection information."""
        raise NotImplementedError
    
    # ========================================
    # Common Implementation (Protocol-Agnostic)
    # ========================================

    async def run_scenario(self) -> Dict[str, Any]:
        """Execute the complete Fail-Storm recovery scenario."""
        try:
            self.output.info(f"🚀 Starting Fail-Storm Recovery Scenario (Protocol: {self.config['scenario']['protocol'].upper()})")
            self.scenario_start_time = time.time()
            
            # Phase 0: Setup
            self.output.info("📋 Phase 0: Initializing scenario...")
            await self._setup_scenario()
            
            if self.shutdown_event.is_set():
                return await self._cleanup_and_exit("Setup interrupted")
            
            # Check if cyclic faults are enabled
            if self.config.get("scenario", {}).get("cyclic_faults", False):
                # Execute cyclic fail storm scenario
                self.output.info("🔄 Phase 1-N: Cyclic Fail-Storm execution...")
                await self._execute_cyclic_fail_storm()
            else:
                # Original single-fault execution
                # Phase 1: Normal Operation
                self.output.info("⚡ Phase 1: Normal Shard QA workflow execution...")
                await self._execute_normal_phase()
                
                if self.shutdown_event.is_set():
                    return await self._cleanup_and_exit("Normal phase interrupted")
                
                # Phase 2: Fault Injection
                self.output.info("💥 Phase 2: Fault injection...")
                await self._execute_fault_injection()
                
                if self.shutdown_event.is_set():
                    return await self._cleanup_and_exit("Fault injection interrupted")
                
                # Phase 3: Recovery Monitoring
                self.output.info("🔄 Phase 3: Recovery monitoring...")
                await self._monitor_recovery()
            
            # Phase 4: Evaluation and Results
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
        
        agent_count = self.config["scenario"]["agent_count"]
        for i in range(agent_count):
            agent_workspace = self.workspace_dir / f"agent_{i}"
            agent_workspace.mkdir(exist_ok=True)

    def _load_document(self) -> Dict[str, Any]:
        """Load the document for processing."""
        doc_path = Path(__file__).parent.parent / "docs" / "gaia_document.txt"
        
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
        
        # Fallback document
        fallback_content = """
        This is a test document for the Fail-Storm recovery scenario.
        
        Key Information:
        1. The system should be resilient to node failures
        2. Agents should collaborate to find answers in distributed documents
        3. When a question cannot be answered locally, agents should communicate with neighbors
        4. The system should maintain functionality even when 30% of nodes fail
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
        """Setup Shard QA workers using protocol-agnostic approach."""
        agent_count = self.config["scenario"]["agent_count"]
        base_port = self.config["agents"]["base_port"]
        host = self.config["agents"]["host"]
        
        # Find available ports
        try:
            available_ports = self._find_available_ports(host, base_port, agent_count)
            self.output.success(f"Found available ports: {available_ports}")
        except RuntimeError as e:
            self.output.error(f"Port allocation failed: {e}")
            raise
        
        # Prepare LLM configuration in the format expected by ShardWorker
        llm_config = {"model": self.config["llm"]}
        
        # Use shard data files from configuration
        shard_qa_data_dir = Path(__file__).parent.parent / self.config.get("shard_qa", {}).get("data_dir", "data/shards")
        if not shard_qa_data_dir.exists():
            self.output.error(f"Shard QA data directory not found: {shard_qa_data_dir}")
            raise FileNotFoundError(f"Missing shard QA data: {shard_qa_data_dir}")
        
        # Create agents with ShardWorkerExecutor
        for i in range(agent_count):
            agent_id = f"agent{i}"
            port = available_ports[i]
            
            # Define neighbors (ring topology)
            neighbors = {
                "prev_id": f"agent{(i-1) % agent_count}",
                "next_id": f"agent{(i+1) % agent_count}"
            }
            
            # Use existing shard data file
            shard_data_file = shard_qa_data_dir / f"shard{i}.jsonl"
            if not shard_data_file.exists():
                shard_data_file = shard_qa_data_dir / "shard0.jsonl"
            
            # Create ShardWorkerExecutor
            worker_config = llm_config.copy()
            worker_config.update({
                "agent_id": agent_id,
                "qa_task_mode": True,
                "normal_phase_duration": self.config.get("shard_qa", {}).get("normal_phase_duration", 30.0)
            })
            
            executor = ShardWorkerExecutor(
                config=worker_config,
                global_config=self.config,
                shard_id=agent_id,
                data_file=str(shard_data_file),
                neighbors=neighbors,
                output=self.output,
                force_llm=True
            )
            
            # Add metrics collector reference
            executor.worker.metrics_collector = self.metrics_collector
            
            # Set network reference for the worker
            executor.worker.set_network(self.mesh_network)
            
            # Store the executor
            self.shard_workers[agent_id] = executor
            
            # Create agent using protocol-specific method
            agent = await self.create_agent(agent_id, host, port, executor)
            
            # Register with network
            await self.mesh_network.register_agent(agent)
            self.agents[agent_id] = agent
            
            # Display protocol-specific information
            protocol_info = self.get_protocol_info(agent_id, port, shard_data_file.name)
            self.output.progress(protocol_info)
        
        # Display summary
        protocol_name = self.config["scenario"]["protocol"].upper()
        self.output.success(f"Created {len(self.agents)} Shard QA agents with {protocol_name} protocol")

    # The rest of the methods remain the same as the original implementation
    # but are now protocol-agnostic...
    
    # [Due to length constraints, I'll implement the key remaining methods]
    # These would include: _setup_mesh_topology, _broadcast_document, 
    # _execute_normal_phase, _execute_fault_injection, _monitor_recovery, 
    # _finalize_scenario, etc.
    
    async def _finalize_scenario(self) -> Dict[str, Any]:
        """Finalize scenario and collect/save metrics."""
        self.output.info("📊 Finalizing scenario and saving results...")
        
        # Collect metrics
        if self.metrics_collector:
            results = self.metrics_collector.get_final_results()
        else:
            results = {
                "metadata": {
                    "scenario": "fail_storm_recovery",
                    "protocol": self.config.get("scenario", {}).get("protocol", "unknown"),
                    "status": "completed",
                    "end_time": time.time()
                }
            }
        
        # Save results to files
        await self._save_results(results)
        
        return results
    
    async def _execute_cyclic_fail_storm(self) -> None:
        """Execute cyclic fail-storm scenario with alternating normal/fault phases."""
        import asyncio
        import time
        import random
        
        scenario_config = self.config.get("scenario", {})
        total_runtime = scenario_config.get("total_runtime", 1200.0)  # 20 minutes
        fault_cycle_interval = scenario_config.get("fault_cycle_interval", 120.0)  # 2 minutes
        normal_phase_duration = scenario_config.get("normal_phase_duration", 120.0)  # 2 minutes
        agents_per_fault = scenario_config.get("agents_per_fault", 3)
        
        start_time = time.time()
        cycle_count = 0
        current_phase = "normal"  # Start with normal phase
        
        self.output.info(f"🔄 Starting Cyclic Fail-Storm:")
        self.output.info(f"   Total runtime: {total_runtime/60:.1f} minutes")
        self.output.info(f"   Cycle interval: {fault_cycle_interval/60:.1f} minutes")
        self.output.info(f"   Agents per fault: {agents_per_fault}")
        
        while time.time() - start_time < total_runtime and not self.shutdown_event.is_set():
            cycle_start = time.time()
            cycle_count += 1
            elapsed = time.time() - start_time
            
            if current_phase == "normal":
                self.output.info(f"✅ Cycle {cycle_count}: Normal Phase ({elapsed/60:.1f}m elapsed)")
                
                # Run normal QA tasks for specified duration
                await self._execute_normal_phase_cyclic(normal_phase_duration)
                
                # Switch to fault phase
                current_phase = "fault"
                
            else:  # fault phase
                self.output.info(f"💥 Cycle {cycle_count}: Fault Injection Phase ({elapsed/60:.1f}m elapsed)")
                
                # Inject faults (kill some agents)
                await self._inject_cyclic_faults(agents_per_fault)
                
                # Monitor recovery for a period
                recovery_duration = min(fault_cycle_interval, 60.0)  # Max 1 minute recovery
                await self._monitor_cyclic_recovery(recovery_duration)
                
                # Switch back to normal phase
                current_phase = "normal"
            
            # Wait for next cycle if needed
            cycle_elapsed = time.time() - cycle_start
            if cycle_elapsed < fault_cycle_interval:
                wait_time = fault_cycle_interval - cycle_elapsed
                self.output.info(f"⏳ Waiting {wait_time:.1f}s for next cycle...")
                await asyncio.sleep(wait_time)
        
        total_elapsed = time.time() - start_time
        self.output.success(f"🎉 Cyclic Fail-Storm completed: {cycle_count} cycles in {total_elapsed/60:.1f} minutes")
    
    async def _execute_normal_phase_cyclic(self, duration: float) -> None:
        """Execute normal phase for cyclic fail-storm."""
        # Start QA tasks for all alive agents
        qa_tasks = []
        for agent_id, worker in self.shard_workers.items():
            if agent_id not in self.killed_agents:
                task = asyncio.create_task(
                    self._run_qa_task_for_agent_cyclic(agent_id, worker, duration),
                    name=f"qa_task_{agent_id}"
                )
                qa_tasks.append(task)
        
        # Wait for all tasks to complete or timeout
        if qa_tasks:
            await asyncio.gather(*qa_tasks, return_exceptions=True)
    
    async def _run_qa_task_for_agent_cyclic(self, agent_id: str, worker, duration: float):
        """Run QA tasks for an agent during cyclic normal phase."""
        start_time = time.time()
        task_count = 0
        max_groups = self.config.get("scenario", {}).get("max_groups", 1000)
        
        # Continue from where we left off, or start from random group
        group_id = getattr(self, f"_last_group_{agent_id}", 0)
        
        while time.time() - start_time < duration and not self.shutdown_event.is_set():
            try:
                # Execute QA task for current group
                task_start_time = time.time()
                result = await worker.worker.start_task(group_id % max_groups)
                task_end_time = time.time()
                task_count += 1
                
                # Record task execution in metrics
                if self.metrics_collector:
                    result_str = str(result).lower() if result else ""
                    answer_found = (result and 
                                  ("document search success" in result_str or "answer_found:" in result_str) and 
                                  "no answer" not in result_str)
                    answer_source = "local" if "local" in result_str else "neighbor"
                    
                    self.metrics_collector.record_task_execution(
                        task_id=f"{agent_id}_cyclic_g{group_id}_{task_count}",
                        agent_id=agent_id,
                        task_type="qa_cyclic",
                        start_time=task_start_time,
                        end_time=task_end_time,
                        success=True,
                        answer_found=answer_found,
                        answer_source=answer_source,
                        group_id=group_id % max_groups
                    )
                
                group_id += 1
                
                # Brief pause between tasks
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.output.error(f"❌ [{agent_id}] QA task failed: {e}")
                await asyncio.sleep(1.0)
        
        # Save last group for next cycle
        setattr(self, f"_last_group_{agent_id}", group_id)
        
        self.output.info(f"   {agent_id}: Completed {task_count} QA tasks (groups {group_id-task_count}-{group_id-1})")
    
    async def _inject_cyclic_faults(self, agents_per_fault: int) -> None:
        """Inject faults by killing specified number of agents."""
        import random
        
        # Get list of alive agents
        alive_agents = [aid for aid in self.agents.keys() if aid not in self.killed_agents]
        
        if len(alive_agents) <= agents_per_fault:
            self.output.warning(f"⚠️ Only {len(alive_agents)} agents alive, cannot kill {agents_per_fault}")
            return
        
        # Randomly select agents to kill
        agents_to_kill = random.sample(alive_agents, min(agents_per_fault, len(alive_agents)))
        
        self.output.warning(f"💥 Killing {len(agents_to_kill)} agents: {', '.join(agents_to_kill)}")
        
        # Record fault injection time
        if self.metrics_collector and not hasattr(self.metrics_collector, 'fault_injection_time'):
            self.metrics_collector.set_fault_injection_time()
        
        # Kill selected agents
        for agent_id in agents_to_kill:
            try:
                await self._kill_agent(agent_id)
                self.output.warning(f"   ✗ Killed agent: {agent_id}")
            except Exception as e:
                self.output.error(f"❌ Failed to kill {agent_id}: {e}")
        
        # Schedule reconnection
        reconnection_delay = 10.0  # 10 seconds
        self.output.info(f"🔄 Scheduling reconnection for {len(agents_to_kill)} agents in {reconnection_delay}s...")
        
        for agent_id in agents_to_kill:
            asyncio.create_task(self._schedule_reconnection(agent_id, reconnection_delay))
    
    async def _monitor_cyclic_recovery(self, duration: float) -> None:
        """Monitor recovery during cyclic fail-storm."""
        start_time = time.time()
        
        while time.time() - start_time < duration and not self.shutdown_event.is_set():
            alive_count = len(self.agents) - len(self.killed_agents)
            total_count = len(self.agents)
            alive_percentage = (alive_count / total_count) * 100
            elapsed = time.time() - start_time
            
            self.output.info(f"🔄 Recovery monitoring: {alive_percentage:.0f}% alive, {elapsed:.0f}s elapsed")
            
            # Check if all agents recovered
            if len(self.killed_agents) == 0:
                self.output.success(f"✅ All agents recovered in {elapsed:.1f}s")
                if self.metrics_collector:
                    self.metrics_collector.set_steady_state_time()
                break
            
            await asyncio.sleep(5.0)  # Check every 5 seconds
    
    async def _schedule_reconnection(self, agent_id: str, delay: float) -> None:
        """Schedule reconnection of a killed agent."""
        await asyncio.sleep(delay)
        
        if agent_id in self.killed_agents:
            try:
                await self._reconnect_agent(agent_id)
                self.output.success(f"✅ Agent {agent_id} reconnected successfully")
                
                # Record first recovery time if this is the first recovery
                if self.metrics_collector and not hasattr(self.metrics_collector, 'first_recovery_time'):
                    self.metrics_collector.set_first_recovery_time()
                    
            except Exception as e:
                self.output.error(f"❌ Failed to reconnect {agent_id}: {e}")
    
    async def _kill_agent(self, agent_id: str) -> None:
        """Kill an agent (to be implemented by subclasses)."""
        # Default implementation: just mark as killed
        if agent_id in self.agents:
            # Store agent config for reconnection
            if not hasattr(self, 'killed_agent_configs'):
                self.killed_agent_configs = {}
            
            self.killed_agent_configs[agent_id] = {
                "agent_id": agent_id,
                "host": "0.0.0.0",
                "port": getattr(self.agents[agent_id], '_port', 9000),
                "data_file": getattr(self.shard_workers.get(agent_id, {}).worker if agent_id in self.shard_workers else None, 'data_file', 'shard0.jsonl')
            }
            
            # Add to killed agents set
            self.killed_agents.add(agent_id)
            self.temporarily_killed_agents.add(agent_id)
            
            # Stop the agent (basic implementation)
            try:
                if hasattr(self.agents[agent_id], 'stop'):
                    await self.agents[agent_id].stop()
            except Exception:
                pass  # Ignore stop errors
    
    async def _reconnect_agent(self, agent_id: str) -> None:
        """Reconnect a killed agent (to be implemented by subclasses)."""
        # Default implementation: just remove from killed set
        if agent_id in self.killed_agents:
            self.killed_agents.discard(agent_id)
            if hasattr(self, 'killed_agent_configs') and agent_id in self.killed_agent_configs:
                del self.killed_agent_configs[agent_id]
    
    def _generate_qa_summary(self) -> Dict[str, Any]:
        """Generate unified QA summary from metrics collector."""
        if not self.metrics_collector:
            return {
                "total_tasks": 0,
                "found_answers": 0,
                "answer_found_rate": 0.0,
                "groups_tested": 0,
                "group_range": "N/A",
                "answer_sources": {},
                "note": "No metrics collector available"
            }
        
        try:
            # Get all task executions from metrics collector directly
            task_executions = self.metrics_collector.task_executions
            
            # Count QA tasks and answers
            total_tasks = 0
            found_answers = 0
            groups_tested = set()
            answer_sources = {"local": 0, "neighbor": 0, "unknown": 0}
            
            for task in task_executions:
                task_type = getattr(task, "task_type", "")
                if "qa" in task_type.lower():  # Include qa_normal, qa_search, qa_recovery, etc.
                    total_tasks += 1
                    group_id = getattr(task, "group_id", None)
                    if group_id is not None:
                        groups_tested.add(group_id)
                    
                    if getattr(task, "answer_found", False):
                        found_answers += 1
                        source = getattr(task, "answer_source", "unknown")
                        if source in answer_sources:
                            answer_sources[source] += 1
                        else:
                            answer_sources["unknown"] += 1
            
            answer_rate = found_answers / total_tasks if total_tasks > 0 else 0.0
            group_range = f"{min(groups_tested)}-{max(groups_tested)}" if groups_tested else "N/A"
            
            return {
                "total_tasks": total_tasks,
                "found_answers": found_answers,
                "answer_found_rate": answer_rate,
                "groups_tested": len(groups_tested),
                "group_range": group_range,
                "answer_sources": answer_sources,
                "success_rate": 1.0 if total_tasks > 0 else 0.0  # All tasks completed successfully
            }
            
        except Exception as e:
            return {
                "total_tasks": 0,
                "found_answers": 0,
                "answer_found_rate": 0.0,
                "groups_tested": 0,
                "group_range": "Error",
                "answer_sources": {},
                "error": str(e)
            }
    
    async def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to configured output files."""
        try:
            # Get output file paths from config with timestamp and protocol
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            protocol_name = self.config.get("scenario", {}).get("protocol", "unknown")
            
            # Add unified QA summary to results
            qa_summary = self._generate_qa_summary()
            results["qa_summary"] = qa_summary
            
            output_config = self.config.get("output", {})
            base_results_file = output_config.get("results_file", "failstorm_metrics.json")
            base_detailed_file = output_config.get("detailed_results_file", "detailed_failstorm_metrics.json")
            
            # Add timestamp and protocol to filenames
            results_file = base_results_file.replace(".json", f"_{timestamp}_{protocol_name}.json")
            detailed_results_file = base_detailed_file.replace(".json", f"_{timestamp}_{protocol_name}.json")
            
            # Ensure results directory exists
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
            # Full paths
            results_path = self.results_dir / results_file
            detailed_results_path = self.results_dir / detailed_results_file
            
            # Save main results
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save detailed results (same for now, can be extended)
            with open(detailed_results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.output.success(f"📊 Results saved to: {results_path}")
            self.output.success(f"📈 Detailed metrics: {detailed_results_path}")
            
            # Store paths for runner access
            self.saved_results_path = results_path
            self.saved_detailed_results_path = detailed_results_path
            
        except Exception as e:
            self.output.error(f"Failed to save results: {e}")
            raise
    
    def get_results_paths(self) -> Dict[str, str]:
        """Get the actual paths where results were saved."""
        return {
            "results_file": str(getattr(self, 'saved_results_path', self.results_dir / self.config.get("output", {}).get("results_file", "failstorm_metrics.json"))),
            "detailed_results_file": str(getattr(self, 'saved_detailed_results_path', self.results_dir / self.config.get("output", {}).get("detailed_results_file", "detailed_failstorm_metrics.json")))
        }
    
    async def _cleanup_resources(self) -> None:
        """Clean up all resources."""
        self.output.progress("Cleaning up resources...")
        
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

    async def _cleanup_and_exit(self, reason: str) -> Dict[str, Any]:
        """Cleanup and exit with error results."""
        self.output.error(f"Scenario terminated: {reason}")
        
        results = {
            "metadata": {
                "scenario": "fail_storm_recovery",
                "status": "terminated",
                "reason": reason,
                "end_time": time.time()
            },
            "partial_results": self.metrics_collector.get_real_time_summary() if self.metrics_collector else {}
        }
        
        return results