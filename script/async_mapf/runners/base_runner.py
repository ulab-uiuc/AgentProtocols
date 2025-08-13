# script/async_mapf/runners/base_runner.py
import asyncio
import logging
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import uvicorn
from abc import ABC, abstractmethod

# Core coordinator
from script.async_mapf.core.network_base import NetworkBase
from script.async_mapf.utils.log_utils import get_log_manager


class SimpleOutput:
    """Lightweight output wrapper that uses project LogManager if available."""
    def __init__(self, logger_name: str = "Runner"):
        lm = get_log_manager()
        if lm:
            self.logger = lm.get_network_logger()
        else:
            self.logger = logging.getLogger(logger_name)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                fmt = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(fmt)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
                self.logger.propagate = False

    def info(self, msg: str) -> None: self.logger.info(msg)
    def warning(self, msg: str) -> None: self.logger.warning(msg)
    def error(self, msg: str) -> None: self.logger.error(msg)
    def success(self, msg: str) -> None: self.logger.info(msg)
    def progress(self, msg: str) -> None: self.logger.info(msg)


class RunnerBase(ABC):
    """
    Protocol-agnostic runner that:
      - Loads YAML config
      - Creates NetworkBase coordinator
      - Allocates dynamic ports for network/agents
      - Starts uvicorn servers (network + agents)
      - Installs signal handlers
      - Performs graceful shutdown

    Subclasses must implement:
      - build_network_app(network_coordinator, agent_urls)
      - build_agent_app(agent_id, agent_cfg, port, network_base_port)
    """

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.sim_config = self._load_config(self.config_path)

        self.output = SimpleOutput("Runner")
        self.network_coordinator: Optional[NetworkBase] = None

        self.network_base_port: Optional[int] = None
        self.agent_ports: Dict[int, int] = {}

        self.shutdown_event = asyncio.Event()
        self.servers: List[uvicorn.Server] = []
        self.server_tasks: List[asyncio.Task] = []
        self.bg_tasks: List[asyncio.Task] = []

        # Reduce noisy logs from HTTP stacks
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    # -----------------------------
    # Config & Coordinator
    # -----------------------------
    def _load_config(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def create_network_coordinator(self) -> NetworkBase:
        network_cfg = self.sim_config["network"]
        return NetworkBase(network_cfg)

    # -----------------------------
    # Dynamic Ports
    # -----------------------------
    def get_dynamic_port(self) -> int:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", 0))
            return s.getsockname()[1]

    def preallocate_agent_ports(self) -> Dict[int, int]:
        ports: Dict[int, int] = {}
        for agent_cfg in self.sim_config["network"]["agents"]:
            agent_id = agent_cfg["id"]
            ports[agent_id] = self.get_dynamic_port()
        return ports

    def agent_urls_from_ports(self, ports: Dict[int, int]) -> Dict[int, str]:
        return {aid: f"http://localhost:{port}" for aid, port in ports.items()}

    # -----------------------------
    # Server Scheduling
    # -----------------------------
    async def _start_uvicorn(self, app, port: int, log_level: str = "info") -> uvicorn.Server:
        """
        Start an ASGI app with uvicorn. If the object exposes .build(), use it.
        """
        built_app = app.build() if hasattr(app, "build") else app
        cfg = uvicorn.Config(built_app, host="0.0.0.0", port=port, log_level=log_level)
        server = uvicorn.Server(cfg)
        self.servers.append(server)

        task = asyncio.create_task(server.serve())
        self.server_tasks.append(task)
        return server

    def _schedule_bg(self, coro: "asyncio.coroutines") -> asyncio.Task:
        """Schedule a background coroutine and track it for shutdown."""
        task = asyncio.create_task(coro)
        self.bg_tasks.append(task)
        return task

    # -----------------------------
    # Signal Handling
    # -----------------------------
    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()

        def _trigger_shutdown() -> None:
            self.output.info("Shutdown signal received.")
            self.shutdown_event.set()

        if hasattr(signal, "SIGTERM"):
            try:
                loop.add_signal_handler(signal.SIGTERM, _trigger_shutdown)
            except NotImplementedError:
                pass
        if hasattr(signal, "SIGINT"):
            try:
                loop.add_signal_handler(signal.SIGINT, _trigger_shutdown)
            except NotImplementedError:
                # Windows fallback
                signal.signal(
                    signal.SIGINT,
                    lambda s, f: asyncio.get_running_loop().call_soon_threadsafe(
                        self.shutdown_event.set
                    ),
                )

    # -----------------------------
    # Abstract hooks (must implement)
    # -----------------------------
    @abstractmethod
    def build_network_app(
        self, network_coordinator: NetworkBase, agent_urls: Dict[int, str]
    ):
        """Return ASGI app (or object with .build()) for NetworkBase endpoint."""
        raise NotImplementedError

    @abstractmethod
    def build_agent_app(
        self, agent_id: int, agent_cfg: Dict[str, Any], port: int, network_base_port: int
    ):
        """Return ASGI app (or object with .build()) for each agent endpoint."""
        raise NotImplementedError

    # -----------------------------
    # Orchestration
    # -----------------------------
    async def run(self) -> None:
        """Main entry point to start the simulation (protocol-agnostic)."""
        # Install signal handlers first
        self._install_signal_handlers()
        self.output.info("Initializing NetworkBase coordinator...")

        # 1) Create coordinator
        self.network_coordinator = self.create_network_coordinator()
        # NEW: when all agents reach their goals, trigger graceful shutdown
        self.network_coordinator.set_on_simulation_complete(lambda: self.shutdown_event.set())

        # 2) Pre-allocate ports for agents and build URLs for the network app
        self.agent_ports = self.preallocate_agent_ports()
        agent_urls = self.agent_urls_from_ports(self.agent_ports)

        # 3) Dynamic port for NetworkBase ASGI endpoint
        self.network_base_port = self.get_dynamic_port()
        self.output.info(f"NetworkBase will listen on port {self.network_base_port}")

        # 4) Start NetworkBase ASGI (protocol-specific)
        net_app = self.build_network_app(self.network_coordinator, agent_urls)
        await self._start_uvicorn(net_app, port=self.network_base_port, log_level="info")
        self.output.success(f"NetworkBase ASGI is up at http://localhost:{self.network_base_port}/")

        # 5) Start NetworkBase main loop (concurrent mode)
        self.output.info("Starting NetworkBase main loop...")
        self._schedule_bg(self.network_coordinator.run())

        # 6) Start each agent ASGI app (protocol-specific)
        self.output.info("Starting agent ASGI apps...")
        for agent_cfg in self.sim_config["network"]["agents"]:
            agent_id = agent_cfg["id"]
            port = self.agent_ports[agent_id]
            agent_app = self.build_agent_app(agent_id, agent_cfg, port, self.network_base_port)
            await self._start_uvicorn(agent_app, port=port, log_level="error")
            self.output.success(f"Agent {agent_id} is up at http://localhost:{port}/")

        self.output.info("Simulation running. Press Ctrl+C to stop.")
        # 7) Wait for either Ctrl+C/SIGTERM or the completion callback above
        await self.shutdown_event.wait()
        await self.shutdown()


    async def shutdown(self) -> None:
        """Gracefully stop all uvicorn servers and background tasks."""
        self.output.info("Shutting down gracefully...")
        # Signal uvicorn servers to exit
        for server in self.servers:
            server.should_exit = True

        # Wait for server tasks to exit
        if self.server_tasks:
            try:
                await asyncio.wait(self.server_tasks, timeout=2.5)
            except asyncio.TimeoutError:
                self.output.warning("Some uvicorn servers did not exit within timeout.")

        # Cancel background tasks
        for task in self.bg_tasks:
            if not task.done():
                task.cancel()
        if self.bg_tasks:
            try:
                await asyncio.wait(self.bg_tasks, timeout=2.0)
            except asyncio.TimeoutError:
                self.output.warning("Some background tasks did not cancel within timeout.")

        # Stop the coordinator
        if self.network_coordinator:
            self.network_coordinator.stop()

        self.output.success("Shutdown complete.")
