# -*- coding: utf-8 -*-
"""
RunnerBase - Protocol-agnostic runner skeleton
Responsible for:
    * Loading configuration / colored output
    * Delegating protocol-specific work to subclasses: create network / start agents / protocol-level communication (send commands to coordinator)
    * Setting up topology, health checks, dispatching/saving results, and cleanup

Subclasses must implement at least:
    - create_network(self) -> AgentNetwork
    - setup_agents(self) -> List[str]               # Return worker_ids
    - send_command_to_coordinator(self, command:str) -> Dict|None

Optional overrides:
    - setup_topology(self)
    - run_health_check(self)
    - dispatch_questions_dynamically(self, questions)
    - save_results(self, results)
    - cleanup(self)
"""

from __future__ import annotations

import asyncio
import json
import time
import yaml
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Colored output (graceful degradation if dependency missing)
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except Exception:
    class _F: RED=GREEN=YELLOW=BLUE=CYAN=WHITE=""
    class _S: BRIGHT=RESET_ALL=""
    Fore, Style = _F(), _S()


class ColoredOutput:
    @staticmethod
    def info(message: str) -> None:
        print(f"{Fore.BLUE}{Style.BRIGHT}ℹ️  {message}{Style.RESET_ALL}")

    @staticmethod
    def success(message: str) -> None:
        print(f"{Fore.GREEN}{Style.BRIGHT}✅ {message}{Style.RESET_ALL}")

    @staticmethod
    def warning(message: str) -> None:
        print(f"{Fore.YELLOW}{Style.BRIGHT}⚠️  {message}{Style.RESET_ALL}")

    @staticmethod
    def error(message: str) -> None:
        print(f"{Fore.RED}{Style.BRIGHT}❌ {message}{Style.RESET_ALL}")

    @staticmethod
    def system(message: str) -> None:
        print(f"{Fore.CYAN}{Style.BRIGHT}🔧 {message}{Style.RESET_ALL}")

    @staticmethod
    def progress(message: str) -> None:
        print(f"{Fore.WHITE}   {message}{Style.RESET_ALL}")


class RunnerBase:
    """Base class for protocol-agnostic Runners."""

    def __init__(self, config_path: str = "config.yaml"):
        self.output = ColoredOutput()
        self.config = self._load_config(config_path)
        self.network = None          # assigned by subclass create_network()
        self.coordinator = None      # set by subclass setup_agents()
        self.coordinator_executor = None  # optional: retained by subclass
        self.workers = []            # set by subclass setup_agents()
        self._started = False

    # ---------- Configuration ----------
    def _load_config(self, config_path: str) -> dict:
        cfg_file = Path(__file__).parent.parent / config_path
        if not cfg_file.exists():
            # Try config.yaml located in the runner's directory
            cfg_file = Path(__file__).parent / config_path
        with open(cfg_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ---------- Methods to be implemented by subclasses ----------
    async def create_network(self):
        """Return an AgentNetwork instance (contains the protocol's CommBackend)."""
        raise NotImplementedError

    async def setup_agents(self) -> List[str]:
        """Create and register Coordinator and Workers. Return worker_ids."""
        raise NotImplementedError

    async def send_command_to_coordinator(self, command: str) -> Optional[Dict[str, Any]]:
        """Send protocol-level commands to the coordinator, e.g. 'status' / 'dispatch'."""
        raise NotImplementedError

    # ---------- Reusable / Overridable ----------
    async def setup_topology(self) -> None:
        """Create the topology according to configuration. Subclasses may override."""
        topo = self.config.get("qa", {}).get("network", {}).get("topology", "star")
        self.output.info("=== Setting up Network Topology ===")
        if topo == "star":
            self.network.setup_star_topology("Coordinator-1")
            self.output.success("Setup star topology with center: Coordinator-1")
        elif topo == "mesh":
            self.network.setup_mesh_topology()
            self.output.success("Setup mesh topology")
        else:
            self.output.warning(f"Unknown topology '{topo}', skip wiring")

        await asyncio.sleep(1)
        info = self.network.get_topology()
        edge_count = sum(len(e) for e in info.values())
        self.output.system(f"Current connection count: {edge_count}")
        self.output.info("Connections:")
        for aid, conns in info.items():
            if conns:
                self.output.progress(f"{aid} → {list(conns)}")

    async def run_health_check(self) -> None:
        self.output.info("=== Health Check ===")
        status = await self.network.health_check()
        healthy = sum(1 for ok in status.values() if ok)
        total = len(status)
        self.output.system(f"Health: {healthy}/{total} healthy")
        for aid, ok in status.items():
            (self.output.success if ok else self.output.error)(f"{aid}: {'Healthy' if ok else 'Failed'}")

    async def load_questions(self) -> List[Dict[str, Any]]:
        """By default RunnerBase only performs a status check; the Coordinator is expected to load questions."""
        self.output.info("Checking coordinator status...")
        resp = await self.send_command_to_coordinator("status")
        if resp:
            self.output.success("Coordinator status checked")
        return []

    async def dispatch_questions_dynamically(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Delegate dispatching to the Coordinator (typically via a 'dispatch' command)."""
        self.output.info("Starting dispatch process...")
        resp = await self.send_command_to_coordinator("dispatch")
        if resp:
            # The dispatch was sent successfully, results are handled internally
            self.output.success("Dispatch completed!")
            return [] # MODIFIED: Return response
        else:
            self.output.error("Failed to communicate with coordinator")
            return []

    async def save_results(self, results: List[Dict[str, Any]]) -> None:
        """In most protocols the Coordinator saves results internally; this is just a placeholder."""
        self.output.info("Results are saved internally by Coordinator")

    async def cleanup(self) -> None:
        """Stop agents and close the network."""
        self.output.system("Cleaning up resources...")
        try:
            if self.coordinator and hasattr(self.coordinator, "stop"):
                await self.coordinator.stop()
            for w in self.workers:
                if hasattr(w, "stop"):
                    await w.stop()
        finally:
            if self.network and hasattr(self.network, "close"):
                await self.network.close()
        self.output.success("Resource cleanup completed")

    # ---------- Main flow ----------
    async def run(self) -> None:
        self.output.info("Runner started")
        print("=" * 60)
        try:
            # 1) 网络 & agent
            self.network = await self.create_network()
            worker_ids = await self.setup_agents()

            # 2) Topology
            await self.setup_topology()

            # 3) Health check
            await self.run_health_check()

            # 4) Let the Coordinator prepare (or load questions)
            questions = await self.load_questions()

            # 5) Send dispatch command
            self.output.info("=== Starting Q&A Processing ===")
            t0 = time.time()
            results = await self.dispatch_questions_dynamically(questions)
            t1 = time.time()

            # 6) Save results (in most cases the Coordinator already persisted them internally; this is a placeholder)
            await self.save_results(results)

            # 7) Display summary
            self.output.success("Runner finished!")
            self.output.system(f"Total time: {t1 - t0:.2f} seconds")

            # 8) Run health check again / print metrics
            await self.run_health_check()
            if hasattr(self.network, "snapshot_metrics"):
                metrics = self.network.snapshot_metrics()
                self.output.info("Network metrics:")
                self.output.progress(f"Agent count: {metrics.get('agent_count')}")
                self.output.progress(f"Connection count: {metrics.get('edge_count')}")

        except Exception as e:
            self.output.error(f"Runner error: {e}")
            import traceback; traceback.print_exc()
        finally:
            await self.cleanup()


# Allow running the base runner directly (it will do nothing and indicate a concrete runner is required)
async def _main():
    rb = RunnerBase()
    try:
        await rb.run()
    except NotImplementedError:
        print("RunnerBase is abstract. Please run a concrete protocol runner instead.")

if __name__ == "__main__":
    asyncio.run(_main())
