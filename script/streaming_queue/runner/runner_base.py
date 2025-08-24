# -*- coding: utf-8 -*-
"""
RunnerBase - 协议无关的运行器骨架
负责：
  * 加载配置 / 彩色输出
  * 调用子类实现的：创建网络 / 启动 agent / 协议通信（发指令给协调者）
  * 建拓扑、健康检查、调度/结果保存、清理

子类需要至少实现：
  - create_network(self) -> AgentNetwork
  - setup_agents(self) -> List[str]               # 返回 worker_ids
  - send_command_to_coordinator(self, command:str) -> Dict|None

可选覆盖：
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

# 颜色输出（无依赖降级）
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
    """协议无关 Runner 基类。"""

    def __init__(self, config_path: str = "config.yaml"):
        self.output = ColoredOutput()
        self.config = self._load_config(config_path)
        self.network = None          # 由子类 create_network() 赋值
        self.coordinator = None      # 由子类 setup_agents() 赋值
        self.coordinator_executor = None  # 可选：子类保留引用
        self.workers = []            # 由子类 setup_agents() 赋值
        self._started = False

    # ---------- 配置 ----------
    def _load_config(self, config_path: str) -> dict:
        cfg_file = Path(__file__).parent.parent / config_path
        if not cfg_file.exists():
            # 尝试与 runner 同目录的 config.yaml
            cfg_file = Path(__file__).parent / config_path
        with open(cfg_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ---------- 需子类实现 ----------
    async def create_network(self):
        """返回 AgentNetwork 实例（内含对应协议的 CommBackend）。"""
        raise NotImplementedError

    async def setup_agents(self) -> List[str]:
        """创建并注册 Coordinator + Workers。返回 worker_ids。"""
        raise NotImplementedError

    async def send_command_to_coordinator(self, command: str) -> Optional[Dict[str, Any]]:
        """向协调者发协议层命令，如 'status' / 'dispatch'。"""
        raise NotImplementedError

    # ---------- 可复用/可覆盖 ----------
    async def setup_topology(self) -> None:
        """按配置创建拓扑。子类可覆盖。"""
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
        """RunnerBase 默认只做 status；问题由 Coordinator 自行加载。"""
        self.output.info("Checking coordinator status...")
        resp = await self.send_command_to_coordinator("status")
        if resp:
            self.output.success("Coordinator status checked")
        return []

    async def dispatch_questions_dynamically(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将调度交给 Coordinator（通常是 'dispatch' 指令）。"""
        self.output.info("Starting dispatch process...")
        resp = await self.send_command_to_coordinator("dispatch")
        if resp:
            # The dispatch was sent successfully, results are handled internally
            self.output.success("Dispatch completed!")
            return []
        else:
            self.output.error("Failed to communicate with coordinator")
            return []

    async def save_results(self, results: List[Dict[str, Any]]) -> None:
        """多数协议里结果由 Coordinator 内部保存；这里仅提示。"""
        self.output.info("Results are saved internally by Coordinator")

    async def cleanup(self) -> None:
        """停止 agent / 关闭网络。"""
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

    # ---------- 主流程 ----------
    async def run(self) -> None:
        self.output.info("Runner started")
        print("=" * 60)
        try:
            # 1) 网络 & agent
            self.network = await self.create_network()
            worker_ids = await self.setup_agents()

            # 2) 拓扑
            await self.setup_topology()

            # 3) 健康检查
            await self.run_health_check()

            # 4) 让 Coordinator 做准备（或载问题）
            questions = await self.load_questions()

            # 5) 下发调度指令
            self.output.info("=== Starting Q&A Processing ===")
            t0 = time.time()
            results = await self.dispatch_questions_dynamically(questions)
            t1 = time.time()

            # 6) 保存（大多数时候 Coordinator 已经内部写盘，这里只是占位）
            await self.save_results(results)

            # 7) 展示概览
            self.output.success("Runner finished!")
            self.output.system(f"Total time: {t1 - t0:.2f} seconds")

            # 8) 再次健康检查 / 打印指标
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


# 允许直接运行 base（不会做任何事，只提示需要子类）
async def _main():
    rb = RunnerBase()
    try:
        await rb.run()
    except NotImplementedError:
        print("RunnerBase is abstract. Please run a concrete protocol runner instead.")

if __name__ == "__main__":
    asyncio.run(_main())
