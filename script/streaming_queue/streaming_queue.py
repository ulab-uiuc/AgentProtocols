#!/usr/bin/env python3
"""
Real AgentNetwork Demo - Using real AgentNetwork class and BaseAgent
"""
import asyncio
import json
import yaml
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
import httpx

# Add colorama for colored output
try:
    from colorama import init, Fore, Back, Style

    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    # Fallback if colorama is not available
    COLORS_AVAILABLE = False

    class Fore:
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        CYAN = ""
        WHITE = ""

    class Style:
        BRIGHT = ""
        RESET_ALL = ""

import sys
from pathlib import Path

import sys
from pathlib import Path

# 获取当前文件的路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
src_path = project_root / "src"
script_path = project_root / "script"

# 添加必要的路径
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(script_path))

# 导入基础模块
from network import AgentNetwork
from base_agent import BaseAgent

# 如果文件存在，尝试导入
try:
    from qa_coordinator.agent_executor import QACoordinatorExecutor
except ImportError as e:
    print(f"导入 QACoordinatorExecutor 失败: {e}")

try:
    from qa_worker.agent_executor import QAAgentExecutor
except ImportError as e:
    print(f"导入 QAAgentExecutor 失败: {e}")

class ColoredOutput:
    """Helper class for colored console output"""

    @staticmethod
    def info(message: str) -> None:
        """Print info message in blue"""
        print(f"{Fore.BLUE}{Style.BRIGHT}ℹ️  {message}{Style.RESET_ALL}")

    @staticmethod
    def success(message: str) -> None:
        """Print success message in green"""
        print(f"{Fore.GREEN}{Style.BRIGHT}✅ {message}{Style.RESET_ALL}")

    @staticmethod
    def warning(message: str) -> None:
        """Print warning message in yellow"""
        print(f"{Fore.YELLOW}{Style.BRIGHT}⚠️  {message}{Style.RESET_ALL}")

    @staticmethod
    def error(message: str) -> None:
        """Print error message in red"""
        print(f"{Fore.RED}{Style.BRIGHT}❌ {message}{Style.RESET_ALL}")

    @staticmethod
    def system(message: str) -> None:
        """Print system status in cyan"""
        print(f"{Fore.CYAN}{Style.BRIGHT}🔧 {message}{Style.RESET_ALL}")

    @staticmethod
    def progress(message: str) -> None:
        """Print progress message in white"""
        print(f"{Fore.WHITE}   {message}{Style.RESET_ALL}")


class RealAgentNetworkDemo:
    """Real AgentNetwork Demo Class"""

    def __init__(self, config_path="config.yaml"):
        self.config = self.load_config(config_path)
        self.network = AgentNetwork()  # Use real AgentNetwork
        self.coordinator = None
        self.workers = []
        self.httpx_client = httpx.AsyncClient(timeout=30.0)
        self.output = ColoredOutput()

    def load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        config_file = Path(__file__).parent / config_path
        with open(config_file, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def _convert_config_for_qa_agent(self, config):
        """Convert configuration format to adapt QA Agent"""
        if not config:
            return None

        core_config = config.get("core", {})
        if core_config.get("type") == "openai":
            return {
                "model": {
                    "type": "openai",
                    "name": core_config.get("name", "gpt-4o"),
                    "openai_api_key": core_config.get("openai_api_key"),
                    "openai_base_url": core_config.get("openai_base_url", "https://api.openai.com/v1"),
                    "temperature": core_config.get("temperature", 0.0),
                }
            }
        elif core_config.get("type") == "local":
            return {
                "model": {"type": "local", "name": core_config.get("name", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"), "temperature": core_config.get("temperature", 0.0)},
                "base_url": core_config.get("base_url", "http://localhost:8000/v1"),
                "port": core_config.get("port", 8000),
            }
        return None

    async def setup_agents(self):
        """Setup real A2A Agents"""
        self.output.info("Initializing real AgentNetwork and A2A Agents...")

        qa_config = self._convert_config_for_qa_agent(self.config)

        # Create Coordinator A2A Agent
        coordinator_executor = QACoordinatorExecutor(self.config, self.output)
        self.coordinator = await BaseAgent.create_a2a(
            agent_id="Coordinator-1", host="localhost", port=9998, executor=coordinator_executor, httpx_client=self.httpx_client  # Get port from configuration
        )
        await self.network.register_agent(self.coordinator)
        self.output.success("Coordinator-1 created and registered to AgentNetwork")

        # Store coordinator's executor for easy access
        self.coordinator_executor = coordinator_executor

        # Create Worker A2A Agents
        worker_count = self.config["qa"]["worker"]["count"]
        start_port = self.config["qa"]["worker"]["start_port"]
        worker_ids = []

        for i in range(worker_count):
            worker_id = f"Worker-{i+1}"
            port = start_port + i

            # Create Worker executor
            worker_executor = QAAgentExecutor(qa_config)

            # Create Worker A2A Agent
            worker = await BaseAgent.create_a2a(agent_id=worker_id, host="localhost", port=port, executor=worker_executor, httpx_client=self.httpx_client)

            await self.network.register_agent(worker)
            self.workers.append(worker)
            worker_ids.append(worker_id)
            self.output.success(f"{worker_id} created and registered to AgentNetwork (port: {port})")

        # Set up coordinator with network and worker information
        self.coordinator_executor.coordinator.set_network(self.network, worker_ids)

        return worker_ids

    async def setup_topology(self):
        """Setup network topology"""
        self.output.info("=== Setting up Network Topology ===")

        topology = self.config["qa"]["network"]["topology"]

        if topology == "star":
            self.network.setup_star_topology("Coordinator-1")
            self.output.success("Setup star topology with center node: Coordinator-1")
        elif topology == "mesh":
            self.network.setup_mesh_topology()
            self.output.success("Setup mesh topology")
        else:
            self.output.error("Unknown topology type")
            return

        # Wait for topology setup to complete
        await asyncio.sleep(1)

        # Display topology information
        topology_info = self.network.get_topology()
        edge_count = sum(len(edges) for edges in topology_info.values())
        self.output.system(f"Current topology connection count: {edge_count}")

        self.output.info("Detailed connection information:")
        for agent_id, connections in topology_info.items():
            if connections:
                self.output.progress(f"{agent_id} → {list(connections)}")

    async def send_message_to_coordinator(self, command: str):
        """Send HTTP message directly to coordinator"""
        coordinator_url = "http://localhost:9998"

        # Use A2A message format based on the example
        message_payload = {"id": str(time.time_ns()), "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": command}], "messageId": str(time.time_ns())}}}

        try:
            # Use correct A2A endpoint: /message
            response = await self.httpx_client.post(f"{coordinator_url}/message", json=message_payload, timeout=60.0)
            response.raise_for_status()
            result = response.json()

            # Extract text response from events array
            if "events" in result and result["events"]:
                for event in result["events"]:
                    if event.get("type") == "agent_text_message":
                        return {"result": event.get("data", event.get("text", str(event)))}

            return {"result": "Command processed"}

        except Exception as e:
            self.output.error(f"HTTP request to coordinator failed: {e}")
            return None

    async def load_questions(self):
        """Check coordinator status"""
        self.output.info("Checking coordinator status...")
        response = await self.send_message_to_coordinator("status")
        if response:
            self.output.success("Coordinator status checked")
        return []

    async def dispatch_questions_dynamically(self, questions: List[Dict]):
        """Start dispatch process via HTTP"""
        self.output.info("Starting dispatch process via HTTP...")

        # Send dispatch command to coordinator
        response = await self.send_message_to_coordinator("dispatch")
        if response and "result" in response:
            self.output.success("Dispatch completed!")
            self.output.system(response["result"])
            # Return empty list since results are handled internally by coordinator
            return []
        else:
            self.output.error("Failed to start dispatch")
            return []

    async def save_results(self, results):
        """Save results (coordinator handles result saving internally)"""
        self.output.info("Results are handled and saved by coordinator internally")

    async def run_health_check(self):
        """Run health check"""
        self.output.info("=== Health Check ===")

        health_status = await self.network.health_check()
        healthy_count = sum(1 for status in health_status.values() if status)
        total_count = len(health_status)

        self.output.system(f"Health check results ({healthy_count}/{total_count} healthy):")
        for agent_id, status in health_status.items():
            if status:
                self.output.success(f"{agent_id}: Healthy")
            else:
                self.output.error(f"{agent_id}: Failed")

    async def run_demo(self):
        """Run complete demo"""
        self.output.info("Real A2A AgentNetwork QA System Demo")
        print("=" * 60)

        try:
            # 1. Setup A2A Agents
            worker_ids = await self.setup_agents()

            # 2. Setup network topology
            await self.setup_topology()

            # 3. Health check
            await self.run_health_check()

            # 4. Check coordinator status via HTTP
            questions = await self.load_questions()

            # 5. Start dispatch process via HTTP
            self.output.info("=== Starting Q&A Processing via HTTP ===")
            start_time = time.time()
            results = await self.dispatch_questions_dynamically(questions)
            end_time = time.time()

            # 6. Save results
            await self.save_results(results)

            # 7. Display completion
            self.output.success("Demo completed!")
            self.output.system(f"Total time: {end_time - start_time:.2f} seconds")

            # 8. Final health check
            await self.run_health_check()

            # 9. Display network metrics
            metrics = self.network.snapshot_metrics()
            self.output.info("Network metrics:")
            self.output.progress(f"Agent count: {metrics['agent_count']}")
            self.output.progress(f"Connection count: {metrics['edge_count']}")

        except Exception as e:
            self.output.error(f"Error during demo: {e}")
            import traceback

            traceback.print_exc()

        finally:
            # Cleanup resources
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        self.output.system("Cleaning up resources...")

        # Stop all Agents
        if self.coordinator:
            await self.coordinator.stop()

        for worker in self.workers:
            await worker.stop()

        # Close HTTP client
        await self.httpx_client.aclose()

        self.output.success("Resource cleanup completed")


async def main():
    demo = RealAgentNetworkDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
