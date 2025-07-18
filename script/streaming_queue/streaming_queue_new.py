#!/usr/bin/env python3
"""
Real AgentNetwork Demo - Using real AgentNetwork class and BaseAgent with Agora support
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

# Add paths to import AgentNetwork and protocols
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add root directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.network import AgentNetwork
from src.base_agent import BaseAgent

# Import real QA Agent Executors
sys.path.insert(0, str(Path(__file__).parent / "qa_worker"))
sys.path.insert(0, str(Path(__file__).parent / "qa_coordinator"))
from qa_worker.agent_executor import QAAgentExecutor
from qa_coordinator.agent_executor import QACoordinatorExecutor
from agent_adapters.agora_adapter import AgoraServerAdapter

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
    """Real AgentNetwork Demo Class with Agora support"""
    
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
        with open(config_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def _convert_config_for_qa_agent(self, config):
        """Convert configuration format to adapt QA Agent"""
        if not config:
            return None
            
        core_config = config.get('core', {})
        protocol = core_config.get('protocol', 'a2a')
        
        base_config = {
            "model": {
                "type": core_config.get('type', 'openai'),
                "name": core_config.get('name', 'gpt-4o'),
                "openai_api_key": core_config.get('openai_api_key'),
                "openai_base_url": core_config.get('openai_base_url', 'https://api.openai.com/v1'),
                "temperature": core_config.get('temperature', 0.0)
            },
            "protocol": protocol
        }
        
        if core_config.get('type') == 'local':
            base_config.update({
                "model": {
                    "type": "local",
                    "name": core_config.get('name', 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'),
                    "temperature": core_config.get('temperature', 0.0)
                },
                "base_url": core_config.get('base_url', 'http://localhost:8000/v1'),
                "port": core_config.get('port', 8000)
            })
        
        if protocol == 'agora':
            if not core_config.get('openai_api_key'):
                raise ValueError("openai_api_key is required for Agora protocol")
            base_config['openai_api_key'] = core_config.get('openai_api_key')
        
        return base_config
    
    async def setup_agents(self):
        """Setup real A2A or Agora Agents"""
        self.output.info("Initializing real AgentNetwork and Agents...")
        
        qa_config = self._convert_config_for_qa_agent(self.config)
        protocol = qa_config.get('protocol', 'a2a')
        
        # Create Coordinator Agent
        coordinator_executor = QACoordinatorExecutor(qa_config, self.output)
        server_adapter = AgoraServerAdapter() if protocol == 'agora' else None
        self.coordinator = await BaseAgent.create_a2a(
            agent_id="Coordinator-1",
            host="localhost",
            port=9998,
            executor=coordinator_executor,
            httpx_client=self.httpx_client,
            server_adapter=server_adapter,
            protocol=protocol,
            openai_api_key=qa_config.get('openai_api_key'),
            model=qa_config['model']['name']
        )
        await self.network.register_agent(self.coordinator)
        self.output.success("Coordinator-1 created and registered to AgentNetwork")
        
        # Store coordinator's executor for easy access
        self.coordinator_executor = coordinator_executor
        
        # Create Worker Agents
        worker_count = self.config['qa']['worker']['count']
        start_port = self.config['qa']['worker']['start_port']
        worker_ids = []
        
        for i in range(worker_count):
            worker_id = f"Worker-{i+1}"
            port = start_port + i
            
            # Create Worker executor
            worker_executor = QAAgentExecutor(qa_config)
            
            # Create Worker Agent
            worker = await BaseAgent.create_a2a(
                agent_id=worker_id,
                host="localhost",
                port=port,
                executor=worker_executor,
                httpx_client=self.httpx_client,
                server_adapter=server_adapter,
                protocol=protocol,
                openai_api_key=qa_config.get('openai_api_key'),
                model=qa_config['model']['name']
            )
            
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
        
        topology = self.config['qa']['network']['topology']
        
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
    
    async def send_message_to_coordinator(self, command: str, task_type: str = "general"):
        """Send message to coordinator using appropriate protocol"""
        coordinator_url = "http://localhost:9998"
        
        protocol = self.config.get('core', {}).get('protocol', 'a2a')
        
        if protocol == "agora":
            # Agora message format
            message_payload = {
                "message": command,
                "type": task_type,
                "context": {"source": "demo"}
            }
        else:
            # A2A message format
            message_payload = {
                "id": str(time.time_ns()),
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [
                            {
                                "kind": "text",
                                "text": command
                            }
                        ],
                        "messageId": str(time.time_ns())
                    }
                }
            }
        
        try:
            if protocol == "agora":
                # Use network routing for Agora
                response = await self.network.route_message(
                    src_id="ExternalClient",
                    dst_id="Coordinator-1",
                    payload=message_payload
                )
                return {"result": response}
            else:
                # Use HTTP POST for A2A
                response = await self.httpx_client.post(
                    f"{coordinator_url}/message",
                    json=message_payload,
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract text response from events array
                if "events" in result and result["events"]:
                    for event in result["events"]:
                        if event.get("type") == "agent_text_message":
                            return {"result": event.get("data", event.get("text", str(event)))}
                
                return {"result": "Command processed"}
            
        except Exception as e:
            self.output.error(f"Message to coordinator failed: {e}")
            return None
    
    async def load_questions(self):
        """Check coordinator status and load demo questions"""
        self.output.info("Checking coordinator status...")
        response = await self.send_message_to_coordinator("status")
        if response:
            self.output.success("Coordinator status checked")
        
        # Demo questions for Agora
        protocol = self.config.get('core', {}).get('protocol', 'a2a')
        if protocol == "agora":
            return [
                {"type": "weather", "location": "New York", "date": "today"},
                {"type": "booking", "service": "hotel", "datetime": "2025-07-20", "details": {"guests": 2}},
                {"type": "data", "query_type": "search", "parameters": {"term": "AI trends"}},
                {"type": "general", "message": "Hello, how can you assist me?"}
            ]
        return []
    
    async def dispatch_questions_dynamically(self, questions: List[Dict]):
        """Dispatch questions via coordinator"""
        self.output.info("Starting dispatch process...")
        
        results = []
        for question in questions:
            response = await self.send_message_to_coordinator(
                command=question.get("message", json.dumps(question)),
                task_type=question.get("type", "general")
            )
            if response and "result" in response:
                self.output.success(f"Processed question: {question.get('type', 'general')}")
                self.output.system(f"Response: {response['result']}")
                results.append(response)
            else:
                self.output.error(f"Failed to process question: {question}")
        
        return results
    
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
        self.output.info("Real AgentNetwork QA System Demo with Agora support")
        print("=" * 60)
        
        try:
            # 1. Setup Agents
            worker_ids = await self.setup_agents()
            
            # 2. Setup network topology
            await self.setup_topology()
            
            # 3. Health check
            await self.run_health_check()
            
            # 4. Load questions and check coordinator status
            questions = await self.load_questions()
            
            # 5. Start dispatch process
            self.output.info("=== Starting Q&A Processing ===")
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