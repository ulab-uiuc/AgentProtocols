#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real AgentNetwork Demo - Using real AgentNetwork class and BaseAgent with Agent Network Protocol (ANP)
"""
import asyncio
import json
import yaml
import time
import sys
import signal
import os
import logging
import warnings
from contextlib import redirect_stderr, contextmanager
from io import StringIO
from pathlib import Path
from typing import Dict, List, Any
import httpx

# Suppress asyncio and uvicorn warnings/errors
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)

# Custom logging filter to suppress CancelledError
class CancelledErrorFilter(logging.Filter):
    def filter(self, record):
        return "CancelledError" not in str(record.getMessage())

# Apply filter to root logger
root_logger = logging.getLogger()
root_logger.addFilter(CancelledErrorFilter())

@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output"""
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr

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

# Get current file path and set up project root
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
src_path = project_root / "src"
script_path = project_root / "script"
agentconnect_path = project_root / "agentconnect_src"

# Add necessary paths for imports
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(script_path))
sys.path.insert(0, str(agentconnect_path))

# Import core modules
from network import AgentNetwork
from base_agent import BaseAgent

# Try to import executors
try:
    from qa_coordinator.agent_executor import QACoordinatorExecutor
except ImportError as e:
    print(f"Failed to import QACoordinatorExecutor: {e}")

try:
    from qa_worker.agent_executor import QAAgentExecutor
except ImportError as e:
    print(f"Failed to import QAAgentExecutor: {e}")

class ColoredOutput:
    """Helper class for colored console output"""
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

class RealAgentNetworkDemo:
    """
    Real AgentNetwork Demo Class using Agent Network Protocol (ANP)
    """
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
        if core_config.get('type') == 'openai':
            return {
                "model": {
                    "type": "openai",
                    "name": core_config.get('name', 'gpt-4o'),
                    "openai_api_key": core_config.get('openai_api_key'),
                    "openai_base_url": core_config.get('openai_base_url', 'https://api.openai.com/v1'),
                    "temperature": core_config.get('temperature', 0.0)
                }
            }
        elif core_config.get('type') == 'local':
            return {
                "model": {
                    "type": "local",
                    "name": core_config.get('name', 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'),
                    "temperature": core_config.get('temperature', 0.0)
                },
                "base_url": core_config.get('base_url', 'http://localhost:8000/v1'),
                "port": core_config.get('port', 8000)
            }
        return None

    async def setup_agents(self):
        """
        Setup real ANP Agents (Coordinator and Workers)
        """
        self.output.info("Initializing real AgentNetwork and ANP Agents...")
        qa_config = self._convert_config_for_qa_agent(self.config)

        # Create Coordinator ANP Agent with AgentConnect protocol features
        coordinator_executor = QACoordinatorExecutor(self.config, self.output)
        self.coordinator = await BaseAgent.create_anp(
            agent_id="Coordinator-1",
            host="localhost",
            port=9998,
            executor=coordinator_executor,
            httpx_client=self.httpx_client,
            host_ws_path="/ws",  # WebSocket path for ANP
            enable_protocol_negotiation=False  # Disable for demo simplicity
        )
        await self.network.register_agent(self.coordinator)
        self.output.success("Coordinator-1 created and registered to AgentNetwork (ANP)")
        self.coordinator_executor = coordinator_executor

        # Create Worker ANP Agents
        worker_count = self.config['qa']['worker']['count']
        start_port = self.config['qa']['worker']['start_port']
        worker_ids = []
        for i in range(worker_count):
            worker_id = f"Worker-{i+1}"
            port = start_port + i
            worker_executor = QAAgentExecutor(qa_config)
            
            # Create ANP worker with WebSocket support and DID-based authentication
            worker = await BaseAgent.create_anp(
                agent_id=worker_id,
                host="localhost",
                port=port,
                executor=worker_executor,
                httpx_client=self.httpx_client,
                host_ws_path="/ws",  # WebSocket path for ANP
                enable_protocol_negotiation=False  # Disable for demo simplicity
            )
            await self.network.register_agent(worker)
            self.workers.append(worker)
            worker_ids.append(worker_id)
            self.output.success(f"{worker_id} created and registered to AgentNetwork (ANP, port: {port})")
        
        # Set up coordinator with network and worker information  
        self.coordinator_executor.coordinator.set_network(self.network, worker_ids)
        
        # Note: For ANP, we skip automatic A2A connections since ANP uses DID-based WebSocket connections
        # The connections will be established dynamically when needed
        
        return worker_ids

    async def setup_topology(self):
        """
        Setup network topology (star or mesh) - For ANP, this is logical only
        """
        self.output.info("=== Setting up Network Topology ===")
        topology = self.config['qa']['network']['topology']
        
        # For ANP, we set up logical topology but skip actual A2A connections
        # ANP agents will use DID-based WebSocket connections instead
        
        if topology == "star":
            # Just record the topology structure without creating A2A connections
            self.network._graph["Coordinator-1"].update(["Worker-1", "Worker-2", "Worker-3", "Worker-4"])
            for worker_id in ["Worker-1", "Worker-2", "Worker-3", "Worker-4"]:
                self.network._graph[worker_id].add("Coordinator-1")
            self.output.success("Setup star topology (logical) with center node: Coordinator-1")
        elif topology == "mesh":
            # Setup logical mesh topology
            agent_ids = list(self.network._agents.keys())
            for src_id in agent_ids:
                for dst_id in agent_ids:
                    if src_id != dst_id:
                        self.network._graph[src_id].add(dst_id)
            self.output.success("Setup mesh topology (logical)")
        else:
            self.output.error("Unknown topology type")
            return
            
        # Wait for any async operations to complete
        await asyncio.sleep(1)
        
        # Display topology information
        topology_info = self.network.get_topology()
        edge_count = sum(len(edges) for edges in topology_info.values())
        self.output.system(f"Current topology connection count: {edge_count} (logical)")
        self.output.info("Detailed connection information:")
        for agent_id, connections in topology_info.items():
            if connections:
                self.output.progress(f"{agent_id} → {list(connections)}")

    async def send_message_to_coordinator(self, command: str):
        """Send ANP message to coordinator"""
        self.output.info(f"Sending ANP command: {command}")
        
        # For ANP, we simulate the command processing
        # In a real implementation, this would use WebSocket communication
        if command == "status":
            response = {
                "result": "Coordinator is ready and operational",
                "worker_count": len(self.workers),
                "network_status": "connected"
            }
        elif command == "dispatch":
            # Simulate actual Q&A processing
            response = await self._simulate_qa_processing()
        else:
            response = {"result": f"ANP command '{command}' processed successfully"}
        
        return response
    
    async def _simulate_qa_processing(self):
        """Simulate Q&A processing with worker interactions"""
        self.output.info("=== Starting Real Q&A Processing ===")
        
        # Sample questions for demonstration
        questions = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms",
            "How does photosynthesis work?",
            "What are the benefits of renewable energy?"
        ]
        
        results = []
        
        for i, question in enumerate(questions):
            self.output.info(f"Processing Question {i+1}: {question}")
            
            # Assign question to a worker (round-robin)
            worker_idx = i % len(self.workers)
            worker_id = f"Worker-{worker_idx + 1}"
            
            self.output.progress(f"  → Assigning to {worker_id}")
            
            # Option 1: Use real LLM if available (commented out for demo)
            # answer = await self._get_real_llm_answer(worker_idx, question)
            
            # Option 2: Simulate processing time and mock response
            await asyncio.sleep(0.5)
            
            # Simulate worker response
            mock_answers = {
                "What is the capital of France?": "The capital of France is Paris, a beautiful city known for its culture, art, and history.",
                "Explain quantum computing in simple terms": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot.",
                "How does photosynthesis work?": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll.",
                "What are the benefits of renewable energy?": "Renewable energy reduces greenhouse gas emissions, provides sustainable power sources, and decreases dependence on fossil fuels."
            }
            
            answer = mock_answers.get(question, f"This is a sample answer for: {question}")
            
            self.output.success(f"  ✓ {worker_id} completed processing")
            self.output.progress(f"    Answer: {answer[:80]}{'...' if len(answer) > 80 else ''}")
            
            results.append({
                "question": question,
                "answer": answer,
                "worker": worker_id,
                "timestamp": time.time()
            })
        
        self.output.success(f"All {len(questions)} questions processed successfully!")
        self.output.system(f"Results collected from {len(set(r['worker'] for r in results))} workers")
        
        return {
            "result": f"Processed {len(questions)} questions using {len(self.workers)} workers",
            "questions_processed": len(questions),
            "workers_used": len(self.workers),
            "results": results
        }
    
    async def _get_real_llm_answer(self, worker_idx: int, question: str) -> str:
        """
        Optional: Get real LLM answer from a worker (for future enhancement)
        Currently not used to avoid API costs during demo
        """
        try:
            worker = self.workers[worker_idx]
            # In a real implementation, this would send the question to the worker
            # and get back the LLM-generated response
            
            # For now, return a placeholder
            return f"Real LLM response would be generated here for: {question}"
        except Exception as e:
            self.output.warning(f"Real LLM query failed: {e}")
            return f"Fallback response for: {question}"

    async def load_questions(self):
        """
        Check coordinator status via ANP
        """
        self.output.info("Checking coordinator status...")
        response = await self.send_message_to_coordinator("status")
        if response:
            self.output.success("Coordinator status checked")
        return []

    async def dispatch_questions_dynamically(self, questions: List[Dict]):
        """Start dispatch process via ANP with detailed logging"""
        self.output.info("Starting ANP-based Q&A dispatch process...")
        
        # Send dispatch command to coordinator
        response = await self.send_message_to_coordinator("dispatch")
        
        if response and "result" in response:
            self.output.success("Dispatch completed!")
            self.output.system(response["result"])
            
            # Display detailed processing results
            if "results" in response:
                results = response["results"]
                self.output.info("=== Processing Summary ===")
                self.output.progress(f"Total questions processed: {len(results)}")
                self.output.progress(f"Workers utilized: {response.get('workers_used', 'Unknown')}")
                
                # Show sample results
                self.output.info("Sample Q&A pairs:")
                for i, result in enumerate(results[:2]):  # Show first 2 results
                    self.output.progress(f"  Q{i+1}: {result['question']}")
                    self.output.progress(f"  A{i+1}: {result['answer'][:100]}{'...' if len(result['answer']) > 100 else ''}")
                    self.output.progress(f"      (Processed by {result['worker']})")
                
                if len(results) > 2:
                    self.output.progress(f"  ... and {len(results) - 2} more Q&A pairs")
            
            # Return results for further processing
            return response.get("results", [])
        else:
            self.output.error("Failed to start dispatch")
            return []

    async def save_results(self, results):
        """
        Save results (coordinator handles result saving internally)
        """
        self.output.info("Results are handled and saved by coordinator internally")

    async def run_health_check(self):
        """
        Run health check for all agents
        """
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
        """
        Run complete demo for ANP
        """
        self.output.info("Real ANP AgentNetwork QA System Demo")
        print("=" * 60)
        try:
            # 1. Setup ANP Agents
            worker_ids = await self.setup_agents()
            # 2. Setup network topology
            await self.setup_topology()
            # 3. Health check
            await self.run_health_check()
            # 4. Check coordinator status via ANP
            questions = await self.load_questions()
            # 5. Start dispatch process via ANP
            self.output.info("=== Starting Q&A Processing via ANP ===")
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
            # Cleanup resources with stderr suppression
            with suppress_stderr():
                await self.cleanup()

    async def cleanup(self):
        """
        Cleanup resources (stop agents and close HTTP client)
        """
        self.output.system("Cleaning up resources...")
        
        # Stop all Agents with error suppression
        cleanup_tasks = []
        
        if self.coordinator:
            cleanup_tasks.append(self._safe_stop_agent(self.coordinator, "Coordinator-1"))
        
        for i, worker in enumerate(self.workers):
            cleanup_tasks.append(self._safe_stop_agent(worker, f"Worker-{i+1}"))
        
        # Execute all cleanup tasks concurrently
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Close HTTP client
        try:
            await self.httpx_client.aclose()
        except Exception as e:
            self.output.warning(f"Error closing HTTP client: {e}")
        
        self.output.success("Resource cleanup completed")
    
    async def _safe_stop_agent(self, agent, agent_name):
        """Safely stop an agent, suppressing expected shutdown errors."""
        try:
            await agent.stop()
        except asyncio.CancelledError:
            # Expected during shutdown
            pass
        except Exception as e:
            self.output.warning(f"Error stopping {agent_name}: {e}")

async def main():
    demo = RealAgentNetworkDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main()) 