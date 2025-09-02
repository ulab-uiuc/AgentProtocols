"""
Meta Protocol Runner - BaseAgent integration with streaming_queue protocols

Uses src/core/base_agent.py and src/core/network.py to create a unified
network with workers from different protocols (ACP, ANP, Agora, A2A).
"""

import asyncio
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

# Setup paths
current_file = Path(__file__).resolve()
streaming_queue_path = current_file.parents[2]
project_root = streaming_queue_path.parent.parent
src_path = project_root / "src"

# Add paths - src first for priority
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(streaming_queue_path) not in sys.path:
    sys.path.insert(0, str(streaming_queue_path))

# Import src components (BaseAgent from src/core)
from core.base_agent import BaseAgent

# Import streaming_queue network (NOT src network)
from core.network_base import NetworkBase

# Import executor wrappers
from protocol_backend.meta_protocol.executor_wrappers import create_protocol_worker, validate_executor_interface

from runner_base import ColoredOutput


class MetaProtocolRunner:
    """
    Meta Protocol Runner using src/core architecture
    
    Creates BaseAgent instances with different protocol worker executors:
    - Each worker uses its native SDK (ACP 1.0.3, ANP AgentConnect, Agora, A2A)
    - All workers are managed through src/core/base_agent.py
    - Network topology managed by src/core/network.py
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.output = ColoredOutput()
        
        # Use streaming_queue/core/network_base.py
        self.network_base = NetworkBase()
        
        # Track BaseAgent instances
        self.base_agents: Dict[str, BaseAgent] = {}
        self.protocol_map: Dict[str, str] = {}  # agent_id -> protocol
        
        print("[MetaProtocolRunner] Initialized with src/core architecture")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        config_file = Path(config_path)
        if not config_file.exists():
            config_file = streaming_queue_path / config_path
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def _convert_config_for_qa(self) -> Dict[str, Any]:
        """Convert config for QA workers"""
        core = self.config.get("core", {})
        if core.get("type") == "openai":
            return {
                "model": {
                    "type": "openai",
                    "name": core.get("name", "gpt-4o"),
                    "openai_api_key": core.get("openai_api_key"),
                    "openai_base_url": core.get("openai_base_url", "https://api.openai.com/v1"),
                    "temperature": core.get("temperature", 0.0),
                }
            }
        return {}
    
    async def create_protocol_workers(self) -> None:
        """Create BaseAgent workers with different protocol executors"""
        self.output.info("Creating BaseAgent instances with protocol-specific executors...")
        
        qa_config = self._convert_config_for_qa()
        worker_start_port = self.config.get("qa", {}).get("worker", {}).get("start_port", 10001)
        
        # Define protocol assignments
        protocol_assignments = [
            ("acp", "ACP-Worker-1"),
            ("anp", "ANP-Worker-2"), 
            ("agora", "Agora-Worker-3"),
            ("a2a", "A2A-Worker-4")
        ]
        
        for i, (protocol, agent_id) in enumerate(protocol_assignments):
            try:
                self.output.info(f"Creating {protocol.upper()} worker: {agent_id}")
                
                # Create protocol-specific executor wrapper
                executor_wrapper = create_protocol_worker(protocol, qa_config)
                
                # Validate executor interface
                if not validate_executor_interface(executor_wrapper):
                    raise RuntimeError(f"Invalid executor interface for {protocol}")
                
                # Create BaseAgent with protocol executor
                agent_port = worker_start_port + i
                
                # Import appropriate server adapter
                if protocol == "acp":
                    from server_adapters.acp_adapter import ACPServerAdapter
                    server_adapter = ACPServerAdapter()
                elif protocol == "anp":
                    from server_adapters.anp_adapter import ANPServerAdapter
                    server_adapter = ANPServerAdapter()
                elif protocol == "agora":
                    from server_adapters.agora_adapter import AgoraServerAdapter
                    server_adapter = AgoraServerAdapter()
                else:  # a2a
                    from server_adapters.a2a_adapter import A2AServerAdapter
                    server_adapter = A2AServerAdapter()
                
                # Create BaseAgent
                base_agent = BaseAgent(
                    agent_id=agent_id,
                    host="localhost",
                    port=agent_port,
                    server_adapter=server_adapter
                )
                
                # Set the wrapped executor
                base_agent._executor = executor_wrapper
                
                # Initialize BaseAgent
                await base_agent.initialize()
                
                # Register with NetworkBase
                await self.network_base.register_agent(agent_id, base_agent.base_url)
                
                # Track agent
                self.base_agents[agent_id] = base_agent
                self.protocol_map[agent_id] = protocol
                
                self.output.success(f"✅ {protocol.upper()} worker created: {agent_id} @ {base_agent.base_url}")
                
            except Exception as e:
                self.output.error(f"❌ Failed to create {protocol} worker: {e}")
                # Continue with other protocols
                continue
    
    async def setup_network_topology(self) -> None:
        """Setup network topology using src/core/network.py"""
        self.output.info("Setting up meta protocol network topology...")
        
        # Get all worker IDs
        worker_ids = [aid for aid in self.base_agents.keys() if "worker" in aid.lower()]
        
        if len(worker_ids) < 2:
            self.output.error("Not enough workers for topology setup")
            return
        
        # Setup full mesh topology to enable cross-protocol communication
        await self.network_base.setup_full_mesh()
        
        self.output.success("Full mesh topology established for cross-protocol communication")
    
    async def test_cross_protocol_communication(self) -> None:
        """Test communication between different protocol workers"""
        self.output.info("=== Cross-Protocol Communication Test ===")
        
        worker_ids = [aid for aid in self.base_agents.keys() if "worker" in aid.lower()]
        
        if len(worker_ids) < 2:
            self.output.error("Need at least 2 workers for cross-protocol test")
            return
        
        # Test questions for different protocols
        test_cases = [
            ("What is machine learning?", "acp"),
            ("Explain blockchain technology", "anp"),
            ("What are the benefits of AI?", "agora"),
            ("How does quantum computing work?", "a2a")
        ]
        
        results = []
        
        for question, target_protocol in test_cases:
            # Find worker with target protocol
            target_worker = None
            for worker_id in worker_ids:
                if self.protocol_map.get(worker_id) == target_protocol:
                    target_worker = worker_id
                    break
            
            if not target_worker:
                self.output.error(f"No {target_protocol} worker found")
                continue
            
            try:
                self.output.info(f"Sending to {target_protocol.upper()} worker: {question[:50]}...")
                
                # Use NetworkBase to route message
                response = await self.network_base.route_message(
                    src_id=worker_ids[0],  # Use first worker as sender
                    dst_id=target_worker,
                    message=question
                )
                
                results.append({
                    "protocol": target_protocol,
                    "worker": target_worker,
                    "question": question,
                    "response": response,
                    "status": "success"
                })
                
                self.output.success(f"✅ {target_protocol.upper()} worker responded")
                
            except Exception as e:
                results.append({
                    "protocol": target_protocol,
                    "worker": target_worker,
                    "question": question,
                    "error": str(e),
                    "status": "failed"
                })
                self.output.error(f"❌ {target_protocol.upper()} worker failed: {e}")
        
        # Display results summary
        self.output.info("=== Test Results Summary ===")
        success_count = sum(1 for r in results if r["status"] == "success")
        total_count = len(results)
        self.output.info(f"Success rate: {success_count}/{total_count}")
        
        for result in results:
            if result["status"] == "success":
                self.output.success(f"✅ {result['protocol'].upper()}: Communication successful")
            else:
                self.output.error(f"❌ {result['protocol'].upper()}: {result.get('error', 'Unknown error')}")
    
    async def display_network_status(self) -> None:
        """Display network status"""
        self.output.info("=== Meta Protocol Network Status ===")
        
        # BaseAgent health checks
        for agent_id, agent in self.base_agents.items():
            protocol = self.protocol_map[agent_id]
            try:
                # Use BaseAgent's health check
                is_healthy = await agent.health_check() if hasattr(agent, 'health_check') else True
                status_icon = "✅" if is_healthy else "❌"
                self.output.info(f"{status_icon} {protocol.upper()}: {agent_id} @ {agent.base_url}")
            except Exception as e:
                self.output.error(f"❌ {protocol.upper()}: {agent_id} - Error: {e}")
        
        # Network metrics from streaming_queue/core/network_base.py
        try:
            self.output.info("=== Network Metrics (streaming_queue/core) ===")
            self.output.info(f"Total agents: {len(self.base_agents)}")
            self.output.info(f"Protocol diversity: {len(set(self.protocol_map.values()))} protocols")
            
            # Get connection info from NetworkBase
            connections = getattr(self.network_base, '_connections', {})
            self.output.info(f"Total connections: {len(connections)}")
        except Exception as e:
            self.output.error(f"Error getting network metrics: {e}")
    
    async def run(self) -> None:
        """Run meta protocol demonstration"""
        try:
            self.output.info("🚀 Meta Protocol Integration with src/core")
            self.output.info("=" * 60)
            
            # Create protocol workers
            await self.create_protocol_workers()
            
            # Setup network topology
            await self.setup_network_topology()
            
            # Display status
            await self.display_network_status()
            
            # Test cross-protocol communication
            await self.test_cross_protocol_communication()
            
            # Final status
            await self.display_network_status()
            
            self.output.success("Meta protocol integration demonstration completed!")
            
        except Exception as e:
            self.output.error(f"Meta protocol runner error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup BaseAgents
            self.output.info("Cleaning up BaseAgent instances...")
            for agent_id, agent in self.base_agents.items():
                try:
                    await agent.close()
                    protocol = self.protocol_map[agent_id]
                    self.output.info(f"Closed {protocol} BaseAgent: {agent_id}")
                except Exception as e:
                    self.output.error(f"Error closing {agent_id}: {e}")
            
            self.output.success("Meta protocol cleanup completed")


# Main execution
async def main():
    runner = MetaProtocolRunner()
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
