#!/usr/bin/env python3
"""
Simplified Fail-Storm Meta Network Demo

This demo shows the meta protocol structure without requiring all SDK dependencies.
It demonstrates the configuration, network topology, and basic functionality.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List

# Colorama for output
try:
    from colorama import init, Fore, Style
    init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    class Fore:
        GREEN = BLUE = YELLOW = RED = WHITE = CYAN = ""
    class Style:
        RESET_ALL = BRIGHT = ""


class ColoredOutput:
    """Colored output helper."""
    
    def success(self, message: str) -> None:
        print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")
    
    def info(self, message: str) -> None:
        print(f"{Fore.BLUE}ℹ️  {message}{Style.RESET_ALL}")
    
    def warning(self, message: str) -> None:
        print(f"{Fore.YELLOW}⚠️  {message}{Style.RESET_ALL}")
    
    def error(self, message: str) -> None:
        print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")
    
    def progress(self, message: str) -> None:
        print(f"{Fore.WHITE}   {message}{Style.RESET_ALL}")


class MockMetaAgent:
    """Mock meta agent for demonstration."""
    
    def __init__(self, agent_id: str, protocol: str, port: int):
        self.agent_id = agent_id
        self.protocol = protocol
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.status = "healthy"
    
    async def start(self):
        """Mock start method."""
        await asyncio.sleep(0.1)  # Simulate startup time
        return True
    
    async def health_check(self):
        """Mock health check."""
        return {"status": self.status, "url": self.url, "protocol": self.protocol}
    
    async def send_task(self, task_data: Dict[str, Any]):
        """Mock task processing."""
        await asyncio.sleep(0.2)  # Simulate processing time
        question = task_data.get("questions", ["No question"])[0]
        
        # Mock different responses based on protocol
        responses = {
            "acp": f"ACP response: {question} - Processed with streaming capabilities",
            "anp": f"ANP response: {question} - Processed with DID authentication", 
            "agora": f"Agora response: {question} - Processed with tool integration",
            "a2a": f"A2A response: {question} - Processed with event streaming"
        }
        
        return {
            "answers": [responses.get(self.protocol, f"Generic response: {question}")],
            "protocol": self.protocol,
            "agent_id": self.agent_id,
            "response_time": 0.2,
            "success": True
        }
    
    async def stop(self):
        """Mock stop method."""
        await asyncio.sleep(0.1)
        self.status = "stopped"


class SimplifiedMetaCoordinator:
    """Simplified meta coordinator for demonstration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents: Dict[str, MockMetaAgent] = {}
        self.protocol_types: Dict[str, str] = {}
        self.output = ColoredOutput()
        
        # Stats tracking
        self.stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "protocol_stats": {}
        }
    
    async def create_agents(self):
        """Create mock agents for each protocol."""
        protocols_config = self.config.get("protocols", {})
        base_port = self.config.get("network", {}).get("base_port", 9000)
        
        port_offset = 0
        for protocol in ["acp", "anp", "agora", "a2a"]:
            protocol_settings = protocols_config.get(protocol, {})
            if protocol_settings.get("enabled", True):
                agent_count = protocol_settings.get("agent_count", 2)
                
                for i in range(agent_count):
                    agent_id = f"{protocol.upper()}-Agent-{i+1}"
                    port = base_port + port_offset
                    
                    agent = MockMetaAgent(agent_id, protocol, port)
                    self.agents[agent_id] = agent
                    self.protocol_types[agent_id] = protocol
                    
                    # Initialize protocol stats
                    if protocol not in self.stats["protocol_stats"]:
                        self.stats["protocol_stats"][protocol] = {
                            "agents": 0,
                            "tasks_processed": 0,
                            "avg_response_time": 0.0
                        }
                    self.stats["protocol_stats"][protocol]["agents"] += 1
                    
                    port_offset += 1
        
        self.output.success(f"Created {len(self.agents)} mock agents across {len(set(self.protocol_types.values()))} protocols")
    
    async def start_all_agents(self):
        """Start all agents."""
        self.output.info("Starting all agents...")
        
        for agent_id, agent in self.agents.items():
            protocol = self.protocol_types[agent_id]
            await agent.start()
            self.output.progress(f"  {protocol.upper()}: {agent_id} @ {agent.url}")
        
        self.output.success("All agents started successfully")
    
    async def health_check_all(self):
        """Check health of all agents."""
        self.output.info("Performing health checks...")
        
        healthy_count = 0
        for agent_id, agent in self.agents.items():
            health = await agent.health_check()
            protocol = self.protocol_types[agent_id]
            
            if health["status"] == "healthy":
                self.output.success(f"  {protocol.upper()}: {agent_id} - Healthy")
                healthy_count += 1
            else:
                self.output.error(f"  {protocol.upper()}: {agent_id} - {health['status']}")
        
        self.output.info(f"Health Check: {healthy_count}/{len(self.agents)} agents healthy")
        return healthy_count == len(self.agents)
    
    async def test_cross_protocol_communication(self):
        """Test cross-protocol task distribution."""
        self.output.info("Testing Cross-Protocol Communication...")
        
        test_questions = self.config.get("shard_qa", {}).get("test_questions", [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain quantum computing.",
            "What is blockchain technology?"
        ])
        
        results = []
        for i, (agent_id, agent) in enumerate(self.agents.items()):
            question = test_questions[i % len(test_questions)]
            protocol = self.protocol_types[agent_id]
            
            self.output.progress(f"  Testing {protocol.upper()} agent {agent_id}...")
            
            task_data = {
                "questions": [question],
                "shard_id": f"test_shard_{i}",
                "context": {"test": True}
            }
            
            result = await agent.send_task(task_data)
            results.append((agent_id, protocol, result))
            
            if result["success"]:
                answer = result["answers"][0][:50] + "..." if len(result["answers"][0]) > 50 else result["answers"][0]
                response_time = result["response_time"]
                self.output.success(f"    ✅ Response: {answer} ({response_time:.2f}s)")
                
                # Update stats
                self.stats["successful_tasks"] += 1
                self.stats["protocol_stats"][protocol]["tasks_processed"] += 1
            else:
                self.output.error(f"    ❌ Error: {result.get('error', 'Unknown error')}")
                self.stats["failed_tasks"] += 1
            
            self.stats["total_tasks"] += 1
        
        successful = sum(1 for _, _, result in results if result["success"])
        self.output.info(f"Communication Test: {successful}/{len(results)} agents responded successfully")
        
        return results
    
    async def simulate_fault_injection(self):
        """Simulate fault injection for fail-storm testing."""
        self.output.info("Simulating Fault Injection...")
        
        # Simulate failure of 25% of agents (1 agent per protocol)
        agent_ids = list(self.agents.keys())
        failed_agents = []
        protocols_failed = set()
        
        for agent_id in agent_ids:
            protocol = self.protocol_types[agent_id]
            if protocol not in protocols_failed and len(failed_agents) < len(agent_ids) // 2:
                failed_agents.append(agent_id)
                protocols_failed.add(protocol)
        
        self.output.warning(f"Simulating failure of {len(failed_agents)} agents:")
        for agent_id in failed_agents:
            protocol = self.protocol_types[agent_id]
            self.agents[agent_id].status = "failed"
            self.output.progress(f"  💥 Failed {protocol.upper()}: {agent_id}")
        
        # Test network resilience
        self.output.info("Testing Network Resilience...")
        healthy_agents = [aid for aid in agent_ids if aid not in failed_agents]
        
        resilience_results = []
        for agent_id in healthy_agents:
            protocol = self.protocol_types[agent_id]
            agent = self.agents[agent_id]
            
            if agent.status == "healthy":
                task_data = {
                    "questions": ["Test resilience question"],
                    "shard_id": "resilience_test",
                    "context": {"resilience_test": True}
                }
                
                result = await agent.send_task(task_data)
                resilience_results.append(result["success"])
                
                if result["success"]:
                    self.output.success(f"    ✅ {protocol.upper()}: {agent_id} still responsive")
                else:
                    self.output.error(f"    ❌ {protocol.upper()}: {agent_id} affected by network failure")
        
        resilient_count = sum(resilience_results)
        self.output.info(f"Resilience Test: {resilient_count}/{len(healthy_agents)} healthy agents still responsive")
    
    async def display_metrics(self):
        """Display comprehensive metrics."""
        self.output.info("📊 Fail-Storm Meta-Protocol Metrics")
        self.output.info("=" * 50)
        
        # Overall stats
        self.output.info(f"Total Agents: {len(self.agents)}")
        self.output.info(f"Protocols: {', '.join(set(self.protocol_types.values()))}")
        self.output.info(f"Total Tasks: {self.stats['total_tasks']}")
        self.output.info(f"Successful: {self.stats['successful_tasks']}")
        self.output.info(f"Failed: {self.stats['failed_tasks']}")
        
        # Protocol-specific metrics
        self.output.info("\n📈 Protocol Performance:")
        for protocol, stats in self.stats["protocol_stats"].items():
            self.output.progress(f"  {protocol.upper()}:")
            self.output.progress(f"    Agents: {stats['agents']}")
            self.output.progress(f"    Tasks Processed: {stats['tasks_processed']}")
        
        # Save metrics
        metrics_file = Path("script/fail_storm_recovery/results/meta_demo_metrics.json")
        metrics_file.parent.mkdir(exist_ok=True)
        
        with open(metrics_file, 'w') as f:
            json.dump({
                "total_agents": len(self.agents),
                "protocols": list(set(self.protocol_types.values())),
                "stats": self.stats,
                "agent_details": {
                    agent_id: {
                        "protocol": self.protocol_types[agent_id],
                        "url": agent.url,
                        "status": agent.status
                    }
                    for agent_id, agent in self.agents.items()
                }
            }, f, indent=2)
        
        self.output.success(f"Metrics saved to {metrics_file}")
    
    async def cleanup(self):
        """Cleanup all agents."""
        self.output.info("Cleaning up agents...")
        
        for agent_id, agent in self.agents.items():
            await agent.stop()
            protocol = self.protocol_types[agent_id]
            self.output.progress(f"  Stopped {protocol.upper()}: {agent_id}")
        
        self.output.success("Cleanup completed")


async def main():
    """Main demo function."""
    # Configuration (same structure as config_meta.yaml)
    config = {
        "core": {
            "protocol": "meta",
            "type": "openai",
            "name": "gpt-4o",
            "temperature": 0.0
        },
        "network": {
            "topology": "mesh",
            "base_port": 9000
        },
        "protocols": {
            "acp": {"enabled": True, "agent_count": 2},
            "anp": {"enabled": True, "agent_count": 2},
            "agora": {"enabled": True, "agent_count": 2},
            "a2a": {"enabled": True, "agent_count": 2}
        },
        "shard_qa": {
            "test_questions": [
                "What is the capital of France?",
                "Explain quantum computing in simple terms.",
                "How does photosynthesis work?",
                "What are the benefits of renewable energy?"
            ]
        }
    }
    
    output = ColoredOutput()
    
    try:
        output.info("🌪️  Fail-Storm Meta-Protocol Network Demo")
        output.info("=" * 70)
        
        # Create coordinator
        coordinator = SimplifiedMetaCoordinator(config)
        
        # Create and start agents
        await coordinator.create_agents()
        await coordinator.start_all_agents()
        
        # Test network functionality
        await coordinator.health_check_all()
        await coordinator.test_cross_protocol_communication()
        
        # Simulate fail-storm scenario
        await coordinator.simulate_fault_injection()
        
        # Display results
        await coordinator.display_metrics()
        
        output.success("\n✅ Fail-Storm Meta-Protocol Demo Completed!")
        output.info("\n📋 What this demo showed:")
        output.progress("  • Meta-protocol network with 8 agents (2 per protocol)")
        output.progress("  • Cross-protocol communication capabilities")
        output.progress("  • Fault injection and resilience testing")
        output.progress("  • Performance metrics collection")
        output.progress("  • Network topology management")
        
        output.info("\n🔧 Implementation Files Created:")
        output.progress("  • protocol_backends/meta_protocol/: Meta protocol integration")
        output.progress("  • config_meta.yaml: Network configuration")
        output.progress("  • runners/run_meta_network.py: Full network runner")
        output.progress("  • This demo shows the concept without SDK dependencies")
        
    except Exception as e:
        output.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'coordinator' in locals():
            await coordinator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
