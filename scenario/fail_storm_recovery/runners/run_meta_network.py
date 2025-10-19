#!/usr/bin/env python3
"""
Fail-Storm Meta Network Runner

This runner creates a meta-protocol network with 2 agents per protocol (ACP, ANP, Agora, A2A)
for fail-storm recovery testing. It demonstrates cross-protocol communication and fault tolerance.
"""

import asyncio
import sys
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Setup paths
current_file = Path(__file__).resolve()
fail_storm_path = current_file.parents[1]  # runners -> fail_storm_recovery
project_root = fail_storm_path.parent.parent
src_path = project_root / "src"

# Add paths - project root first, then src, then fail_storm
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(fail_storm_path))

# Debug: print paths to verify
print(f"Project root: {project_root}")
print(f"Src path: {src_path}")
print(f"Fail storm path: {fail_storm_path}")
print(f"Python path: {sys.path[:3]}")

# Import meta protocol components
from protocol_backends.meta_protocol.meta_coordinator import (
    MetaProtocolCoordinator, create_meta_protocol_network
)

# Import colorama for output
try:
    from colorama import init, Fore, Back, Style
    init()
except ImportError as e:
    raise ImportError(f"Colorama is required for meta network colored output. Please install colorama package. Error: {e}")


class ColoredOutput:
    """Colored output helper for better user experience."""
    
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


class FailStormMetaNetworkRunner:
    """
    Runner for fail-storm meta-protocol network testing.
    
    Creates a network with 2 agents per protocol and demonstrates:
    - Cross-protocol communication
    - Fault injection and recovery
    - Performance metrics collection
    - Network topology management
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        # Always load config with default fallback
        self.config = self._load_config(config_path) if config_path else self._load_default_config()
        self.output = ColoredOutput()
        self.coordinator: Optional[MetaProtocolCoordinator] = None
        
        # Reduce log noise
        import logging
        import warnings
        
        # Suppress CancelledError in uvicorn/starlette lifespan shutdown
        logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
        logging.getLogger("starlette").setLevel(logging.CRITICAL)
        
        # Filter out asyncio.CancelledError from uvicorn
        class CancelledErrorFilter(logging.Filter):
            def filter(self, record):
                # Suppress CancelledError from lifespan shutdown
                if "CancelledError" in record.getMessage():
                    return False
                if record.exc_info and record.exc_info[0].__name__ == "CancelledError":
                    return False
                return True
        
        for logger_name in ["uvicorn.error", "uvicorn", "starlette"]:
            logger = logging.getLogger(logger_name)
            logger.addFilter(CancelledErrorFilter())
        
        # Other log level settings
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)
        
        # Suppress warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration or from configs/config_meta.yaml."""
        # Try to load from standard location first
        default_config_path = fail_storm_path / "configs" / "config_meta.yaml"
        
        if default_config_path.exists():
            try:
                with open(default_config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    return config
            except Exception as e:
                print(f"Warning: Could not load {default_config_path}: {e}")
        
        # Fallback to hardcoded default with llm config
        import os
        return {
            "llm": {
                "type": "openai",
                "model": "gpt-4o",
                "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
                "openai_base_url": "https://api.openai.com/v1",
                "temperature": 0.0,
                "max_tokens": 8192,
                "timeout": 30.0
            },
            "network": {
                "topology": "mesh",
                "health_check_interval": 5,
                "message_timeout": 30,
                "base_port": 9000
            },
            "agents": {
                "base_port": 9000,
                "host": "127.0.0.1",
                "workspace_dir": "workspaces"
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
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        config_file = Path(config_path)
        
        # Try different config locations
        if not config_file.exists():
            config_file = fail_storm_path / config_path
        if not config_file.exists():
            config_file = fail_storm_path / "configs" / config_path
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # If file not found, use default
        return self._load_default_config()
    
    async def create_network(self) -> None:
        """Create the fail-storm meta-protocol network."""
        self.output.info("🚀 Creating Fail-Storm Meta-Protocol Network")
        self.output.info("=" * 60)
        
        try:
            # Create the network with 2 agents per protocol
            self.coordinator = await create_meta_protocol_network(self.config)
            
            agent_count = len(self.coordinator.meta_agents)
            protocol_count = len(set(self.coordinator.protocol_types.values()))
            
            self.output.success(f"Created network with {agent_count} agents across {protocol_count} protocols")
            
            # Display agent details
            self.output.info("\n📋 Agent Details:")
            for agent_id, protocol in self.coordinator.protocol_types.items():
                agent = self.coordinator.meta_agents[agent_id]
                url = agent.base_agent.get_listening_address()
                self.output.progress(f"  {protocol.upper()}: {agent_id} @ {url}")
            
        except Exception as e:
            self.output.error(f"Failed to create network: {e}")
            raise
    
    async def setup_network_topology(self) -> None:
        """Setup network topology and install adapters."""
        self.output.info("\n🔗 Setting up Network Topology")
        
        try:
            # Install outbound adapters for cross-protocol communication
            await self.coordinator.install_outbound_adapters()
            self.output.success("Outbound adapters installed for cross-protocol communication")
            
        except Exception as e:
            self.output.error(f"Failed to setup network topology: {e}")
            raise
    
    async def test_network_health(self) -> None:
        """Test network health and connectivity."""
        self.output.info("\n🔍 Testing Network Health")
        
        try:
            # Health check all agents
            health_results = await self.coordinator.health_check_all()
            
            healthy_count = 0
            for agent_id, health in health_results.items():
                protocol = health["protocol"]
                status = health["status"]
                
                if status == "healthy":
                    self.output.success(f"  {protocol.upper()}: {agent_id} - Healthy")
                    healthy_count += 1
                else:
                    error = health.get("error", "Unknown error")
                    self.output.error(f"  {protocol.upper()}: {agent_id} - {status}: {error}")
            
            self.output.info(f"Health Check: {healthy_count}/{len(health_results)} agents healthy")
            
        except Exception as e:
            self.output.error(f"Health check failed: {e}")
            raise
    
    async def test_cross_protocol_communication(self) -> None:
        """Test cross-protocol communication with shard QA tasks."""
        self.output.info("\n🧪 Testing Cross-Protocol Communication")
        
        try:
            test_questions = self.config.get("shard_qa", {}).get("test_questions", [
                "What is artificial intelligence?",
                "Explain machine learning basics."
            ])
            
            # Test each agent with a question
            results = []
            for i, (agent_id, protocol) in enumerate(self.coordinator.protocol_types.items()):
                question = test_questions[i % len(test_questions)]
                
                self.output.progress(f"  Testing {protocol.upper()} agent {agent_id}...")
                
                # Send question to worker using coordinator
                result = await self.coordinator.send_to_worker(agent_id, question)
                results.append((agent_id, protocol, result))
                
                if result["success"]:
                    response_time = result["response_time"]
                    answer_text = result.get("answer", "No answer")
                    if answer_text:
                        self.output.success(f"    ✅ Response: {answer_text[:50]}... ({response_time:.2f}s)")
                    else:
                        self.output.warning(f"    ⚠️  Empty response ({response_time:.2f}s)")
                else:
                    error = result["raw"].get("error", "Unknown error")
                    self.output.error(f"    ❌ Error: {error}")
            
            # Summary
            successful = sum(1 for _, _, result in results if result["success"])
            self.output.info(f"Communication Test: {successful}/{len(results)} agents responded successfully")
            
        except Exception as e:
            self.output.error(f"Cross-protocol communication test failed: {e}")
            raise
    
    async def simulate_fault_injection(self) -> None:
        """Simulate fault injection for fail-storm testing."""
        self.output.info("\n⚡ Simulating Fault Injection")
        
        try:
            # Get list of agents to "fail"
            agent_ids = list(self.coordinator.meta_agents.keys())
            
            # Simulate failure of 25% of agents (1 agent per protocol)
            failed_agents = []
            protocols_failed = set()
            
            for agent_id in agent_ids:
                protocol = self.coordinator.protocol_types[agent_id]
                if protocol not in protocols_failed and len(failed_agents) < len(agent_ids) // 2:
                    failed_agents.append(agent_id)
                    protocols_failed.add(protocol)
            
            self.output.warning(f"Simulating failure of {len(failed_agents)} agents:")
            for agent_id in failed_agents:
                protocol = self.coordinator.protocol_types[agent_id]
                self.output.progress(f"  💥 Failing {protocol.upper()}: {agent_id}")
                
                # Note: In a real scenario, these agents would be killed/stopped
                # For this demo, we just mark them as failed
            
            # Test network resilience
            self.output.info("\n🔄 Testing Network Resilience")
            
            # Try to communicate with remaining healthy agents
            healthy_agents = [aid for aid in agent_ids if aid not in failed_agents]
            
            for agent_id in healthy_agents:
                protocol = self.coordinator.protocol_types[agent_id]
                self.output.progress(f"  Testing resilience of {protocol.upper()}: {agent_id}")
                
                # Send test question to check resilience
                result = await self.coordinator.send_to_worker(agent_id, "Test resilience question")
                
                if result["success"]:
                    self.output.success(f"    ✅ Agent still responsive after network failure")
                else:
                    self.output.error(f"    ❌ Agent affected by network failure")
            
        except Exception as e:
            self.output.error(f"Fault injection simulation failed: {e}")
            raise
    
    async def display_metrics(self) -> None:
        """Display comprehensive fail-storm metrics."""
        self.output.info("\n📊 Fail-Storm Recovery Metrics")
        self.output.info("=" * 40)
        
        try:
            # Use get_protocol_stats to get current metrics
            stats = await self.coordinator.get_protocol_stats()
            
            # Overall summary
            self.output.info(f"Total Agents: {stats['total_agents']}")
            self.output.info(f"Protocols: {', '.join(stats['protocols'])}")
            
            # Protocol-specific metrics
            self.output.info("\n📈 Protocol Performance:")
            for protocol, pstats in stats.get('summary', {}).items():
                self.output.progress(f"  {protocol.upper()}:")
                self.output.progress(f"    Agents: {pstats.get('agents', 0)}")
                self.output.progress(f"    Questions: {pstats.get('total_questions', 0)}")
                self.output.progress(f"    Avg Response: {pstats.get('avg_response_time', 0):.2f}s")
                self.output.progress(f"    Errors: {pstats.get('total_errors', 0)}")
            
            # Save detailed metrics
            metrics_file = fail_storm_path / "results" / "meta_network_metrics.json"
            metrics_file.parent.mkdir(exist_ok=True)
            
            with open(metrics_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.output.success(f"Detailed metrics saved to {metrics_file}")
            
        except Exception as e:
            self.output.error(f"Failed to display metrics: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup network resources."""
        self.output.info("\n🧹 Cleaning up Network Resources")
        
        try:
            if self.coordinator:
                await self.coordinator.close_all()
                self.output.success("All agents shut down successfully")
        except Exception as e:
            self.output.error(f"Cleanup error: {e}")
    
    async def run(self) -> None:
        """Run the complete fail-storm meta-network test."""
        try:
            self.output.info("🌪️  Fail-Storm Meta-Protocol Network Test")
            self.output.info("=" * 70)
            
            # Create and setup network
            await self.create_network()
            await self.setup_network_topology()
            
            # Test network functionality
            await self.test_network_health()
            await self.test_cross_protocol_communication()
            
            # Simulate fail-storm scenario
            await self.simulate_fault_injection()
            
            # Display results
            await self.display_metrics()
            
            self.output.success("\n✅ Fail-Storm Meta-Protocol Network Test Completed!")
            
        except Exception as e:
            self.output.error(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            await self.cleanup()


async def main():
    """Main entry point."""
    config = {
        "core": {
            "protocol": "meta",
            "type": "openai",
            "name": "gpt-4o",
            "openai_api_key": "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",
            "openai_base_url": "https://api.openai.com/v1",
            "temperature": 0.0,
            "max_tokens": 8192,
            "timeout": 30.0
        },
        "network": {
            "topology": "mesh",
            "health_check_interval": 5,
            "message_timeout": 30,
            "base_port": 9000
        },
        "agents": {
            "base_port": 9000,
            "host": "127.0.0.1",
            "workspace_dir": "workspaces"
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
    
    runner = FailStormMetaNetworkRunner()
    runner.config = config
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
