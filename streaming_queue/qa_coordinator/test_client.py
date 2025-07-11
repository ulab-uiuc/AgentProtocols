#!/usr/bin/env python3
"""
Test client for QA Coordinator Agent - demonstrates AgentNetwork integration
"""

import asyncio
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_network.network import AgentNetwork
from agent_network.base_agent import BaseAgent
from agent_network.streaming_queue.qa_coordinator.agent_executor import QACoordinatorExecutor
from agent_network.streaming_queue.qa_worker.agent_executor import QAAgentExecutor


async def setup_test_network():
    """Set up a test network with Coordinator and Worker agents."""
    print("Setting up test network...")
    
    # Create AgentNetwork instance
    network = AgentNetwork()
    
    # Create Coordinator agent
    coordinator_executor = QACoordinatorExecutor()
    coordinator = await BaseAgent.create_a2a(
        agent_id="Coordinator",
        port=9998,
        executor=coordinator_executor
    )
    
    # Create Worker agents
    worker_agents = []
    worker_ids = []
    
    for i in range(1, 5):  # Create 4 worker agents
        worker_id = f"Worker-{i}"
        worker_executor = QAAgentExecutor()
        worker = await BaseAgent.create_a2a(
            agent_id=worker_id,
            port=10000 + i,
            executor=worker_executor
        )
        worker_agents.append(worker)
        worker_ids.append(worker_id)
    
    # Register all agents in network
    await network.register_agent(coordinator)
    for worker in worker_agents:
        await network.register_agent(worker)
    
    # Setup star topology with Coordinator as center
    network.setup_star_topology(center_id="Coordinator")
    
    # Configure coordinator with network and worker IDs
    coordinator_executor.coordinator.set_network(network, worker_ids)
    
    print(f"Network setup complete:")
    print(f"- Coordinator: {coordinator.get_listening_address()}")
    for worker in worker_agents:
        print(f"- {worker.agent_id}: {worker.get_listening_address()}")
    
    return network, coordinator, worker_agents


async def test_dispatch():
    """Test the dispatch functionality."""
    print("\n" + "="*50)
    print("Testing QA Coordinator Dispatch")
    print("="*50)
    
    # Setup network
    network, coordinator, workers = await setup_test_network()
    
    try:
        # Test status check
        print("\n1. Testing status check...")
        status_payload = {
            "params": {
                "message": {
                    "text": "status",
                    "messageId": "test-status"
                }
            }
        }
        
        # Since we can't directly call the executor here, we'll access it through the coordinator
        coordinator_executor = coordinator._server_adapter._executor_ref
        if hasattr(coordinator_executor, 'coordinator'):
            result = await coordinator_executor.coordinator.dispatch_round("marco_1000.jsonl")
            print(f"Dispatch result:\n{result}")
        
        # Wait a bit for processing
        await asyncio.sleep(1)
        
        # Show network metrics
        metrics = network.snapshot_metrics()
        print(f"\nNetwork metrics:")
        print(f"- Agents: {metrics['agent_count']}")
        print(f"- Edges: {metrics['edge_count']}")
        print(f"- Topology: {metrics['topology']}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        await coordinator.stop()
        for worker in workers:
            await worker.stop()


async def test_manual_dispatch():
    """Test manual dispatch without full network setup."""
    print("\n" + "="*50)
    print("Testing Manual Dispatch")
    print("="*50)
    
    # Create coordinator instance
    coordinator = QACoordinatorExecutor()
    
    # Create mock network
    class MockNetwork:
        def __init__(self):
            self.messages = []
            
        async def route_message(self, src, dst, payload):
            print(f"MockNetwork: {src} -> {dst}")
            print(f"Payload: {payload}")
            self.messages.append((src, dst, payload))
            return {"status": "success", "response": f"Mock response from {dst}"}
    
    mock_network = MockNetwork()
    mock_workers = ["Worker-1", "Worker-2", "Worker-3", "Worker-4"]
    
    # Configure coordinator
    coordinator.coordinator.set_network(mock_network, mock_workers)
    
    # Test dispatch
    result = await coordinator.coordinator.dispatch_round("marco_1000.jsonl")
    print(f"\nDispatch result:\n{result}")
    
    print(f"\nMessages sent: {len(mock_network.messages)}")
    for i, (src, dst, payload) in enumerate(mock_network.messages[:5]):  # Show first 5
        msg_text = payload['params']['message']['text'][:50]
        print(f"{i+1}. {src} -> {dst}: {msg_text}...")


if __name__ == "__main__":
    print("QA Coordinator Test Client")
    print("Choose test mode:")
    print("1. Full network test (requires A2A SDK)")
    print("2. Manual dispatch test (mock network)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(test_dispatch())
    elif choice == "2":
        asyncio.run(test_manual_dispatch())
    else:
        print("Invalid choice. Running manual dispatch test...")
        asyncio.run(test_manual_dispatch()) 