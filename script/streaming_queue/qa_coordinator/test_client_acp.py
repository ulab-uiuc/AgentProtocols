#!/usr/bin/env python3
"""
Test client for QA Coordinator Agent (ACP version) - demonstrates AgentNetwork integration with ACP protocol
"""

import asyncio
import sys
from pathlib import Path

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "script" / "streaming_queue"))

from network import AgentNetwork
from base_agent import BaseAgent
from qa_coordinator.agent_executor_acp import QACoordinatorExecutorACP
from qa_worker.agent_executor_acp import QAAgentExecutorACP


async def setup_test_network_acp():
    """Set up a test network with ACP Coordinator and Worker agents."""
    print("Setting up ACP test network...")

    # Create AgentNetwork instance
    network = AgentNetwork()

    # Create Coordinator agent using ACP protocol
    coordinator_executor_instance = QACoordinatorExecutorACP(coordinator_id="Coordinator-ACP")

    # Create callable executor for ACP interface
    async def coordinator_executor(messages, context):
        async for result in coordinator_executor_instance.execute(messages, context):
            yield result

    coordinator = await BaseAgent.create_acp(
        agent_id="Coordinator-ACP",
        port=9998,
        executor=coordinator_executor
    )

    # Create Worker agents using ACP protocol
    worker_agents = []
    worker_ids = []

    for i in range(1, 5):  # Create 4 worker agents
        worker_id = f"Worker-ACP-{i}"
        worker_executor_instance = QAAgentExecutorACP()

        # Create callable executor for ACP interface
        async def worker_executor(messages, context):
            async for result in worker_executor_instance.execute(messages, context):
                yield result

        worker = await BaseAgent.create_acp(
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
    await network.setup_star_topology(center_id="Coordinator-ACP")

    # Configure coordinator with network and worker IDs
    coordinator_executor_instance.set_agent_network(network)
    coordinator_executor_instance.coordinator.worker_ids = worker_ids

    print("ACP Network setup complete:")
    print(f"- Coordinator: {coordinator.get_listening_address()}")
    for worker in worker_agents:
        print(f"- {worker.agent_id}: {worker.get_listening_address()}")

    return network, coordinator, worker_agents, coordinator_executor_instance


async def test_dispatch_acp():
    """Test the ACP dispatch functionality."""
    print("\n" + "="*50)
    print("Testing ACP QA Coordinator Dispatch")
    print("="*50)

    # Setup network
    network, coordinator, workers, coordinator_executor_instance = await setup_test_network_acp()

    try:
        # Test status check
        print("\n1. Testing status check...")

        # Get the coordinator executor
        coordinator_executor_ref = coordinator_executor_instance
        if coordinator_executor_ref and hasattr(coordinator_executor_ref, 'coordinator'):
            result = await coordinator_executor_ref.coordinator.dispatch_round()
            print(f"Dispatch result:\n{result}")
        else:
            print("Warning: Could not access coordinator executor directly")
            print("Testing via ACP message...")

            # Test via ACP message format
            from acp_sdk.models import Message, MessagePart

            # Create test message
            message = Message(parts=[MessagePart(text="dispatch")])

            # Create mock context
            from acp_sdk.server import Context
            context = Context(
                session=None,
                store=None,
                loader=None,
                executor=None,
                request=None,
                yield_queue=None,
                yield_resume_queue=None
            )

            # Get executor and test it
            executor = coordinator_executor_ref or QACoordinatorExecutorACP()
            executor.set_agent_network(network)

            # Execute the dispatch command
            async for result in executor.execute([message], context):
                print(f"ACP Result: {result.text if hasattr(result, 'text') else result}")

        # Wait a bit for processing
        await asyncio.sleep(2)

        # Show network metrics
        metrics = network.snapshot_metrics()
        print("\nNetwork metrics:")
        print(f"- Agents: {metrics['agent_count']}")
        print(f"- Edges: {metrics['edge_count']}")
        print(f"- Topology: {metrics['topology']}")

        # Test health check
        print("\n2. Testing health check...")
        health_status = await network.health_check()
        print(f"Health status: {health_status}")

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


async def test_manual_dispatch_acp():
    """Test manual ACP dispatch without full network setup."""
    print("\n" + "="*50)
    print("Testing Manual ACP Dispatch")
    print("="*50)

    # Create coordinator instance
    coordinator_executor = QACoordinatorExecutorACP()

    # Create mock network
    class MockNetworkACP:
        def __init__(self):
            self.messages = []

        async def route_message(self, src, dst, payload):
            print(f"MockNetworkACP: {src} -> {dst}")
            print(f"Payload: {payload}")
            self.messages.append((src, dst, payload))
            # Return mock ACP response
            return {
                "results": [
                    {
                        "text": f"Mock ACP response from {dst}",
                        "type": "text"
                    }
                ],
                "status": "success"
            }

    mock_network = MockNetworkACP()
    mock_workers = ["Worker-ACP-1", "Worker-ACP-2", "Worker-ACP-3", "Worker-ACP-4"]

    # Configure coordinator
    coordinator_executor.set_agent_network(mock_network)
    coordinator_executor.coordinator.worker_ids = mock_workers

    # Update data path to use correct location
    coordinator_executor.coordinator.data_path = str(Path(__file__).parent.parent / "data" / "top1000_simplified.jsonl")

    # Test dispatch using ACP message format
    from acp_sdk.models import Message, MessagePart
    from acp_sdk.server import Context

    message = Message(parts=[MessagePart(text="dispatch")])

    # Create context with required parameters (using None for mock testing)
    context = Context(
        session=None,
        store=None,
        loader=None,
        executor=None,
        request=None,
        yield_queue=None,
        yield_resume_queue=None
    )

    print("Executing ACP dispatch command...")
    async for result in coordinator_executor.execute([message], context):
        print(f"ACP Result: {result.text if hasattr(result, 'text') else result}")

    print(f"\nMessages sent: {len(mock_network.messages)}")
    for i, (src, dst, payload) in enumerate(mock_network.messages[:5]):  # Show first 5
        if 'messages' in payload and payload['messages']:
            msg_content = payload['messages'][0]['parts'][0].get('content', '')[:50]
            print(f"{i+1}. {src} -> {dst}: {msg_content}...")


async def test_single_question_acp():
    """Test processing a single question through ACP."""
    print("\n" + "="*50)
    print("Testing Single Question ACP Processing")
    print("="*50)

    # Create coordinator instance
    coordinator_executor = QACoordinatorExecutorACP()

    # Create a more realistic mock network
    class RealisticMockNetwork:
        def __init__(self):
            self.messages = []

        async def route_message(self, src, dst, payload):
            print(f"Network: {src} -> {dst}")
            self.messages.append((src, dst, payload))

            # Extract question from ACP message format
            question = "Unknown question"
            if 'messages' in payload and payload['messages']:
                for msg in payload['messages']:
                    if 'parts' in msg and msg['parts']:
                        question = msg['parts'][0].get('content', 'Unknown question')
                        break

            # Return realistic mock answer
            return {
                "results": [
                    {
                        "text": f"Mock answer from {dst}: This is a simulated response to '{question[:30]}...'",
                        "type": "text"
                    }
                ],
                "status": "success"
            }

    mock_network = RealisticMockNetwork()
    mock_workers = ["Worker-ACP-1", "Worker-ACP-2"]

    # Configure coordinator
    coordinator_executor.set_agent_network(mock_network)
    coordinator_executor.coordinator.worker_ids = mock_workers

    # Update data path to use correct location
    coordinator_executor.coordinator.data_path = str(Path(__file__).parent.parent / "data" / "top1000_simplified.jsonl")

    # Test single question processing
    test_question = {
        "id": 1,
        "q": "What is the capital of France?"
    }

    print(f"Processing test question: {test_question}")
    result = await coordinator_executor.coordinator.process_single_question(test_question)
    print(f"Result: {result}")

    print(f"\nNetwork messages sent: {len(mock_network.messages)}")
    for src, dst, payload in mock_network.messages:
        print(f"- {src} -> {dst}")


if __name__ == "__main__":
    print("QA Coordinator ACP Test Client")
    print("Choose test mode:")
    print("1. Full ACP network test (requires ACP SDK)")
    print("2. Manual ACP dispatch test (mock network)")
    print("3. Single question processing test")

    choice = input("Enter choice (1, 2, or 3): ").strip()

    if choice == "1":
        asyncio.run(test_dispatch_acp())
    elif choice == "2":
        asyncio.run(test_manual_dispatch_acp())
    elif choice == "3":
        asyncio.run(test_single_question_acp())
    else:
        print("Invalid choice. Running manual dispatch test...")
        asyncio.run(test_manual_dispatch_acp())
