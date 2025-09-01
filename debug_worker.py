#!/usr/bin/env python3
# Debug script to test worker components directly

import sys
sys.path.append('/Users/ldq/Work/1_Helping_Others/Multiagent-Protocol')

import yaml
import asyncio
from pathlib import Path

# Load config
config_path = Path('/Users/ldq/Work/1_Helping_Others/Multiagent-Protocol/script/streaming_queue/config.yaml')
with open(config_path, 'r') as f:
    full_config = yaml.safe_load(f)

# Prepare core config like the runner does
core_section = full_config.get("core", {})
qa_section = full_config.get("qa", {})

core_config = {
    "model": {
        "type": core_section.get("type", "openai"),
        "name": core_section.get("name", "gpt-3.5-turbo"),
        "temperature": core_section.get("temperature", 0.3),
        "openai_api_key": core_section.get("openai_api_key"),
        "openai_base_url": core_section.get("openai_base_url", "https://api.openai.com/v1")
    },
    "base_url": core_section.get("base_url", "http://localhost:8000/v1"),
    "port": core_section.get("port", 8000),
    "qa": qa_section
}

print("Config loaded.")

async def test_acp_worker():
    """Test the ACPWorkerExecutor directly"""
    try:
        from script.streaming_queue.protocol_backend.acp.worker import ACPWorkerExecutor

        # Create ACP worker executor with same config as the runner
        acp_worker = ACPWorkerExecutor(core_config)
        print("✅ ACPWorkerExecutor created successfully!")

        print("🔧 Testing ACPWorkerExecutor.execute...")

        # This is the format that the coordinator sends
        acp_input = {
            "content": [{"type": "text", "text": "What is the capital of France?"}]
        }

        result = await acp_worker.execute(acp_input)
        print(f"📝 ACP Worker result: {result}")

        # Extract the text from the ACP response
        content = result.get("content", [])
        text = content[0].get("text", "") if content else ""
        print(f"📄 Extracted text: '{text}'")
        print(f"📏 Text length: {len(text) if text else 'None'}")

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_coordinator_to_worker_communication():
    """Test the full coordinator->worker communication flow"""
    try:
        from script.streaming_queue.protocol_backend.acp.coordinator import ACPCoordinatorForACP

        # Create coordinator
        coordinator = ACPCoordinatorForACP(core_config)
        coordinator.worker_ids = ["worker-1", "worker-2", "worker-3", "worker-4"]

        print("✅ ACPCoordinatorForACP created successfully!")
        print("🔧 Testing coordinator send_to_worker...")

        # Test sending to a worker (this simulates what happens during dispatch)
        result = await coordinator.send_to_worker("worker-1", "What is the capital of France?")
        print(f"📝 Coordinator->Worker result: {result}")

        answer = result.get("answer", "")
        print(f"📄 Answer: '{answer}'")
        print(f"📏 Answer length: {len(answer) if answer else 'None'}")

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    print("=" * 60)
    print("🚀 Testing Worker Components")
    print("=" * 60)

    print("\n1️⃣ Testing ACPWorkerExecutor directly...")
    await test_acp_worker()

    print("\n2️⃣ Testing Coordinator->Worker communication...")
    await test_coordinator_to_worker_communication()

    print("\n✅ Tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
