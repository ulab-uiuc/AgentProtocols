"""Simple test script for network deliver functionality."""
import asyncio
import sys
from pathlib import Path
import json
import time
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from protocols.protocol_factory import protocol_factory

async def test_deliver(protocol: str = 'dummy'):
    """Test network deliver functionality."""
    print(f"🧪 Testing Network Deliver Functionality - Protocol: {protocol.upper()}")
    print("-" * 50)
    
    # Create agent configurations
    agents_config = [
        {'id': i, 'name': f'TestAgent_{i}', 'tool': 'create_chat_completion', 'port': 9000 + i}
        for i in range(3)
    ]
    
    # Create multi-agent system using protocol factory
    network, agents = protocol_factory.create_multi_agent_system(
        agents_config, 
        task_id="test_protocol_001", 
        protocol=protocol
    )
    
    for i, agent in enumerate(agents):
        print(f"✅ Created agent {i}: {agent.name} (Protocol: {protocol})")
    
    try:
        # Start network
        print("\n🌐 Starting network...")
        await network.start()
        
        # Test message delivery
        print("\n📤 Testing message delivery...")
        
        test_messages = [
            {"type": "greeting", "content": "Hello from test!"},
            {"type": "task", "content": "Process this data", "priority": 1},
            {"type": "status", "content": "System ready"}
        ]
        
        # Send messages between agents
        for i, msg in enumerate(test_messages):
            src_agent = i % len(agents)
            dst_agent = (i + 1) % len(agents)
            
            print(f"📨 Sending message {i+1}: Agent {src_agent} -> Agent {dst_agent}")
            print(f"   Message: {msg}")
            
            await network.deliver(dst_agent, msg)
            await asyncio.sleep(0.2)  # Small delay for processing
        
        # Wait for message processing
        await asyncio.sleep(1.0)
        
        # Check message pool
        print("\n📊 Checking message pool...")
        summary = network.get_message_pool_summary()
        print(f"   Total input messages: {summary['total_input_messages']}")
        print(f"   Active agents: {summary['total_agents']}")
        print(f"   Conversation turns: {summary['conversation_turns']}")
        
        # Test polling
        print("\n📥 Testing message polling...")
        messages = await network.poll()
        print(f"   Polled {len(messages)} messages")
        
        if summary['total_input_messages'] > 0:
            print("✅ Message delivery test PASSED")
            print("   - Messages were successfully delivered")
            print("   - Message pool is recording correctly")
        else:
            print("❌ Message delivery test FAILED")
            print("   - No messages recorded in message pool")
        
        # Test message pool logging
        print("\n� Testing message pool logging...")
        await network._log_message_pool_to_workspace()
        print("✅ Message pool logging completed")
        
        print(f"\n� Network metrics:")
        print(f"   - Packets sent: {network.pkt_cnt}")
        print(f"   - Bytes TX: {network.bytes_tx}")
        print(f"   - Bytes RX: {network.bytes_rx}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Stop network
        print("\n🛑 Stopping network...")
        await network.stop()
        print("✅ Test completed")


async def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Network Deliver Functionality")
    parser.add_argument('--protocol', type=str, default='dummy', 
                        help='Protocol to test (default: dummy)')
    parser.add_argument('--list', action='store_true', 
                        help='List available protocols')
    
    args = parser.parse_args()
    
    try:
        # List available protocols if requested
        if args.list:
            print("📋 Available protocols:")
            protocols = protocol_factory.get_available_protocols()
            for proto in protocols:
                info = protocol_factory.get_protocol_info(proto)
                print(f"  🔌 {proto.upper()}: {info.get('description', 'No description')}")
            return
        
        # Validate protocol
        available_protocols = protocol_factory.get_available_protocols()
        if args.protocol not in available_protocols:
            print(f"❌ Error: Protocol '{args.protocol}' not available.")
            print(f"Available protocols: {', '.join(available_protocols)}")
            return
        
        # Run test with selected protocol
        await test_deliver(args.protocol)
        print("\n🎉 All tests completed!")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
