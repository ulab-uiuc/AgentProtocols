#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-protocol compatibility test - Test A2A and Agent Protocol integration
"""

import asyncio
import json
import logging
from typing import Any, Dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import framework components
from src.core.base_agent import BaseAgent
from src.server_adapters import A2AServerAdapter, AgentProtocolServerAdapter

# Simple test executor
class TestExecutor:
    """Test executor that supports both A2A and Agent Protocol interfaces"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
    
    # A2A SDK native interface
    async def execute(self, context, event_queue):
        """A2A SDK native execution interface"""
        message = getattr(context, 'message', 'Unknown message')
        logger.info(f"[A2A] {self.agent_name} processing message: {message}")
        
        # Send response event
        try:
            from a2a.utils import new_agent_text_message
            response = new_agent_text_message(f"A2A response from {self.agent_name}: processed '{message}'")
            await event_queue.put(response)
        except ImportError:
            # If no A2A SDK, create simple response
            response = {
                "type": "text",
                "content": f"A2A response from {self.agent_name}: processed '{message}'"
            }
            await event_queue.put(response)
    
    # Agent Protocol interface
    async def handle_task_creation(self, task):
        """Handle Agent Protocol task creation"""
        logger.info(f"[Agent Protocol] {self.agent_name} creating task: {task.task_id}")
        task.status = "ready"
    
    async def execute_step(self, step):
        """Execute Agent Protocol step"""
        logger.info(f"[Agent Protocol] {self.agent_name} executing step: {step.step_id}")
        
        result = {
            "output": f"Agent Protocol response from {self.agent_name}: processed '{step.input}'",
            "status": "completed",
            "is_last": True,
            "artifacts": []
        }
        
        return result


async def test_a2a_to_a2a():
    """Test communication between A2A agents"""
    print("\n🧪 Testing A2A to A2A communication")
    print("=" * 50)
    
    try:
        # Create two A2A agents
        agent1 = await BaseAgent.create_a2a(
            agent_id="a2a-agent-1",
            host="localhost",
            port=8001,
            executor=TestExecutor("A2A Agent 1")
        )
        
        agent2 = await BaseAgent.create_a2a(
            agent_id="a2a-agent-2", 
            host="localhost",
            port=8002,
            executor=TestExecutor("A2A Agent 2")
        )
        
        # Establish connection
        await agent1.add_connection(
            dst_id="a2a-agent-2",
            base_url="http://localhost:8002",
            protocol="a2a"
        )
        
        # Send message
        response = await agent1.send("a2a-agent-2", {
            "message": "Hello from A2A Agent 1!"
        })
        
        print(f"✅ A2A to A2A communication successful")
        print(f"   Response: {response}")
        
        # Cleanup
        await agent1.stop()
        await agent2.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ A2A to A2A communication failed: {e}")
        return False


async def test_agent_protocol_to_agent_protocol():
    """Test communication between Agent Protocol agents"""
    print("\n🧪 Testing Agent Protocol to Agent Protocol communication")
    print("=" * 50)
    
    try:
        # Create two Agent Protocol agents
        agent1 = await BaseAgent.create_agent_protocol(
            agent_id="ap-agent-1",
            host="localhost", 
            port=8003,
            executor=TestExecutor("AP Agent 1")
        )
        
        agent2 = await BaseAgent.create_agent_protocol(
            agent_id="ap-agent-2",
            host="localhost",
            port=8004,
            executor=TestExecutor("AP Agent 2")
        )
        
        # Establish connection
        await agent1.add_connection(
            dst_id="ap-agent-2",
            base_url="http://localhost:8004",
            protocol="agent_protocol"
        )
        
        # Send message
        response = await agent1.send("ap-agent-2", {
            "message": "Hello from Agent Protocol Agent 1!"
        })
        
        print(f"✅ Agent Protocol to Agent Protocol communication successful")
        print(f"   Response: {response}")
        
        # Cleanup
        await agent1.stop()
        await agent2.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ Agent Protocol to Agent Protocol communication failed: {e}")
        return False


async def test_a2a_to_agent_protocol():
    """Test A2A to Agent Protocol cross-protocol communication"""
    print("\n🧪 Testing A2A to Agent Protocol cross-protocol communication")
    print("=" * 50)
    
    try:
        # Create A2A agent
        a2a_agent = await BaseAgent.create_a2a(
            agent_id="a2a-client",
            host="localhost",
            port=8005,
            executor=TestExecutor("A2A Client")
        )
        
        # Create Agent Protocol agent
        ap_agent = await BaseAgent.create_agent_protocol(
            agent_id="ap-server",
            host="localhost",
            port=8006,
            executor=TestExecutor("AP Server")
        )
        
        # A2A agent connects to Agent Protocol agent
        await a2a_agent.add_connection(
            dst_id="ap-server",
            base_url="http://localhost:8006",
            protocol="agent_protocol"  # explicitly specify protocol
        )
        
        # Send message
        response = await a2a_agent.send("ap-server", {
            "message": "Cross-protocol message from A2A to Agent Protocol!"
        })
        
        print(f"✅ A2A to Agent Protocol cross-protocol communication successful")
        print(f"   Response: {response}")
        
        # Cleanup
        await a2a_agent.stop()
        await ap_agent.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ A2A to Agent Protocol cross-protocol communication failed: {e}")
        return False


async def test_agent_protocol_to_a2a():
    """Test Agent Protocol to A2A cross-protocol communication"""
    print("\n🧪 Testing Agent Protocol to A2A cross-protocol communication")
    print("=" * 50)
    
    try:
        # Create Agent Protocol agent
        ap_agent = await BaseAgent.create_agent_protocol(
            agent_id="ap-client",
            host="localhost",
            port=8007,
            executor=TestExecutor("AP Client")
        )
        
        # Create A2A agent
        a2a_agent = await BaseAgent.create_a2a(
            agent_id="a2a-server",
            host="localhost",
            port=8008,
            executor=TestExecutor("A2A Server")
        )
        
        # Agent Protocol agent connects to A2A agent
        await ap_agent.add_connection(
            dst_id="a2a-server",
            base_url="http://localhost:8008",
            protocol="a2a"  # explicitly specify protocol
        )
        
        # Send message
        response = await ap_agent.send("a2a-server", {
            "message": "Cross-protocol message from Agent Protocol to A2A!"
        })
        
        print(f"✅ Agent Protocol to A2A cross-protocol communication successful")
        print(f"   Response: {response}")
        
        # Cleanup
        await ap_agent.stop()
        await a2a_agent.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ Agent Protocol to A2A cross-protocol communication failed: {e}")
        return False


async def test_auto_protocol_detection():
    """Test automatic protocol detection"""
    print("\n🧪 Testing automatic protocol detection")
    print("=" * 50)
    
    try:
        # Create agents for different protocols
        a2a_agent = await BaseAgent.create_a2a(
            agent_id="a2a-auto",
            host="localhost",
            port=8009,
            executor=TestExecutor("A2A Auto")
        )
        
        ap_agent = await BaseAgent.create_agent_protocol(
            agent_id="ap-auto",
            host="localhost",
            port=8010,
            executor=TestExecutor("AP Auto")
        )
        
        # Create client agent
        client = await BaseAgent.create_a2a(
            agent_id="auto-client",
            host="localhost",
            port=8011,
            executor=TestExecutor("Auto Client")
        )
        
        # Auto-detect A2A protocol
        await client.add_connection(
            dst_id="a2a-auto",
            base_url="http://localhost:8009",
            protocol="auto"  # auto-detect
        )
        
        # Auto-detect Agent Protocol protocol
        await client.add_connection(
            dst_id="ap-auto",
            base_url="http://localhost:8010",
            protocol="auto"  # auto-detect
        )
        
        # Test connections
        response1 = await client.send("a2a-auto", {"message": "Auto-detected A2A!"})
        response2 = await client.send("ap-auto", {"message": "Auto-detected Agent Protocol!"})
        
        print(f"✅ Automatic protocol detection succeeded")
        print(f"   A2A Response: {response1}")
        print(f"   Agent Protocol Response: {response2}")
        
        # Show connection info
        connections = client.get_connection_info()
        print("\n📊 Connection info:")
        for dst_id, info in connections.items():
            print(f"   {dst_id}: {info['protocol']} protocol")
        
        # Cleanup
        await client.stop()
        await a2a_agent.stop()
        await ap_agent.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ Automatic protocol detection failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 Multi-protocol compatibility tests")
    print("============================================================")
    
    tests = [
        ("A2A to A2A", test_a2a_to_a2a),
        ("Agent Protocol to Agent Protocol", test_agent_protocol_to_agent_protocol),
        ("A2A to Agent Protocol", test_a2a_to_agent_protocol),
        ("Agent Protocol to A2A", test_agent_protocol_to_a2a),
        ("Automatic protocol detection", test_auto_protocol_detection),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
            
            # Short delay to avoid port conflicts
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"❌ Test '{test_name}' raised an exception: {e}")
            results.append((test_name, False))
    
    # Summary results
    print("\n📋 Test results summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status} - {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The two protocols are successfully interoperable.")
    else:
        print("⚠️ Some tests failed, please check the error messages.")


if __name__ == "__main__":
    asyncio.run(main())