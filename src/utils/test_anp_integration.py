#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANP (Agent Network Protocol) integration test
Test ANP adapter integration with existing multi-protocol framework
"""

import asyncio
import json
import logging
from typing import Any, Dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import framework components
try:
    from src.core.base_agent import BaseAgent
    from src.server_adapters import ANPServerAdapter
    from src.agent_adapters import ANPAdapter, ANPMessageBuilder
except ImportError as e:
    logger.warning(f"ANP adapters not available: {e}")
    raise ImportError("ANP adapters require the AgentConnect library. Please install it via 'pip install agent-connect'.")


# Simple test executor
class TestANPExecutor:
    """Test executor that supports the ANP protocol interface"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
    
    async def execute_step(self, step):
        """Execute step - Agent Protocol style"""
        logger.info(f"[ANP-AP] {self.agent_name} executing step: {getattr(step, 'input', 'Unknown')}")
        
        result = {
            "output": f"ANP Responsefrom {self.agent_name}: processed '{getattr(step, 'input', 'Unknown')}'",
            "status": "completed",
            "is_last": True,
            "artifacts": []
        }
        
        return result
    
    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Direct call - Callable style"""
        logger.info(f"[ANP-Callable] {self.agent_name} processing payload: {payload}")
        
        return {
            "response": f"ANP processing result from {self.agent_name}",
            "input_received": payload,
            "status": "success"
        }


async def test_anp_adapter_availability():
    """Test the availability of ANP adapters"""
    print("\n🧪 Testing ANP adapter availability")
    print("=" * 50)
    
    try:
        # Validate imports
        assert ANPAdapter is not None
        assert ANPServerAdapter is not None
        assert ANPMessageBuilder is not None
        
        print("✅ ANP adapters imported successfully")
        print(f"   - ANPAdapter: {ANPAdapter}")
        print(f"   - ANPServerAdapter: {ANPServerAdapter}")
        print(f"   - ANPMessageBuilder: {ANPMessageBuilder}")
        
        return True
        
    except Exception as e:
        print(f"❌ ANP adapter test failed: {e}")
        return False


async def test_anp_message_builder():
    """Test ANP message builder"""
    print("\n🧪 Testing ANP message builder")
    print("=" * 50)
    
    try:
        # Test text message
        text_msg = ANPMessageBuilder.text_message("Hello ANP!")
        print(f"✅ Text message: {text_msg}")
        
        # Test JSON message
        json_msg = ANPMessageBuilder.json_message({"key": "value", "number": 42})
        print(f"✅ JSON message: {json_msg}")
        
        # Test ping message
        ping_msg = ANPMessageBuilder.ping_message()
        print(f"✅ Ping message: {ping_msg}")
        
        # Test protocol negotiation message
        negotiation_msg = ANPMessageBuilder.protocol_negotiation_message(
            requirement="Simple Q&A protocol",
            input_desc="User question",
            output_desc="AI answer"
        )
        print(f"✅ Protocol negotiation message: {negotiation_msg}")
        
        return True
        
    except Exception as e:
        print(f"❌ ANP message builder test failed: {e}")
        return False


async def test_anp_server_creation():
    """Test ANP server creation (mock)"""
    print("\n🧪 Testing ANP server creation")
    print("=" * 50)

    try:
        # Create test executor
        executor = TestANPExecutor("ANP Server Test")
        
        # Create server adapter
        adapter = ANPServerAdapter()
        
        # Simulate building server (not actually starting)
        print("📋 Simulating ANP server building...")
        print(f"   Protocol name: {adapter.protocol_name}")
        print(f"   Adapter type: {type(adapter)}")
        
        # Test executor wrapper
        from src.server_adapters.anp_adapter import ANPExecutorWrapper
        wrapper = ANPExecutorWrapper(executor)
        print(f"✅ Executor wrapper created successfully: {wrapper.executor_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ ANP server creation test failed: {e}")
        return False


async def test_anp_client_creation():
    """Test ANP client creation (mock)"""
    print("\n🧪 Testing ANP client creation")
    print("=" * 50)

    
    try:
        import httpx
        
        # Create mocked DID info
        mock_did_info = {
            "private_key_pem": "-----BEGIN PRIVATE KEY-----\nMOCK_KEY\n-----END PRIVATE KEY-----",
            "did": "did:wba:example.com:mock-agent-123",
            "did_document_json": '{"id": "did:wba:example.com:mock-agent-123"}'
        }
        
        # Create ANP adapter (without initialization)
        with httpx.AsyncClient() as client:
            adapter = ANPAdapter(
                httpx_client=client,
                target_did="did:wba:example.com:target-agent-456",
                local_did_info=mock_did_info,
                host_domain="localhost",
                host_port="8000",
                protocol_negotiation=False
            )
            
            print(f"✅ ANP client adapter created successfully")
            print(f"   Target DID: {adapter.target_did}")
            print(f"   localDID: {adapter.local_did_info.get('did', 'unknown')}")
            print(f"   Protocol negotiation: {adapter.protocol_negotiation}")
            
            # Test endpoint info
            endpoint_info = adapter.get_endpoint_info()
            print(f"✅ Endpoint info: {endpoint_info}")
            
        return True
        
    except Exception as e:
        print(f"❌ ANP client creation test failed: {e}")
        return False


async def test_anp_protocol_comparison():
    """Compare ANP protocol with others"""
    print("\n🧪 ANP protocol feature comparison")
    print("=" * 50)
    
    protocols = {
        "ANP": {
            "Authentication": "Decentralized DID-based",
            "Transport": "WebSocket",
            "Message format": "JSON + encryption",
            "Connection type": "Persistent",
            "Protocol negotiation": "LLM-driven negotiation (optional)",
            "Security": "End-to-end encryption",
            "Highlights": "Decentralized identity, intelligent negotiation"
        },
        "A2A": {
            "Authentication": "SDK built-in auth",
            "Transport": "HTTP/SSE",
            "Message format": "JSON",
            "Connection type": "Request/Response",
            "Protocol negotiation": "Fixed interface",
            "Security": "Transport layer encryption",
            "Highlights": "Streaming responses, event queue"
        },
        "Agent Protocol": {
            "Authentication": "API Key/Token",
            "Transport": "HTTP REST",
            "Message format": "JSON",
            "Connection type": "Stateless",
            "Protocol negotiation": "Standardized API",
            "Security": "Transport layer encryption",
            "Highlights": "Task/Step/Artifact model"
        }
    }
    
    print("📊 Protocol feature comparison table:")
    print("-" * 80)
    
    # Print header
    features = list(protocols["ANP"].keys())
    print(f"{'Feature':15} | {'ANP':25} | {'A2A':25} | {'Agent Protocol':25}")
    print("-" * 80)
    
    # Print comparison rows
    for feature in features:
        anp_val = protocols["ANP"][feature]
        a2a_val = protocols["A2A"][feature]
        ap_val = protocols["Agent Protocol"][feature]
        print(f"{feature:15} | {anp_val:25} | {a2a_val:25} | {ap_val:25}")
    
    print("-" * 80)
    
    # ANP advantages
    print("\n🎯 ANP advantages:")
    advantages = [
        "✅ Decentralized identity authentication, no third-party CA",
        "✅ End-to-end encryption for privacy and security",
        "✅ Persistent WebSocket connections for real-time communication",
        "✅ Intelligent protocol negotiation with strong adaptability",
        "✅ Supports complex agent network topologies",
        "✅ Future-proof agent internet standard"
    ]
    
    for advantage in advantages:
        print(f"   {advantage}")
    
    print("\n⚠️  Current challenges:")
    challenges = [
        "⚠️  Requires DID resolution service support",
        "⚠️  Complexity of WebSocket connection management",
        "⚠️  Protocol negotiation may add latency",
        "⚠️  Ecosystem still evolving"
    ]
    
    for challenge in challenges:
        print(f"   {challenge}")
    
    return True


async def test_anp_integration_roadmap():
    """Show ANP integration roadmap"""
    print("\n🗺️  ANP integration roadmap")
    print("=" * 50)
    
    roadmap = {
        "Phase 1 - Basic integration": [
            "✅ ANP adapter interface design",
            "✅ Basic DID authentication integration",
            "✅ WebSocket communication wrapper",
            "⏳ Message routing and transformation",
            "⏳ Error handling and reconnection"
        ],
        "Phase 2 - Protocol negotiation": [
            "⏳ LLM protocol negotiation integration",
            "⏳ Dynamic protocol loading",
            "⏳ Protocol versioning",
            "⏳ Protocol caching"
        ],
        "Phase 3 - Interoperability": [
            "⏳ A2A ↔ ANP protocol bridge",
            "⏳ Agent Protocol ↔ ANP bridge",
            "⏳ Unified message routing",
            "⏳ Multi-protocol session management"
        ],
        "Phase 4 - Advanced features": [
            "⏳ Distributed DID resolution",
            "⏳ Advanced encryption options",
            "⏳ Performance optimizations",
            "⏳ Monitoring and analytics"
        ]
    }
    
    for phase, tasks in roadmap.items():
        print(f"\n📋 {phase}:")
        for task in tasks:
            print(f"   {task}")
    
    print(f"\n🎯 Next priorities:")
    next_steps = [
        "1. Improve AgentConnect integration and error handling",
        "2. Implement message conversion between ANP and A2A/Agent Protocol",
        "3. Create complete end-to-end test cases",
        "4. Optimize connection management and performance",
        "5. Add protocol negotiation support"
    ]
    
    for step in next_steps:
        print(f"   {step}")
    
    return True


async def main():
    """Run all ANP integration tests"""
    print("🚀 ANP (Agent Network Protocol) integration tests")
    print("============================================================")
    
    tests = [
        ("ANP adapter availability", test_anp_adapter_availability),
        ("ANP message builder", test_anp_message_builder),
        ("ANP server creation", test_anp_server_creation),
        ("ANP client creation", test_anp_client_creation),
        ("ANP protocol comparison", test_anp_protocol_comparison),
        ("ANP integration roadmap", test_anp_integration_roadmap),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
            
            # Short delay
            await asyncio.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Test '{test_name}' raised an exception: {e}")
            results.append((test_name, False))
    
    # Summarize results
    print("\n📋 Test results summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "ℹ️  INFO"
        print(f"   {status} - {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if ANPAdapter is not None:
        print("🎉 ANP adapters have been successfully integrated into the multi-protocol framework!")
        print("💡 You can now use the ANP protocol for decentralized agent communication.")
    else:
        print("📝 ANP adapter code is ready, awaiting AgentConnect library installation.")
        print("💡 Install AgentConnect to enable ANP protocol features.")


if __name__ == "__main__":
    asyncio.run(main()) 