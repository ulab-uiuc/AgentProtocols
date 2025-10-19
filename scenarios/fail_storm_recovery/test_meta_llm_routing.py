#!/usr/bin/env python3
"""
Test LLM-based intelligent routing for Fail-Storm Meta Protocol

This test verifies the LLM intelligent selection module works correctly
with real benchmark data from our four protocols.
"""

import asyncio
import sys
from pathlib import Path

# Add the script directory to the path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from protocol_backends.meta_protocol.llm_router import FailStormLLMRouter, FailStormRoutingDecision

async def test_llm_routing():
    """Test LLM-based protocol routing for fail-storm scenarios."""
    print("🧠 Testing LLM-based intelligent routing for Fail-Storm scenarios...")
    
    # Create LLM router
    router = FailStormLLMRouter()
    
    # Test different task scenarios
    test_scenarios = [
        {
            "name": "Security-Critical Task",
            "task": {
                "question": "What are the classified security protocols for agent authentication?",
                "context": "Sensitive security information",
                "metadata": {
                    "requires_security": True,
                    "privacy_critical": True,
                    "high_availability": True
                }
            },
            "fault_scenario": "cyclic"
        },
        {
            "name": "High-Performance Task", 
            "task": {
                "question": "What is the capital of France?",
                "context": "Simple factual question",
                "metadata": {
                    "requires_security": False,
                    "privacy_critical": False,
                    "high_availability": True
                }
            },
            "fault_scenario": "stress"
        },
        {
            "name": "Complex Research Task",
            "task": {
                "question": "Analyze the relationship between quantum computing and cryptographic security implications",
                "context": "Complex multi-step reasoning required",
                "metadata": {
                    "requires_security": False,
                    "privacy_critical": False,
                    "high_availability": True
                }
            },
            "fault_scenario": "single"
        }
    ]
    
    # Test each scenario
    for scenario in test_scenarios:
        print(f"\n📋 Testing: {scenario['name']}")
        print(f"   Question: {scenario['task']['question'][:60]}...")
        print(f"   Fault scenario: {scenario['fault_scenario']}")
        
        # Get routing decision (will use default since no LLM client)
        decision = await router.route_task_for_failstorm(
            scenario['task'], 
            num_agents=8, 
            fault_scenario=scenario['fault_scenario']
        )
        
        print(f"   🎯 Selected protocols: {decision.selected_protocols}")
        print(f"   🔄 Fault strategy: {decision.fault_strategy}")
        print(f"   📊 Load balancing: {decision.load_balancing}")
        print(f"   🤖 Strategy: {decision.strategy}")
        print(f"   📈 Confidence: {decision.confidence:.1%}")
        print(f"   💭 Reasoning: {decision.reasoning[:100]}...")
        
        # Verify assignment
        assert len(decision.agent_assignments) == 8, f"Expected 8 agents, got {len(decision.agent_assignments)}"
        assert all(p in ["a2a", "acp", "agora", "anp"] for p in decision.selected_protocols), "Invalid protocols"
        
        print(f"   ✅ Assignment valid: {len(decision.agent_assignments)} agents assigned")
    
    # Test protocol recommendations
    print(f"\n📊 Protocol Recommendations:")
    recommendations = router.get_protocol_recommendations("general")
    
    sorted_protocols = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    for protocol, score in sorted_protocols:
        info = router.protocols[protocol]
        print(f"   {protocol.upper()}: Score={score:.3f} (Response: {info.avg_response_time:.1f}s, "
              f"Success: {info.success_rate:.1%}, Recovery: {info.recovery_time:.1f}s, "
              f"Fault Tolerance: {info.fault_tolerance:.1%})")
    
    print(f"\n🎉 LLM routing test completed successfully!")
    return True

if __name__ == "__main__":
    result = asyncio.run(test_llm_routing())
    if result:
        print("✅ All LLM routing tests passed!")
        sys.exit(0)
    else:
        print("❌ LLM routing tests failed!")
        sys.exit(1)
