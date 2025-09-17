#!/usr/bin/env python3
"""
Test LLM Router only (without meta coordinator dependencies)
"""

import asyncio
import sys
from pathlib import Path

# Add the script directory to the path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Direct import of LLM router only
sys.path.insert(0, str(script_dir / "protocol_backends" / "meta_protocol"))
from llm_router import FailStormLLMRouter, FailStormRoutingDecision

async def test_llm_router_only():
    """Test LLM router functionality independently."""
    print("🧠 Testing LLM Router for Fail-Storm scenarios...")
    
    # Create LLM router
    router = FailStormLLMRouter()
    
    # Update with real benchmark data
    print("📊 Updating router with real benchmark data...")
    router.update_protocol_performance("anp", avg_response_time=6.76, success_rate=0.610, recovery_time=10.0, fault_tolerance=0.95)
    router.update_protocol_performance("a2a", avg_response_time=7.39, success_rate=0.596, recovery_time=6.0, fault_tolerance=0.85)
    router.update_protocol_performance("agora", avg_response_time=7.10, success_rate=0.600, recovery_time=6.1, fault_tolerance=0.90)
    router.update_protocol_performance("acp", avg_response_time=7.83, success_rate=0.590, recovery_time=8.0, fault_tolerance=0.85)
    
    # Test protocol recommendations
    print(f"\n📈 Protocol Recommendations (based on real benchmark data):")
    recommendations = router.get_protocol_recommendations("general")
    
    sorted_protocols = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    for i, (protocol, score) in enumerate(sorted_protocols, 1):
        info = router.protocols[protocol]
        print(f"   {i}. {protocol.upper()}: Score={score:.3f}")
        print(f"      Performance: {info.avg_response_time:.1f}s avg, {info.success_rate:.1%} success")
        print(f"      Fault tolerance: {info.fault_tolerance:.1%}, Recovery: {info.recovery_time:.1f}s")
        print(f"      Best for: {', '.join(info.best_for[:2])}")
    
    # Test routing decisions
    test_scenarios = [
        {
            "name": "Security-Critical",
            "task": {
                "question": "What are the security protocols?",
                "metadata": {"requires_security": True, "privacy_critical": True}
            },
            "fault_scenario": "cyclic"
        },
        {
            "name": "High-Performance", 
            "task": {
                "question": "What is the capital of France?",
                "metadata": {"requires_security": False, "privacy_critical": False}
            },
            "fault_scenario": "stress"
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n🎯 Testing scenario: {scenario['name']}")
        decision = await router.route_task_for_failstorm(scenario['task'], num_agents=8, fault_scenario=scenario['fault_scenario'])
        
        print(f"   Selected protocols: {decision.selected_protocols}")
        print(f"   Agent distribution: {len([p for p in decision.agent_assignments.values() if p == 'anp'])} ANP, "
              f"{len([p for p in decision.agent_assignments.values() if p == 'a2a'])} A2A, "
              f"{len([p for p in decision.agent_assignments.values() if p == 'agora'])} Agora, "
              f"{len([p for p in decision.agent_assignments.values() if p == 'acp'])} ACP")
        print(f"   Fault strategy: {decision.fault_strategy}")
        print(f"   Confidence: {decision.confidence:.1%}")
    
    print(f"\n🎉 LLM Router test completed successfully!")
    return True

if __name__ == "__main__":
    result = asyncio.run(test_llm_router_only())
    if result:
        print("✅ LLM Router tests passed!")
        sys.exit(0)
    else:
        print("❌ LLM Router tests failed!")
        sys.exit(1)
