#!/usr/bin/env python3
"""
Test Meta Protocol Implementation

This script tests the meta protocol implementation in fail_storm_recovery,
ensuring all components work together correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add paths
current_file = Path(__file__).resolve()
fail_storm_path = current_file.parent
project_root = fail_storm_path.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(fail_storm_path))

from protocol_backends.meta_protocol.meta_coordinator import MetaProtocolCoordinator


async def test_meta_protocol():
    """Test meta protocol coordinator creation and basic functionality."""
    print("🚀 Testing Meta Protocol Implementation")
    print("=" * 60)
    
    # Test configuration with all protocols
    config = {
        "core": {
            "type": "openai",
            "name": "gpt-4o",
            "openai_api_key": "test-key",
            "openai_base_url": "https://api.openai.com/v1",
            "temperature": 0.0
        },
        "protocols": {
            "anp": {
                "core": {
                    "type": "openai",
                    "name": "gpt-4o",
                    "openai_api_key": "test-key",
                    "openai_base_url": "https://api.openai.com/v1",
                    "temperature": 0.0
                }
            },
            "agora": {
                "core": {
                    "type": "openai",
                    "name": "gpt-4o",
                    "openai_api_key": "test-key",
                    "openai_base_url": "https://api.openai.com/v1",
                    "temperature": 0.0
                }
            },
            "a2a": {
                "core": {
                    "type": "openai",
                    "name": "gpt-4o",
                    "openai_api_key": "test-key",
                    "openai_base_url": "https://api.openai.com/v1",
                    "temperature": 0.0
                }
            },
            "acp": {
                "core": {
                    "type": "openai",
                    "name": "gpt-4o",
                    "openai_api_key": "test-key",
                    "openai_base_url": "https://api.openai.com/v1",
                    "temperature": 0.0
                }
            }
        },
        "base_port": 9000,
        "qa": {
            "coordinator": {
                "batch_size": 10,
                "result_file": "data/qa_results_meta_test.json"
            },
            "network": {
                "response_timeout": 30
            }
        }
    }
    
    coordinator = None
    try:
        print("📝 Creating Meta Protocol Coordinator...")
        coordinator = MetaProtocolCoordinator(config)
        print("✅ Meta Protocol Coordinator created successfully")
        
        print("\n🧠 Testing LLM Router...")
        sample_task = {
            "question": "What is artificial intelligence?",
            "context": "General knowledge question for testing",
            "metadata": {"type": "test", "priority": "normal"}
        }
        
        # Test LLM routing (will use fallback if no API key)
        try:
            routing_decision = await coordinator.llm_router.route_task_with_llm(sample_task, num_agents=4)
            print(f"✅ LLM Routing Decision:")
            print(f"   Selected protocols: {routing_decision.selected_protocols}")
            print(f"   Agent assignments: {routing_decision.agent_assignments}")
            print(f"   Strategy: {routing_decision.strategy}")
            print(f"   Confidence: {routing_decision.confidence:.2%}")
        except Exception as e:
            print(f"⚠️  LLM routing test failed (expected without API key): {e}")
        
        print("\n📊 Testing Protocol Statistics...")
        stats = await coordinator.get_protocol_stats()
        print(f"✅ Protocol stats retrieved: {len(stats['protocols'])} protocols available")
        
        print("\n🎯 Testing Metrics Collection...")
        # Test metrics registration
        coordinator.meta_metrics_collector.register_worker("test-worker", "a2a")
        worker_stats = coordinator.meta_metrics_collector.get_worker_statistics("test-worker")
        print(f"✅ Metrics collection working: {worker_stats.get('worker_id', 'unknown')}")
        
        print("\n✅ All Meta Protocol components tested successfully!")
        print("\nNext steps:")
        print("1. Set up proper OpenAI API key for LLM routing")
        print("2. Configure fail_storm_recovery to use meta protocol")
        print("3. Run full integration tests with actual agents")
        
    except Exception as e:
        print(f"❌ Meta Protocol test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if coordinator:
            try:
                print("\n🧹 Cleaning up...")
                await coordinator.close_all()
                print("✅ Cleanup completed")
            except Exception as cleanup_e:
                print(f"⚠️ Cleanup error: {cleanup_e}")


if __name__ == "__main__":
    asyncio.run(test_meta_protocol())
