#!/usr/bin/env python3
"""
Test script for Fail-Storm Meta-Protocol Integration

This script tests the basic functionality of the meta-protocol integration
without requiring full network setup or external dependencies.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Setup paths
current_file = Path(__file__).resolve()
fail_storm_path = current_file.parent
project_root = fail_storm_path.parent.parent
src_path = project_root / "src"

# Add paths
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(fail_storm_path))

# Test imports
print("🔍 Testing Meta-Protocol Integration Imports...")

try:
    # Test meta protocol imports
    from protocol_backends.meta_protocol import (
        FailStormMetaCoordinator,
        A2AMetaAgent, create_a2a_meta_worker,
        ANPMetaAgent, create_anp_meta_worker,
        ACPMetaAgent, create_acp_meta_worker,
        AgoraMetaAgent, create_agora_meta_worker
    )
    print("✅ Meta-protocol imports successful")
except ImportError as e:
    print(f"❌ Meta-protocol import failed: {e}")
    sys.exit(1)

try:
    # Test BaseAgent import
    from src.core.base_agent import BaseAgent
    print("✅ BaseAgent import successful")
except ImportError as e:
    print(f"❌ BaseAgent import failed: {e}")
    sys.exit(1)

try:
    # Test shard worker import
    from shard_qa.shard_worker.agent_executor import ShardWorkerExecutor
    print("✅ ShardWorkerExecutor import successful")
except ImportError as e:
    print(f"❌ ShardWorkerExecutor import failed: {e}")
    print("   This is expected if shard_qa components are not set up")

print("\n🧪 Testing Meta-Agent Creation...")

async def test_meta_agent_creation():
    """Test creating meta agents without full network setup."""
    
    # Test configuration
    test_config = {
        "core": {
            "type": "test",
            "name": "test-model",
            "temperature": 0.0
        }
    }
    
    print("📝 Testing A2A Meta Agent...")
    try:
        a2a_agent = A2AMetaAgent("test-a2a-agent", test_config)
        print(f"   ✅ A2A Meta Agent created: {a2a_agent.agent_id}")
    except Exception as e:
        print(f"   ❌ A2A Meta Agent creation failed: {e}")
    
    print("📝 Testing ANP Meta Agent...")
    try:
        anp_agent = ANPMetaAgent("test-anp-agent", test_config)
        print(f"   ✅ ANP Meta Agent created: {anp_agent.agent_id}")
    except Exception as e:
        print(f"   ❌ ANP Meta Agent creation failed: {e}")
    
    print("📝 Testing ACP Meta Agent...")
    try:
        acp_agent = ACPMetaAgent("test-acp-agent", test_config)
        print(f"   ✅ ACP Meta Agent created: {acp_agent.agent_id}")
    except Exception as e:
        print(f"   ❌ ACP Meta Agent creation failed: {e}")
    
    print("📝 Testing Agora Meta Agent...")
    try:
        agora_agent = AgoraMetaAgent("test-agora-agent", test_config)
        print(f"   ✅ Agora Meta Agent created: {agora_agent.agent_id}")
    except Exception as e:
        print(f"   ❌ Agora Meta Agent creation failed: {e}")

async def test_meta_coordinator():
    """Test creating meta coordinator."""
    
    print("\n📝 Testing Meta Coordinator...")
    try:
        coordinator = FailStormMetaCoordinator()
        print(f"   ✅ Meta Coordinator created with {len(coordinator.meta_agents)} agents")
        
        # Test metrics initialization
        metrics = await coordinator.get_failstorm_metrics()
        print(f"   ✅ Metrics collection working: {len(metrics)} metric categories")
        
    except Exception as e:
        print(f"   ❌ Meta Coordinator creation failed: {e}")

async def test_configuration_loading():
    """Test configuration loading."""
    
    print("\n📝 Testing Configuration Loading...")
    try:
        import yaml
        
        # Test loading the meta config
        config_file = fail_storm_path / "config_meta.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            protocols = config.get("protocols", {})
            print(f"   ✅ Config loaded with {len(protocols)} protocols")
            
            for protocol in protocols:
                print(f"      - {protocol.upper()}")
        else:
            print(f"   ⚠️  Config file not found: {config_file}")
            
    except Exception as e:
        print(f"   ❌ Configuration loading failed: {e}")

async def main():
    """Run all tests."""
    print("🌪️  Fail-Storm Meta-Protocol Integration Test")
    print("=" * 60)
    
    await test_meta_agent_creation()
    await test_meta_coordinator()
    await test_configuration_loading()
    
    print("\n✅ Integration test completed!")
    print("\n📋 Next Steps:")
    print("   1. Set up environment variables (OPENAI_API_KEY, etc.)")
    print("   2. Run: python script/fail_storm_recovery/runners/run_meta_network.py")
    print("   3. Check results in script/fail_storm_recovery/results/")

if __name__ == "__main__":
    asyncio.run(main())
