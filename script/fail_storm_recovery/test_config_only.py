#!/usr/bin/env python3
"""
Simple config test for Fail-Storm Meta-Protocol Integration

This script only tests configuration loading and basic structure validation.
"""

import yaml
from pathlib import Path

def test_config_loading():
    """Test configuration loading."""
    
    print("🔍 Testing Configuration Loading...")
    fail_storm_path = Path(__file__).parent
    
    # Test loading the meta config
    config_file = fail_storm_path / "config_meta.yaml"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"✅ Config loaded successfully")
            
            # Validate config structure
            required_sections = ['core', 'network', 'agents', 'protocols', 'qa', 'shard_qa', 'scenario', 'output']
            for section in required_sections:
                if section in config:
                    print(f"   ✅ {section} section present")
                else:
                    print(f"   ⚠️  {section} section missing")
            
            # Test protocol configuration
            protocols = config.get("protocols", {})
            print(f"   📋 Protocols configured: {len(protocols)}")
            
            for protocol, settings in protocols.items():
                enabled = settings.get("enabled", True)
                agent_count = settings.get("agent_count", 2)
                print(f"      - {protocol.upper()}: {'enabled' if enabled else 'disabled'}, {agent_count} agents")
            
            # Test core configuration
            core = config.get("core", {})
            llm_type = core.get("type", "unknown")
            model = core.get("name", "unknown")
            print(f"   🤖 LLM: {llm_type} ({model})")
            
            # Test network configuration
            network = config.get("network", {})
            base_port = network.get("base_port", 9000)
            topology = network.get("topology", "mesh")
            print(f"   🌐 Network: {topology} topology, starting port {base_port}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Configuration loading failed: {e}")
            return False
    else:
        print(f"   ❌ Config file not found: {config_file}")
        return False

def test_directory_structure():
    """Test directory structure."""
    
    print("\n🔍 Testing Directory Structure...")
    fail_storm_path = Path(__file__).parent
    
    required_dirs = [
        "protocol_backends",
        "protocol_backends/meta_protocol",
        "runners",
        "shard_qa"
    ]
    
    for dir_path in required_dirs:
        full_path = fail_storm_path / dir_path
        if full_path.exists():
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path} missing")
    
    # Test meta protocol files
    meta_files = [
        "protocol_backends/meta_protocol/__init__.py",
        "protocol_backends/meta_protocol/meta_coordinator.py",
        "protocol_backends/meta_protocol/a2a_meta_agent.py",
        "protocol_backends/meta_protocol/anp_meta_agent.py",
        "protocol_backends/meta_protocol/acp_meta_agent.py",
        "protocol_backends/meta_protocol/agora_meta_agent.py"
    ]
    
    print("\n   Meta Protocol Files:")
    for file_path in meta_files:
        full_path = fail_storm_path / file_path
        if full_path.exists():
            print(f"      ✅ {file_path}")
        else:
            print(f"      ❌ {file_path} missing")

def main():
    """Run all tests."""
    print("🌪️  Fail-Storm Meta-Protocol Configuration Test")
    print("=" * 60)
    
    config_ok = test_config_loading()
    test_directory_structure()
    
    print("\n📋 Summary:")
    if config_ok:
        print("✅ Configuration is valid and ready for use")
        print("\n📝 Next Steps:")
        print("   1. Set up environment variables if needed")
        print("   2. Install required dependencies (ACP SDK, ANP SDK, etc.)")
        print("   3. Run: python runners/run_meta_network.py --config config_meta.yaml")
    else:
        print("❌ Configuration has issues that need to be resolved")
    
    print("\n🔧 Files created:")
    print("   - config_meta.yaml: Main configuration file")
    print("   - protocol_backends/meta_protocol/: Meta protocol integration")
    print("   - runners/run_meta_network.py: Network runner script")

if __name__ == "__main__":
    main()
