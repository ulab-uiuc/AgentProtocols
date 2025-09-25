#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta Protocol Runner for Safety Testing
Runs privacy protection tests using intelligent protocol selection.
"""

import asyncio
import sys
from pathlib import Path

# Add paths
HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent
PROJECT_ROOT = SAFETY_TECH.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SAFETY_TECH) not in sys.path:
    sys.path.insert(0, str(SAFETY_TECH))

# Import meta protocol coordinator
from protocol_backends.meta_protocol.meta_coordinator import SafetyMetaCoordinator


async def main():
    """Run meta protocol safety testing."""
    print("🚀 Starting Safety Meta Protocol Testing")
    print("=" * 60)
    
    # Use meta protocol config
    config_path = SAFETY_TECH / "configs" / "config_meta_a2a.yaml"  # Use existing config as base
    
    coordinator = SafetyMetaCoordinator(str(config_path))
    
    try:
        # Create network with intelligent protocol selection
        await coordinator.create_network()
        
        # Setup agents based on selected protocol
        agents = await coordinator.setup_agents()
        
        # Run health checks
        await coordinator.run_health_checks()
        
        # Run privacy protection tests
        results = await coordinator.run_privacy_test()
        
        # Display results
        coordinator.display_results(results)
        
        print("\n✅ Meta Protocol Safety Testing Completed Successfully!")
        
    except Exception as e:
        print(f"\n❌ Meta Protocol Testing Failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup resources
        await coordinator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())



