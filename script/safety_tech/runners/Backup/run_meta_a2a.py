#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta A2A Privacy Testing Runner
Runs A2A privacy testing using meta-protocol wrapper for unified interface.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path for imports
HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent
sys.path.insert(0, str(SAFETY_TECH))

from protocol_backends.meta_protocol.meta_coordinator import SafetyMetaCoordinator


class MetaA2ARunner(SafetyMetaCoordinator):
    """Meta A2A Privacy Testing Runner"""

    def __init__(self):
        super().__init__("config_meta_a2a.yaml")


async def main():
    """Main entry point for Meta A2A privacy testing"""
    try:
        runner = MetaA2ARunner()
        await runner.run()
    except KeyboardInterrupt:
        print("\n🛑 Meta A2A testing interrupted by user")
    except Exception as e:
        print(f"❌ Meta A2A testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
