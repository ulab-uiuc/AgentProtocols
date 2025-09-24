#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta ACP Privacy Testing Runner
Runs ACP privacy testing using meta-protocol wrapper for unified interface.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path for imports
HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent
sys.path.insert(0, str(SAFETY_TECH))

from protocol_backends.meta_protocol.meta_coordinator import SafetyMetaCoordinator


class MetaACPRunner(SafetyMetaCoordinator):
    """Meta ACP Privacy Testing Runner"""

    def __init__(self):
        super().__init__("config_meta_acp.yaml")


async def main():
    """Main entry point for Meta ACP privacy testing"""
    try:
        runner = MetaACPRunner()
        await runner.run()
    except KeyboardInterrupt:
        print("\n🛑 Meta ACP testing interrupted by user")
    except Exception as e:
        print(f"❌ Meta ACP testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
