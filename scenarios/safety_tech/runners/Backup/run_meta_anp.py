#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta ANP Privacy Testing Runner
Runs ANP privacy testing using meta-protocol wrapper for unified interface.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path for imports
HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent
sys.path.insert(0, str(SAFETY_TECH))

from protocol_backends.meta_protocol.meta_coordinator import SafetyMetaCoordinator


class MetaANPRunner(SafetyMetaCoordinator):
    """Meta ANP Privacy Testing Runner"""

    def __init__(self):
        super().__init__("config_meta_anp.yaml")


async def main():
    """Main entry point for Meta ANP privacy testing"""
    try:
        runner = MetaANPRunner()
        await runner.run()
    except KeyboardInterrupt:
        print("\n🛑 Meta ANP testing interrupted by user")
    except Exception as e:
        print(f"❌ Meta ANP testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
