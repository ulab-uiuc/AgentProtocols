#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta Agora Privacy Testing Runner
Runs Agora privacy testing using meta-protocol wrapper for unified interface.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path for imports
HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent
sys.path.insert(0, str(SAFETY_TECH))

from protocol_backends.meta_protocol.meta_coordinator import SafetyMetaCoordinator


class MetaAgoraRunner(SafetyMetaCoordinator):
    """Meta Agora Privacy Testing Runner"""

    def __init__(self):
        super().__init__("config_meta_agora.yaml")


async def main():
    """Main entry point for Meta Agora privacy testing"""
    try:
        runner = MetaAgoraRunner()
        await runner.run()
    except KeyboardInterrupt:
        print("\n🛑 Meta Agora testing interrupted by user")
    except Exception as e:
        print(f"❌ Meta Agora testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
