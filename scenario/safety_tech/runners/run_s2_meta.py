#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2 Meta Protocol Runner Launcher for Safety_Tech

Main entry point for running S2 security testing with intelligent protocol routing.
Note: E2E encryption testing requires sudo privileges for network packet capture.
"""

import asyncio
import sys
from pathlib import Path

# Add project paths
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent  # 从runners到script到Multiagent-Protocol
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE.parent))  # 加入safety_tech目录

from protocol_backends.meta_protocol.s2_meta_runner import S2MetaProtocolRunner, main


if __name__ == "__main__":
    asyncio.run(main())
