#!/usr/bin/env python3
"""
Fail-Storm Meta Network Launcher

This script properly sets up the Python path and runs the meta network from the project root.
"""

import sys
import os
from pathlib import Path

# Setup project root and add to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "script" / "fail_storm_recovery"))

# Add agentconnect to path
agentconnect_path = project_root / "agentconnect_src"
if agentconnect_path.exists():
    sys.path.insert(0, str(agentconnect_path))

print(f"🚀 Starting Fail-Storm Meta Network from {project_root}")
print(f"Python path configured: {len(sys.path)} paths")

# Import and run the meta network
try:
    from script.fail_storm_recovery.runners.run_meta_network import main
    import asyncio
    
    # Run the meta network
    asyncio.run(main())
    
except Exception as e:
    print(f"❌ Error running meta network: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
