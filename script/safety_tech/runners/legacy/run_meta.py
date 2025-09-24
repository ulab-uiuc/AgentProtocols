#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Meta Privacy Testing Runner
Runs privacy testing using meta-protocol wrapper with configurable protocol selection.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add current directory to path for imports
HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent
sys.path.insert(0, str(SAFETY_TECH))

from protocol_backends.meta_protocol.meta_coordinator import SafetyMetaCoordinator


class UnifiedMetaRunner(SafetyMetaCoordinator):
    """Unified Meta Privacy Testing Runner with protocol selection"""

    def __init__(self, protocol: str = "acp"):
        """
        Initialize unified meta runner with specified protocol
        
        Args:
            protocol: Protocol type ("acp", "anp", "agora", "a2a")
        """
        # Map protocol to config file
        config_map = {
            "acp": "config_meta_acp.yaml",
            "anp": "config_meta_anp.yaml", 
            "agora": "config_meta_agora.yaml",
            "a2a": "config_meta_a2a.yaml"
        }
        
        if protocol not in config_map:
            raise ValueError(f"Unsupported protocol: {protocol}. Supported: {list(config_map.keys())}")
        
        config_file = config_map[protocol]
        super().__init__(config_file)
        
        self.selected_protocol = protocol


def main():
    """Main entry point for unified meta privacy testing"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Unified Meta Privacy Testing Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runners/run_meta.py --protocol acp      # Run ACP meta testing
  python runners/run_meta.py --protocol agora    # Run Agora meta testing
  python runners/run_meta.py --protocol anp      # Run ANP meta testing
  python runners/run_meta.py --protocol a2a      # Run A2A meta testing
  python runners/run_meta.py                     # Run ACP meta testing (default)
        """
    )
    
    parser.add_argument(
        "--protocol", "-p",
        choices=["acp", "anp", "agora", "a2a"],
        default="acp",
        help="Protocol to use for meta testing (default: acp)"
    )
    
    parser.add_argument(
        "--list-protocols", "-l",
        action="store_true",
        help="List available protocols and exit"
    )
    
    args = parser.parse_args()
    
    # List protocols if requested
    if args.list_protocols:
        print("Available Meta Protocols:")
        print("  acp    - Agent Communication Protocol")
        print("  anp    - Agent Network Protocol")  
        print("  agora  - Agora Protocol")
        print("  a2a    - Agent-to-Agent Protocol")
        sys.exit(0)
    
    # Run meta testing
    async def run_meta_test():
        try:
            print(f"🚀 Starting Meta {args.protocol.upper()} Privacy Testing")
            print("=" * 60)
            
            runner = UnifiedMetaRunner(args.protocol)
            await runner.run()
            
            print("=" * 60)
            print(f"✅ Meta {args.protocol.upper()} Privacy Testing completed successfully!")
            
        except KeyboardInterrupt:
            print(f"\n🛑 Meta {args.protocol.upper()} testing interrupted by user")
        except Exception as e:
            print(f"❌ Meta {args.protocol.upper()} testing failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Run the async function
    asyncio.run(run_meta_test())


if __name__ == "__main__":
    main()
