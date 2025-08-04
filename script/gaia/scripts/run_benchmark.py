"""Main script for running Gaia benchmark with multi-agent framework."""
import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core import TaskPlanner, MeshNetwork
from protocols import JsonAdapter, AgentProtocolAdapter


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run Gaia benchmark with multi-agent framework")
    parser.add_argument("--task-file", default="gaia_task.txt", help="Gaia task document file")
    parser.add_argument("--strategy", default="adaptive", choices=["simple", "adaptive"], 
                       help="Planning strategy")
    parser.add_argument("--protocol", default="json", choices=["json", "agent_protocol"],
                       help="Communication protocol")
    parser.add_argument("--timeout", type=int, default=300, help="Execution timeout in seconds")
    
    args = parser.parse_args()
    
    print("🌟 Starting Gaia Multi-Agent Framework")
    print(f"📄 Task file: {args.task_file}")
    print(f"🧠 Strategy: {args.strategy}")
    print(f"🔌 Protocol: {args.protocol}")
    print(f"⏰ Timeout: {args.timeout}s")
    print("-" * 50)
    
    try:
        # 1. Initialize protocol adapter
        if args.protocol == "agent_protocol":
            adapter = AgentProtocolAdapter()
        else:
            adapter = JsonAdapter()
        
        print(f"🔌 Protocol adapter initialized: {adapter.__class__.__name__}")
        
        # 2. Load Gaia task document
        task_file_path = Path(args.task_file)
        if not task_file_path.exists():
            print(f"❌ Task file not found: {task_file_path}")
            print("💡 Creating sample task file...")
            sample_task = """
            Question: What is the current population of Tokyo, Japan in 2024?
            
            Please provide:
            1. The most recent population figure
            2. Data sources and verification
            3. Comparison with previous years if available
            
            Use web search to find accurate and up-to-date information.
            """
            task_file_path.write_text(sample_task.strip(), encoding='utf-8')
            print(f"✅ Sample task created at {task_file_path}")
        
        with open(task_file_path, "r", encoding='utf-8') as f:
            gaia_task_document = f.read()
        
        print(f"📄 Loaded task document ({len(gaia_task_document)} characters)")
        
        # 3. Initialize intelligent planner
        planner = TaskPlanner(strategy_type=args.strategy)
        
        # 4. Analyze Gaia task and generate agent configuration
        print("🧠 Analyzing Gaia task and planning agents...")
        config_path = await planner.plan_agents(gaia_task_document, args.strategy)
        print(f"📋 Agent configuration generated: {config_path}")
        
        # 5. Create dynamic network manager
        network = MeshNetwork(adapter)
        
        # 6. Create agents based on configuration
        print("🤖 Creating agents based on configuration...")
        await network.load_and_create_agents(config_path)
        
        # 7. Start network
        print("🚀 Starting network communication...")
        await network.start()
        
        # 8. Begin task execution
        print("📄 Broadcasting Gaia task document...")
        await network.broadcast_init(gaia_task_document)
        
        # 9. Monitor execution with timeout
        print("⏳ Monitoring task execution...")
        try:
            await asyncio.wait_for(network._monitor_done(), timeout=args.timeout)
        except asyncio.TimeoutError:
            print("⏰ Execution timeout reached")
            network.done_ts = time.time() * 1000
            network.done_payload = "TIMEOUT: Framework execution exceeded time limit"
            await network._evaluate()
        
        print("✅ Task completed")
        print("📊 Results:")
        print("  - Check metrics.json for performance metrics")
        print("  - Check run_artifacts.tar.gz for detailed logs")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'network' in locals():
            await network.stop()
        print("🧹 Cleanup completed")


if __name__ == "__main__":
    # Import time here to avoid issues
    import time
    asyncio.run(main())
