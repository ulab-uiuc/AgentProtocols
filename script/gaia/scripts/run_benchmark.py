"""Main script for running Gaia benchmark with multi-agent framework."""
import argparse
import asyncio
import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core import TaskPlanner
from protocols.dummy.agent import DummyAgent
from protocols.dummy.network import DummyNetwork, create_dummy_agent
from typing import Dict, Any

async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run Gaia benchmark with multi-agent framework")
    parser.add_argument("--task-file", default="/root/Multiagent-Protocol/script/gaia/GAIABench/2023/validation/metadata.jsonl", help="Gaia task document file")
    parser.add_argument("--protocol", default="json", choices=["json", "agent_protocol"],
                       help="Communication protocol")
    parser.add_argument("--timeout", type=int, default=300, help="Execution timeout in seconds")
    parser.add_argument("--debug", default=True, action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    print("🌟 Starting Gaia Multi-Agent Framework")
    print(f"📄 Task file: {args.task_file}")
    print(f"🔌 Protocol: {args.protocol}")
    print(f"⏰ Timeout: {args.timeout}s")
    print("-" * 50)
    
    try:
        # 1. Load Gaia task document
        task_file_path = Path(args.task_file)
        if not task_file_path.exists():
            raise FileNotFoundError(f"Task file '{args.task_file}' not found")
        
        with open(task_file_path, "r", encoding='utf-8') as f:
            task_dict_list = [json.loads(line) for line in f if line.strip()]
        
        print(f"📄 Loaded task document ({len(task_dict_list)} tasks)")
        
        if args.debug:
            print("🔍 Debug mode enabled: Test with only one case")
            task_dict = task_dict_list[0]  # Use only the first task for debugging
            task_id = task_dict.get("task_id", "debug_task")
            task_doc = task_dict.get("Question", "What is the answer?")
            level = task_dict.get("Level", 1)
        else:
            raise NotImplementedError("Debug mode is not enabled, please set --debug flag")
        
        # 2. Initialize intelligent planner
        # planner = TaskPlanner(task_id=task_id, level=level)
        
        # 3. Analyze Gaia task and generate agent configuration
        print("🧠 Analyzing Gaia task and planning agents...")
        # config_path = await planner.plan_agents(task_doc)
        config_path = "/root/Multiagent-Protocol/script/gaia/workspaces/c61d22de-5f6c-4958-a7f6-5e9707bd3466/agent_config.json"
        print(f"📋 Agent configuration generated: {config_path}")
        
        # 4. Create dynamic network manager
        network = DummyNetwork()
        with open(config_path, "r", encoding='utf-8') as f:
            general_config = json.load(f)
            agent_config = general_config['agents']
        print(f"🤖 Agent configuration loaded: {agent_config}")

        # 5. Create agents based on configuration
        print("🤖 Creating agents based on configuration...")
        for agent_info in agent_config:
            agent = create_dummy_agent(agent_config=agent_info, task_id=task_id)
            network.register_agent(agent)
        
        # 6. Start network
        await network.start()
        
        # 7. Execute workflow with the Gaia task
        print("🎯 Executing workflow with Gaia task...")
        try:
            result = await asyncio.wait_for(
                network.execute_workflow(general_config, task_doc), 
                timeout=args.timeout
            )
            print("✅ Workflow execution completed")
            print(f"📋 Final Result: {result}")
            
            # Store the result for evaluation
            network.done_payload = result
            network.done_ts = time.time() * 1000
            
        except asyncio.TimeoutError:
            print("⏰ Execution timeout reached")
            network.done_ts = time.time() * 1000
            network.done_payload = "TIMEOUT: Framework execution exceeded time limit"
        
        # 8. Evaluate results
        print("📊 Evaluating results...")
        await network.evaluate()
        
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
