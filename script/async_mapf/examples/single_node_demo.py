#!/usr/bin/env python3
"""
Single-node MAPF demonstration.

Simple example showing basic MAPF execution using the dummy protocol.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from script.async_mapf.runners.local_runner import LocalRunner
from script.async_mapf.metrics.recorder import MetricsRecorder
from script.async_mapf.metrics.dashboard import RealtimeDashboard, MAPFDashboardConnector


async def basic_demo():
    """Basic MAPF demonstration with 4 agents."""
    print("Starting basic MAPF demo...")
    
    # Configuration for dummy protocol
    config_path = Path(__file__).parent.parent / "config" / "dummy.yaml"
    
    # Create and run scenario
    runner = LocalRunner(str(config_path))
    
    try:
        results = await runner.run()
        print(f"\nDemo completed successfully!")
        print(f"Success rate: {results['success_rate']:.1%}")
        print(f"Execution time: {results['execution_time']:.2f}s")
        
    except Exception as e:
        print(f"Demo failed: {e}")


async def demo_with_metrics():
    """Demo with metrics recording and analysis."""
    print("Starting MAPF demo with metrics recording...")
    
    config_path = Path(__file__).parent.parent / "config" / "dummy.yaml"
    
    # Setup metrics recording
    recorder = MetricsRecorder("demo_metrics")
    recorder.start_recording()
    
    # Create runner and inject metrics collection
    runner = LocalRunner(str(config_path))
    
    try:
        # Setup scenario but don't run yet
        await runner.setup()
        
        # Record initial metrics
        recorder.record_custom_event("demo_started", {
            "num_agents": len(runner.agents),
            "world_size": runner.world.size,
            "protocol": "dummy"
        })
        
        # Run scenario with periodic metrics collection
        import time
        start_time = time.time()
        
        # Create tasks for agents and network
        tasks = []
        tasks.append(asyncio.create_task(runner.network.run()))
        
        for agent in runner.agents:
            tasks.append(asyncio.create_task(agent.run()))
        
        # Monitor progress
        while not runner.network.is_scenario_complete():
            # Record metrics
            if runner.network:
                net_metrics = runner.network.get_performance_metrics()
                recorder.record_network_metrics(net_metrics)
            
            for agent in runner.agents:
                agent_metrics = {
                    "goal_reached": agent.is_at_goal(),
                    "current_position": agent.current_pos,
                    "path_length": len(agent.path) if agent.path else 0,
                    "is_active": agent.is_active
                }
                recorder.record_agent_metrics(agent.aid, agent_metrics)
            
            # Record world state
            recorder.record_world_snapshot(runner.world.get_world_state())
            
            await asyncio.sleep(0.5)  # Update every 500ms
        
        # Cancel remaining tasks
        for task in tasks:
            task.cancel()
        
        # Record completion
        end_time = time.time()
        recorder.record_custom_event("demo_completed", {
            "duration": end_time - start_time,
            "final_conflicts": runner.world.detect_conflicts()
        })
        
        print(f"Demo completed in {end_time - start_time:.2f}s")
        
    except Exception as e:
        recorder.record_custom_event("demo_error", {"error": str(e)})
        print(f"Demo failed: {e}")
        
    finally:
        # Stop recording and generate report
        recorder.stop_recording()
        print("Metrics saved and report generated")


async def demo_with_dashboard():
    """Demo with real-time dashboard monitoring."""
    print("Starting MAPF demo with real-time dashboard...")
    
    config_path = Path(__file__).parent.parent / "config" / "dummy.yaml"
    
    # Setup dashboard
    dashboard = RealtimeDashboard(update_interval=0.5)
    connector = MAPFDashboardConnector(dashboard)
    
    # Setup alert thresholds
    dashboard.set_alert_threshold("network.conflict_count", "max", 50, 
                                 "High conflict count detected!")
    dashboard.set_alert_threshold("network.active_agents", "min", 1,
                                 "No active agents remaining!")
    
    # Create runner
    runner = LocalRunner(str(config_path))
    
    try:
        # Setup scenario
        await runner.setup()
        
        # Connect dashboard to components
        connector.connect_network(runner.network)
        connector.connect_agents(runner.agents)
        
        # Start dashboard
        dashboard.start()
        
        # Add dashboard callback to print updates
        def print_progress(metric):
            if metric.name == "network.active_agents":
                print(f"Active agents: {metric.value}")
        
        dashboard.add_metric_callback("network.active_agents", print_progress)
        
        # Run scenario
        results = await runner.run()
        
        # Print final dashboard state
        dashboard.print_summary()
        
        # Generate web dashboard
        html_file = dashboard.generate_web_dashboard("demo_dashboard.html")
        print(f"Web dashboard saved to: {html_file}")
        
        print(f"\nDemo completed!")
        print(f"Success rate: {results['success_rate']:.1%}")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        
    finally:
        dashboard.stop()


async def interactive_demo():
    """Interactive demo allowing user to control execution."""
    print("Starting interactive MAPF demo...")
    print("Commands: step, run, status, agents, world, quit")
    
    config_path = Path(__file__).parent.parent / "config" / "dummy.yaml"
    runner = LocalRunner(str(config_path))
    
    await runner.setup()
    
    print(f"Setup complete: {len(runner.agents)} agents in {runner.world.size}x{runner.world.size} world")
    runner.world.print_grid()
    
    step_count = 0
    
    while True:
        try:
            command = input("\nCommand> ").strip().lower()
            
            if command == "quit":
                break
                
            elif command == "step":
                # Single step execution
                step_count += 1
                print(f"Executing step {step_count}...")
                
                # Process one round of agent actions
                for agent in runner.agents:
                    if agent.is_active and not agent.is_at_goal():
                        await agent.process_messages()
                        if agent.path and agent.path_index < len(agent.path):
                            await agent.move_next()
                
                # Update network
                runner.network.update_global_clock()
                await runner.network.process_messages()
                
                print("Step completed")
                
            elif command == "run":
                # Run to completion
                print("Running to completion...")
                results = await runner.run()
                print(f"Execution completed: {results['success_rate']:.1%} success")
                break
                
            elif command == "status":
                # Show status
                active_agents = sum(1 for a in runner.agents if a.is_active)
                completed_agents = sum(1 for a in runner.agents if a.is_at_goal())
                print(f"Step: {step_count}")
                print(f"Active agents: {active_agents}")
                print(f"Completed agents: {completed_agents}")
                print(f"Conflicts: {len(runner.world.detect_conflicts())}")
                
            elif command == "agents":
                # Show agent details
                for agent in runner.agents:
                    status = "GOAL" if agent.is_at_goal() else ("ACTIVE" if agent.is_active else "INACTIVE")
                    print(f"Agent {agent.aid}: {agent.current_pos} -> {agent.goal} [{status}]")
                    
            elif command == "world":
                # Show world state
                runner.world.print_grid()
                
            else:
                print("Unknown command. Available: step, run, status, agents, world, quit")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Interactive demo ended")


def main():
    """Main entry point for demonstrations."""
    if len(sys.argv) > 1:
        demo_type = sys.argv[1]
    else:
        print("Available demos:")
        print("  basic - Basic MAPF execution")
        print("  metrics - Demo with metrics recording")
        print("  dashboard - Demo with real-time dashboard")
        print("  interactive - Interactive step-by-step demo")
        demo_type = input("Select demo type: ").strip()
    
    if demo_type == "basic":
        asyncio.run(basic_demo())
    elif demo_type == "metrics":
        asyncio.run(demo_with_metrics())
    elif demo_type == "dashboard":
        asyncio.run(demo_with_dashboard())
    elif demo_type == "interactive":
        asyncio.run(interactive_demo())
    else:
        print(f"Unknown demo type: {demo_type}")
        sys.exit(1)


if __name__ == "__main__":
    main() 