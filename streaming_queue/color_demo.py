#!/usr/bin/env python3
"""
Color Output Demo - Demonstrates the colored console output functionality
"""

import sys
from pathlib import Path

# Add parent directory to path to import ColoredOutput
sys.path.insert(0, str(Path(__file__).parent))

try:
    from streaming_queue import ColoredOutput
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    print("Warning: Could not import ColoredOutput. Running without colors.")
    
    # Fallback ColoredOutput for demo
    class ColoredOutput:
        @staticmethod
        def info(message): print(f"ℹ️  {message}")
        
        @staticmethod
        def success(message): print(f"✅ {message}")
        
        @staticmethod
        def warning(message): print(f"⚠️  {message}")
        
        @staticmethod
        def error(message): print(f"❌ {message}")
        
        @staticmethod
        def system(message): print(f"🔧 {message}")
        
        @staticmethod
        def progress(message): print(f"   {message}")


def demo_colored_output():
    """Demonstrate all types of colored output."""
    output = ColoredOutput()
    
    print("=" * 60)
    print("🎨 Colored Output Demo")
    print("=" * 60)
    
    output.info("This is an informational message (Blue)")
    output.success("This is a success message (Green)")
    output.warning("This is a warning message (Yellow)")
    output.error("This is an error message (Red)")
    output.system("This is a system status message (Cyan)")
    output.progress("This is a progress/detail message (White)")
    
    print("\n" + "=" * 60)
    output.info("Demo completed!")
    
    if COLORS_AVAILABLE:
        output.success("Colors are working properly!")
    else:
        output.warning("Colors are not available. Install colorama: pip install colorama")


def demo_agent_network_messages():
    """Demonstrate messages similar to those in the actual agent network."""
    output = ColoredOutput()
    
    print("\n" + "=" * 60)
    output.info("🎉 Real A2A AgentNetwork QA System Demo")
    print("=" * 60)
    
    # Agent setup simulation
    output.info("Initializing real AgentNetwork and A2A Agents...")
    output.success("Coordinator-1 created and registered to AgentNetwork")
    output.success("Worker-1 created and registered to AgentNetwork (port: 10001)")
    output.success("Worker-2 created and registered to AgentNetwork (port: 10002)")
    
    # Topology setup
    output.info("=== Setting up Network Topology ===")
    output.success("Setup star topology with center node: Coordinator-1")
    output.system("Current topology connection count: 4")
    
    # Health check
    output.info("=== Health Check ===")
    output.system("Health check results (3/3 healthy):")
    output.success("Coordinator-1: Healthy")
    output.success("Worker-1: Healthy")
    output.success("Worker-2: Healthy")
    
    # Question processing
    output.system("Loaded 50 questions")
    output.info("Starting dynamic load balancing: 50 questions, 2 workers")
    
    # Progress messages
    output.progress("Worker-1 starting to process question 1: What is artificial intelligence...")
    output.progress("Worker-2 starting to process question 2: How does machine learning work...")
    output.progress("Worker-1 completed question 1, continuing to next...")
    output.progress("Worker-2 completed question 2, continuing to next...")
    
    # Results
    output.system("Collected 50/50 results")
    output.success("Result collection completed, collected 50 results")
    output.success("Results saved to: data/qa_results_real.json")
    
    # Final stats
    output.success("Demo completed!")
    output.system("Total time: 45.67 seconds")
    output.system("Successfully processed: 48 questions")
    output.system("Failed: 2 questions")
    
    # Cleanup
    output.system("Cleaning up resources...")
    output.success("Resource cleanup completed")


if __name__ == "__main__":
    demo_colored_output()
    demo_agent_network_messages() 