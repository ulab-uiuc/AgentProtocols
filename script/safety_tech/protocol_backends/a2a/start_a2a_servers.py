# -*- coding: utf-8 -*-
"""
A2A Agent Servers Starter
Start local A2A agent servers for privacy testing
"""

import asyncio
import uvicorn
import sys
import threading
from pathlib import Path

# Add path for imports
HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent.parent  # Up two levels: a2a -> protocol_backend -> safety_tech
sys.path.insert(0, str(SAFETY_TECH))

# A2A SDK imports
try:
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentProvider
    A2A_SDK_AVAILABLE = True
    print("[A2A Server] A2A SDK available for server creation")
except ImportError as e:
    A2A_SDK_AVAILABLE = False
    print(f"[A2A Server] A2A SDK not available: {e}")
    print("Cannot start A2A servers without SDK")
    sys.exit(1)

# Import our executors
from protocol_backends.a2a import A2AReceptionistExecutor, A2ADoctorExecutor
from runners.runner_base import SimpleOutput
import yaml

def load_config():
    """Load A2A configuration"""
    try:
        # Look for config in runner directory
        config_path = SAFETY_TECH / "runner" / "config_a2a.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config: {e}")
        return {}

def create_receptionist_app(config):
    """Create A2A app for receptionist agent"""
    output = SimpleOutput("A2A-Receptionist-Server")
    
    # Create executor
    executor = A2AReceptionistExecutor(config, "A2A_Receptionist", output)
    
    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    
    # Create agent card
    agent_card = AgentCard(
        name="Privacy-Aware Medical Receptionist",
        description="A2A-enabled medical receptionist focused on privacy protection",
        url="http://127.0.0.1:8001/",
        version="1.0.0",
        provider=AgentProvider(
            name="Privacy Testing Framework",
            organization="Agent Research Lab",
            url="http://127.0.0.1:8001/",
            email="privacy@example.com",
        ),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="privacy_protection",
                name="Privacy Protection",
                description="Protects patient privacy in medical communications",
                tags=["privacy", "medical", "protection"],
                inputModes=["text"],
                outputModes=["text"],
                examples=["Protect patient information", "Filter sensitive data"]
            )
        ]
    )
    
    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

def create_doctor_app(config):
    """Create A2A app for doctor agent"""
    output = SimpleOutput("A2A-Doctor-Server")
    
    # Create executor
    executor = A2ADoctorExecutor(config, "A2A_Doctor", output)
    
    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    
    # Create agent card
    agent_card = AgentCard(
        name="Privacy-Testing Medical Doctor",
        description="A2A-enabled medical doctor for privacy testing scenarios",
        url="http://127.0.0.1:8002/",
        version="1.0.0",
        provider=AgentProvider(
            name="Privacy Testing Framework",
            organization="Agent Research Lab", 
            url="http://127.0.0.1:8002/",
            email="privacy@example.com",
        ),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="information_extraction",
                name="Information Extraction",
                description="Attempts to extract patient information for privacy testing",
                tags=["extraction", "medical", "testing"],
                inputModes=["text"],
                outputModes=["text"], 
                examples=["Extract patient details", "Request personal information"]
            )
        ]
    )
    
    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

def run_server(app, port, name):
    """Run A2A server on specified port"""
    print(f"[A2A Server] Starting {name} on port {port}")
    # Get the actual ASGI app from A2AStarletteApplication
    asgi_app = app.build()
    uvicorn.run(asgi_app, host="127.0.0.1", port=port, log_level="info")

async def main():
    """Start both A2A servers"""
    if not A2A_SDK_AVAILABLE:
        print("Cannot start servers without A2A SDK")
        return
    
    config = load_config()
    print("[A2A Server] Configuration loaded")
    
    # Create apps
    receptionist_app = create_receptionist_app(config)
    doctor_app = create_doctor_app(config)
    
    print("[A2A Server] Apps created, starting servers...")
    
    # Start servers in separate threads
    receptionist_thread = threading.Thread(
        target=run_server, 
        args=(receptionist_app, 8001, "Receptionist"),
        daemon=True
    )
    doctor_thread = threading.Thread(
        target=run_server, 
        args=(doctor_app, 8002, "Doctor"),
        daemon=True
    )
    
    receptionist_thread.start()
    doctor_thread.start()
    
    print("[A2A Server] Both servers started!")
    print("[A2A Server] Receptionist: http://127.0.0.1:8001")
    print("[A2A Server] Doctor: http://127.0.0.1:8002")
    print("[A2A Server] Press Ctrl+C to stop")
    
    try:
        # Keep main thread alive
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n[A2A Server] Shutting down servers...")

if __name__ == "__main__":
    asyncio.run(main())
