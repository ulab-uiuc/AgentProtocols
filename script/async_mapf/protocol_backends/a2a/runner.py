# script/async_mapf/protocol_backends/a2a/runner.py
from pathlib import Path
from typing import Dict, Any

from script.async_mapf.runners.base_runner import RunnerBase, SimpleOutput

# A2A SDK
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentProvider

# Protocol-specific executors
from script.async_mapf.protocol_backends.a2a.agents.mapf_worker_executor import MAPFAgentExecutor
from script.async_mapf.protocol_backends.a2a.agents.network_executor import NetworkBaseExecutor


class A2ARunner(RunnerBase):
    """
    A2A protocol runner.
    Implements how to build ASGI apps for NetworkBase and MAPF agents using A2A SDK.
    """

    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self.output = SimpleOutput("A2ARunner")

    # -----------------------------
    # Hooks required by RunnerBase
    # -----------------------------
    def build_network_app(self, network_coordinator, agent_urls: Dict[int, str]):
        """
        Build ASGI app for the NetworkBase endpoint using A2A executors.
        Use positional args for NetworkBaseExecutor to be compatible
        with older constructor signatures.
        """
        # Create NetworkBase executor with knowledge of agent URLs
        # NOTE: pass `network_coordinator` POSITIONALLY for compatibility.
        network_executor = NetworkBaseExecutor(
            network_coordinator,       # coordinator (positional)
            agent_urls,                # agent_urls (positional)
            output=self.output         # keep named; this is supported
        )

        # Build the "coordinator" AgentCard
        agent_card = AgentCard(
            name="MAPF Network Coordinator",
            description="Central coordinator for Multi-Agent Path Finding simulation",
            url="http://localhost:0/",
            version="1.0.0",
            provider=AgentProvider(
                name="MAPF Simulation",
                organization="Agent Research Lab",
                url="http://localhost:0/",
                email="admin@example.com",
            ),
            defaultInputModes=["text"],
            defaultOutputModes=["text"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[
                AgentSkill(
                    id="coordinate_mapf",
                    name="coordinate_mapf",
                    description="Coordinate multi-agent path finding and collision detection",
                    tags=["mapf", "coordination", "pathfinding"],
                    inputModes=["text"],
                    outputModes=["text"],
                    examples=[],
                )
            ],
            documentationUrl="",
            security=[],
            securitySchemes={},
            supportsAuthenticatedExtendedCard=False,
        )

        request_handler = DefaultRequestHandler(
            agent_executor=network_executor,
            task_store=InMemoryTaskStore(),
        )

        return A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

    def build_agent_app(self, agent_id: int, agent_cfg: Dict[str, Any], port: int, network_base_port: int):
        """
        Build ASGI app for a single agent endpoint using A2A executors.
        """
        # Build agent skill/card
        skill = AgentSkill(
            id=f"mapf_agent_{agent_id}",
            name=f"MAPF Agent {agent_id}",
            description=f"Multi-Agent Path Finding robot agent {agent_id}",
            tags=["mapf", "pathfinding", "robot"],
            inputModes=["text"],
            outputModes=["text"],
            examples=["PLAN", "STEP", "STATUS"],
        )

        agent_card = AgentCard(
            name=f"MAPF Agent {agent_id}",
            description=f"Multi-Agent Path Finding robot agent {agent_id}",
            url=f"http://localhost:{port}/",
            version="1.0.0",
            provider=AgentProvider(
                name="MAPF Simulation",
                organization="Agent Research Lab",
                url=f"http://localhost:{port}/",
                email="admin@example.com",
            ),
            defaultInputModes=["text"],
            defaultOutputModes=["text"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[skill],
            documentationUrl="",
            security=[],
            securitySchemes={},
            supportsAuthenticatedExtendedCard=False,
        )

        # Make agent talk to NetworkBase via its dynamic port
        network_base_url = f"http://localhost:{network_base_port}"
        executor = MAPFAgentExecutor(
            cfg=agent_cfg,
            global_cfg=self.sim_config,
            agent_id=agent_id,
            router_url=network_base_url,
            output=self.output,
        )

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )

        return A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )
