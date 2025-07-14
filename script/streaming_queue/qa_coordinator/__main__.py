import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from agent_executor import (
    QACoordinatorExecutor,  # type: ignore[import-untyped]
)


if __name__ == '__main__':
    # --8<-- [start:AgentSkill]
    skill = AgentSkill(
        id='qa_coordination',
        name='QA Task Coordination',
        description='Coordinates question-answer tasks by dispatching questions from jsonl files to worker agents using round-robin scheduling',
        tags=['coordination', 'qa', 'task distribution', 'round-robin'],
        examples=['dispatch marco_1000.jsonl', 'status', 'start coordination'],
    )
    # --8<-- [end:AgentSkill]

    extended_skill = AgentSkill(
        id='advanced_coordination',
        name='Advanced QA Coordination',
        description='Advanced coordination features including batch size configuration, worker health monitoring, and performance metrics.',
        tags=['coordination', 'qa', 'advanced', 'metrics', 'monitoring'],
        examples=['configure batch size 100', 'monitor workers', 'get performance metrics'],
    )

    # --8<-- [start:AgentCard]
    # This will be the public-facing agent card
    public_agent_card = AgentCard(
        name='QA Coordinator Agent',
        description='Coordinates question-answer tasks by dispatching questions to registered worker agents using A2A protocol',
        url='http://localhost:9998/',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],  # Only the basic skill for the public card
        supportsAuthenticatedExtendedCard=True,
    )
    # --8<-- [end:AgentCard]

    # This will be the authenticated extended agent card
    # It includes the additional 'extended_skill'
    specific_extended_agent_card = public_agent_card.model_copy(
        update={
            'name': 'QA Coordinator Agent - Extended Edition',  # Different name for clarity
            'description': 'Full-featured QA coordinator with advanced monitoring and configuration capabilities for authenticated users.',
            'version': '1.0.1',  # Could even be a different version
            # Capabilities and other fields like url, defaultInputModes, defaultOutputModes,
            # supportsAuthenticatedExtendedCard are inherited from public_agent_card unless specified here.
            'skills': [
                skill,
                extended_skill,
            ],  # Both skills for the extended card
        }
    )

    request_handler = DefaultRequestHandler(
        agent_executor=QACoordinatorExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
        extended_agent_card=specific_extended_agent_card,
    )

    uvicorn.run(server.build(), host='0.0.0.0', port=9998) 