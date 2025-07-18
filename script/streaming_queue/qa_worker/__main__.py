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
    QAAgentExecutor,  # type: ignore[import-untyped]
)


if __name__ == '__main__':
    # Define QA skill
    skill = AgentSkill(
        id='question_answering',
        name='Question Answering',
        description='Answers questions using advanced language models',
        tags=['qa', 'question', 'answer', 'llm'],
        examples=[
            'What is artificial intelligence?',
            'How does machine learning work?',
            'Explain quantum computing'
        ],
    )

    extended_skill = AgentSkill(
        id='advanced_qa',
        name='Advanced Question Answering',
        description='Provides detailed explanations and analysis for complex questions.',
        tags=['qa', 'advanced', 'analysis', 'detailed'],
        examples=[
            'Provide a detailed analysis of neural networks',
            'Explain the mathematical foundations of machine learning'
        ],
    )

    # This will be the public-facing agent card
    public_agent_card = AgentCard(
        name='QA Worker Agent',
        description='A question-answering agent using Core LLM',
        url='http://localhost:8000/',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],  # Only the basic skill for the public card
        supportsAuthenticatedExtendedCard=True,
    )

    # This will be the authenticated extended agent card
    # It includes the additional 'extended_skill'
    specific_extended_agent_card = public_agent_card.model_copy(
        update={
            'name': 'QA Worker Agent - Extended Edition',  # Different name for clarity
            'description': 'The full-featured QA agent with advanced capabilities.',
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
        agent_executor=QAAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
        extended_agent_card=specific_extended_agent_card,
    )

    uvicorn.run(server.build(), host='0.0.0.0', port=8000) 