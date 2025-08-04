"""Prompt templates for multi-agent system planning and agent coordination."""
from typing import Dict, List, Any


class PromptTemplates:
    """Central repository for all prompt templates used in the multi-agent system."""

    TASK_ANALYSIS_SYSTEM = """You are an expert multi-agent system architect. Analyze the given task with deep understanding and provide a comprehensive analysis.

Consider these aspects:
1. TASK TYPE - Classify precisely:
   - qa_with_reasoning: Question-answering requiring logical reasoning
   - multi_step_analysis: Complex analysis requiring multiple processing stages  
   - content_generation: Creating new content, documents, reports
   - computational_task: Mathematical calculations, data processing
   - research_task: In-depth information gathering and synthesis
   - general_qa: Simple question-answering

2. COMPLEXITY ASSESSMENT:
   - low: Simple, straightforward tasks requiring 1-2 steps
   - medium: Moderate complexity requiring 3-5 processing steps
   - high: Complex tasks requiring 6+ steps, domain expertise, or sophisticated reasoning

3. REQUIRED TOOLS - Select all applicable:
   - web_search: Online information retrieval
   - file_operators: File I/O, data processing  
   - python_execute: Code execution, calculations
   - content_creation: Writing, summarization, report generation and reasoning

4. DOMAIN EXPERTISE needed (technology, science, business, finance, healthcare, etc.)

5. PROCESSING REQUIREMENTS:
   - Sequential vs parallel processing needs
   - Validation/verification requirements
   - Error handling complexity

IMPORTANT: You must limit your agent recommendations to a maximum of {max_agents} agents total. Plan efficiently within this constraint.

Respond with detailed JSON analysis including your reasoning."""

    TASK_ANALYSIS_USER = """Analyze this task for optimal multi-agent system design:

TASK DOCUMENT:
{document}

Provide comprehensive analysis with specific reasoning for each decision. Consider edge cases and potential challenges."""

    # Agent role prompts
    AGENT_SYSTEM_CONTEXT = """
TASK CONTEXT:
- Task Type: {task_type}
- Complexity: {complexity}
- Domain Areas: {domain_areas}
- Total Agents in System: {total_agents}
- Execution Pattern: {execution_pattern}

COMMUNICATION PROTOCOL:
- Always provide clear, structured responses
- Include confidence levels in your analysis when applicable
- Cite sources when using external information
- Indicate when you need clarification or additional information
- Format responses appropriately for the next agent in the workflow
"""

    COORDINATOR_ROLE = """You are {agent_name}, a task coordination specialist. Your primary role is to:

1. ANALYZE the incoming task and break it down into actionable components
2. PLAN the execution strategy and identify key requirements
3. COORDINATE with specialized agents by providing clear, structured instructions
4. MONITOR progress and adjust plans as needed

Your specialization in {specialization} means you excel at understanding complex tasks and organizing efficient workflows."""

    RESEARCHER_ROLE = """You are {agent_name}, a research specialist focused on {specialization}. Your role is to:

1. SEARCH for relevant, accurate, and up-to-date information
2. EVALUATE source credibility and information quality
3. SYNTHESIZE findings from multiple sources
4. PROVIDE comprehensive research results with proper attribution

Your expertise in {specialization} enables you to find the most relevant and reliable information sources."""

    SPECIALIST_ROLE = """You are {agent_name}, a domain specialist in {specialization}. Your responsibilities include:

1. EXECUTE specialized tasks requiring domain expertise
2. ANALYZE complex problems within your area of specialization
3. PROVIDE expert-level insights and recommendations
4. VALIDATE results using domain-specific knowledge

Your specialization in {specialization} makes you the go-to expert for tasks requiring deep technical knowledge."""

    WORKER_ROLE = """You are {agent_name}, a processing specialist focused on {specialization}. Your tasks include:

1. PROCESS data and information according to specifications
2. EXECUTE operational tasks efficiently and accurately  
3. TRANSFORM information into required formats
4. MAINTAIN quality standards throughout processing

Your expertise in {specialization} ensures reliable and accurate processing of assigned tasks."""

    CREATOR_ROLE = """You are {agent_name}, a content creation specialist in {specialization}. You are responsible for:

1. CREATE high-quality content based on provided information
2. STRUCTURE information in clear, logical formats
3. ADAPT writing style and tone to task requirements
4. ENSURE content accuracy and coherence

Your specialization in {specialization} enables you to produce professional, engaging content."""

    SYNTHESIZER_ROLE = """You are {agent_name}, a synthesis and reasoning specialist. Your critical role is to:

1. INTEGRATE information from all previous agents
2. REASON through complex problems and draw logical conclusions
3. SYNTHESIZE findings into coherent, comprehensive responses
4. PROVIDE final answers that address the original task completely

Your expertise in {specialization} makes you responsible for delivering the final, well-reasoned response."""

    # Tool usage instructions
    TOOL_USAGE_TEMPLATE = """

TOOL USAGE:
You have access to the '{tool_name}' tool with the following capabilities:
{tool_description}

Use this tool strategically to accomplish your tasks. Always:
- Understand the tool's parameters and expected outputs
- Provide appropriate inputs based on your analysis
- Interpret tool results accurately
- Report any tool errors or limitations encountered"""

    # Workflow context template
    WORKFLOW_CONTEXT_TEMPLATE = """
WORKFLOW POSITION:
{workflow_info}"""

    # Quality standards
    QUALITY_STANDARDS_TEMPLATE = """

QUALITY STANDARDS:
- Maximum response length: {max_tokens} tokens
- Provide accurate, relevant information
- Use clear, professional language
- Include confidence levels when making assessments
- Acknowledge limitations or uncertainties
- Follow the established communication protocol"""

    # Final agent prompt template
    AGENT_PROMPT_TEMPLATE = """{system_context}

{base_prompt}{tool_instructions}{workflow_context}{quality_standards}

Remember: Your success is measured by how well you contribute to solving the overall task while fulfilling your specific role in the multi-agent system."""

    @classmethod
    def get_task_analysis_prompts(cls, max_agents: int = 5) -> tuple[str, str]:
        """Get system and user prompts for task analysis."""
        system_prompt = cls.TASK_ANALYSIS_SYSTEM.format(max_agents=max_agents)
        user_template = cls.TASK_ANALYSIS_USER
        
        return system_prompt, user_template

    @classmethod
    def get_role_prompt(cls, role: str) -> str:
        """Get the base prompt template for a specific role."""
        role_prompts = {
            "coordinator": cls.COORDINATOR_ROLE,
            "researcher": cls.RESEARCHER_ROLE,
            "specialist": cls.SPECIALIST_ROLE,
            "worker": cls.WORKER_ROLE,
            "creator": cls.CREATOR_ROLE,
            "synthesizer": cls.SYNTHESIZER_ROLE
        }
        return role_prompts.get(role, "You are {agent_name}, a {role} specialist.")

    @classmethod
    def build_agent_prompt(
        cls,
        agent_name: str,
        role: str,
        specialization: str,
        tool_name: str,
        tool_description: str,
        max_tokens: int,
        task_type: str,
        complexity: str,
        domain_areas: List[str],
        total_agents: int,
        execution_pattern: str,
        workflow_info: str
    ) -> str:
        """Build a complete agent prompt from templates."""
        
        # System context
        system_context = cls.AGENT_SYSTEM_CONTEXT.format(
            task_type=task_type,
            complexity=complexity,
            domain_areas=", ".join(domain_areas),
            total_agents=total_agents,
            execution_pattern=execution_pattern
        )
        
        # Base role prompt
        role_template = cls.get_role_prompt(role)
        base_prompt = role_template.format(
            agent_name=agent_name,
            specialization=specialization
        )
        
        # Tool instructions
        tool_instructions = ""
        if tool_name and tool_description:
            tool_instructions = cls.TOOL_USAGE_TEMPLATE.format(
                tool_name=tool_name,
                tool_description=tool_description
            )
        
        # Workflow context
        workflow_context = cls.WORKFLOW_CONTEXT_TEMPLATE.format(
            workflow_info=workflow_info
        ) if workflow_info else ""
        
        # Quality standards
        quality_standards = cls.QUALITY_STANDARDS_TEMPLATE.format(
            max_tokens=max_tokens
        )
        
        # Combine all parts
        return cls.AGENT_PROMPT_TEMPLATE.format(
            system_context=system_context,
            base_prompt=base_prompt,
            tool_instructions=tool_instructions,
            workflow_context=workflow_context,
            quality_standards=quality_standards
        ).strip()


class PromptBuilder:
    """Helper class for building prompts dynamically."""
    
    def __init__(self):
        self.templates = PromptTemplates()
    
    def build_task_analysis_messages(self, document: str, max_agents: int = 5) -> List[Dict[str, str]]:
        """Build messages for task analysis."""
        system_prompt, user_template = self.templates.get_task_analysis_prompts(max_agents)
        user_prompt = user_template.format(document=document)
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def build_agent_prompt(self, agent_config: Dict[str, Any], analysis: Dict[str, Any], 
                          workflow: Dict[str, Any], tool_description: str = "",
                          workflow_info: str = "") -> str:
        """Build a complete agent prompt."""
        return self.templates.build_agent_prompt(
            agent_name=agent_config.get("name", "Agent"),
            role=agent_config.get("role", "worker"),
            specialization=agent_config.get("specialization", "general"),
            tool_name=agent_config.get("tool", ""),
            tool_description=tool_description,
            max_tokens=agent_config.get("max_tokens", 500),
            task_type=analysis.get("task_type", "general_qa"),
            complexity=analysis.get("complexity", "medium"),
            domain_areas=analysis.get("domain_areas", ["general"]),
            total_agents=len(analysis.get("agents", [])),
            execution_pattern=workflow.get("execution_pattern", "sequential"),
            workflow_info=workflow_info
        )
