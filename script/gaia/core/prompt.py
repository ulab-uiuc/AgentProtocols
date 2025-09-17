"""Prompt templates for system planning and agent coordination."""
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

3. REQUIRED TOOLS - Select from available tools:
   Available tools: {available_tools}

4. AGENT CONFIGURATION - For each required tool, specify:
   - name: Descriptive agent name (e.g., "WebResearcher", "DataAnalyst", "CodeExecutor")
   - role: Create meaningful, task-specific roles (e.g., "information_gatherer", "computational_specialist", "data_processor", "final_synthesizer", "document_analyzer", "web_navigator", etc.)
   - Be creative with roles - they should reflect the agent's specific function in solving the task

   Example role types you can use as inspiration:
   * information_gatherer: Searches for and collects relevant information from various sources
   * computational_specialist: Executes calculations, data processing, and analytical tasks
   * document_analyzer: Processes and extracts information from documents and files
   * evidence_synthesizer: Integrates information from multiple sources into coherent conclusions
   * task_coordinator: Breaks down complex tasks and manages workflow execution
   * content_creator: Generates reports, summaries, and structured outputs
   * domain_expert: Provides specialized knowledge in specific fields
   * data_processor: Handles data transformation, cleaning, and formatting
   * web_navigator: Specializes in web search and online information retrieval
   * final_synthesizer: Provides comprehensive final answers and conclusions


5. DOMAIN EXPERTISE needed (technology, science, business, finance, healthcare, etc.)

6. PROCESSING REQUIREMENTS:
   - Sequential vs parallel processing needs
   - Validation/verification requirements
   - Error handling complexity

IMPORTANT HARD RULES:
- The tool 'create_chat_completion' is reserved for the FINAL agent only. Include it exactly once and position it as the LAST step in the workflow. Do NOT assign or call it in intermediate steps or by non-final agents.

IMPORTANT: Based on the GAIA task level {level}, we recommend using approximately {recommended_agents} agents for optimal performance. However, you can adjust this number based on task complexity:
- Use fewer agents (1-2) for very simple, single-step tasks
- Use the recommended number ({recommended_agents}) for typical level {level} tasks  
- Use more agents (up to {max_agents}) only if the task genuinely requires complex multi-step processing

You must limit your agent recommendations to a maximum of {max_agents} agents total. Plan efficiently within this constraint.
Respond with detailed JSON analysis including your reasoning.

Analyze the task and respond with a JSON object containing:
{{
  "task_type": "general_qa|research_task|computational_task|multi_step_analysis",
  "complexity": "low|medium|high", 
  "required_tools": ["tool1", "tool2"],
  "agents": [
    {{
      "tool": "tool_name",
      "name": "AgentName", 
      "role": "specific_role_based_on_function",

    }}
  ],
  "estimated_steps": number,
  "domain_areas": ["domain1", "domain2"]
}}

Example:
{{
  "task_type": "research_task",
  "complexity": "medium",
  "required_tools": ["browser_use", "create_chat_completion"],
  "agents": [
    {{
      "tool": "browser_use",
      "name": "WebResearcher",
      "role": "academic_information_gatherer", 

    }},
    {{
      "tool": "create_chat_completion",
      "name": "ReasoningSynthesizer",
      "role": "evidence_synthesizer",

    }}
  ],
  "estimated_steps": 3,
  "domain_areas": ["general_knowledge"]
}}
"""

    TASK_ANALYSIS_USER = """Analyze this task for optimal multi-agent system design:

TASK DOCUMENT:
{document}

Provide comprehensive analysis with specific reasoning for each decision. Consider edge cases and potential challenges."""

    # Agent system context with original task requirement
    AGENT_SYSTEM_CONTEXT = """
🎯 ORIGINAL TASK REQUIREMENT (NEVER FORGET):
{original_query}

TASK CONTEXT:
- Task Type: {task_type}
- Complexity: {complexity}
- Domain Areas: {domain_areas}
- Total Agents in System: {total_agents}
- Execution Pattern: {execution_pattern}

⚠️ CRITICAL REMINDERS:
- ALWAYS keep the original task requirement in mind throughout your execution
- Your work must contribute to answering the original query
- Validate that your actions align with the original task before proceeding
- If unsure about relevance, ask yourself: "Does this help answer the original question?"

COMMUNICATION PROTOCOL:
- Always provide clear, structured responses
- Include confidence levels in your analysis when applicable
- Cite sources when using external information
- Indicate when you need clarification or additional information
- Format responses appropriately for the next agent in the workflow
"""

    # Tool usage instructions
    TOOL_USAGE_TEMPLATE = """

TOOL USAGE:
You have access to the '{tool_name}' tool with the following capabilities:
{tool_description}

Use this tool strategically to accomplish your tasks. Always:
- Understand the tool's parameters and expected outputs
- Provide appropriate inputs based on your analysis
- Interpret tool results accurately
- Report any tool errors or limitations encountered

SPECIAL RULES:
- If your tool is 'create_chat_completion': You are the FINAL agent. Use this tool exactly once at the very end to format and emit the final answer. Do NOT use it for intermediate steps or partial results. Non-final agents must not call this tool.

SEARCH OPTIMIZATION (for browser_use tool):
When using web_search action, use TARGETED SEARCH STRATEGY based on the original task:

1. ANALYZE THE ORIGINAL QUERY FIRST:
   - Identify key terms, dates, specific names, and requirements
   - Understand what type of content is needed (academic papers, articles, specific documents)
   
2. FOR ARXIV/ACADEMIC PAPERS:
   - If the original query mentions specific years (e.g., "2022", "2016"), ALWAYS include those years in search
   - For AI/ML papers: Use "AI regulation 2022 site:arxiv.org" or "machine learning policy 2022 arxiv"
   - For Physics papers: Use "Physics Society 2016 arxiv" or "American Physical Society 2016"
   - For specific organizations: Include organization name + year + "arxiv"
   - VALIDATE: Check if returned papers are from the correct year and topic
   
3. USE STRATEGIC SEARCH PROGRESSION:
   - Start with specific terms from the original query + year + site:arxiv.org
   - If no results: try variations without site restriction
   - If still no results: try synonyms or broader terms
   - Always verify results match the original requirements
   
4. VALIDATE RESULTS:
   - Before accepting any search results, check if they actually relate to the original query
   - Verify dates, authorship, and content relevance  
   - If results don't match what's needed, try different search terms
   - For academic queries: Ensure results are from reputable academic sources
   
5. COMMON PATTERNS:
   - "2022 AI regulation paper" → Try "AI regulation 2022 arxiv", "AI governance 2022 policy paper"
   - "Physics Society article 2016" → Try "American Physical Society 2016", "Physics Society 2016 arxiv" 
   - "Machine learning research" → Try "machine learning arxiv", "ML research paper"

❌ AVOID: Generic broad terms that may return irrelevant results from wrong time periods
✅ PREFER: Specific terms with years/dates that directly relate to what the original query asks for
⚠️  CRITICAL: Always double-check that search results match the year and topic requirements from the original query"""

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
    def get_task_analysis_prompts(cls, recommended_agents: int = 4, max_agents: int = 10, 
                                 level: int = 2, available_tools: List[str] = None) -> tuple[str, str]:
        """Get system and user prompts for task analysis."""
        if available_tools is None:
            available_tools = ["web_search", "python_execute", "file_operators", "create_chat_completion"]
        
        system_prompt = cls.TASK_ANALYSIS_SYSTEM.format(
            recommended_agents=recommended_agents,
            max_agents=max_agents,
            level=level,
            available_tools=", ".join(available_tools)
        )
        user_template = cls.TASK_ANALYSIS_USER
        
        return system_prompt, user_template

    @classmethod
    def get_role_prompt(cls, role: str, agent_name: str) -> str:
        """Generate a dynamic role prompt based on the role name."""
        # Create a generic but descriptive prompt based on the role
        role_words = role.replace("_", " ").replace("-", " ").title()
        
        return f"""You are {agent_name}, a {role_words.lower()} specialist. Your primary responsibilities include:

1. EXECUTE tasks related to your {role_words.lower()} expertise
2. PROVIDE expert-level insights and analysis within your domain
3. PROCESS information efficiently and accurately according to your role
4. COLLABORATE effectively with other agents in the workflow
5. DELIVER high-quality results that contribute to the overall task completion

Your expertise in {role_words.lower()} makes you an essential part of the multi-agent system."""

    @classmethod
    def build_agent_prompt(
        cls,
        agent_name: str,
        role: str,
        tool_name: str,
        tool_description: str,
        max_tokens: int,
        task_type: str,
        complexity: str,
        domain_areas: List[str],
        total_agents: int,
        execution_pattern: str,
        workflow_info: str,
        original_query: str = ""  # Add original query parameter
    ) -> str:
        """Build a complete agent prompt from templates."""
        
        # System context with original query
        system_context = cls.AGENT_SYSTEM_CONTEXT.format(
            original_query=original_query or "No specific query provided",
            task_type=task_type,
            complexity=complexity,
            domain_areas=", ".join(domain_areas),
            total_agents=total_agents,
            execution_pattern=execution_pattern
        )
        
        # Base role prompt
        role_template = cls.get_role_prompt(role, agent_name)
        base_prompt = role_template
        
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
    
    def build_task_analysis_messages(self, document: str, recommended_agents: int = 4, 
                                   max_agents: int = 10, level: int = 2, 
                                   available_tools: List[str] = None) -> List[Dict[str, str]]:
        """Build messages for task analysis."""
        system_prompt, user_template = self.templates.get_task_analysis_prompts(
            recommended_agents, max_agents, level, available_tools)
        user_prompt = user_template.format(document=document)
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def build_agent_prompt(self, agent_config: Dict[str, Any], analysis: Dict[str, Any], 
                          workflow: Dict[str, Any], tool_description: str = "",
                          workflow_info: str = "", original_query: str = "") -> str:
        """Build a complete agent prompt."""
        return self.templates.build_agent_prompt(
            agent_name=agent_config.get("name", "Agent"),
            role=agent_config.get("role", "worker"),
            tool_name=agent_config.get("tool", ""),
            tool_description=tool_description,
            max_tokens=agent_config.get("max_tokens", 500),
            task_type=analysis.get("task_type", "general_qa"),
            complexity=analysis.get("complexity", "medium"),
            domain_areas=analysis.get("domain_areas", ["general"]),
            total_agents=len(analysis.get("agents", [])),
            execution_pattern=workflow.get("execution_pattern", "sequential"),
            workflow_info=workflow_info,
            original_query=original_query  # Pass original query through
        )
