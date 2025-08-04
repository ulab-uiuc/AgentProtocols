"""Intelligent planner for analyzing Gaia tasks and generating agent configurations."""
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.registry import ToolRegistry
from core.llm import call_llm


class PlanningStrategy(ABC):
    """Abstract base class for planning strategies."""
    
    @abstractmethod
    async def analyze_task(self, document: str) -> Dict[str, Any]:
        """Analyze the task document and extract characteristics."""
        pass
    
    @abstractmethod
    async def generate_config(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate agent configuration based on task analysis."""
        pass


class SimplePlanningStrategy(PlanningStrategy):
    """Simple strategy that creates basic agent configurations."""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.llm = call_llm()
    
    async def analyze_task(self, document: str) -> Dict[str, Any]:
        """LLM-based task analysis for multi-agent planning."""
        # System prompt for task analysis
        system_prompt = """You are an expert multi-agent system planner. Analyze the given task and determine:

1. Task type (qa_with_reasoning, multi_step_analysis, content_generation, computational_task, research_task)
2. Complexity level (low, medium, high) based on steps required and domain expertise needed
3. Required capabilities from these options:
   - web_search: For finding information online
   - file_operators: For reading/writing files, data processing
   - python_execute: For code execution, calculations, data analysis
   - planning: For breaking down complex tasks
   - reasoning: For logi cal analysis and synthesis
   - research: For in-depth information gathering
   - data_analysis: For processing and analyzing data
   - content_creation: For generating text, reports, summaries

4. Estimated number of processing steps
5. Key domain areas involved (technology, science, business, general knowledge, etc.)

Respond with a JSON object containing your analysis. Be specific and accurate."""

        user_prompt = f"""Please analyze this task for multi-agent planning:

TASK:
{document}

Provide a detailed analysis focusing on what types of specialized agents would be needed to complete this task effectively."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Get LLM analysis
            response = await self.llm.ask(messages=messages, temperature=0.3)
            
            # Try to parse JSON response
            try:
                # Extract JSON from response if it's wrapped in text
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    llm_analysis = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError):
                # Fallback to keyword-based analysis if JSON parsing fails
                print("Warning: LLM response parsing failed, falling back to keyword analysis")
                return await self._fallback_keyword_analysis(document)
            
            # Validate and structure the analysis
            analysis = self._validate_and_structure_analysis(llm_analysis, document)
            return analysis
            
        except Exception as e:
            print(f"Warning: LLM analysis failed ({e}), falling back to keyword analysis")
            return await self._fallback_keyword_analysis(document)
    
    def _validate_and_structure_analysis(self, llm_analysis: Dict[str, Any], document: str) -> Dict[str, Any]:
        """Validate and structure LLM analysis results."""
        # Valid options
        valid_task_types = ["qa_with_reasoning", "multi_step_analysis", "content_generation", 
                           "computational_task", "research_task", "general_qa"]
        valid_complexities = ["low", "medium", "high"]
        valid_capabilities = ["web_search", "file_operators", "python_execute", "planning", 
                             "reasoning", "research", "data_analysis", "content_creation"]
        
        # Extract and validate fields
        task_type = llm_analysis.get("task_type", "general_qa")
        if task_type not in valid_task_types:
            task_type = "general_qa"
        
        complexity = llm_analysis.get("complexity", "medium")
        if complexity not in valid_complexities:
            complexity = "medium"
        
        required_capabilities = llm_analysis.get("required_capabilities", [])
        if not isinstance(required_capabilities, list):
            required_capabilities = []
        
        # Filter valid capabilities
        required_capabilities = [cap for cap in required_capabilities if cap in valid_capabilities]
        
        # Ensure minimum capabilities
        if not required_capabilities:
            required_capabilities = ["reasoning", "planning"]
        
        # Always include reasoning for complex tasks
        if complexity in ["medium", "high"] and "reasoning" not in required_capabilities:
            required_capabilities.append("reasoning")
        
        estimated_steps = llm_analysis.get("estimated_steps", len(required_capabilities) + 1)
        if not isinstance(estimated_steps, int) or estimated_steps < 1:
            estimated_steps = len(required_capabilities) + 1
        
        domain_areas = llm_analysis.get("domain_areas", ["general_knowledge"])
        if not isinstance(domain_areas, list):
            domain_areas = ["general_knowledge"]
        
        return {
            "task_type": task_type,
            "complexity": complexity,
            "required_capabilities": required_capabilities,
            "estimated_steps": estimated_steps,
            "domain_areas": domain_areas,
            "llm_analysis": llm_analysis,  # Keep original analysis for reference
            "document_length": len(document),
            "has_web_search": "web_search" in required_capabilities,
            "has_file_ops": "file_operators" in required_capabilities,
            "has_code_exec": "python_execute" in required_capabilities,
            "has_planning": "planning" in required_capabilities
        }
    
    async def _fallback_keyword_analysis(self, document: str) -> Dict[str, Any]:
        """Fallback keyword-based analysis when LLM fails."""
        doc_lower = document.lower()
        
        # Detect task characteristics
        has_web_search = any(keyword in doc_lower for keyword in ["search", "web", "internet", "online", "find"])
        has_file_ops = any(keyword in doc_lower for keyword in ["file", "read", "write", "csv", "json", "data"])
        has_code_exec = any(keyword in doc_lower for keyword in ["code", "python", "execute", "run", "calculate"])
        has_planning = any(keyword in doc_lower for keyword in ["plan", "step", "strategy", "approach"])
        
        complexity = "low"
        if len(document) > 1000:
            complexity = "high"
        elif len(document) > 500:
            complexity = "medium"
        
        required_capabilities = []
        if has_web_search:
            required_capabilities.append("web_search")
        if has_file_ops:
            required_capabilities.append("file_operators")
        if has_code_exec:
            required_capabilities.append("python_execute")
        if has_planning:
            required_capabilities.append("planning")
        
        # Default capabilities if none detected
        if not required_capabilities:
            required_capabilities = ["web_search", "planning"]
            
        # Always add reasoning for fallback
        if "reasoning" not in required_capabilities:
            required_capabilities.append("reasoning")
        
        return {
            "task_type": "general_qa",
            "complexity": complexity,
            "required_capabilities": required_capabilities,
            "estimated_steps": len(required_capabilities) + 1,
            "domain_areas": ["general_knowledge"],
            "llm_analysis": None,
            "document_length": len(document),
            "has_web_search": has_web_search,
            "has_file_ops": has_file_ops,
            "has_code_exec": has_code_exec,
            "has_planning": has_planning
        }
    
    async def generate_config(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate agent configuration based on LLM analysis."""
        agents = []
        agent_id = 0
        port = 9000
        
        capabilities = analysis["required_capabilities"]
        complexity = analysis["complexity"]
        task_type = analysis["task_type"]
        
        # Token allocation based on complexity
        token_allocation = {
            "low": {"planner": 400, "worker": 300, "reasoner": 500},
            "medium": {"planner": 600, "worker": 500, "reasoner": 800},
            "high": {"planner": 800, "worker": 600, "reasoner": 1200}
        }
        tokens = token_allocation.get(complexity, token_allocation["medium"])
        
        # Always start with a planning agent for complex tasks
        if "planning" in capabilities or complexity in ["medium", "high"]:
            agents.append({
                "id": agent_id,
                "name": "TaskPlanner",
                "tool": "planning",
                "port": port,
                "priority": 1,
                "max_tokens": tokens["planner"],
                "specialization": "task_decomposition",
                "capabilities": ["planning", "task_analysis"],
                "role": "coordinator"
            })
            agent_id += 1
            port += 1
        
        # Enhanced capability mapping with role definitions
        capability_agents = {
            "web_search": {
                "name": "WebResearcher", 
                "tool": "web_search", 
                "specialization": "information_retrieval",
                "role": "researcher",
                "capabilities": ["web_search", "information_synthesis"]
            },
            "research": {
                "name": "ResearchAgent",
                "tool": "web_search",
                "specialization": "deep_research",
                "role": "specialist",
                "capabilities": ["research", "fact_checking", "source_validation"]
            },
            "file_operators": {
                "name": "DataProcessor", 
                "tool": "file_operators", 
                "specialization": "data_management",
                "role": "worker",
                "capabilities": ["file_ops", "data_processing"]
            },
            "python_execute": {
                "name": "CodeExecutor", 
                "tool": "python_execute", 
                "specialization": "computation",
                "role": "specialist",
                "capabilities": ["code_execution", "calculation", "data_analysis"]
            },
            "data_analysis": {
                "name": "DataAnalyst",
                "tool": "python_execute",
                "specialization": "statistical_analysis",
                "role": "specialist", 
                "capabilities": ["data_analysis", "visualization", "statistical_modeling"]
            },
            "content_creation": {
                "name": "ContentCreator",
                "tool": "create_chat_completion",
                "specialization": "content_generation",
                "role": "creator",
                "capabilities": ["writing", "summarization", "content_structuring"]
            }
        }
        
        # Add specialized agents based on capabilities
        for capability in capabilities:
            if capability in capability_agents and capability != "planning":
                agent_config = capability_agents[capability]
                agents.append({
                    "id": agent_id,
                    "name": agent_config["name"],
                    "tool": agent_config["tool"],
                    "port": port,
                    "priority": 2,
                    "max_tokens": tokens["worker"],
                    "specialization": agent_config["specialization"],
                    "role": agent_config["role"],
                    "capabilities": agent_config["capabilities"]
                })
                agent_id += 1
                port += 1
        
        # Always add a reasoning/synthesis agent for final processing
        if "reasoning" in capabilities or len(agents) > 1:
            agents.append({
                "id": agent_id,
                "name": "ReasoningSynthesizer",
                "tool": "create_chat_completion",
                "port": port,
                "priority": 3,
                "max_tokens": tokens["reasoner"],
                "specialization": "reasoning_synthesis", 
                "role": "synthesizer",
                "capabilities": ["reasoning", "synthesis", "final_answer_generation"]
            })
        
        # Generate enhanced workflow
        workflow = self._generate_enhanced_workflow(agents, analysis)
        
        # Generate agent prompts
        agent_prompts = await self._generate_agent_prompts(agents, analysis, workflow)
        
        return {
            "task_id": f"task_{int(time.time())}",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "task_analysis": analysis,
            "agents": agents,
            "workflow": workflow,
            "agent_prompts": agent_prompts,
            "communication_rules": self._generate_communication_rules(agents),
            "performance_targets": self._set_performance_targets(analysis),
            "execution_strategy": self._determine_execution_strategy(analysis, agents)
        }
    
    def _generate_enhanced_workflow(self, agents: List[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced workflow based on agents and task analysis."""
        if len(agents) <= 1:
            return {
                "start_agent": 0,
                "message_flow": [],
                "parallel_execution": [],
                "fallback_agents": [],
                "execution_pattern": "single_agent"
            }
        
        # Determine execution pattern based on task complexity and type
        task_type = analysis.get("task_type", "general_qa")
        complexity = analysis.get("complexity", "medium")
        
        if complexity == "high" and len(agents) > 3:
            execution_pattern = "parallel_then_sequential"
        elif task_type in ["multi_step_analysis", "research_task"]:
            execution_pattern = "sequential_with_feedback"
        else:
            execution_pattern = "sequential"
        
        message_flow = []
        parallel_execution = []
        
        if execution_pattern == "parallel_then_sequential":
            # First, run research/data agents in parallel
            parallel_agents = [i for i, agent in enumerate(agents) 
                             if agent.get("role") in ["researcher", "worker", "specialist"] 
                             and i > 0]  # Skip planner
            
            if parallel_agents:
                parallel_execution.append({
                    "agents": parallel_agents,
                    "sync_after": True,
                    "timeout": 120
                })
                
                # Then sequential flow to synthesizer
                synthesizer_idx = next((i for i, agent in enumerate(agents) 
                                      if agent.get("role") == "synthesizer"), len(agents) - 1)
                
                message_flow.append({
                    "from": "parallel_group_0",
                    "to": [synthesizer_idx],
                    "message_type": "aggregated_results"
                })
        
        elif execution_pattern == "sequential_with_feedback":
            # Standard sequential flow with feedback loops
            for i in range(len(agents) - 1):
                message_flow.append({
                    "from": i,
                    "to": [i + 1],
                    "message_type": "task_result",
                    "allow_feedback": True
                })
        
        else:  # Standard sequential
            for i in range(len(agents) - 1):
                message_flow.append({
                    "from": i,
                    "to": [i + 1],
                    "message_type": "task_result"
                })
        
        # Final output flow
        final_agent_idx = len(agents) - 1
        message_flow.append({
            "from": final_agent_idx,
            "to": "final",
            "message_type": "final_answer"
        })
        
        return {
            "start_agent": 0,
            "message_flow": message_flow,
            "parallel_execution": parallel_execution,
            "fallback_agents": self._identify_fallback_agents(agents),
            "execution_pattern": execution_pattern
        }
    
    async def _generate_agent_prompts(self, agents: List[Dict[str, Any]], analysis: Dict[str, Any], workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed prompts for each agent based on their role and tools."""
        agent_prompts = {}
        
        # Get tool descriptions from registry
        tool_descriptions = {}
        for agent in agents:
            tool_name = agent.get("tool", "")
            if tool_name and tool_name not in tool_descriptions:
                tool_info = self.tool_registry.get_tool_info(tool_name)
                tool_descriptions[tool_name] = tool_info
        
        # System context for all agents
        system_context = f"""
TASK CONTEXT:
- Task Type: {analysis.get('task_type', 'general_qa')}
- Complexity: {analysis.get('complexity', 'medium')}
- Domain Areas: {', '.join(analysis.get('domain_areas', ['general']))}
- Total Agents in System: {len(agents)}
- Execution Pattern: {workflow.get('execution_pattern', 'sequential')}

COMMUNICATION PROTOCOL:
- Always provide clear, structured responses
- Include confidence levels in your analysis when applicable
- Cite sources when using external information
- Indicate when you need clarification or additional information
- Format responses appropriately for the next agent in the workflow
"""

        for i, agent in enumerate(agents):
            agent_id = agent["id"]
            agent_name = agent["name"]
            role = agent.get("role", "worker")
            specialization = agent.get("specialization", "general")
            capabilities = agent.get("capabilities", [])
            tool_name = agent.get("tool", "")
            max_tokens = agent.get("max_tokens", 500)
            
            # Get next agents in workflow
            next_agents = self._get_next_agents(agent_id, workflow)
            prev_agents = self._get_previous_agents(agent_id, workflow)
            
            # Role-specific prompt templates
            role_prompts = {
                "coordinator": f"""You are {agent_name}, a task coordination specialist. Your primary role is to:

1. ANALYZE the incoming task and break it down into actionable components
2. PLAN the execution strategy and identify key requirements
3. COORDINATE with specialized agents by providing clear, structured instructions
4. MONITOR progress and adjust plans as needed

Your specialization in {specialization} means you excel at understanding complex tasks and organizing efficient workflows.""",

                "researcher": f"""You are {agent_name}, a research specialist focused on {specialization}. Your role is to:

1. SEARCH for relevant, accurate, and up-to-date information
2. EVALUATE source credibility and information quality
3. SYNTHESIZE findings from multiple sources
4. PROVIDE comprehensive research results with proper attribution

Your expertise in {specialization} enables you to find the most relevant and reliable information sources.""",

                "specialist": f"""You are {agent_name}, a domain specialist in {specialization}. Your responsibilities include:

1. EXECUTE specialized tasks requiring domain expertise
2. ANALYZE complex problems within your area of specialization
3. PROVIDE expert-level insights and recommendations
4. VALIDATE results using domain-specific knowledge

Your specialization in {specialization} makes you the go-to expert for tasks requiring deep technical knowledge.""",

                "worker": f"""You are {agent_name}, a processing specialist focused on {specialization}. Your tasks include:

1. PROCESS data and information according to specifications
2. EXECUTE operational tasks efficiently and accurately  
3. TRANSFORM information into required formats
4. MAINTAIN quality standards throughout processing

Your expertise in {specialization} ensures reliable and accurate processing of assigned tasks.""",

                "creator": f"""You are {agent_name}, a content creation specialist in {specialization}. You are responsible for:

1. CREATE high-quality content based on provided information
2. STRUCTURE information in clear, logical formats
3. ADAPT writing style and tone to task requirements
4. ENSURE content accuracy and coherence

Your specialization in {specialization} enables you to produce professional, engaging content.""",

                "synthesizer": f"""You are {agent_name}, a synthesis and reasoning specialist. Your critical role is to:

1. INTEGRATE information from all previous agents
2. REASON through complex problems and draw logical conclusions
3. SYNTHESIZE findings into coherent, comprehensive responses
4. PROVIDE final answers that address the original task completely

Your expertise in {specialization} makes you responsible for delivering the final, well-reasoned response."""
            }
            
            base_prompt = role_prompts.get(role, f"You are {agent_name}, a {role} specialist.")
            
            # Add tool-specific instructions
            tool_instructions = ""
            if tool_name and tool_name in tool_descriptions:
                tool_desc = tool_descriptions[tool_name]
                tool_instructions = f"""

TOOL USAGE:
You have access to the '{tool_name}' tool with the following capabilities:
{tool_desc}

Use this tool strategically to accomplish your tasks. Always:
- Understand the tool's parameters and expected outputs
- Provide appropriate inputs based on your analysis
- Interpret tool results accurately
- Report any tool errors or limitations encountered"""

            # Add workflow context
            workflow_context = ""
            if prev_agents:
                prev_agent_names = [agents[idx]['name'] for idx in prev_agents if isinstance(idx, int) and idx < len(agents)]
                workflow_context += f"\nYou will receive input from: {prev_agent_names}"
            if next_agents:
                next_agent_names = [agents[idx]['name'] for idx in next_agents if isinstance(idx, int) and idx < len(agents)]
                if "final" in next_agents:
                    next_agent_names.append("final_output")
                workflow_context += f"\nYour output will be sent to: {next_agent_names}"
            if not next_agents or "final" in str(next_agents):
                workflow_context += "\nYou are responsible for generating the final response."

            # Add capability-specific guidelines
            capability_guidelines = ""
            if capabilities:
                capability_guidelines = f"""

YOUR CAPABILITIES: {', '.join(capabilities)}

CAPABILITY-SPECIFIC GUIDELINES:"""
                
                capability_instructions = {
                    "web_search": "- Search for recent, authoritative sources\n- Verify information across multiple reliable sources\n- Include source URLs and publication dates",
                    "reasoning": "- Apply logical thinking and analysis\n- Explain your reasoning process clearly\n- Consider multiple perspectives and potential counterarguments",
                    "planning": "- Break down complex tasks into manageable steps\n- Prioritize actions based on importance and dependencies\n- Anticipate potential challenges and prepare contingencies",
                    "research": "- Conduct thorough, systematic information gathering\n- Cross-reference multiple sources for accuracy\n- Maintain objectivity and identify potential biases",
                    "data_analysis": "- Apply appropriate analytical methods\n- Identify patterns, trends, and anomalies\n- Present findings with appropriate visualizations when possible",
                    "synthesis": "- Integrate information from multiple sources\n- Identify connections and relationships between concepts\n- Create coherent, comprehensive summaries",
                    "file_ops": "- Handle files safely and efficiently\n- Maintain data integrity during processing\n- Use appropriate file formats for the task",
                    "code_execution": "- Write clean, efficient, and well-documented code\n- Test code thoroughly before execution\n- Handle errors gracefully and provide meaningful error messages"
                }
                
                for cap in capabilities:
                    if cap in capability_instructions:
                        capability_guidelines += f"\n{capability_instructions[cap]}"

            # Add quality standards
            quality_standards = f"""

QUALITY STANDARDS:
- Maximum response length: {max_tokens} tokens
- Provide accurate, relevant information
- Use clear, professional language
- Include confidence levels when making assessments
- Acknowledge limitations or uncertainties
- Follow the established communication protocol"""

            # Combine all parts
            full_prompt = f"""{system_context}

{base_prompt}{tool_instructions}{workflow_context}{capability_guidelines}{quality_standards}

Remember: Your success is measured by how well you contribute to solving the overall task while fulfilling your specific role in the multi-agent system."""

            agent_prompts[str(agent_id)] = {
                "agent_name": agent_name,
                "role": role,
                "specialization": specialization,
                "capabilities": capabilities,
                "tool": tool_name,
                "system_prompt": full_prompt.strip(),
                "max_tokens": max_tokens,
                "workflow_position": {
                    "receives_from": [agents[idx]["name"] for idx in prev_agents if idx < len(agents)],
                    "sends_to": [agents[idx]["name"] for idx in next_agents if idx < len(agents)] if next_agents and "final" not in str(next_agents) else ["final_output"]
                }
            }

        return agent_prompts
    
    def _get_next_agents(self, agent_id: int, workflow: Dict[str, Any]) -> List[int]:
        """Get the list of agents that receive output from the given agent."""
        next_agents = []
        message_flows = workflow.get("message_flow", [])
        
        for flow in message_flows:
            if flow.get("from") == agent_id:
                to_agents = flow.get("to", [])
                if isinstance(to_agents, list):
                    next_agents.extend([agent for agent in to_agents if isinstance(agent, int)])
                elif isinstance(to_agents, int):
                    next_agents.append(to_agents)
                elif to_agents == "final":
                    next_agents.append("final")
        
        return next_agents
    
    def _get_previous_agents(self, agent_id: int, workflow: Dict[str, Any]) -> List[int]:
        """Get the list of agents that send input to the given agent."""
        prev_agents = []
        message_flows = workflow.get("message_flow", [])
        
        for flow in message_flows:
            to_agents = flow.get("to", [])
            if isinstance(to_agents, list) and agent_id in to_agents:
                from_agent = flow.get("from")
                if isinstance(from_agent, int):
                    prev_agents.append(from_agent)
            elif to_agents == agent_id:
                from_agent = flow.get("from")
                if isinstance(from_agent, int):
                    prev_agents.append(from_agent)
        
        return prev_agents
    
    def _identify_fallback_agents(self, agents: List[Dict[str, Any]]) -> List[int]:
        """Identify agents that can serve as fallbacks."""
        fallbacks = []
        
        # Agents with general capabilities can serve as fallbacks
        for i, agent in enumerate(agents):
            capabilities = agent.get("capabilities", [])
            if any(cap in capabilities for cap in ["reasoning", "web_search", "planning"]):
                fallbacks.append(i)
        
        return fallbacks
        """Identify agents that can serve as fallbacks."""
        fallbacks = []
        
        # Agents with general capabilities can serve as fallbacks
        for i, agent in enumerate(agents):
            capabilities = agent.get("capabilities", [])
            if any(cap in capabilities for cap in ["reasoning", "web_search", "planning"]):
                fallbacks.append(i)
        
        return fallbacks
    
    def _determine_execution_strategy(self, analysis: Dict[str, Any], agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine the execution strategy based on analysis and agents."""
        complexity = analysis.get("complexity", "medium")
        task_type = analysis.get("task_type", "general_qa")
        num_agents = len(agents)
        
        strategy = {
            "approach": "adaptive",
            "max_iterations": 3 if complexity == "high" else 2,
            "error_handling": "graceful_degradation",
            "timeout_strategy": "progressive",
            "resource_allocation": "balanced"
        }
        
        if complexity == "high":
            strategy.update({
                "approach": "multi_stage",
                "validation_steps": True,
                "intermediate_checkpoints": True
            })
        elif task_type in ["research_task", "multi_step_analysis"]:
            strategy.update({
                "approach": "iterative_refinement",
                "feedback_loops": True,
                "quality_gates": True
            })
        
        return strategy
    
    def _generate_communication_rules(self, agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate communication rules."""
        return {
            "broadcast_types": ["doc_init", "system_status"],
            "direct_routing": {
                "task_result": [i + 1 for i in range(len(agents) - 1)],
                "final_answer": ["final"]
            },
            "timeout_seconds": 60
        }
    
    def _set_performance_targets(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Set performance targets based on analysis."""
        complexity_multiplier = {"low": 1, "medium": 2, "high": 3}
        multiplier = complexity_multiplier.get(analysis["complexity"], 2)
        
        return {
            "max_execution_time": 120000 * multiplier,  # milliseconds
            "target_accuracy": 0.8,
            "max_total_tokens": 3000 * multiplier
        }


class AdaptivePlanningStrategy(SimplePlanningStrategy):
    """Adaptive strategy with more sophisticated LLM-based analysis."""
    
    async def analyze_task(self, document: str) -> Dict[str, Any]:
        """Enhanced task analysis with more sophisticated LLM-based detection."""
        # Enhanced system prompt for more sophisticated analysis
        system_prompt = """You are an expert multi-agent system architect. Analyze the given task with deep understanding and provide a comprehensive analysis.

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

3. REQUIRED CAPABILITIES - Select all applicable:
   - web_search: Online information retrieval
   - file_operators: File I/O, data processing  
   - python_execute: Code execution, calculations
   - planning: Task decomposition, strategy formation
   - reasoning: Logical analysis, inference
   - research: Deep information gathering, source verification
   - data_analysis: Statistical analysis, pattern recognition
   - content_creation: Writing, summarization, report generation

4. DOMAIN EXPERTISE needed (technology, science, business, finance, healthcare, etc.)

5. PROCESSING REQUIREMENTS:
   - Sequential vs parallel processing needs
   - Validation/verification requirements
   - Error handling complexity

Respond with detailed JSON analysis including your reasoning."""

        user_prompt = f"""Analyze this task for optimal multi-agent system design:

TASK DOCUMENT:
{document}

Provide comprehensive analysis with specific reasoning for each decision. Consider edge cases and potential challenges."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Get enhanced LLM analysis with lower temperature for more consistent results
            response = await self.llm.ask(messages=messages, temperature=0.1)
            
            # Try to parse JSON response
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    llm_analysis = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError):
                print("Warning: Enhanced LLM response parsing failed, trying simple analysis")
                return await super().analyze_task(document)
            
            # Enhanced validation and structuring
            analysis = self._enhanced_validate_and_structure(llm_analysis, document)
            return analysis
            
        except Exception as e:
            print(f"Warning: Enhanced LLM analysis failed ({e}), falling back to simple analysis")
            return await super().analyze_task(document)
    
    def _enhanced_validate_and_structure(self, llm_analysis: Dict[str, Any], document: str) -> Dict[str, Any]:
        """Enhanced validation with additional sophistication."""
        # Get base validation
        analysis = self._validate_and_structure_analysis(llm_analysis, document)
        
        # Enhanced fields
        processing_requirements = llm_analysis.get("processing_requirements", {})
        domain_expertise = llm_analysis.get("domain_expertise", [])
        challenges = llm_analysis.get("challenges", [])
        validation_needs = llm_analysis.get("validation_needs", [])
        
        # Enhanced complexity assessment
        doc_lower = document.lower()
        question_complexity = len([w for w in ["what", "how", "why", "analyze", "compare", "evaluate"] if w in doc_lower])
        technical_indicators = len([w for w in ["algorithm", "formula", "calculate", "compute", "analyze", "model"] if w in doc_lower])
        
        if question_complexity > 3 or technical_indicators > 2 or len(document) > 1500:
            analysis["complexity"] = "high"
        
        # Add enhanced fields
        analysis.update({
            "processing_requirements": processing_requirements,
            "domain_expertise": domain_expertise if isinstance(domain_expertise, list) else [],
            "potential_challenges": challenges if isinstance(challenges, list) else [],
            "validation_needs": validation_needs if isinstance(validation_needs, list) else [],
            "question_complexity_score": question_complexity,
            "technical_complexity_score": technical_indicators,
            "enhanced_analysis": True
        })
        
        # Adjust capabilities based on enhanced analysis
        if technical_indicators > 1 and "python_execute" not in analysis["required_capabilities"]:
            analysis["required_capabilities"].append("python_execute")
        
        if question_complexity > 2 and "reasoning" not in analysis["required_capabilities"]:
            analysis["required_capabilities"].append("reasoning")
        
        if len(analysis["required_capabilities"]) > 3 and "planning" not in analysis["required_capabilities"]:
            analysis["required_capabilities"].append("planning")
        
        return analysis


class TaskPlanner:
    """Main planner class that coordinates task analysis and agent configuration generation."""
    
    def __init__(self, strategy_type: str = "adaptive"):
        self.tool_registry = ToolRegistry()
        self.strategies = {
            "simple": SimplePlanningStrategy(self.tool_registry),
            "adaptive": AdaptivePlanningStrategy(self.tool_registry)
        }
        self.strategy = self.strategies.get(strategy_type, self.strategies["adaptive"])
    
    async def analyze_and_plan(self, gaia_task_document: str) -> Dict[str, Any]:
        """Analyze Gaia task and generate optimal agent configuration."""
        print("🧠 Analyzing Gaia task...")
        task_analysis = await self.strategy.analyze_task(gaia_task_document)
        
        print("📋 Generating agent configuration...")
        agent_config = await self.strategy.generate_config(task_analysis)
        
        print("💾 Saving configuration...")
        await self._save_config(agent_config)
        
        return agent_config
    
    async def plan_agents(self, gaia_task_doc: str, strategy: str = "adaptive") -> str:
        """Plan agents and return configuration file path."""
        if strategy in self.strategies:
            self.strategy = self.strategies[strategy]
        
        config = await self.analyze_and_plan(gaia_task_doc)
        return "config/agent_config.json"
    
    async def _save_config(self, config: Dict[str, Any]):
        """Save configuration to JSON file."""
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        config_path = config_dir / "agent_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Configuration saved to {config_path}")


# Standalone testing
if __name__ == "__main__":
    import asyncio
    
    async def test_planner():
        """Test the planner with a sample Gaia task."""
        sample_task = """
        Question: What is the population of Tokyo in 2023? 
        Please search for the most recent data and provide a detailed answer with sources.
        
        Additional requirements:
        - Use web search to find current information
        - Verify the data from multiple sources
        - Provide reasoning for your answer
        """
        
        print("=== TaskPlanner Test ===")
        planner = TaskPlanner(strategy_type="adaptive")
        
        # Test analysis
        config = await planner.analyze_and_plan(sample_task)
        
        print("\n=== Generated Configuration ===")
        print(json.dumps(config, indent=2, ensure_ascii=False))
        
        print("\n=== Available Tools ===")
        tools = planner.tool_registry.get_available_tools()
        print(f"Available tools: {tools}")
        
        for tool_name in tools[:3]:  # Show first 3 tools
            info = planner.tool_registry.get_tool_info(tool_name)
            print(f"- {tool_name}: {info}")
    
    # Run the test
    asyncio.run(test_planner())
