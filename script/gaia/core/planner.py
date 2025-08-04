"""Intelligent planner for analyzing Gaia tasks and generating agent configurations."""
import json
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.registry import ToolRegistry
from core.llm import call_llm
from core.prompt import PromptBuilder


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return {}


class TaskPlanner:
    """Simplified planner that directly analyzes tasks and creates agent configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.tool_registry = ToolRegistry()
        self.llm = call_llm()
        self.prompt_builder = PromptBuilder()
        self.config = load_config(config_path)
        self.max_agents = self.config.get("agents", {}).get("max_agent_num", 5)
        
        # Predefined agent templates with roles and tools
        self.agent_templates = {
            "web_search": {
                "name": "WebResearcher",
                "role": "researcher", 
                "specialization": "information_retrieval"
            },
            "python_execute": {
                "name": "CodeExecutor",
                "role": "specialist",
                "specialization": "computation"
            },
            "file_operators": {
                "name": "DataProcessor",
                "role": "worker",
                "specialization": "data_management"
            },
            "reasoning": {
                "name": "ReasoningSynthesizer", 
                "role": "synthesizer",
                "specialization": "reasoning_synthesis"
            }
        }
    
    async def analyze_and_plan(self, gaia_task_document: str) -> Dict[str, Any]:
        """Analyze Gaia task and generate optimal agent configuration."""
        print("🧠 Analyzing Gaia task...")
        
        # Create system message for task analysis
        # system_message_content = (
        #     "You are a task analysis assistant. Analyze the given task and determine "
        #     "what capabilities and tools are needed. Focus on identifying the key "
        #     "requirements rather than detailed sub-steps. Optimize for clarity and efficiency."
        # )
        
        # Get available tools description
        available_tools = self.tool_registry.get_available_tools()
        planning_message_content = self.prompt_builder.build_task_analysis_messages(
            gaia_task_document, max_agents=self.max_agents
        )
        planning_message_content['user'] += ('The available tools are: ' + ', '.join(available_tools) + '.\n')

        # agents_description = []
        # for tool_name, template in self.agent_templates.items():
        #     if tool_name in tool_mapping or tool_name == "reasoning":
        #         agents_description.append({
        #             "tool": tool_name,
        #             "name": template["name"],
        #             "role": template["role"],
        #             "description": template['specialization']
        #         })
        
        # system_message_content += (
        #     f"\nAvailable agent types: {json.dumps(agents_description, indent=2)}\n"
        #     "Analyze the task and respond with a JSON object containing:\n"
        #     "{\n"
        #     "  \"task_type\": \"general_qa|research_task|computational_task|multi_step_analysis\",\n"
        #     "  \"complexity\": \"low|medium|high\",\n"
        #     "  \"required_tools\": [\"web_search\", \"python_execute\", \"file_operators\", \"reasoning\"],\n"
        #     "  \"estimated_steps\": number,\n"
        #     "  \"domain_areas\": [\"domain1\", \"domain2\"]\n"
        #     "}"
        # )
        
        # Analyze task using LLM
        task_analysis = await self._analyze_task_with_llm(gaia_task_document, planning_message_content)
        
        print("📋 Generating agent configuration...")
        agent_config = await self._generate_config(task_analysis, gaia_task_document)
        
        print("💾 Saving configuration...")
        config_path = await self._save_config(agent_config, agent_config.get("task_id"))
        
        return agent_config, config_path 
    
    async def _analyze_task_with_llm(self, document: str, messages: str) -> Dict[str, Any]:
        """Use LLM to analyze the task and determine requirements."""
        try:
            
            # Get LLM analysis
            response = await self.llm.ask(messages=messages, temperature=0.2)
            
            # Parse JSON response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                analysis = json.loads(json_str)
                return self._validate_analysis(analysis, document)
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"Warning: LLM analysis failed ({e}), using fallback analysis")
            return self._fallback_analysis(document)
    
    def _validate_analysis(self, analysis: Dict[str, Any], document: str) -> Dict[str, Any]:
        """Validate and clean up LLM analysis."""
        valid_task_types = ["general_qa", "research_task", "computational_task", "multi_step_analysis"]
        valid_complexities = ["low", "medium", "high"]
        valid_tools = ["web_search", "python_execute", "file_operators", "reasoning"]
        
        # Validate and fix fields
        task_type = analysis.get("task_type", "general_qa")
        if task_type not in valid_task_types:
            task_type = "general_qa"
        
        complexity = analysis.get("complexity", "medium")
        if complexity not in valid_complexities:
            complexity = "medium"
        
        required_tools = analysis.get("required_tools", [])
        if not isinstance(required_tools, list):
            required_tools = []
        required_tools = [tool for tool in required_tools if tool in valid_tools]
        
        # Ensure minimum tools
        if not required_tools:
            required_tools = ["reasoning"]
        if "reasoning" not in required_tools:
            required_tools.append("reasoning")
        
        estimated_steps = analysis.get("estimated_steps", len(required_tools))
        if not isinstance(estimated_steps, int) or estimated_steps < 1:
            estimated_steps = len(required_tools)
        
        domain_areas = analysis.get("domain_areas", ["general_knowledge"])
        if not isinstance(domain_areas, list):
            domain_areas = ["general_knowledge"]
        
        return {
            "task_type": task_type,
            "complexity": complexity,
            "required_tools": required_tools,
            "estimated_steps": estimated_steps,
            "domain_areas": domain_areas,
            "document_length": len(document)
        }
    
    def _fallback_analysis(self, document: str) -> Dict[str, Any]:
        """Fallback analysis when LLM fails."""
        doc_lower = document.lower()
        
        # Simple keyword detection
        required_tools = []
        if any(keyword in doc_lower for keyword in ["search", "web", "internet", "find"]):
            required_tools.append("web_search")
        if any(keyword in doc_lower for keyword in ["calculate", "compute", "code", "python"]):
            required_tools.append("python_execute") 
        if any(keyword in doc_lower for keyword in ["file", "data", "csv", "json"]):
            required_tools.append("file_operators")
        
        # Always include reasoning
        if "reasoning" not in required_tools:
            required_tools.append("reasoning")
        
        complexity = "low"
        if len(document) > 1000:
            complexity = "high"
        elif len(document) > 500:
            complexity = "medium"
        
        return {
            "task_type": "general_qa",
            "complexity": complexity,
            "required_tools": required_tools,
            "estimated_steps": len(required_tools),
            "domain_areas": ["general_knowledge"],
            "document_length": len(document)
        }
    
    async def _generate_config(self, analysis: Dict[str, Any], original_document: str) -> Dict[str, Any]:
        """Generate agent configuration based on analysis."""
        agents = []
        agent_id = 0
        port = 9000
        
        required_tools = analysis["required_tools"]
        complexity = analysis["complexity"]
        
        # Token allocation based on complexity
        token_allocation = {
            "low": {"default": 400, "synthesizer": 600},
            "medium": {"default": 500, "synthesizer": 800}, 
            "high": {"default": 600, "synthesizer": 1200}
        }
        tokens = token_allocation.get(complexity, token_allocation["medium"])
        
        # Get tool mapping
        tool_mapping = self.tool_registry.get_tool_mapping()
        
        # Create agents based on required tools
        for tool_name in required_tools:
            if tool_name in self.agent_templates:
                template = self.agent_templates[tool_name]
                
                # Map tool name to actual tool
                actual_tool = tool_mapping.get(tool_name, "create_chat_completion")
                if not self.tool_registry.validate_tool_name(actual_tool):
                    actual_tool = "create_chat_completion"
                
                # Determine tokens
                max_tokens = tokens["synthesizer"] if template["role"] == "synthesizer" else tokens["default"]
                
                agents.append({
                    "id": agent_id,
                    "name": template["name"],
                    "tool": actual_tool,
                    "port": port,
                    "priority": 2,
                    "max_tokens": max_tokens,
                    "specialization": template["specialization"],
                    "role": template["role"]
                })
                
                agent_id += 1
                port += 1
        
        # Limit agents if needed
        if len(agents) > self.max_agents:
            agents = self._reduce_agents_to_limit(agents, self.max_agents)
        
        # Generate workflow
        workflow = self._generate_workflow(agents)
        
        # Generate agent prompts
        agent_prompts = await self._generate_agent_prompts(agents, analysis, workflow, original_document)
        
        return {
            "task_id": time.strftime("%Y-%m-%d-%H-%M"),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "task_analysis": analysis,
            "agents": agents,
            "workflow": workflow,
            "agent_prompts": agent_prompts,
            "communication_rules": self._generate_communication_rules(agents),
            "performance_targets": self._set_performance_targets(analysis)
        }
    
    def _reduce_agents_to_limit(self, agents: List[Dict[str, Any]], max_agents: int) -> List[Dict[str, Any]]:
        """Reduce agents to fit limit, prioritizing essential roles."""
        if len(agents) <= max_agents:
            return agents
        
        print(f"Reducing {len(agents)} agents to {max_agents} limit")
        
        # Priority: synthesizer > specialist > worker > researcher
        priority_order = {"synthesizer": 1, "specialist": 2, "worker": 3, "researcher": 4}
        
        # Sort by priority
        agents_with_priority = []
        for agent in agents:
            role = agent.get("role", "worker")
            priority = priority_order.get(role, 5)
            agents_with_priority.append((priority, agent))
        
        agents_with_priority.sort(key=lambda x: x[0])
        selected_agents = [agent for _, agent in agents_with_priority[:max_agents]]
        
        # Reassign IDs and ports
        for i, agent in enumerate(selected_agents):
            agent["id"] = i
            agent["port"] = 9000 + i
        
        return selected_agents
    
    def _generate_workflow(self, agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate simple sequential workflow."""
        if len(agents) <= 1:
            return {
                "start_agent": 0,
                "message_flow": [],
                "execution_pattern": "single_agent"
            }
        
        # Simple sequential flow
        message_flow = []
        for i in range(len(agents) - 1):
            message_flow.append({
                "from": i,
                "to": [i + 1],
                "message_type": "task_result"
            })
        
        # Final output
        message_flow.append({
            "from": len(agents) - 1,
            "to": "final",
            "message_type": "final_answer"
        })
        
        return {
            "start_agent": 0,
            "message_flow": message_flow,
            "execution_pattern": "sequential"
        }
    
    async def _generate_agent_prompts(self, agents: List[Dict[str, Any]], analysis: Dict[str, Any], 
                                    workflow: Dict[str, Any], original_document: str) -> Dict[str, Any]:
        """Generate prompts for each agent."""
        agent_prompts = {}
        
        for agent in agents:
            agent_id = agent["id"]
            tool_name = agent.get("tool", "")
            tool_info = self.tool_registry.get_tool_info(tool_name)
            
            # Build workflow context
            is_first = agent_id == 0
            is_last = agent_id == len(agents) - 1
            
            workflow_context = ""
            if is_first:
                workflow_context = "You are the first agent in the workflow."
            elif is_last:
                workflow_context = "You are the final agent responsible for generating the complete answer."
            else:
                workflow_context = f"You will receive input from previous agents and pass results to the next agent."
            
            # Create system prompt using prompt builder
            system_prompt = self.prompt_builder.build_agent_prompt(
                agent_config=agent,
                analysis=analysis,
                workflow=workflow,
                tool_description=tool_info,
                workflow_info=workflow_context
            )
            
            agent_prompts[str(agent_id)] = {
                "agent_name": agent["name"],
                "role": agent.get("role", "worker"),
                "specialization": agent.get("specialization", "general"),
                "tool": tool_name,
                "system_prompt": system_prompt,
                "max_tokens": agent.get("max_tokens", 500)
            }
        
        return agent_prompts
    
    def _generate_communication_rules(self, agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate simple communication rules."""
        return {
            "timeout_seconds": 60,
            "max_retries": 2,
            "routing": "sequential"
        }
    
    def _set_performance_targets(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Set performance targets."""
        complexity_multiplier = {"low": 1, "medium": 2, "high": 3}
        multiplier = complexity_multiplier.get(analysis["complexity"], 2)
        
        return {
            "max_execution_time": 120000 * multiplier,
            "target_accuracy": 0.8,
            "max_total_tokens": 3000 * multiplier
        }
    
    async def plan_agents(self, gaia_task_doc: str) -> str:
        """Plan agents and return configuration file path."""
        _, config_path = await self.analyze_and_plan(gaia_task_doc)
        return config_path
    
    async def _save_config(self, config: Dict[str, Any], task_id: Optional[str] = None):
        """Save configuration to JSON file in workspace with task-specific directory."""
        # Use provided task_id or generate one with current timestamp
        if task_id is None:
            task_id = time.strftime("%Y-%m-%d-%H-%M")
        
        # Create workspace directory structure: workspaces/task_id/
        workspace_dir = Path("workspaces")
        task_workspace_dir = workspace_dir / task_id
        task_workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Save agent config in the task workspace
        config_path = task_workspace_dir / "agent_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Configuration saved to {config_path}")
        print(f"📁 Task workspace created at {task_workspace_dir}")

        return config_path


# Standalone testing
if __name__ == "__main__":
    import asyncio
    
    async def test_planner():
        """Test the simplified planner."""
        sample_task = """
        Question: What is the population of Tokyo in 2023? 
        Please search for the most recent data and provide a detailed answer with sources.
        
        Additional requirements:
        - Use web search to find current information
        - Verify the data from multiple sources
        - Provide reasoning for your answer
        """
        
        print("=== TaskPlanner Test ===")
        planner = TaskPlanner()
        
        # Test analysis and planning
        config = await planner.analyze_and_plan(sample_task)
        
        print("\n=== Generated Configuration ===")
        print(json.dumps(config, indent=2, ensure_ascii=False))
        
        print("\n=== Available Tools ===")
        tools = planner.tool_registry.get_available_tools()
        print(f"Available tools: {tools}")
    
    # Run the test
    asyncio.run(test_planner())
