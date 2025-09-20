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
from core.schema import Colors

# GAIA 根目录 (script/gaia)
GAIA_ROOT = Path(__file__).resolve().parent.parent

class TaskPlanner:
    """Simplified planner that directly analyzes tasks and creates agent configurations."""
    
    # Class-level port tracker to avoid conflicts across tasks
    _port_offset = 0
    _max_ports_per_task = 20  # Reserve 20 ports per task

    def __init__(self, config_path: Optional[str] = None, task_id: Optional[List[str]] = None, level: Optional[int] = 1, protocol_name: Optional[str] = None):
        # Ensure protocol_name is established before loading config so load_config
        # can pick a protocol-specific default file (e.g. config/meta_protocol.yaml).
        self.protocol_name = protocol_name or "dummy"
        self.tool_registry = ToolRegistry()
        self.llm = call_llm()
        self.prompt_builder = PromptBuilder()
        self.config = self.load_config(config_path)

        # Get level-based agent recommendation
        agents_config = self.config.get("agents", {})
        level_recommendations = agents_config.get("level_recommendations", {1: 2, 2: 4, 3: 8})
        self.recommended_agents = level_recommendations.get(level, 4)  # Default to 4 if level not found
        self.max_agents = agents_config.get("max_agent_num", 10)  # Hard limit

        self.task_id = task_id
        self.level = level
        # 协议名用于构造工作区路径，默认 general；允许外部覆盖
        self.protocol_name = protocol_name or self.config.get("protocol", "general")
        
        # Assign unique port range for this task
        self.port_base = 9000 + TaskPlanner._port_offset
        TaskPlanner._port_offset += TaskPlanner._max_ports_per_task
        print(f"🔌 Task {task_id} assigned port base: {self.port_base}")

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file with flexible relative resolution.

        Resolution strategy:
        - If config_path is provided and absolute, use it.
        - If config_path is provided and relative, try in order:
            1. GAIA_ROOT/config/<config_path>
            2. GAIA_ROOT/<config_path>
            3. runners_dir/<config_path>
        - If config_path is None, try protocol-specific candidates in order:
            1. GAIA_ROOT/config/{protocol_name}.yaml
            2. GAIA_ROOT/config/{protocol_name}_protocol.yaml
            3. GAIA_ROOT/config/meta_protocol.yaml
            4. GAIA_ROOT/config/general.yaml
        Raises FileNotFoundError with attempted paths if none found.
        """
        runners_dir = Path(__file__).parent
        gaia_root = runners_dir.parent

        candidates: List[Path] = []
        if config_path:
            p = Path(config_path)
            if p.is_absolute():
                candidates = [p]
            else:
                candidates = [
                    gaia_root / 'config' / p,
                    gaia_root / p,
                    runners_dir / p,
                ]
        else:
            # Try protocol-specific and common fallbacks
            candidates = [
                gaia_root / 'config' / f"{self.protocol_name}.yaml",
                gaia_root / 'config' / f"{self.protocol_name}_protocol.yaml",
                gaia_root / 'config' / 'meta_protocol.yaml',
                gaia_root / 'config' / 'general.yaml',
            ]

        # Remove duplicates while preserving order
        seen = set()
        uniq: List[Path] = []
        for c in candidates:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        candidates = uniq

        tried = []
        target: Optional[Path] = None
        for cand in candidates:
            tried.append(str(cand))
            if cand.exists():
                target = cand
                break

        if target is None:
            raise FileNotFoundError("Config file not found; tried:\n" + "\n".join(tried))

        try:
            with open(target, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise FileNotFoundError(f"Config file not found: {target} ({e})")

    async def analyze_and_plan(self, gaia_task_document: str, workspace_dir: Optional[Path] = None) -> tuple[Dict[str, Any], str]:
        """
        Analyze Gaia task and generate optimal agent configuration.
        
        Args:
            gaia_task_document: Task description
            workspace_dir: Optional pre-created workspace directory
            
        Returns:
            tuple: (config dict, config file path)
        """
        # Plan reuse control via general.yaml: planner.reuse_plan (bool, default: True)
        task_id = self.task_id or time.strftime("%Y-%m-%d-%H-%M")
        # Ensure task_id is a simple string for path construction
        task_id_str = str(task_id)       
        existing_path = GAIA_ROOT / "workspaces" / "agent_config" / f'{task_id_str}.json'
        reuse_plan = True

        try:
            planner_cfg = (self.config.get("planner") if isinstance(self.config, dict) else None) or {}
            reuse_plan = bool(planner_cfg.get("reuse_plan", True))
        except Exception:
            reuse_plan = True

        # Early exit only when reuse_plan enabled and file exists
        if reuse_plan and existing_path.exists():
            try:
                print(f"{Colors.GREEN}\nPlanner: Reuse enabled. Using existing plan at {existing_path}.{Colors.RESET}")
                with open(existing_path, 'r', encoding='utf-8') as f:
                    existing_cfg = json.load(f)
            except Exception:
                existing_cfg = {}
            return existing_cfg, str(existing_path)
        elif existing_path.exists() and not reuse_plan:
            print(f"{Colors.YELLOW}\nPlanner: Reuse disabled. Ignoring existing plan at {existing_path} and regenerating.{Colors.RESET}")

        print("🧠 Analyzing Gaia task...")
        
        # Get available tools
        available_tools = self.tool_registry.get_available_tools()
        
        # Build analysis messages with available tools
        messages = self.prompt_builder.build_task_analysis_messages(
            gaia_task_document, 
            recommended_agents=self.recommended_agents,
            max_agents=self.max_agents,
            level=self.level,
            available_tools=available_tools
        )
        
        # Analyze task using LLM
        task_analysis = await self._analyze_task_with_llm(gaia_task_document, messages)
        
        print("📋 Generating agent configuration...")
        agent_config = await self._generate_config(task_analysis, gaia_task_document)
        
        print("💾 Saving configuration...")
        config_path = await self._save_config(agent_config, agent_config.get("task_id"), workspace_dir)
        
        return agent_config, str(config_path)  # 将 PosixPath 转换为字符串 
    
    async def _analyze_task_with_llm(self, document: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
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
        available_tools = self.tool_registry.get_available_tools()
        
        # Validate and fix fields
        task_type = analysis.get("task_type", "general_qa")
        if task_type not in valid_task_types:
            task_type = "general_qa"
        
        level_to_complexity = {1: "low", 2: "medium", 3: "high"}
        complexity = level_to_complexity.get(self.level, "medium")
        
        # Validate required_tools
        required_tools = analysis.get("required_tools", [])
        if not isinstance(required_tools, list):
            required_tools = []
        required_tools = [tool for tool in required_tools if tool in available_tools]
        
        # Validate agents configuration - no role validation, let LLM decide
        agents_config = analysis.get("agents", [])
        if not isinstance(agents_config, list):
            agents_config = []
        
        validated_agents = []
        for agent in agents_config:
            if isinstance(agent, dict):
                tool = agent.get("tool", "")
                if tool in available_tools:
                    validated_agent = {
                        "tool": tool,
                        "name": agent.get("name", f"Agent_{tool}"),
                        "role": agent.get("role", "processor")  # Generic fallback role
                    }
                    validated_agents.append(validated_agent)
        
        # Ensure minimum configuration
        if not validated_agents:
            validated_agents = [{
                "tool": "create_chat_completion",
                "name": "ReasoningSynthesizer",
                "role": "synthesizer"
            }]
        
        # Update required_tools based on validated agents
        required_tools = list(set([agent["tool"] for agent in validated_agents]))
        
        estimated_steps = analysis.get("estimated_steps", len(required_tools))
        if not isinstance(estimated_steps, int) or estimated_steps < 1:
            estimated_steps = len(required_tools)
        
        domain_areas = analysis.get("domain_areas", ["general_knowledge"])
        if not isinstance(domain_areas, list):
            domain_areas = ["general_knowledge"]
        
        return {
            "task_type": task_type,
            "complexity": complexity,
            "level": self.level,  # 添加level信息到分析结果
            "required_tools": required_tools,
            "agents": validated_agents,
            "estimated_steps": estimated_steps,
            "domain_areas": domain_areas,
            "document_length": len(document)
        }
    
    def _fallback_analysis(self, document: str) -> Dict[str, Any]:
        """Fallback analysis when LLM fails."""
        doc_lower = document.lower()
        available_tools = self.tool_registry.get_available_tools()
        
        # Simple keyword detection for tool selection
        agents_config = []
        if any(keyword in doc_lower for keyword in ["search", "web", "internet", "find"]) and "web_search" in available_tools:
            agents_config.append({
                "tool": "web_search",
                "name": "WebResearcher",
                "role": "information_gatherer"
            })
        if any(keyword in doc_lower for keyword in ["calculate", "compute", "code", "python"]) and "python_execute" in available_tools:
            agents_config.append({
                "tool": "python_execute", 
                "name": "CodeExecutor",
                "role": "computational_specialist"
            })
        if any(keyword in doc_lower for keyword in ["file", "data", "csv", "json"]) and "file_operators" in available_tools:
            agents_config.append({
                "tool": "file_operators",
                "name": "DataProcessor", 
                "role": "data_handler"
            })
        
        # Always include a reasoning agent as final step
        if "create_chat_completion" in available_tools:
            agents_config.append({
                "tool": "create_chat_completion",
                "name": "ReasoningSynthesizer",
                "role": "final_synthesizer"
            })
        
        required_tools = [agent["tool"] for agent in agents_config]
        
        # 根据level参数映射complexity，而不是文档长度
        level_to_complexity = {1: "low", 2: "medium", 3: "high"}
        complexity = level_to_complexity.get(self.level, "medium")
        
        return {
            "task_type": "general_qa",
            "complexity": complexity,
            "level": self.level,  # 添加level信息
            "required_tools": required_tools,
            "agents": agents_config,
            "estimated_steps": len(required_tools),
            "domain_areas": ["general_knowledge"],
            "document_length": len(document)
        }
    
    async def _generate_config(self, analysis: Dict[str, Any], original_document: str) -> Dict[str, Any]:
        """Generate agent configuration based on analysis."""
        agents = []
        agent_id = 0
        port = self.port_base  # Use task-specific port base
        task_id = self.task_id or time.strftime("%Y-%m-%d-%H-%M")
        
        agents_config = analysis["agents"]
        complexity = analysis["complexity"]
        
        # Get token allocation from config instead of hardcoding
        agents_config_section = self.config.get("agents", {})
        default_max_tokens = agents_config_section.get("default_max_tokens", 500)
        
        # Use config-based complexity thresholds for token allocation
        complexity_thresholds = agents_config_section.get("complexity_thresholds", {
            "low": 2000,
            "medium": 4000,
            "high": 8000
        })
        max_tokens = complexity_thresholds.get(complexity, complexity_thresholds["medium"])
        
        # Create agents based on LLM analysis
        for agent_config in agents_config:
            tool_name = agent_config["tool"]
            
            # Validate tool availability
            if not self.tool_registry.validate_tool_name(tool_name):
                print(f"Warning: Tool {tool_name} not available, skipping agent")
                continue
            
            # Use uniform max_tokens for all agents
            agent_max_tokens = min(max_tokens, default_max_tokens)
            
            agents.append({
                "id": agent_id,
                "name": agent_config["name"],
                "tool": tool_name,
                "port": port,
                "priority": 2,
                "max_tokens": agent_max_tokens,
                "role": agent_config["role"]
            })
            
            agent_id += 1
            port += 1
        
        # Limit agents if needed
        if len(agents) > self.max_agents:
            agents = await self._reduce_agents_to_limit(agents, self.max_agents)
        
        # Generate workflow
        workflow = self._generate_workflow(agents)
        
        # Generate agent prompts
        agent_prompts = await self._generate_agent_prompts(agents, analysis, workflow, original_document)
        
        return {
            "task_id": task_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "task_analysis": analysis,
            "agents": agents,
            "workflow": workflow,
            "agent_prompts": agent_prompts,
            "communication_rules": self._generate_communication_rules(agents),
            "performance_targets": self._set_performance_targets(analysis)
        }
    
    async def _reduce_agents_to_limit(self, agents: List[Dict[str, Any]], max_agents: int) -> List[Dict[str, Any]]:
        """Reduce agents to fit limit by asking LLM to replan."""
        if len(agents) <= max_agents:
            return agents
        
        print(f"Reducing {len(agents)} agents to {max_agents} limit using LLM replanning")
        
        # Create agents summary for LLM
        agents_summary = []
        for agent in agents:
            agents_summary.append({
                "name": agent.get("name"),
                "tool": agent.get("tool"),
                "role": agent.get("role"),
            })
        
        # Ask LLM to select the most important agents
        reduction_prompt = f"""
You have {len(agents)} agents but need to reduce to {max_agents} agents maximum.

Current agents:
{json.dumps(agents_summary, indent=2, ensure_ascii=False)}

Select the {max_agents} most essential agents for completing the task effectively. 
Consider:
1. Tool diversity and coverage
2. Workflow efficiency 
3. Essential capabilities for the task

Respond with JSON containing only the selected agents in the same format:
{{"selected_agents": [...]}}
"""

        try:
            messages = [{"role": "user", "content": reduction_prompt}]
            response = await self.llm.ask(messages=messages, temperature=0.1)
            
            # Parse LLM response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                selection_result = json.loads(json_str)
                selected_agent_specs = selection_result.get("selected_agents", [])
                
                # Match selected specs back to original agents
                selected_agents = []
                for spec in selected_agent_specs:
                    for agent in agents:
                        if (agent.get("name") == spec.get("name") or 
                            agent.get("tool") == spec.get("tool")):
                            selected_agents.append(agent)
                            break
                
                # If LLM selection worked, use it
                if len(selected_agents) <= max_agents:
                    # Reassign IDs and ports using task-specific port base
                    for i, agent in enumerate(selected_agents):
                        agent["id"] = i
                        agent["port"] = self.port_base + i
                    
                    print(f"✅ LLM selected {len(selected_agents)} agents")
                    return selected_agents
        
        except Exception as e:
            print(f"⚠️ LLM reduction failed ({e}), using simple fallback")
        
        # Fallback: simple truncation keeping first and last agents
        if max_agents >= 2:
            selected_agents = [agents[0]] + agents[-(max_agents-1):]
        else:
            selected_agents = agents[:max_agents]
        
        # Reassign IDs and ports using task-specific port base
        for i, agent in enumerate(selected_agents):
            agent["id"] = i
            agent["port"] = self.port_base + i
        
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
                workflow_context = """You are the final agent responsible for generating the complete answer.

CRITICAL: Your response must contain the EXACT FINAL ANSWER to the original question. 
- Do NOT just describe the process or summarize what needs to be done
- Do NOT just explain the methodology 
- You MUST provide the specific answer that directly answers the question
- If the question asks for a word, number, or specific piece of information, provide EXACTLY that
- Format your response as: "FINAL ANSWER: [your exact answer here]"

Example:
- If asked "What is the capital of France?", respond: "FINAL ANSWER: Paris"
- If asked "How many...?", respond: "FINAL ANSWER: [number]"
- If asked "Which word...?", respond: "FINAL ANSWER: [the specific word]"

Remember: The success of the entire workflow depends on you providing the precise, actionable final answer."""
            else:
                workflow_context = f"You will receive input from previous agents and pass results to the next agent."
            
            # Create system prompt using prompt builder
            system_prompt = self.prompt_builder.build_agent_prompt(
                agent_config=agent,
                analysis=analysis,
                workflow=workflow,
                tool_description=tool_info,
                workflow_info=workflow_context,
                original_query=original_document  # Pass original document as query
            )
            
            agent_prompts[str(agent_id)] = {
                "agent_name": agent["name"],
                "role": agent.get("role", "worker"),
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
        """Set performance targets based on configuration."""
        # Get performance config from general.yaml
        performance_config = self.config.get("performance", {})
        
        complexity = analysis.get("complexity", "medium")
        
        # Use config-based values with fallbacks
        base_execution_time = performance_config.get("max_execution_time", 120000)
        base_token_limit = performance_config.get("max_total_tokens", 3000)
        target_accuracy = performance_config.get("target_accuracy", 0.8)
        
        # Apply complexity multiplier
        complexity_multiplier = {"low": 1, "medium": 2, "high": 3}
        multiplier = complexity_multiplier.get(complexity, 2)
        
        return {
            "max_execution_time": base_execution_time * multiplier,
            "target_accuracy": target_accuracy,
            "max_total_tokens": base_token_limit * multiplier
        }
    
    async def plan_agents(self, gaia_task_doc: str, workspace_dir: Optional[Path] = None) -> str:
        """
        Plan agents and return configuration file path.
        
        Args:
            gaia_task_doc: Task description
            workspace_dir: Optional pre-created workspace directory
            
        Returns:
            str: Path to agent configuration file
        """
        _, config_path = await self.analyze_and_plan(gaia_task_doc, workspace_dir=workspace_dir)
        return config_path
    
    async def _save_config(self, config: Dict[str, Any], task_id: Optional[str] = None, workspace_dir: Optional[Path] = None):
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration data
            task_id: Task ID
            workspace_dir: Pre-created workspace directory (if provided)
            
        Returns:
            Path: Configuration file path
        """
        if task_id is None:
            task_id = time.strftime("%Y-%m-%d-%H-%M")
        
        if workspace_dir is not None:
            # 使用提供的工作区目录
            task_workspace_dir = workspace_dir
        else:
            # 使用默认路径创建工作区
            protocol_name = self.protocol_name or "general"
            workspace_dir = GAIA_ROOT / "workspaces" / protocol_name
            task_workspace_dir = workspace_dir / task_id
            task_workspace_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = task_workspace_dir / "agent_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ Configuration saved to {config_path}")
        return config_path


# Standalone testing
if __name__ == "__main__":
    import asyncio
    
    async def test_planner():
        """Test the simplified planner."""
        input_file = '/root/Multiagent-Protocol/script/gaia/GAIABench/2023/validation/metadata.jsonl'
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        task_dict = data[0]
        prompt = task_dict.get("question", "")
        task_id = task_dict.get("task_id", None)
        level = task_dict.get("level", 1)
        print("=== TaskPlanner Test ===")
        planner = TaskPlanner(task_id=task_id, level=level)
        
        # Test analysis and planning
        config, config_path = await planner.analyze_and_plan(prompt)

        print("\n=== Generated Configuration ===")
        print(json.dumps(config, indent=2, ensure_ascii=False))
        
        print(f"\n=== Configuration saved to: {config_path} ===")
        
        print("\n=== Available Tools ===")
        tools = planner.tool_registry.get_available_tools()
        print(f"Available tools: {tools}")
    
    # Run the test
    asyncio.run(test_planner())
