"""
ProtoRouter Benchmark - LLM-based protocol selection evaluation system.
"""

import json
import sys
import os
from typing import Dict, List, Any, Optional

# Add src to path for importing utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from utils.core import Core
from prompt_template import format_scenario_prompt, PROTOCOL_SELECTION_FUNCTION


class ProtoRouterBenchmark:
    """
    LLM-based protocol selector that uses the Core utility to call LLM
    and evaluate against ground truth data.
    """
    
    def __init__(self, config):
        self.core = Core(config)
        self.results = []
    
    def process_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Process a scenario using LLM and return the protocol selections"""
        scenario_id = scenario.get("id", "unknown")
        
        # Format the prompt
        prompt = format_scenario_prompt(scenario)
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": "You are ProtoRouter, an expert protocol selector for multi-agent systems."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Call LLM with function calling
            response = self.core.function_call_execute(
                messages=messages,
                functions=[PROTOCOL_SELECTION_FUNCTION]
            )
            
            # Parse the function call response
            llm_selection = self._parse_llm_response(response)
            
            # Debug: check if we got selections for all modules
            if llm_selection and "module_selections" in llm_selection:
                module_count = len(llm_selection["module_selections"])
                expected_count = scenario.get("modules_count", 1)
                if module_count != expected_count:
                    print(f"[WARNING] Expected {expected_count} modules, got {module_count} selections")
                    print(f"[DEBUG] LLM response: {llm_selection}")
            else:
                print(f"[ERROR] No module_selections found in LLM response: {llm_selection}")
            
            return {
                "scenario_id": scenario_id,
                "llm_response": llm_selection,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            print(f"Error processing scenario {scenario_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "scenario_id": scenario_id,
                "llm_response": None,
                "success": False,
                "error": str(e)
            }
    
    def _parse_llm_response(self, response) -> Dict[str, Any]:
        """Parse the LLM function call response"""
        if not response or not response.choices:
            raise ValueError("Empty response from LLM")
        
        message = response.choices[0].message
        
        # Handle both function_call (old format) and tool_calls (new format)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_call = message.tool_calls[0]
            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Raw arguments: {tool_call.function.arguments}")
                raise ValueError(f"Failed to parse function arguments: {tool_call.function.arguments}")
        elif hasattr(message, 'function_call') and message.function_call:
            try:
                function_args = json.loads(message.function_call.arguments)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Raw arguments: {message.function_call.arguments}")
                raise ValueError(f"Failed to parse function arguments: {message.function_call.arguments}")
        else:
            # Check if there's regular content
            if hasattr(message, 'content') and message.content:
                print(f"LLM returned content instead of function call: {message.content}")
                raise ValueError(f"No function call found, got content: {message.content}")
            else:
                raise ValueError("No function call found in LLM response")
        
        return function_args
