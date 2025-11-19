"""
LLM API compatibility patches for Anthropic and Google APIs.

This module provides wrapper classes that make Anthropic Claude and Google Gemini APIs
compatible with the OpenAI-style interface used in the GAIA framework.

Usage:
    from llm_patches import get_llm_instance
    
    llm = get_llm_instance(config_path="path/to/config.yaml")
    response = await llm.ask(messages=[{"role": "user", "content": "Hello"}])
"""
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic package not installed. Claude models will not be available.")

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Warning: google-generativeai package not installed. Gemini models will not be available.")

# Import the base LLM class
import sys
from pathlib import Path
GAIA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GAIA_ROOT))

from core.llm import LLM, LLMConfig


class AnthropicLLMAdapter(LLM):
    """Adapter for Anthropic Claude API to OpenAI-style interface."""
    
    def __init__(self, config_name: str = "default", config_path: Optional[str] = None):
        """Initialize Anthropic LLM adapter."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required for Claude models. Install with: pip install anthropic")
        
        # Initialize base config
        super().__init__(config_name, config_path)
        
        # Get API key from environment or config
        api_key = os.getenv("ANTHROPIC_API_KEY") or self.config.api_key
        if not api_key or api_key == "":
            raise ValueError("ANTHROPIC_API_KEY environment variable or config api_key must be set")
        
        # Initialize Anthropic client
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        
        print(f"🔧 Initialized Anthropic Claude adapter: {self.model}")
    
    def format_messages_for_anthropic(self, messages: List[dict]) -> tuple[Optional[str], List[dict]]:
        """
        Convert OpenAI-style messages to Anthropic format.
        
        Returns:
            Tuple of (system_message, user_messages)
        """
        system_message = None
        formatted_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                # Anthropic uses a separate system parameter
                if system_message is None:
                    system_message = content
                else:
                    system_message += "\n\n" + content
            elif role in ["user", "assistant"]:
                formatted_messages.append({
                    "role": role,
                    "content": content
                })
            # Skip tool messages or convert them to user messages
            elif role == "tool":
                formatted_messages.append({
                    "role": "user",
                    "content": f"[Tool result] {content}"
                })
        
        return system_message, formatted_messages
    
    async def ask(
        self,
        messages: List[Union[dict, Any]],
        system_msgs: Optional[List[Union[dict, Any]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """Ask Claude using Anthropic API."""
        try:
            # Format messages
            formatted_messages = self.format_messages(messages)
            if system_msgs:
                system_formatted = self.format_messages(system_msgs)
                formatted_messages = system_formatted + formatted_messages
            
            # Convert to Anthropic format
            system_message, anthropic_messages = self.format_messages_for_anthropic(formatted_messages)
            
            # Calculate input tokens (approximate)
            input_tokens = self.count_message_tokens(formatted_messages)
            
            # Prepare API call parameters
            params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": self.max_tokens,
                "temperature": temperature if temperature is not None else self.temperature,
            }
            
            if system_message:
                params["system"] = system_message
            
            # Make API call
            if stream:
                # Streaming response
                collected_messages = []
                async with self.client.messages.stream(**params) as stream_response:
                    async for text in stream_response.text_stream:
                        collected_messages.append(text)
                        print(text, end="", flush=True)
                print()  # Newline after streaming
                response_text = "".join(collected_messages)
                
                # Estimate completion tokens
                completion_tokens = self.count_tokens(response_text)
            else:
                # Non-streaming response
                response = await self.client.messages.create(**params)
                response_text = response.content[0].text
                
                # Get actual token counts
                input_tokens = response.usage.input_tokens
                completion_tokens = response.usage.output_tokens
            
            # Update token counts
            self.update_token_count(input_tokens, completion_tokens)
            
            return response_text
            
        except Exception as e:
            print(f"Error in Anthropic ask: {str(e)}")
            raise
    
    async def ask_tool(
        self,
        messages: List[Union[dict, Any]],
        system_msgs: Optional[List[Union[dict, Any]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Optional[dict]:
        """Ask Claude with tool support."""
        try:
            # Format messages
            formatted_messages = self.format_messages(messages)
            if system_msgs:
                system_formatted = self.format_messages(system_msgs)
                formatted_messages = system_formatted + formatted_messages
            
            # Convert to Anthropic format
            system_message, anthropic_messages = self.format_messages_for_anthropic(formatted_messages)
            
            # Calculate input tokens
            input_tokens = self.count_message_tokens(formatted_messages)
            
            # Prepare API call parameters
            params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": self.max_tokens,
                "temperature": temperature if temperature is not None else self.temperature,
            }
            
            if system_message:
                params["system"] = system_message
            
            if tools:
                # Convert OpenAI tool format to Anthropic format
                anthropic_tools = []
                for tool in tools:
                    if tool.get("type") == "function":
                        func = tool.get("function", {})
                        anthropic_tools.append({
                            "name": func.get("name"),
                            "description": func.get("description"),
                            "input_schema": func.get("parameters", {})
                        })
                params["tools"] = anthropic_tools
            
            # Make API call
            response = await self.client.messages.create(**params)
            
            # Update token counts
            input_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            self.update_token_count(input_tokens, completion_tokens)
            
            # Convert response to OpenAI format
            result = {
                "role": "assistant",
                "content": None
            }
            
            # Check if there are tool calls
            if response.stop_reason == "tool_use":
                tool_calls = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input)
                            }
                        })
                result["tool_calls"] = tool_calls
            
            # Get text content
            text_content = ""
            for block in response.content:
                if block.type == "text":
                    text_content += block.text
            
            if text_content:
                result["content"] = text_content
            
            return result
            
        except Exception as e:
            print(f"Error in Anthropic ask_tool: {str(e)}")
            raise


class GoogleLLMAdapter(LLM):
    """Adapter for Google Gemini API to OpenAI-style interface."""
    
    def __init__(self, config_name: str = "default", config_path: Optional[str] = None):
        """Initialize Google Gemini adapter."""
        if not GOOGLE_AVAILABLE:
            raise ImportError("google-generativeai package is required for Gemini models. Install with: pip install google-generativeai")
        
        # Initialize base config
        super().__init__(config_name, config_path)
        
        # Get API key from environment or config
        api_key = os.getenv("GOOGLE_API_KEY") or self.config.api_key
        if not api_key or api_key == "":
            raise ValueError("GOOGLE_API_KEY environment variable or config api_key must be set")
        
        # Configure Google API
        genai.configure(api_key=api_key)
        
        # Initialize model
        self.gemini_model = genai.GenerativeModel(self.model)
        
        print(f"🔧 Initialized Google Gemini adapter: {self.model}")
    
    def format_messages_for_google(self, messages: List[dict]) -> List[dict]:
        """Convert OpenAI-style messages to Google Gemini format."""
        formatted_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            # Map roles
            if role == "system":
                # Gemini doesn't have a system role, add as user message
                formatted_messages.append({
                    "role": "user",
                    "parts": [{"text": f"[System] {content}"}]
                })
            elif role == "user":
                formatted_messages.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                formatted_messages.append({
                    "role": "model",  # Gemini uses "model" instead of "assistant"
                    "parts": [{"text": content}]
                })
            elif role == "tool":
                formatted_messages.append({
                    "role": "user",
                    "parts": [{"text": f"[Tool result] {content}"}]
                })
        
        return formatted_messages
    
    async def ask(
        self,
        messages: List[Union[dict, Any]],
        system_msgs: Optional[List[Union[dict, Any]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """Ask Gemini using Google API."""
        try:
            # Format messages
            formatted_messages = self.format_messages(messages)
            if system_msgs:
                system_formatted = self.format_messages(system_msgs)
                formatted_messages = system_formatted + formatted_messages
            
            # Convert to Google format
            google_messages = self.format_messages_for_google(formatted_messages)
            
            # Calculate input tokens (approximate)
            input_tokens = self.count_message_tokens(formatted_messages)
            
            # Configure generation
            generation_config = {
                "temperature": temperature if temperature is not None else self.temperature,
                "max_output_tokens": self.max_tokens,
            }
            
            # Create chat session
            chat = self.gemini_model.start_chat(history=google_messages[:-1] if len(google_messages) > 1 else [])
            
            # Get last message
            last_message = google_messages[-1]["parts"][0]["text"] if google_messages else ""
            
            # Make API call
            if stream:
                # Streaming response
                collected_messages = []
                response = await chat.send_message_async(
                    last_message,
                    generation_config=generation_config,
                    stream=True
                )
                async for chunk in response:
                    text = chunk.text
                    collected_messages.append(text)
                    print(text, end="", flush=True)
                print()  # Newline after streaming
                response_text = "".join(collected_messages)
            else:
                # Non-streaming response
                response = await chat.send_message_async(
                    last_message,
                    generation_config=generation_config
                )
                response_text = response.text
            
            # Estimate completion tokens
            completion_tokens = self.count_tokens(response_text)
            
            # Update token counts
            self.update_token_count(input_tokens, completion_tokens)
            
            return response_text
            
        except Exception as e:
            print(f"Error in Google ask: {str(e)}")
            raise
    
    async def ask_tool(
        self,
        messages: List[Union[dict, Any]],
        system_msgs: Optional[List[Union[dict, Any]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Optional[dict]:
        """Ask Gemini with tool support."""
        # For now, fall back to regular ask without tool support
        # Full tool support for Gemini requires more complex integration
        response_text = await self.ask(messages, system_msgs, False, temperature)
        
        return {
            "role": "assistant",
            "content": response_text
        }


def get_llm_instance(config_path: Optional[str] = None, config_name: str = "default") -> LLM:
    """
    Get appropriate LLM instance based on configuration.
    
    This factory function automatically detects the API type and returns
    the appropriate adapter (OpenAI, Anthropic, or Google).
    
    Args:
        config_path: Path to configuration file
        config_name: Configuration name for instance caching
    
    Returns:
        LLM instance (OpenAI, Anthropic, or Google adapter)
    """
    # Load config to determine API type
    config = LLMConfig(config_path)
    
    # Check model config for api_type
    api_type = None
    if hasattr(config, 'config') and isinstance(config.config, dict):
        model_config = config.config.get('model', {})
        api_type = model_config.get('api_type')
        
        # Also check llm.default for api_type
        if not api_type:
            llm_config = config.config.get('llm', {})
            default_config = llm_config.get('default', {})
            api_type = default_config.get('api_type')
    
    # Determine which adapter to use
    if api_type == 'anthropic' or 'claude' in config.name.lower():
        return AnthropicLLMAdapter(config_name, config_path)
    elif api_type == 'google' or 'gemini' in config.name.lower():
        return GoogleLLMAdapter(config_name, config_path)
    else:
        # Default to OpenAI
        return LLM(config_name, config_path)


# Test function
async def test_adapters():
    """Test LLM adapters."""
    print("=== LLM Adapter Tests ===\n")
    
    # Test with different configs
    configs = [
        ("configs/gpt4o.yaml", "GPT-4o"),
        ("configs/claude3.5.yaml", "Claude 3.5"),
        ("configs/gemini2.5flash.yaml", "Gemini 2.5 Flash"),
    ]
    
    test_message = [{"role": "user", "content": "What is 2+2? Answer briefly."}]
    
    for config_file, model_name in configs:
        print(f"--- Testing {model_name} ---")
        config_path = Path(__file__).parent / config_file
        
        if not config_path.exists():
            print(f"Config not found: {config_path}")
            continue
        
        try:
            llm = get_llm_instance(str(config_path))
            print(f"Adapter type: {type(llm).__name__}")
            
            response = await llm.ask(test_message)
            print(f"Response: {response[:100]}...")
            print(f"✅ Success\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
    
    print("=== Tests Complete ===")


if __name__ == "__main__":
    asyncio.run(test_adapters())
