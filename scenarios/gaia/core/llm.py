"""LLM calling utilities for the multi-agent framework."""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
import aiohttp
import tiktoken
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.base import BaseTool, ToolResult
from tools.exceptions import ToolError


class TokenLimitExceeded(Exception):
    """Exception raised when token limits are exceeded."""
    pass


class TokenCounter:
    """Token counting utilities."""
    
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        """Calculate tokens for a text string"""
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """Calculate tokens for message content"""
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """Calculate the total number of tokens in a message list"""
        total_tokens = self.FORMAT_TOKENS  # Base format tokens

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # Base tokens per message

            # Add role tokens
            tokens += self.count_text(message.get("role", ""))

            # Add content tokens
            if "content" in message:
                tokens += self.count_content(message["content"])

            # Add name tokens
            tokens += self.count_text(message.get("name", ""))

            total_tokens += tokens

        return total_tokens


class LLMConfig:
    """Configuration class for LLM parameters."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize LLM configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "general.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Extract model configuration
        model_config = self.config.get("model", {})
        self.name = model_config.get("name", "gpt-4o")
        # Prioritize environment variables over config file
        self.base_url = os.getenv("OPENAI_BASE_URL") or model_config.get("base_url", "https://api.openai.com/v1")
        self.api_key = os.getenv("OPENAI_API_KEY") or model_config.get("api_key", "")
        self.max_tokens = model_config.get("max_tokens", 1000)
        self.temperature = model_config.get("temperature", 0.9)
        self.max_input_tokens = model_config.get("max_input_tokens", None)
        
        # Additional parameters
        self.timeout = model_config.get("timeout", 30)
        self.top_p = model_config.get("top_p", 1.0)
        self.frequency_penalty = model_config.get("frequency_penalty", 0.0)
        self.presence_penalty = model_config.get("presence_penalty", 0.0)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }


class LLM:
    """Main LLM class for making API calls."""
    
    _instances: Dict[str, "LLM"] = {}

    def __new__(cls, config_name: str = "default", config_path: Optional[str] = None):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, config_path)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(self, config_name: str = "default", config_path: Optional[str] = None):
        if not hasattr(self, "config"):  # Only initialize if not already initialized
            self.config = LLMConfig(config_path)
            self.model = self.config.name
            self.max_tokens = self.config.max_tokens
            self.temperature = self.config.temperature
            self.api_key = self.config.api_key
            self.base_url = self.config.base_url
            self.max_input_tokens = self.config.max_input_tokens

            # Token tracking
            self.total_input_tokens = 0
            self.total_completion_tokens = 0

            # Initialize tokenizer
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # If the model is not in tiktoken's presets, use cl100k_base as default
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

            self.token_counter = TokenCounter(self.tokenizer)

    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)

    def reset_token_counts(self) -> None:
        """Reset the token counters to zero."""
        self.total_input_tokens = 0
        self.total_completion_tokens = 0
        print("Token counters have been reset.")

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """Update token counts"""
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        
        # Eye-catching colors and formatting for token usage display
        GREEN = "\033[92m"      # bright green
        YELLOW = "\033[93m"     # bright yellow
        CYAN = "\033[96m"       # bright cyan
        BOLD = "\033[1m"        # bold
        RESET = "\033[0m"       # reset

        # Create a visible separator and formatting
        separator = "=" * 60
        current_total = input_tokens + completion_tokens
        cumulative_total = self.total_input_tokens + self.total_completion_tokens

        print(f"\n{CYAN}{BOLD}{separator}{RESET}")
        print(f"{GREEN}{BOLD}🚀 TOKEN USAGE REPORT - Model: {self.model}{RESET}")
        print(f"{CYAN}{BOLD}{separator}{RESET}")
        print(f"{YELLOW}📥 Input Tokens:      {BOLD}{input_tokens:,}{RESET} (this call)")
        print(f"{YELLOW}📤 Output Tokens:     {BOLD}{completion_tokens:,}{RESET} (this call)")
        print(f"{GREEN}📊 Call Total:        {BOLD}{current_total:,}{RESET} tokens")
        print(f"{CYAN}📈 Cumulative Input:  {BOLD}{self.total_input_tokens:,}{RESET} tokens")
        print(f"{CYAN}📈 Cumulative Output: {BOLD}{self.total_completion_tokens:,}{RESET} tokens")
        print(f"{GREEN}{BOLD}🎯 GRAND TOTAL:       {cumulative_total:,} tokens{RESET}")
        print(f"{CYAN}{BOLD}{separator}{RESET}\n")

    def check_token_limit(self, input_tokens: int) -> bool:
        """Check if token limits are exceeded"""
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        """Generate error message for token limit exceeded"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"
        return "Token limit exceeded"

    @staticmethod
    def format_messages(messages: List[Union[dict, Any]]) -> List[dict]:
        """Format messages for LLM by converting them to OpenAI message format."""
        formatted_messages = []

        for message in messages:
            # Convert to dict if needed
            if hasattr(message, 'to_dict'):
                message = message.to_dict()
            elif hasattr(message, '__dict__'):
                message = message.__dict__

            if isinstance(message, dict):
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # OpenAI API rejects standalone messages with role 'tool' unless they are
                # responses to preceding 'tool_calls'. To avoid invalid requests, convert
                # outgoing 'tool' role messages to 'assistant' role and embed tool metadata
                # into the content so context is preserved.
                role = message.get("role")
                if role == "tool":
                    # Build a content string that includes tool name/id if present
                    content_parts = []
                    if "name" in message and message.get("name"):
                        content_parts.append(f"[tool:{message.get('name')}]")
                    if "tool_call_id" in message and message.get("tool_call_id"):
                        content_parts.append(f"[call_id:{message.get('tool_call_id')}]")
                    if "content" in message and message.get("content"):
                        content_parts.append(str(message.get("content")))
                    new_content = " ".join(content_parts) if content_parts else ""
                    safe_msg = {"role": "assistant", "content": new_content}
                    formatted_messages.append(safe_msg)
                else:
                    if "content" in message or "tool_calls" in message:
                        formatted_messages.append(message)
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((Exception,))
    )
    async def ask(
        self,
        messages: List[Union[dict, Any]],
        system_msgs: Optional[List[Union[dict, Any]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to the LLM and get the response.
        
        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream: Whether to stream the response
            temperature: Sampling temperature for the response
            
        Returns:
            str: The generated response
        """
        try:
            # Format messages
            formatted_messages = self.format_messages(messages)
            if system_msgs:
                system_formatted = self.format_messages(system_msgs)
                formatted_messages = system_formatted + formatted_messages

            # Calculate input token count
            input_tokens = self.count_message_tokens(formatted_messages)

            # Check token limits
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                raise TokenLimitExceeded(error_message)

            # Prepare API parameters
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": self.max_tokens,
                "temperature": temperature if temperature is not None else self.temperature,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty
            }

            # Make API call
            response = await self._call_api(params, stream)

            # Update token counts
            if not stream:
                # For non-streaming, we can get actual token counts from response
                prompt_tokens = response.get("usage", {}).get("prompt_tokens", input_tokens)
                completion_tokens = response.get("usage", {}).get("completion_tokens", 0)
                self.update_token_count(prompt_tokens, completion_tokens)
                
                if "choices" in response and response["choices"]:
                    return response["choices"][0]["message"]["content"]
                else:
                    raise ValueError("Empty or invalid response from LLM")
            else:
                # For streaming, calculate completion tokens from the response content
                try:
                    completion_tokens = self.count_tokens(response)
                except Exception:
                    completion_tokens = 0
                self.update_token_count(input_tokens, completion_tokens)
                return response  # response is already the content string for streaming

        except TokenLimitExceeded:
            raise
        except Exception as e:
            print(f"Error in ask: {str(e)}")
            raise

    async def _call_api(self, params: Dict[str, Any], stream: bool = False) -> Union[str, dict]:
        """Make the actual API call."""
        if not self.config.api_key:
            raise ToolError("API key not configured. Set api_key in config or OPENAI_API_KEY environment variable.")

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        params["stream"] = stream

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, headers=headers, json=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ToolError(f"API call failed with status {response.status}: {error_text}")

                if stream:
                    # Handle streaming response
                    collected_messages = []
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            line = line[6:]  # Remove 'data: ' prefix
                            if line == '[DONE]':
                                break
                            try:
                                chunk = json.loads(line)
                                if "choices" in chunk and chunk["choices"]:
                                    delta = chunk["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        collected_messages.append(content)
                                        print(content, end="", flush=True)
                            except json.JSONDecodeError:
                                continue
                    
                    print()  # Newline after streaming
                    return "".join(collected_messages)
                else:
                    # Handle non-streaming response
                    result = await response.json()
                    return result
    
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((Exception,))
    )
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
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            dict: The model's response message

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or messages are invalid
            Exception: For unexpected errors
        """
        try:
            # Format messages
            formatted_messages = self.format_messages(messages)
            if system_msgs:
                system_formatted = self.format_messages(system_msgs)
                formatted_messages = system_formatted + formatted_messages

            # Calculate input token count
            input_tokens = self.count_message_tokens(formatted_messages)

            # If there are tools, calculate token count for tool descriptions
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))

            input_tokens += tools_tokens

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                raise TokenLimitExceeded(error_message)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            # Set up the completion request
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": self.max_tokens,
                "temperature": temperature if temperature is not None else self.temperature,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
                **kwargs,
            }

            if tools:
                params["tools"] = tools
                params["tool_choice"] = tool_choice

            # Make API call
            response = await self._call_api_with_tools(params, timeout)

            # Check if response is valid
            if not response or not response.get("choices") or not response["choices"][0].get("message"):
                return None

            # Update token counts
            prompt_tokens = response.get("usage", {}).get("prompt_tokens", input_tokens)
            completion_tokens = response.get("usage", {}).get("completion_tokens", 0)
            self.update_token_count(prompt_tokens, completion_tokens)

            return response["choices"][0]["message"]

        except TokenLimitExceeded:
            raise
        except Exception as e:
            print(f"Error in ask_tool: {str(e)}")
            raise

    async def _call_api_with_tools(self, params: Dict[str, Any], timeout: int = 300) -> dict:
        """Make API call with tools support."""
        if not self.config.api_key:
            raise ToolError("API key not configured. Set api_key in config or OPENAI_API_KEY environment variable.")

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        timeout_config = aiohttp.ClientTimeout(total=timeout)

        params["stream"] = False  # always use non-streaming for tool requests

        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.post(endpoint, headers=headers, json=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ToolError(f"API call failed with status {response.status}: {error_text}")

                result = await response.json()
                return result


# Utility function to get appropriate LLM tool
# def get_llm_tool(config_path: Optional[str] = None) -> BaseTool:
#     """Get appropriate LLM tool based on configuration."""
#     config = LLMConfig(config_path)
    
#     if config.api_key and config.api_key.strip() and config.api_key != "Your API Key":
#         return LLM(config_path)
#     else:
#         raise ToolError("API key not configured. Set api_key in config or OPENAI_API_KEY environment variable.")


# Utility function to get LLM instance
def call_llm(config_name: str = "default", config_path: Optional[str] = None) -> LLM:
    """Get LLM instance."""
    return LLM(config_name, config_path)


# Test function
async def test_llm():
    """Test the LLM tools."""
    print("=== LLM Tool Test ===")
    
    # Test configuration loading
    config = LLMConfig()
    print(f"Configuration loaded: {config.to_dict()}")
    
    # Test LLM class
    print("\n--- Test 1: LLM Class ---")
    llm = call_llm()
    print(f"Using LLM: {llm.__class__.__name__}")
    
    # Test simple message
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    try:
        result1 = await llm.ask(messages=messages)
        print(f"Result: {result1[:200]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    # # Test simple prompt
    # result2 = await llm_tool.ask(prompt="Explain quantum computing in simple terms.")
    # if result2.output:
    #     print(f"Result: {result2.output[:200]}...")
    # else:
    #     print(f"Error: {result2.error}")
    
    # Test messages format
    print("\n--- Test 2: Messages Format ---")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ]
    result3 = await llm.ask(messages=messages, stream=True)
    if result3:
        print(f"Result: {result3[:200]}...")
    else:
        raise ValueError("No response from LLM")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    """Run LLM tool tests."""
    asyncio.run(test_llm())
