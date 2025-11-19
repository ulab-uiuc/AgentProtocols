"""
Multi-vendor LLM compatibility patch for GAIA experiments.

This module extends the base LLM class to support:
- Anthropic Claude API
- Google Gemini API
- OpenAI-compatible APIs

Usage:
    from llm_patch import create_llm
    
    llm = create_llm(config_path="path/to/config.yaml")
    response = await llm.ask(messages=[...])
"""
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Union

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

# Import base classes
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "core"))

from llm import LLM, LLMConfig, TokenLimitExceeded
from tools.exceptions import ToolError


class AnthropicLLM(LLM):
    """Anthropic Claude API implementation."""
    
    def __init__(self, config_name: str = "default", config_path: Optional[str] = None):
        super().__init__(config_name, config_path)
        # Override API key from environment if available
        self.api_key = os.getenv("ANTHROPIC_API_KEY") or self.config.api_key
        
    async def _call_api(self, params: Dict[str, Any], stream: bool = False) -> Union[str, dict]:
        """Make API call to Anthropic."""
        if not self.api_key:
            raise ToolError("ANTHROPIC_API_KEY not set")
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        # Convert OpenAI format to Anthropic format
        messages = params["messages"]
        system_msg = None
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        anthropic_params = {
            "model": params["model"],
            "messages": anthropic_messages,
            "max_tokens": params.get("max_tokens", 4096),
            "temperature": params.get("temperature", 0.0),
            "stream": stream
        }
        
        if system_msg:
            anthropic_params["system"] = system_msg
        
        endpoint = f"{self.config.base_url.rstrip('/')}/messages"
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, headers=headers, json=anthropic_params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ToolError(f"Anthropic API error {response.status}: {error_text}")
                
                if stream:
                    collected_messages = []
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            line = line[6:]
                            if line == '[DONE]':
                                break
                            try:
                                chunk = json.loads(line)
                                if chunk.get("type") == "content_block_delta":
                                    delta = chunk.get("delta", {})
                                    text = delta.get("text", "")
                                    if text:
                                        collected_messages.append(text)
                                        print(text, end="", flush=True)
                            except json.JSONDecodeError:
                                continue
                    print()
                    return "".join(collected_messages)
                else:
                    result = await response.json()
                    # Convert Anthropic response to OpenAI format
                    return {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": result["content"][0]["text"]
                            }
                        }],
                        "usage": {
                            "prompt_tokens": result.get("usage", {}).get("input_tokens", 0),
                            "completion_tokens": result.get("usage", {}).get("output_tokens", 0)
                        }
                    }


class GoogleLLM(LLM):
    """Google Gemini API implementation."""
    
    def __init__(self, config_name: str = "default", config_path: Optional[str] = None):
        super().__init__(config_name, config_path)
        # Override API key from environment if available
        self.api_key = os.getenv("GOOGLE_API_KEY") or self.config.api_key
        
    async def _call_api(self, params: Dict[str, Any], stream: bool = False) -> Union[str, dict]:
        """Make API call to Google Gemini."""
        if not self.api_key:
            raise ToolError("GOOGLE_API_KEY not set")
        
        # Convert OpenAI format to Gemini format
        messages = params["messages"]
        system_instruction = None
        gemini_contents = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                gemini_contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })
        
        gemini_params = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": params.get("temperature", 0.0),
                "maxOutputTokens": params.get("max_tokens", 8192),
                "topP": params.get("top_p", 1.0),
            }
        }
        
        if system_instruction:
            gemini_params["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
        
        model_name = params["model"]
        endpoint = f"{self.config.base_url.rstrip('/')}/models/{model_name}:generateContent"
        
        # Add API key to URL
        url = f"{endpoint}?key={self.api_key}"
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        headers = {
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=gemini_params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ToolError(f"Google API error {response.status}: {error_text}")
                
                result = await response.json()
                
                # Extract text from Gemini response
                if "candidates" not in result or not result["candidates"]:
                    raise ToolError("No candidates in Gemini response")
                
                candidate = result["candidates"][0]
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                
                if not parts:
                    raise ToolError("No parts in Gemini response")
                
                text = parts[0].get("text", "")
                
                # Convert to OpenAI format
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": text
                        }
                    }],
                    "usage": {
                        "prompt_tokens": result.get("usageMetadata", {}).get("promptTokenCount", 0),
                        "completion_tokens": result.get("usageMetadata", {}).get("candidatesTokenCount", 0)
                    }
                }


def create_llm(config_path: Optional[str] = None) -> LLM:
    """
    Create appropriate LLM instance based on config.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        LLM instance (OpenAI, Anthropic, or Google)
    """
    config = LLMConfig(config_path)
    api_type = config.config.get("model", {}).get("api_type", "openai")
    
    if api_type == "anthropic":
        return AnthropicLLM(config_path=config_path)
    elif api_type == "google":
        return GoogleLLM(config_path=config_path)
    else:
        # Default to OpenAI-compatible
        return LLM(config_path=config_path)


async def test_multi_vendor():
    """Test all three vendors."""
    configs = [
        ("OpenAI", "configs/config_gpt4o.yaml"),
        ("Anthropic", "configs/config_claude.yaml"),
        ("Google", "configs/config_gemini.yaml"),
    ]
    
    test_message = [{"role": "user", "content": "Say 'Hello, I am working!' in one sentence."}]
    
    for vendor, config_path in configs:
        print(f"\n{'='*60}")
        print(f"Testing {vendor}")
        print(f"{'='*60}")
        
        try:
            llm = create_llm(config_path)
            response = await llm.ask(messages=test_message)
            print(f"✓ {vendor} response: {response[:100]}...")
        except Exception as e:
            print(f"✗ {vendor} error: {e}")


if __name__ == "__main__":
    asyncio.run(test_multi_vendor())
