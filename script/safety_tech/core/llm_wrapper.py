# -*- coding: utf-8 -*-
"""
Simple LLM Wrapper - Alternative to Core class
Provides basic LLM functionality without complex dependencies.
"""

import requests
import json
from typing import List, Dict, Any, Optional


class SimpleLLMWrapper:
    """Simple LLM wrapper using direct HTTP requests."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model_config = config.get("model", {})
        
        self.api_key = model_config.get("openai_api_key", "")
        self.base_url = model_config.get("openai_base_url", "https://api.openai.com/v1")
        self.model_name = model_config.get("name", "gpt-4o")
        self.temperature = model_config.get("temperature", 0.3)
        
        if not self.api_key:
            raise ValueError("API key is required")
        
        print(f"[SimpleLLM] Initialized with model: {self.model_name}")
    
    def execute(self, messages: List[Dict[str, str]]) -> str:
        """Execute LLM call using direct HTTP request."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return content.strip()
            else:
                print(f"[SimpleLLM] HTTP Error {response.status_code}: {response.text}")
                return f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            print(f"[SimpleLLM] Request failed: {e}")
            return f"Error: {str(e)}"


def create_llm(config: Dict[str, Any]) -> Optional[SimpleLLMWrapper]:
    """Create LLM instance from config."""
    try:
        return SimpleLLMWrapper(config)
    except Exception as e:
        print(f"[SimpleLLM] Failed to create LLM: {e}")
        return None

