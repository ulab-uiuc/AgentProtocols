#!/usr/bin/env python3
"""
Configuration loader with environment variable support.

This module handles loading YAML configuration files and substituting
environment variables in the format ${VAR_NAME}.
"""

import os
import re
import yaml
from typing import Dict, Any


def load_config_with_env_vars(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file and substitute environment variables.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
        
    Returns
    -------
    Dict[str, Any]
        Loaded configuration with environment variables substituted
        
    Raises
    ------
    FileNotFoundError
        If configuration file doesn't exist
    KeyError
        If required environment variable is not set
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Substitute environment variables
    content = substitute_env_vars(content)
    
    # Load YAML
    return yaml.safe_load(content)


def substitute_env_vars(content: str) -> str:
    """
    Substitute environment variables in format ${VAR_NAME} in content.
    
    Parameters
    ----------
    content : str
        Content string with potential environment variable references
        
    Returns
    -------
    str
        Content with environment variables substituted
        
    Raises
    ------
    KeyError
        If referenced environment variable is not set
    """
    # Pattern to match ${VAR_NAME}
    pattern = r'\$\{([^}]+)\}'
    
    def replace_var(match):
        var_name = match.group(1)
        env_value = os.getenv(var_name)
        
        if env_value is None:
            raise KeyError(f"Environment variable '{var_name}' is not set")
        
        return env_value
    
    return re.sub(pattern, replace_var, content)


def get_required_env_vars() -> Dict[str, str]:
    """
    Get list of required environment variables for fail_storm_recovery.
    Note: At least one of the LLM API keys must be set.
    
    Returns
    -------
    Dict[str, str]
        Dictionary of available environment variables and their descriptions
    """
    return {
        "OPENAI_API_KEY": "OpenAI API key for LLM access (obtain from https://platform.openai.com/api-keys)",
        "NVIDIA_API_KEY": "NVIDIA API key for LLM access (obtain from https://build.nvidia.com/)"
    }


def check_env_vars() -> bool:
    """
    Check if at least one LLM API key is available.
    
    Returns
    -------
    bool
        True if at least one API key is set, False if none are available
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    
    if openai_key or nvidia_key:
        if openai_key and nvidia_key:
            print("✅ Both OpenAI and NVIDIA API keys detected")
        elif openai_key:
            print("✅ OpenAI API key detected")
        elif nvidia_key:
            print("✅ NVIDIA API key detected")
        return True
    else:
        available_vars = get_required_env_vars()
        print("❌ No LLM API keys found. Please set at least one of:")
        for var_name, description in available_vars.items():
            print(f"   {var_name}: {description}")
        print("\nExample setup (choose one or both):")
        print("   set OPENAI_API_KEY=your_openai_key_here")
        print("   set NVIDIA_API_KEY=your_nvidia_key_here")
        return False


def get_available_llm_config() -> Dict[str, Any]:
    """
    Get LLM configuration compatible with Core class format.
    Prefers OpenAI if both are available.
    
    Returns
    -------
    Dict[str, Any]
        LLM configuration dict in Core class format
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    
    if openai_key:
        # Prefer OpenAI if available - format for Core class
        return {
            "model": {
                "type": "openai",
                "name": "gpt-4o-mini",
                "openai_api_key": openai_key,
                "openai_base_url": "https://api.openai.com/v1",
                "temperature": 0.1,
                "max_tokens": 1000
            }
        }
    elif nvidia_key:
        # Fallback to NVIDIA - format for Core class
        return {
            "model": {
                "type": "nvidia", 
                "name": "nvdev/nvidia/llama-3.1-nemotron-70b-instruct",
                "nvidia_api_key": nvidia_key,
                "nvidia_base_url": "https://integrate.api.nvidia.com/v1",
                "temperature": 0.1,
                "max_tokens": 8192,
                "top_p": 0.7
            }
        }
    else:
        raise ValueError("No LLM API keys available")


def create_core_instance():
    """
    Create a Core instance with available LLM configuration.
    
    Returns
    -------
    Core
        Initialized Core instance
    """
    import sys
    from pathlib import Path
    
    # Add src to path to import Core
    # From fail_storm_recovery/utils/config_loader.py to project root
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root / "src"))
    
    try:
        from utils.core import Core
        config = get_available_llm_config()
        return Core(config)
    except ImportError as e:
        raise ImportError(f"Could not import Core class: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to create Core instance: {e}")