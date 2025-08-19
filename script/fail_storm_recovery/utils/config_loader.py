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
    
    Returns
    -------
    Dict[str, str]
        Dictionary of required environment variables and their descriptions
    """
    return {
        "NVIDIA_API_KEY": "NVIDIA API key for LLM access (obtain from https://build.nvidia.com/)"
    }


def check_env_vars() -> bool:
    """
    Check if all required environment variables are set.
    
    Returns
    -------
    bool
        True if all required variables are set, False otherwise
    """
    required_vars = get_required_env_vars()
    missing_vars = []
    
    for var_name in required_vars:
        if not os.getenv(var_name):
            missing_vars.append(var_name)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var_name in missing_vars:
            description = required_vars[var_name]
            print(f"   {var_name}: {description}")
        print("\nPlease set these variables before running:")
        for var_name in missing_vars:
            print(f"   export {var_name}=your_value_here")
        return False
    
    return True