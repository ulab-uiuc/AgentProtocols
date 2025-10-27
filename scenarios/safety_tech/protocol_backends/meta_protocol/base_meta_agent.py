# -*- coding: utf-8 -*-
"""
Base Meta Agent for Safety Testing using src/core/base_agent.py
"""

from __future__ import annotations

import os
import asyncio
import time
import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

# Add src to path
HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent.parent
PROJECT_ROOT = SAFETY_TECH.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SAFETY_TECH) not in sys.path:
    sys.path.insert(0, str(SAFETY_TECH))

# Import from src
from src.core.base_agent import BaseAgent


class BaseSafetyMetaAgent(ABC):
    """
    Base class for all safety testing meta agents using src/core/base_agent.py
    
    Provides common interface for wrapping protocol-specific privacy testing agents
    into BaseAgent instances for proper meta-protocol integration.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any], agent_type: str, output=None):
        self.agent_id = agent_id
        self.config = config
        self.agent_type = agent_type  # "doctor" (S2 testing uses dual doctor architecture)
        self.output = output
        
        # Meta agent state
        self.is_initialized = False
        self.protocol_name = "base"
        
        # BaseAgent instance
        self.base_agent: Optional[BaseAgent] = None
        
        # Performance tracking
        self.message_count = 0
        self.total_response_time = 0.0
        self.error_count = 0
        self.start_time = time.time()

    @abstractmethod
    async def create_base_agent(self, host: str = "0.0.0.0", port: Optional[int] = None) -> BaseAgent:
        """Create and return a BaseAgent instance for this protocol"""
        pass

    @abstractmethod
    async def process_message_direct(self, message: str, sender_id: str = "external") -> str:
        """Process a message directly (fallback when AgentNetwork routing fails)"""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources used by the agent"""
        pass

    def _log(self, message: str) -> None:
        """Log message with agent context"""
        if self.output:
            self.output.info(f"[{self.agent_id}] {message}")
        else:
            print(f"[{self.agent_id}] {message}")

    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information and statistics"""
        uptime = time.time() - self.start_time
        avg_response_time = (
            self.total_response_time / self.message_count 
            if self.message_count > 0 else 0.0
        )
        
        info = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "protocol": self.protocol_name,
            "is_initialized": self.is_initialized,
            "uptime_seconds": uptime,
            "message_count": self.message_count,
            "error_count": self.error_count,
            "avg_response_time": avg_response_time,
            "total_response_time": self.total_response_time,
            "has_base_agent": self.base_agent is not None
        }
        
        # Add BaseAgent info if available
        if self.base_agent:
            try:
                info["base_agent_address"] = self.base_agent.get_listening_address()
                info["base_agent_card"] = self.base_agent.get_agent_card()
            except Exception as e:
                info["base_agent_error"] = str(e)
        
        return info

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        uptime = time.time() - self.start_time
        
        return {
            "messages_per_second": self.message_count / uptime if uptime > 0 else 0.0,
            "error_rate": self.error_count / self.message_count if self.message_count > 0 else 0.0,
            "avg_response_time": (
                self.total_response_time / self.message_count 
                if self.message_count > 0 else 0.0
            ),
            "uptime_minutes": uptime / 60.0
        }

    async def health_check(self) -> bool:
        """Check if the agent is healthy and responsive"""
        try:
            if not self.is_initialized or not self.base_agent:
                return False
            
            # Use BaseAgent's health check method instead of get_agent_card
            if hasattr(self.base_agent, 'health_check'):
                return await self.base_agent.health_check()
            else:
                # Fallback: check if agent is initialized and has a listening address
                return bool(self.base_agent.get_listening_address())
            
        except Exception as e:
            self._log(f"Health check failed: {e}")
            return False

    def _convert_config_for_executor(self) -> Dict[str, Any]:
        """Convert safety_tech config to format expected by protocol executors"""
        core = self.config.get("core", {})
        if core.get("type") == "openai":
            # Prioritize environment variables
            api_key = os.getenv("OPENAI_API_KEY") or core.get("openai_api_key")
            base_url = os.getenv("OPENAI_BASE_URL") or core.get("openai_base_url", "https://api.openai.com/v1")
            return {
                "model": {
                    "type": "openai",
                    "name": core.get("name", "gpt-4o"),
                    "openai_api_key": api_key,
                    "openai_base_url": base_url,
                    "temperature": core.get("temperature", 0.0),
                }
            }
        return {
            "model": {
                "type": "local",
                "name": core.get("name", "default"),
                "temperature": core.get("temperature", 0.3),
                "base_url": core.get("base_url", "http://localhost:8000/v1"),
                "port": core.get("port", 8000)
            }
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.agent_id}, {self.agent_type}, {self.protocol_name})"

    def __repr__(self) -> str:
        return self.__str__()
