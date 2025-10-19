# -*- coding: utf-8 -*-
"""
Direct Communication Backend for Privacy Testing
Simple direct communication without protocol overhead.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

try:
    from ...comm.base import BaseCommBackend
except ImportError:
    try:
        from comm.base import BaseCommBackend
    except ImportError:
        from ...comm.base import BaseCommBackend


class DirectCommBackend(BaseCommBackend):
    """Direct communication backend - no protocol overhead."""

    def __init__(self, **kwargs):
        self._endpoints: Dict[str, str] = {}
        self._agents: Dict[str, Any] = {}  # Store direct references to agent objects

    async def register_endpoint(self, agent_id: str, address: str) -> None:
        """Register direct agent reference."""
        self._endpoints[agent_id] = address
        print(f"[DirectCommBackend] Registered {agent_id}")

    async def register_agent_direct(self, agent_id: str, agent_instance: Any) -> None:
        """Register agent instance directly."""
        self._agents[agent_id] = agent_instance
        self._endpoints[agent_id] = f"direct://{agent_id}"

    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """Send message directly to agent instance."""
        dst_agent = self._agents.get(dst_id)
        if not dst_agent:
            raise RuntimeError(f"Unknown destination agent: {dst_id}")

        try:
            # Extract text from payload
            text = payload.get("text", str(payload))
            
            # Call agent directly
            if hasattr(dst_agent, 'process_message'):
                response = await dst_agent.process_message(src_id, text)
            else:
                response = f"Agent {dst_id} received: {text}"
            
            return {
                "raw": {"direct_response": response},
                "text": response
            }
            
        except Exception as e:
            print(f"[DirectCommBackend] Send failed {src_id} -> {dst_id}: {e}")
            return {"raw": None, "text": ""}

    async def health_check(self, agent_id: str) -> bool:
        """Check if agent instance exists."""
        return agent_id in self._agents

    async def close(self) -> None:
        """Close direct communication."""
        self._endpoints.clear()
        self._agents.clear()

