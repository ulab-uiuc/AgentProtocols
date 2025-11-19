# -*- coding: utf-8 -*-
"""
Comm Backend Base
Abstract the network communication layer so AgentNetwork no longer directly depends on specific protocols/HTTP.
"""

from __future__ import annotations
import abc
from typing import Any, Dict, Optional


class BaseCommBackend(abc.ABC):
    """
        Abstract communication backend:
            - register_endpoint: Register the service address of an agent (or an inproc handle)
            - connect: (optional) establish connections/warm-up
            - send: Send a message from src to dst, return the protocol-native Response
            - health_check: Liveness probe
            - close: Close resources
            - record_retry/record_error: Record connection retries and network errors (optional)
    """
    
    def __init__(self):
        # Optional metrics collector reference
        self.metrics_collector = None
    
    def set_metrics_collector(self, collector):
        """Set metrics collector for recording performance data"""
        self.metrics_collector = collector

    @abc.abstractmethod
    async def register_endpoint(self, agent_id: str, address: str) -> None:
        ...

    async def connect(self, src_id: str, dst_id: str) -> None:
        """Some protocols require explicit connection setup; default to no-op."""
        return None

    @abc.abstractmethod
    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        """
        Send message and return response with timing information.
        Response should include:
        - raw: protocol-native response
        - text: extracted text
        - timing: dict with 'request_start', 'request_end', 'total_time'
        """
        ...

    @abc.abstractmethod
    async def health_check(self, agent_id: str) -> bool:
        ...

    async def close(self) -> None:
        """Close underlying resources (HTTP clients, sockets, etc.). Default to no-op."""
        return None
    
    def record_retry(self, agent_id: str) -> None:
        """Record a connection retry attempt"""
        if self.metrics_collector:
            self.metrics_collector.record_connection_retry(agent_id)
    
    def record_network_error(self, agent_id: str, error_type: str) -> None:
        """Record a network error"""
        if self.metrics_collector:
            self.metrics_collector.record_network_error(agent_id, error_type)
