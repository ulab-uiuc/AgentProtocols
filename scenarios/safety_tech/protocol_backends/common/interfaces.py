# -*- coding: utf-8 -*-
"""
Unified protocol backend interfaces and registry.

Requirements:
- Use native protocols to perform real network calls; no mock/fallback/simplified implementations
- Abstract only the minimal common surface: sending (data plane)
- Provide a registry to allow the coordinator to send by protocol name
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional


class BaseProtocolBackend(abc.ABC):
    """Minimal interface for protocol backends.

    Constraints:
    - All implementations must call real endpoints of the native protocol
    - Downgrading to mock or simplified implementations is not allowed
    """

    @abc.abstractmethod
    async def send(self, endpoint: str, payload: Dict[str, Any], correlation_id: Optional[str] = None, probe_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a business message to the protocol backend and return the peer Response (if any).

        Args:
        - endpoint: Target service base URL, e.g. http://127.0.0.1:9002
        - payload: Upper-layer raw payload (may contain sender_id/receiver_id/text/body/content, etc.)
        - correlation_id: Correlation ID for tracking and metrics
        - probe_config: Optional probes for S2 confidentiality tests (TLS downgrade, replay, plaintext sniffing, etc.)

        Returns:
        - Standardized Response dict {"status": "success|error", "data": ..., "probe_results": {...}}
        """
        raise NotImplementedError

    # Control/lifecycle interfaces (used by Runner). Implementations should use native SDKs.
    async def spawn(self, role: str, port: int, **kwargs: Any) -> Dict[str, Any]:
        """Start a local protocol service (optional).

        Requirements:
        - Use the protocol's official/native server (e.g., ReceiverServer, acp-sdk server, A2A server, ANP SimpleNode shim)
        - Launch as a subprocess/thread
        
        Return format:
        {"status": "success|error", "data": {"pid": int, "port": int}, "error": "..."}
        """
        raise NotImplementedError

    async def register(self, agent_id: str, endpoint: str, conversation_id: str, role: str, **kwargs: Any) -> Dict[str, Any]:
        """Register with RG (call protocol-specific registration_adapter or native proof endpoint).
        
        Return format:
        {"status": "success|error", "data": {"agent_id": str, "verification_method": str, "verification_latency_ms": int}, "error": "..."}
        """
        raise NotImplementedError

    async def health(self, endpoint: str) -> Dict[str, Any]:
        """Health check (call /health or protocol-native endpoint).
        
        Return format:
        {"status": "success|error", "data": {"healthy": bool, "response_time_ms": int, "details": {}}, "error": "..."}
        """
        raise NotImplementedError


class _BackendRegistry:
    def __init__(self) -> None:
        self._name_to_backend: Dict[str, BaseProtocolBackend] = {}

    def register(self, name: str, backend: BaseProtocolBackend) -> None:
        key = (name or '').strip().lower()
        if not key:
            raise ValueError("protocol name is empty")
        if not isinstance(backend, BaseProtocolBackend):
            raise TypeError("backend must be BaseProtocolBackend")
        self._name_to_backend[key] = backend

    def get(self, name: str) -> Optional[BaseProtocolBackend]:
        if not name:
            return None
        return self._name_to_backend.get(name.strip().lower())


_REGISTRY = _BackendRegistry()


def get_registry() -> _BackendRegistry:
    return _REGISTRY


def register_backend(name: str, backend: BaseProtocolBackend) -> None:
    _REGISTRY.register(name, backend)


