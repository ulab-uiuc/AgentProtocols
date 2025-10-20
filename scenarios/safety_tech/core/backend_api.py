# -*- coding: utf-8 -*-
"""
Unified Backend API used by runners to interact with protocol backends.
- spawn_backend(protocol, role, port, **kwargs)
- register_backend(protocol, agent_id, endpoint, conversation_id, role, **kwargs)
- health_backend(protocol, endpoint)
 - send_backend(protocol, endpoint, payload)

Backends are original/native implementations registered via protocol_backends.common.interfaces.register_backend.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _get_backend(protocol: str):
    from scenarios.safety_tech.protocol_backends.common.interfaces import get_registry
    
    # Ensure all protocol backends are imported to trigger their registration code
    # Handle each protocol import in an isolated try/except so one failure won't affect others
    protocols_to_import = [
        ('anp', 'scenario.safety_tech.protocol_backends.anp'),
        ('acp', 'scenario.safety_tech.protocol_backends.acp'),
        ('a2a', 'scenario.safety_tech.protocol_backends.a2a'),
        ('agora', 'scenario.safety_tech.protocol_backends.agora'),
    ]
    
    for proto_name, module_path in protocols_to_import:
        try:
            __import__(module_path)
        except ImportError as e:
            # Raise only if the requested protocol is the one that failed; otherwise log/warn
            if protocol == proto_name:
                print(f"Error: Cannot load {proto_name} backend: {e}")
                raise RuntimeError(f"Protocol backend '{proto_name}' is not available. Missing dependency: {e}")
            else:
                # Import failures for other protocols are warnings only; do not affect current protocol
                pass
        except Exception as e:
            # Handle other exceptions similarly
            if protocol == proto_name:
                print(f"Error: Cannot initialize {proto_name} backend: {e}")
                raise RuntimeError(f"Protocol backend '{proto_name}' failed to initialize: {e}")
    
    registry = get_registry()
    backend = registry.get(protocol)
    if backend is None:
        raise RuntimeError(f"No backend registered for protocol: {protocol}")
    return backend


async def spawn_backend(protocol: str, role: str, port: int, **kwargs: Any) -> Dict[str, Any]:
    backend = _get_backend(protocol)
    return await backend.spawn(role, port, **kwargs)


async def register_backend(protocol: str, agent_id: str, endpoint: str, conversation_id: str, role: str, **kwargs: Any) -> Dict[str, Any]:
    backend = _get_backend(protocol)
    return await backend.register(agent_id, endpoint, conversation_id, role, **kwargs)


async def health_backend(protocol: str, endpoint: str) -> Dict[str, Any]:
    backend = _get_backend(protocol)
    return await backend.health(endpoint)


async def send_backend(protocol: str, endpoint: str, payload: Dict[str, Any], correlation_id: Optional[str] = None, probe_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Send a business message directly to the specified protocol backend (bypassing the coordinator).

    Use cases:
    - Protocol direct-connect load generation/injection probes in Runner
    - Observe protocol data-plane behavior directly without going through the router
    - S1 load testing: concurrency/RPS/backpressure point testing
    - S2 confidentiality testing: TLS downgrade, replay attack, plaintext sniffing probes

    Parameters:
    - correlation_id: Message correlation ID for tracing and metrics aggregation
    - probe_config: Probe configuration, e.g., {"tls_downgrade": True, "replay_nonce": "xxx", "plaintext_sniff": True}
    """
    backend = _get_backend(protocol)
    return await backend.send(endpoint, payload, correlation_id, probe_config)


