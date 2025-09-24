# -*- coding: utf-8 -*-
"""
Meta Protocol Communication Backend for Privacy Testing
Implements meta protocol communication with intelligent routing for the privacy testing framework.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

# Import base communication interface
try:
    from ...comm.base import BaseCommBackend
except ImportError:
    try:
        from comm.base import BaseCommBackend
    except ImportError:
        # Create minimal base class if not available
        class BaseCommBackend:
            pass


@dataclass
class MetaAgentHandle:
    """Handle for meta protocol agent with routing information."""
    agent_id: str
    host: str
    port: int
    base_url: str
    selected_protocol: str
    routing_confidence: float
    server_task: Optional[asyncio.Task] = None


class MetaCommBackend(BaseCommBackend):
    """Meta protocol communication backend for privacy testing with intelligent routing."""

    def __init__(self, selected_protocol: str, routing_decision=None, **kwargs):
        self.selected_protocol = selected_protocol
        self.routing_decision = routing_decision
        self._endpoints: Dict[str, str] = {}  # agent_id -> endpoint uri
        self._client = httpx.AsyncClient(timeout=30.0)
        self._local_agents = {}  # For locally spawned agents
        self._agent_handles: Dict[str, MetaAgentHandle] = {}  # agent_id -> MetaAgentHandle
        
        # Routing metrics
        self._routing_stats = {
            "total_messages": 0,
            "successful_routes": 0,
            "fallback_used": 0,
            "routing_errors": 0
        }

    async def register_endpoint(self, agent_id: str, address: str) -> None:
        """Register meta protocol agent endpoint with routing information."""
        self._endpoints[agent_id] = address
        
        # Extract host and port from address
        if "://" in address:
            # Format: http://host:port
            parts = address.split("://")[1].split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 80
        else:
            # Format: host:port
            parts = address.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 80
        
        # Create meta agent handle
        handle = MetaAgentHandle(
            agent_id=agent_id,
            host=host,
            port=port,
            base_url=address,
            selected_protocol=self.selected_protocol,
            routing_confidence=self.routing_decision.confidence if self.routing_decision else 0.8
        )
        
        self._agent_handles[agent_id] = handle
        
        print(f"[MetaComm] Registered {self.selected_protocol.upper()} agent: {agent_id} @ {address}")

    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send message through meta protocol with intelligent routing.
        
        Args:
            src_id: Source agent ID
            dst_id: Destination agent ID  
            payload: Message payload
            
        Returns:
            Response from destination agent
        """
        self._routing_stats["total_messages"] += 1
        
        try:
            # Add meta protocol routing information to payload
            meta_payload = {
                **payload,
                "meta_routing": {
                    "selected_protocol": self.selected_protocol,
                    "routing_confidence": self.routing_decision.confidence if self.routing_decision else 0.8,
                    "routing_strategy": self.routing_decision.strategy if self.routing_decision else "heuristic",
                    "src_agent": src_id,
                    "dst_agent": dst_id,
                    "timestamp": time.time()
                }
            }
            
            # Route message based on selected protocol
            if dst_id not in self._endpoints:
                raise ValueError(f"Destination agent {dst_id} not registered")
            
            dst_endpoint = self._endpoints[dst_id]
            
            # Send message with protocol-specific handling
            response = await self._send_with_protocol_handling(
                src_id, dst_id, dst_endpoint, meta_payload
            )
            
            self._routing_stats["successful_routes"] += 1
            return response
            
        except Exception as e:
            self._routing_stats["routing_errors"] += 1
            print(f"[MetaComm] Routing error {src_id} -> {dst_id}: {e}")
            
            # Try fallback communication
            return await self._send_fallback(src_id, dst_id, payload)

    async def _send_with_protocol_handling(self, src_id: str, dst_id: str, 
                                         dst_endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send message with protocol-specific handling based on selected protocol."""
        
        # Prepare headers with meta protocol information
        headers = {
            "Content-Type": "application/json",
            "X-Meta-Protocol": f"meta_{self.selected_protocol}",
            "X-Selected-Protocol": self.selected_protocol.upper(),
            "X-Routing-Confidence": str(self.routing_decision.confidence if self.routing_decision else 0.8),
            "X-Source-Agent": src_id,
            "X-Target-Agent": dst_id
        }
        
        # Try protocol-specific endpoints first
        endpoints_to_try = self._get_protocol_endpoints(self.selected_protocol)
        
        for endpoint_path in endpoints_to_try:
            try:
                full_url = f"{dst_endpoint.rstrip('/')}{endpoint_path}"
                
                response = await self._client.post(
                    full_url,
                    json=payload,
                    headers=headers,
                    timeout=30.0
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                # Add meta protocol information to response
                response_data["meta_routing_info"] = {
                    "protocol_used": self.selected_protocol,
                    "endpoint_used": endpoint_path,
                    "routing_successful": True
                }
                
                return response_data
                
            except Exception as e:
                if endpoint_path == endpoints_to_try[-1]:  # Last endpoint
                    raise e  # Re-raise the last error
                continue  # Try next endpoint
        
        raise ConnectionError(f"All {self.selected_protocol} endpoints failed for {dst_endpoint}")

    def _get_protocol_endpoints(self, protocol: str) -> List[str]:
        """Get protocol-specific endpoints to try."""
        endpoint_map = {
            "a2a": ["/", "/message", "/a2a/message"],
            "acp": ["/acp/message", "/message", "/"],
            "agora": ["/agora/message", "/message", "/"],
            "anp": ["/anp/message", "/message", "/"]
        }
        
        return endpoint_map.get(protocol, ["/message", "/"])

    async def _send_fallback(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback communication when primary routing fails."""
        self._routing_stats["fallback_used"] += 1
        
        try:
            if dst_id not in self._endpoints:
                raise ValueError(f"Destination agent {dst_id} not registered for fallback")
            
            dst_endpoint = self._endpoints[dst_id]
            
            # Simple HTTP POST fallback
            fallback_payload = {
                **payload,
                "meta_fallback": True,
                "original_protocol": self.selected_protocol,
                "fallback_reason": "Primary routing failed"
            }
            
            headers = {
                "Content-Type": "application/json",
                "X-Meta-Fallback": "true",
                "X-Original-Protocol": self.selected_protocol
            }
            
            response = await self._client.post(
                f"{dst_endpoint.rstrip('/')}/",
                json=fallback_payload,
                headers=headers,
                timeout=30.0
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Mark as fallback response
            response_data["meta_routing_info"] = {
                "protocol_used": "fallback",
                "original_protocol": self.selected_protocol,
                "routing_successful": False,
                "fallback_used": True
            }
            
            print(f"[MetaComm] Fallback communication successful: {src_id} -> {dst_id}")
            return response_data
            
        except Exception as e:
            print(f"[MetaComm] Fallback communication failed: {src_id} -> {dst_id}: {e}")
            return {
                "error": f"Meta protocol communication failed: {str(e)}",
                "meta_routing_info": {
                    "protocol_used": "none",
                    "routing_successful": False,
                    "fallback_failed": True
                }
            }

    async def health_check(self, agent_id: str) -> bool:
        """Check if agent is healthy and reachable."""
        try:
            if agent_id not in self._endpoints:
                return False
            
            endpoint = self._endpoints[agent_id]
            
            # Try health check endpoints
            health_endpoints = ["/health", "/status", "/"]
            
            for health_path in health_endpoints:
                try:
                    response = await self._client.get(
                        f"{endpoint.rstrip('/')}{health_path}",
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            print(f"[MetaComm] Health check failed for {agent_id}: {e}")
            return False

    async def close(self) -> None:
        """Close meta protocol communication backend."""
        try:
            # Close HTTP client
            await self._client.aclose()
            
            # Stop any local agent servers
            for handle in self._agent_handles.values():
                if handle.server_task and not handle.server_task.done():
                    handle.server_task.cancel()
                    try:
                        await handle.server_task
                    except asyncio.CancelledError:
                        pass
            
            self._endpoints.clear()
            self._agent_handles.clear()
            
            print(f"[MetaComm] Communication backend closed for {self.selected_protocol.upper()}")
            
        except Exception as e:
            print(f"[MetaComm] Error closing backend: {e}")

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get meta protocol routing statistics."""
        total = self._routing_stats["total_messages"]
        
        return {
            "selected_protocol": self.selected_protocol,
            "total_messages": total,
            "successful_routes": self._routing_stats["successful_routes"],
            "fallback_used": self._routing_stats["fallback_used"],
            "routing_errors": self._routing_stats["routing_errors"],
            "success_rate": self._routing_stats["successful_routes"] / max(total, 1),
            "fallback_rate": self._routing_stats["fallback_used"] / max(total, 1),
            "error_rate": self._routing_stats["routing_errors"] / max(total, 1),
            "routing_confidence": self.routing_decision.confidence if self.routing_decision else 0.8
        }

    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered agent."""
        if agent_id not in self._agent_handles:
            return None
        
        handle = self._agent_handles[agent_id]
        return {
            "agent_id": handle.agent_id,
            "host": handle.host,
            "port": handle.port,
            "base_url": handle.base_url,
            "selected_protocol": handle.selected_protocol,
            "routing_confidence": handle.routing_confidence,
            "endpoint_registered": agent_id in self._endpoints
        }

    def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get information about all registered agents."""
        return [self.get_agent_info(agent_id) for agent_id in self._agent_handles.keys()]
