"""
Base server adapter for protocol-specific server implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import uvicorn


class BaseServerAdapter(ABC):
    """Abstract base class for protocol-specific server adapters."""
    
    protocol_name: str
    
    @abstractmethod
    def build(
        self,
        host: str,
        port: int,
        agent_id: str,
        executor: Any,
        **kwargs
    ) -> Tuple[uvicorn.Server, Dict[str, Any]]:
        """
        Build and configure a protocol-specific server.
        
        Args:
            host: Server host address
            port: Server port number
            agent_id: Unique agent identifier
            executor: Business logic executor
            **kwargs: Additional protocol-specific configuration
            
        Returns:
            Tuple of (uvicorn.Server instance, agent_card dict)
        """
        pass 