"""
Extensions to NetworkBase for concurrent processing.
"""

import asyncio
import logging
from typing import Dict, List

class NetworkBaseExtensions:
    """Mixin class to add concurrent processing capabilities to NetworkBase"""
    
    async def _cleanup_expired_reservations(self):
        """Periodically clean up expired reservations"""
        while self.is_running:
            try:
                if hasattr(self, 'conflict_manager'):
                    self.conflict_manager.cleanup_expired_reservations()
            except Exception as e:
                self.logger.error(f"Error during reservation cleanup: {e}")
            
            await asyncio.sleep(10.0)  # Clean up every 10 seconds
    
