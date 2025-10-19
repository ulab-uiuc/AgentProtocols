#!/usr/bin/env python3
"""
Simple Data Loader for Medical Q&A Dataset
Focused on loading medical questions for privacy testing scenario.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SimpleDataLoader:
    """Simple loader for medical conversation data."""

    def __init__(self, data_dir: str = "data"):
        """Initialize with data directory."""
        self.data_dir = Path(data_dir)
        self.medical_file = self.data_dir / "medical_conversations_100.json"

    def load_medical_conversations(self) -> list:
        """Load medical conversations from local file."""
        if not self.medical_file.exists():
            logger.error(f"Medical data file not found: {self.medical_file}")
            return []

        try:
            with open(self.medical_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different data formats
            if isinstance(data, dict):
                # Check if it has a 'samples' key or similar
                if 'samples' in data:
                    conversations = data['samples']
                elif 'data' in data:
                    conversations = data['data']
                else:
                    # Look for the actual conversation data
                    conversations = []
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            conversations = value
                            break
            elif isinstance(data, list):
                conversations = data
            else:
                conversations = []

            logger.info(f"Loaded {len(conversations)} medical conversations")
            return conversations
        except Exception as e:
            logger.error(f"Failed to load medical data: {e}")
            return []

    def get_random_question(self) -> dict:
        """Get a random medical question."""
        data = self.load_medical_conversations()
        if not data:
            return {}

        import random
        return random.choice(data)
