#!/usr/bin/env python3
"""
Medical Data Manager for Privacy Protocol Testing
Loads medical Q&A dataset for privacy-focused agent communication testing.
"""

import json
import os
import logging
from pathlib import Path
from datasets import load_dataset

logger = logging.getLogger(__name__)


class MedicalDataManager:
    """Manages medical Q&A dataset for privacy testing."""

    def __init__(self, data_dir: str = "data"):
        """Initialize with data directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Medical dataset info
        self.medical_file = self.data_dir / "medical_conversations_100.json"

    def download_and_store_medical_data(self, num_samples: int = 100) -> bool:
        """Download and store medical Q&A data locally."""
        try:
            logger.info("Starting medical dataset download...")

            # Download medical Q&A dataset
            dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")

            # Take first num_samples
            medical_samples = []
            count = 0
            for item in dataset:
                if count >= num_samples:
                    break

                # Extract Q&A format
                if 'instruction' in item and 'output' in item:
                    medical_samples.append({
                        'id': count + 1,
                        'question': item['instruction'],
                        'answer': item['output'],
                        'input': item.get('input', '')
                    })
                    count += 1

            # Save to local file
            with open(self.medical_file, 'w', encoding='utf-8') as f:
                json.dump(medical_samples, f, indent=2, ensure_ascii=False)

            logger.info(f"Medical dataset saved: {len(medical_samples)} samples")
            return True

        except Exception as e:
            logger.error(f"Failed to download medical dataset: {e}")
            return False

    def load_medical_data(self) -> list:
        """Load medical data from local file."""
        if not self.medical_file.exists():
            logger.info("Medical data file not found, downloading...")
            if not self.download_and_store_medical_data():
                logger.error("Failed to download medical data")
                return []

        try:
            with open(self.medical_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} medical Q&A pairs")
            return data
        except Exception as e:
            logger.error(f"Failed to load medical data: {e}")
            return []

    def get_medical_sample(self, index: int = None) -> dict:
        """Get a single medical Q&A sample."""
        data = self.load_medical_data()
        if not data:
            return {}

        if index is None:
            import random
            index = random.randint(0, len(data) - 1)
        elif index >= len(data):
            index = len(data) - 1

        return data[index]
