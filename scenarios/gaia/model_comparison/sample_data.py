"""
Sample 10% of tasks from each difficulty level in GAIA dataset.

This script reads the GAIA validation metadata and creates a balanced sample
of 10% from each difficulty level (1, 2, 3) for model comparison experiments.
"""
import json
import random
from pathlib import Path
from collections import defaultdict


def load_metadata(metadata_path: str) -> list:
    """Load GAIA metadata from JSONL file."""
    data = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def sample_by_level(data: list, sample_ratio: float = 0.1, seed: int = 42) -> list:
    """
    Sample tasks by difficulty level.
    
    Args:
        data: List of task dictionaries
        sample_ratio: Ratio of tasks to sample from each level (default: 0.1 = 10%)
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled tasks
    """
    random.seed(seed)
    
    # Group tasks by level
    level_tasks = defaultdict(list)
    for task in data:
        level = task.get('Level', 1)
        level_tasks[level].append(task)
    
    # Sample from each level
    sampled_tasks = []
    for level in sorted(level_tasks.keys()):
        tasks = level_tasks[level]
        sample_size = max(1, int(len(tasks) * sample_ratio)) + 1
        sampled = random.sample(tasks, sample_size)
        sampled_tasks.extend(sampled)
        
        print(f"Level {level}: {len(tasks)} tasks -> sampled {len(sampled)} tasks")
    
    return sampled_tasks


def save_sampled_data(tasks: list, output_path: str):
    """Save sampled tasks to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')
    print(f"Saved {len(tasks)} sampled tasks to {output_path}")


def main():
    """Main sampling function."""
    # Paths
    base_dir = Path(__file__).parent.parent
    metadata_path = base_dir / "dataset" / "2023" / "validation" / "metadata.jsonl"
    output_path = Path(__file__).parent / "data" / "sampled_metadata.jsonl"
    
    # Check if input file exists
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        return
    
    # Load and sample data
    print(f"Loading metadata from {metadata_path}")
    data = load_metadata(str(metadata_path))
    print(f"Total tasks: {len(data)}")
    
    # Sample 10% from each level
    sampled_tasks = sample_by_level(data, sample_ratio=0.1)
    
    # Save sampled data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_sampled_data(sampled_tasks, str(output_path))
    
    # Print statistics
    print("\n=== Sampling Statistics ===")
    level_counts = defaultdict(int)
    for task in sampled_tasks:
        level = task.get('Level', 1)
        level_counts[level] += 1
    
    for level in sorted(level_counts.keys()):
        print(f"Level {level}: {level_counts[level]} tasks")
    print(f"Total sampled: {len(sampled_tasks)} tasks")


if __name__ == "__main__":
    main()
