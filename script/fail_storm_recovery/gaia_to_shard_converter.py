#!/usr/bin/env python3
"""
Gaia to Shard Converter

Convert Gaia document into shard_qa compatible data files.
This allows us to use shard_qa's proven architecture directly without any adapters!

Creates:
- 8 shard data files (shard0.jsonl - shard7.jsonl) 
- Each file contains the same question but different document fragments
- Compatible with shard_qa's ring topology and protocols
"""

import json
import math
from pathlib import Path
from typing import List, Dict

def split_document_into_shards(document_content: str, num_shards: int = 8) -> List[str]:
    """Split document into roughly equal shards for distribution."""
    lines = document_content.strip().split('\n')
    lines_per_shard = math.ceil(len(lines) / num_shards)
    
    shards = []
    for i in range(num_shards):
        start_idx = i * lines_per_shard
        end_idx = min((i + 1) * lines_per_shard, len(lines))
        shard_content = '\n'.join(lines[start_idx:end_idx])
        shards.append(shard_content)
    
    return shards

def create_shard_data_files(
    gaia_doc_path: str,
    output_dir: str,
    base_question: str = "What are the key principles and applications of multi-agent systems?"
):
    """Create shard_qa compatible data files from Gaia document."""
    
    # Read Gaia document
    with open(gaia_doc_path, 'r', encoding='utf-8') as f:
        document_content = f.read()
    
    # Split into 8 shards
    shards = split_document_into_shards(document_content, 8)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # For each shard, create a data file
    for shard_idx, shard_content in enumerate(shards):
        shard_file = output_path / f"shard{shard_idx}.jsonl"
        
        # Create data entry in shard_qa format
        data_entry = {
            "group_id": 0,  # Single task for fail_storm scenario
            "question": base_question,
            "answer": f"Information about multi-agent systems from shard {shard_idx}",
            "snippet": shard_content.strip()
        }
        
        # Write as JSON Lines format (one line per group)
        with open(shard_file, 'w', encoding='utf-8') as f:
            json.dump(data_entry, f, ensure_ascii=False)
            f.write('\n')
        
        print(f"Created {shard_file} with {len(shard_content)} characters")
    
    print(f"\n✅ Successfully created 8 shard files in {output_dir}")
    print("🎯 You can now use shard_qa directly with these files!")

def create_fail_storm_config(output_path: str = "config_shard.yaml"):
    """Create shard_qa compatible config for fail_storm scenario."""
    
    config = {
        "core": {
            "base_url": "http://localhost:8000/v1",
            "max_tokens": 4096,
            "name": "gpt-4o",
            "openai_api_key": "sk-your-key-here",  # User needs to update this
            "openai_base_url": "https://api.openai.com/v1",
            "port": 8000,
            "temperature": 0.0,
            "type": "openai"
        },
        "data": {
            "agent_files": {
                f"shard{i}": f"data/gaia_shards/shard{i}.jsonl" for i in range(8)
            },
            "base_dir": "data/gaia_shards",
            "manifest_file": "data/gaia_shards/manifest.json",
            "version": "gaia_v1.0"
        },
        "environment": {
            "data_version": "gaia_v1.0",
            "group_id": 0
        },
        "network": {
            "health_check_interval": 5,
            "message_timeout": 30,
            "topology": "ring"
        },
        "prometheus": {
            "enabled": True,
            "metrics_path": "/metrics",
            "port": 8000
        },
        "shard_qa": {
            "coordinator": {
                "count": 1,
                "metrics": {
                    "avg_hop": True,
                    "first_answer_latency": True,
                    "msg_bytes_total": True,
                    "ttl_exhausted_total": True
                },
                "result_file": "data/gaia_shards/results.json",
                "start_port": 9998,
                "total_groups": 1  # Single Gaia task
            },
            "history": {
                "max_len": 20
            },
            "ring_config": {
                f"shard{i}": {
                    "next_id": f"shard{(i + 1) % 8}",
                    "prev_id": f"shard{(i - 1) % 8}"
                } for i in range(8)
            },
            "timeouts": {
                "max_retries": 3,
                "response_timeout": 30,
                "task_timeout": 60
            },
            "tool_schema": {
                "max_function_calls": 10,
                "max_ttl": 7
            },
            "workers": {
                "count": 8,
                "data_files": [f"script/fail_storm_recovery/data/gaia_shards/shard{i}.jsonl" for i in range(8)],
                "max_pending": 16,
                "start_port": 10001
            }
        },
        "system": {
            "agents_per_group": 8,
            "dataset_name": "Gaia Multi-Agent Document Processing",
            "description": "Fail-Storm recovery test using Gaia document processing with ring topology",
            "total_groups": 1
        }
    }
    
    # Write config file
    import yaml
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ Created fail_storm config: {output_path}")

if __name__ == "__main__":
    # Convert Gaia document to shard files
    gaia_doc = "docs/gaia_document.txt"
    output_dir = "data/gaia_shards"
    
    create_shard_data_files(gaia_doc, output_dir)
    create_fail_storm_config()
    
    print("\n🎉 Conversion complete!")
    print("📋 Next steps:")
    print("1. Update OpenAI API key in config_shard.yaml")
    print("2. Delete gaia_shard_adapter.py (not needed anymore)")
    print("3. Use shard_qa directly with the new config:")
    print("   python script/shard_qa/shard_qa.py --config script/fail_storm_recovery/config_shard.yaml")