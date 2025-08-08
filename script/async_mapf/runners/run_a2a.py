# script/async_mapf/runners/run_a2a.py
import asyncio
import logging
from pathlib import Path

from script.async_mapf.protocol_backends.a2a.runner import A2ARunner

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Default to script/async_mapf/config/demo.yml
    config_path = Path(__file__).resolve().parents[1] / "config" / "demo.yml"

    try:
        asyncio.run(A2ARunner(config_path).run())
    except KeyboardInterrupt:
        pass
