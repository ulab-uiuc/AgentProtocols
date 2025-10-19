"""
Dummy Runner for GAIA Multi-Agent System

该 Runner 负责：
1. 读取 GAIA 任务与协议配置
2. 创建并初始化 Dummy 网络（Agent 创建逻辑目前仍在 DummyNetwork 内部）
3. 调用通用 RunnerBase 执行工作流
4. 统一的日志重定向（由 RunnerBase 提供）

使用方式：
    python -m script.gaia.runners.run_dummy [config_path]

如果未提供 config_path，将尝试使用 script/gaia/config/a2a.yaml (暂复用 a2a 配置，后续可独立 dummy.yaml)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, Any
import sys

# 路径设置
HERE = Path(__file__).resolve().parent
GAIA_ROOT = HERE.parent
sys.path.insert(0, str(GAIA_ROOT))

# 协议网络导入（Agent 创建已在网络内部处理，不在 runner 中创建）
from protocol_backends.dummy.network import DummyNetwork

# 基类 Runner
from .runner_base import RunnerBase


class DummyRunner(RunnerBase):
    """Dummy 协议 Runner，实现 create_network 钩子。"""
    def __init__(self, config_path: str = "dummy.yaml") -> None:
        super().__init__(config_path, protocol_name="dummy")

    def create_network(self, general_config: Dict[str, Any]) -> DummyNetwork:
        """创建并返回 Dummy 网络实例。"""
        try:
            network = DummyNetwork(config=general_config)
            print("🌐 Dummy Network 已创建（内部已处理 Agent 初始化）")
            return network
        except Exception as e:
            print(f"❌ 创建 Dummy Network 失败: {e}")
            raise


async def main():
    """Dummy Runner 入口。"""
    runner = DummyRunner()
    try:
        await runner.run()
    except KeyboardInterrupt:
        print("\n🛑 Dummy 运行被用户中断")
    except Exception as e:
        print(f"❌ Dummy Runner 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
