"""
ACP Runner for GAIA Multi-Agent System

该 Runner 负责：
1. 读取 GAIA 任务与协议配置
2. 创建并初始化 ACP 网络（Agent 创建与服务启动已在 ACPNetwork 内部完成）
3. 调用通用 RunnerBase 执行工作流
4. 统一的日志重定向（由 RunnerBase 提供）

使用方式：
    python -m script.gaia.runners.run_acp [config_path]

如果未提供 config_path，将尝试使用 script/gaia/config/acp.yaml
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

# 协议网络导入（Agent 创建与服务启动在网络内部处理）
from protocol_backends.acp.network import ACPNetwork

# 基类 Runner（提供通用编排与日志重定向）
from .runner_base import RunnerBase


class ACPRunner(RunnerBase):
    """ACP 协议 Runner，实现 create_network 钩子。"""
    def __init__(self, config_path: str = "acp.yaml") -> None:
        super().__init__(config_path, protocol_name="acp")

    def create_network(self, general_config: Dict[str, Any]) -> ACPNetwork:
        """创建并返回 ACP 网络实例。"""
        try:
            print("ℹ️  Initializing NetworkBase and ACP Agents...")
            network = ACPNetwork(config=general_config)
            print("🌐 ACP Network 已创建（内部已处理 Agent 初始化与 ACP Server 启动）")
            return network
        except Exception as e:
            print(f"❌ 创建 ACP Network 失败: {e}")
            raise


async def main():
    """ACP Runner 入口。"""
    runner = ACPRunner()
    try:
        await runner.run()
    except KeyboardInterrupt:
        print("\n🛑 ACP 运行被用户中断")
    except Exception as e:
        print(f"❌ ACP Runner 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
