"""
Agora Runner for GAIA Multi-Agent System

该 Runner 负责：
1. 读取 GAIA 任务与协议配置
2. 创建并初始化 Agora 网络（Agent 创建逻辑已在网络内部完成）
3. 调用通用 RunnerBase 执行工作流
4. 统一的日志重定向（由 RunnerBase 提供）

使用方式：
    python -m script.gaia.runners.run_agora [config_path] [optional_log_file_name]

如果未提供 config_path，将尝试使用 script/gaia/config/agora.yaml
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# 路径设置
HERE = Path(__file__).resolve().parent
GAIA_ROOT = HERE.parent
sys.path.insert(0, str(GAIA_ROOT))

# 协议网络导入（Agent 创建已在网络内部处理，不再在 runner 中创建）
from protocol_backends.agora.network import AgoraNetwork

# 基类 Runner
from runners.runner_base import RunnerBase


class AgoraRunner(RunnerBase):
    """Agora 协议 Runner，实现 create_network 钩子。"""
    def __init__(self, config_path: str = "agora.yaml") -> None:
        super().__init__(config_path, protocol_name="agora")

        # # 仅输出关键信息（其余逻辑在 network 内）
        # print("🔧 Agora Runner 初始化完成")
        # if self.config:
        #     print(f"📦 Agora 协议配置键: {list(self.config.keys())}")

    def create_network(self, general_config: Dict[str, Any]) -> AgoraNetwork:
        """
        创建并返回 Agora 网络实例。
        说明：Agent 的实例化/注册已移动到 AgoraNetwork 内部，不在此处处理。
        """
        try:
            print("ℹ️  Initializing NetworkBase and Agora Agents...")
            network = AgoraNetwork(config=general_config)
            print("🌐 Agora Network 已创建（内部已处理 Agent 初始化）")
            return network
        except Exception as e:
            print(f"❌ 创建 Agora Network 失败: {e}")
            raise


async def main():
    """Agora Runner 入口。"""
    runner = AgoraRunner()

    try:        
        runner = AgoraRunner()
        await runner.run()
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())