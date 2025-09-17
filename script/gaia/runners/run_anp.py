"""
ANP Runner for GAIA Multi-Agent System

该 Runner 负责：
1. 读取 GAIA 任务与协议配置
2. 创建并初始化 ANP 网络（Agent 创建与服务启动已在 ANPNetwork 内部完成）
3. 调用通用 RunnerBase 执行工作流
4. 统一的日志重定向（由 RunnerBase 提供）

使用方式：
    python -m script.gaia.runners.run_anp [config_path]

如果未提供 config_path，将尝试使用 script/gaia/config/anp.yaml
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
from protocol_backends.anp.network import ANPNetwork

# 基类 Runner（提供通用编排与日志重定向）
from .runner_base import RunnerBase


class ANPRunner(RunnerBase):
    """ANP 协议 Runner，实现 create_network 钩子。"""
    def __init__(self, config_path: str = "anp.yaml") -> None:
        super().__init__(config_path, protocol_name="anp")

    def create_network(self, general_config: Dict[str, Any]) -> ANPNetwork:
        """创建并返回 ANP 网络实例。"""
        try:
            print("ℹ️  Initializing NetworkBase and ANP Agents...")
            
            # Merge runner config (which has model info) into general_config
            if hasattr(self, 'config') and 'model' in self.config:
                general_config['model'] = self.config['model']
                print(f"🔧 Added model config to general_config: {self.config['model']['name']}")
            
            # Merge network configuration from runner config into planned config
            network_config = self.config.get("network", {})
            if network_config:
                general_config['network'] = network_config
                print(f"🔗 Added network config: timeout={network_config.get('timeout_seconds', 'default')}")
            
            network = ANPNetwork(config=general_config)
            print("🌐 ANP Network 已创建（内部已处理 Agent 初始化与 ANP Server 启动）")
            return network
        except Exception as e:
            print(f"❌ 创建 ANP Network 失败: {e}")
            raise


async def main():
    """ANP Runner 入口。"""
    runner = ANPRunner()
    try:
        await runner.run()
    except KeyboardInterrupt:
        print("\n🛑 ANP 运行被用户中断")
    except Exception as e:
        print(f"❌ ANP Runner 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
