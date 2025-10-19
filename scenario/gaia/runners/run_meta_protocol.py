"""
Meta Protocol Runner for GAIA Framework.
Implements intelligent protocol selection similar to fail_storm_recovery.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# 路径设置
HERE = Path(__file__).resolve().parent
GAIA_ROOT = HERE.parent
sys.path.insert(0, str(GAIA_ROOT))

# 协议网络导入
from protocol_backends.meta_protocol.network import MetaProtocolNetwork

# 基类 Runner
from runners.runner_base import RunnerBase


class MetaProtocolRunner(RunnerBase):
    """Meta Protocol Runner，实现 create_network 钩子。"""
    
    def __init__(self, config_path: str = "meta_protocol.yaml") -> None:
        super().__init__(config_path, protocol_name="meta_protocol")

    def create_network(self, general_config: Dict[str, Any]) -> MetaProtocolNetwork:
        """
        创建并返回 Meta Protocol 网络实例。
        集成智能协议选择和跨协议通信能力。
        """
        try:
            print("ℹ️  Initializing Meta Protocol Network with intelligent routing...")
            
            # Extract task_id from general_config
            task_id = general_config.get("task_id", "meta_protocol_task")
            
            # Create meta protocol network
            network = MetaProtocolNetwork(general_config, task_id)
            
            # Register agents from config (LLM routing will happen during network.start())
            network.register_agents_from_config()
            
            print(f"✅ Meta Protocol network created for task: {task_id}")
            print(f"📊 Available protocols: {network._available_protocols}")
            
            return network
            
        except Exception as e:
            print(f"❌ Failed to create Meta Protocol network: {e}")
            raise


# Main execution
async def main():
    """Main execution for Meta Protocol runner."""
    import sys
    
    # 确定配置文件路径
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = GAIA_ROOT / "config" / "meta_protocol.yaml"
    
    # 确定日志文件名
    if len(sys.argv) > 2:
        log_file_name = sys.argv[2]
    else:
        log_file_name = None
    
    print(f"🚀 Starting GAIA Meta Protocol Runner")
    print(f"📋 Config: {config_path}")
    print(f"📊 Meta Protocol: Intelligent routing enabled")
    
    # 创建并运行
    runner = MetaProtocolRunner(str(config_path))
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())