# script/streaming_queue/protocol_backend/anp/coordinator.py
"""
ANP Coordinator:
- 继承 QACoordinatorBase
- 用 ANP 协议实现 send_to_worker()
- 同时提供 QACoordinator ANP Executor，便于把协调器暴露为 ANP Agent
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

# 允许多种导入相对路径（避免运行根目录不同导致的导入失败）
try:
    from ...core.qa_coordinator_base import QACoordinatorBase
except ImportError:
    try:
        from core.qa_coordinator_base import QACoordinatorBase
    except ImportError:
        from script.streaming_queue.core.qa_coordinator_base import QACoordinatorBase


# ============ ANP Coordinator ============

class ANPQACoordinator(QACoordinatorBase):
    """
    使用 ANP 协议与 worker 通信。
    约定 self.agent_network 暴露一个协议信道方法：
        await agent_network.route_message(src_id, dst_id, payload) -> Dict
    其中 payload 为 ANP 的标准消息（见实现）。
    """

    async def send_to_worker(self, worker_id: str, question: str) -> Dict[str, Any]:
        if not self.agent_network:
            error_msg = "No ANP network/router set. Call set_network() first."
            raise RuntimeError(error_msg)

        # ANP 消息格式：简单的文本载荷
        payload = {"text": question}

        # 通过 ANP 路由发送（接口与原代码保持一致）
        response = await self.agent_network.route_message(self.coordinator_id, worker_id, payload)

        # 解析 ANP 响应，提取文本答案
        answer: Optional[str] = None
        try:
            if isinstance(response, dict):
                # 首先尝试直接获取 text 字段
                if "text" in response:
                    answer = response["text"]
                # 否则尝试从嵌套结构中提取（兼容ANP消息格式）
                elif "raw" in response:
                    raw_data = response["raw"]
                    if isinstance(raw_data, dict):
                        payload = raw_data.get("payload", {})
                        answer = payload.get("data", "")
        except Exception:
            answer = None

        result = {
            "answer": answer or "No answer received",
            "raw": response,
        }
        return result


# ============ ANP Executor（把 Coordinator 暴露为 ANP Agent） ============

class ANPCoordinatorExecutor:
    """
    ANP 原生 Executor，用于对话式控制协调器。
    支持命令：
      - "dispatch" / "start_dispatch"
      - "status"
      - "setup_network Worker-1,Worker-2,Worker-3,Worker-4"
    """

    def __init__(self, config: Dict | None = None, output=None):
        self.coordinator = ANPQACoordinator(config, output)

    async def execute(self, input_data: Dict) -> Dict:
        """
        处理ANP消息并执行协调器命令
        输入格式可能是：
        1. {"text": "command"}  # 简单格式
        2. ANP消息格式的payload
        """
        # 从输入中提取命令文本
        user_text = ""
        if isinstance(input_data, dict):
            # 支持简单的text格式
            if "text" in input_data:
                user_text = input_data["text"]
            # 支持ANP消息格式
            elif "payload" in input_data:
                payload = input_data.get("payload", {})
                user_text = payload.get("data", "status")
            else:
                user_text = "status"
        else:
            user_text = str(input_data)

        cmd = user_text.strip().lower()
        original_text = user_text.strip()  # 保留原始大小写用于worker IDs

        try:
            if cmd == "dispatch":
                result = await self.coordinator.dispatch_round()
            elif cmd.startswith("setup_network"):
                # 命令格式: "setup_network Worker-1,Worker-2,Worker-3,Worker-4"
                try:
                    worker_list = original_text.split(" ", 1)[1] if " " in original_text else ""
                    worker_ids = [w.strip() for w in worker_list.split(",") if w.strip()]

                    # 设置协调器网络访问
                    self.coordinator.worker_ids = worker_ids
                    self.coordinator.coordinator_id = "Coordinator-1"

                    result = f"ANP network setup complete with {len(worker_ids)} workers: {worker_ids}"
                except Exception as e:
                    result = f"ANP network setup failed: {e}"
            else:
                result = await self.coordinator.get_status()

            # 返回简单的文本格式供ANP CommBackend包装
            return {"text": result}

        except Exception as e:
            return {"text": f"ANP Coordinator error: {str(e)}"}


class ANPCoordinatorHandle:
    """ANP协调器句柄，用于管理协调器生命周期"""
    def __init__(self, executor: ANPCoordinatorExecutor, base_url: str):
        self.executor = executor
        self.base_url = base_url
        
    async def stop(self):
        """停止协调器（如果需要清理资源）"""
        pass
