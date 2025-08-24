# script/streaming_queue/protocol_backend/acp/coordinator.py
"""
ACP Coordinator:
- 继承 QACoordinatorBase
- 用 ACP 原生 SDK 实现 send_to_worker()
- 同时提供 QACoordinator ACP Executor，便于把协调器暴露为 ACP Agent
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from ...core.qa_coordinator_base import QACoordinatorBase

# --- ACP 原生 SDK（可选依赖） ---
try:
    from acp_sdk import Client
    from acp_sdk.models import RunCreateRequest, Input
    ACP_AVAILABLE = True
except ImportError:
    # 兜底（没有 acp-sdk 也能导入，便于单元测试）
    ACP_AVAILABLE = False
    Client = None
    RunCreateRequest = None
    Input = None


# ============ ACP Coordinator ============

class ACPQACoordinator(QACoordinatorBase):
    """
    使用 ACP 协议与 worker 通信。
    约定 self.agent_network 暴露一个协议信道方法：
        await agent_network.route_message(src_id, dst_id, payload) -> Dict
    其中 payload 为 ACP 的标准消息（见实现）。
    """

    async def send_to_worker(self, worker_id: str, question: str) -> Dict[str, Any]:
        if not self.agent_network:
            error_msg = "No ACP network/router set. Call set_network() first."
            raise RuntimeError(error_msg)

        # ACP 用户消息格式
        payload = {
            "input": {
                "content": [{"type": "text", "text": question}]
            }
        }

        # 通过 ACP 路由发送（接口与原代码保持一致）
        response = await self.agent_network.route_message(self.coordinator_id, worker_id, payload)

        # 解析 ACP 响应，提取文本答案
        answer: Optional[str] = None
        try:
            if isinstance(response, dict):
                # 首先尝试直接获取 text 字段
                if "text" in response:
                    answer = response["text"]
                # 否则尝试从嵌套结构中提取
                elif "raw" in response:
                    raw_data = response["raw"]
                    if isinstance(raw_data, dict):
                        output = raw_data.get("output", {})
                        content = output.get("content", [])
                        if content and isinstance(content[0], dict):
                            answer = content[0].get("text", "")
        except Exception:
            answer = None

        result = {
            "answer": answer or "No answer received",
            "raw": response,
        }
        return result


# ============ ACP Executor（把 Coordinator 暴露为 ACP Agent） ============

class ACPCoordinatorExecutor:
    """
    ACP 原生 Executor，用于对话式控制协调器。
    支持命令：
      - "dispatch" / "start_dispatch"
      - "status"
      - "setup_network Worker-1,Worker-2,Worker-3,Worker-4"
    """

    def __init__(self, config: Dict | None = None, output=None):
        self.coordinator = ACPQACoordinator(config, output)

    async def execute(self, input_data: Dict) -> Dict:
        # ACP input: Extract text from message parts
        parts = input_data.get("content", [])
        user_text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text") or "status"
        cmd = user_text.strip().lower()

        # For setup_network, preserve the original case for worker IDs
        original_text = user_text.strip()

        if cmd == "dispatch":
            result = await self.coordinator.dispatch_round()
        elif cmd.startswith("setup_network"):
            # Command format: "setup_network Worker-1,Worker-2,Worker-3,Worker-4"
            # Use original_text to preserve case
            try:
                worker_list = original_text.split(" ", 1)[1] if " " in original_text else ""
                worker_ids = [w.strip() for w in worker_list.split(",") if w.strip()]

                # Set up the coordinator with network access
                self.coordinator.worker_ids = worker_ids
                self.coordinator.coordinator_id = "Coordinator-1"

                result = f"Network setup complete with {len(worker_ids)} workers: {worker_ids}"
            except Exception as e:
                result = f"Network setup failed: {e}"
        else:
            result = await self.coordinator.get_status()

        # Return ACP output format
        return {"content": [{"type": "text", "text": result}]}
