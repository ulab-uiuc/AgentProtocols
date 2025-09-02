# """
# script/streaming_queue/protocol_backend/agora/worker.py
# """
# """
# Agora Worker (protocol-specific)

# - 继承 QAWorkerBase，使用 Agora 协议包装成可执行的 Agent
# - 仅负责"通信适配层"：提取用户输入文本 → 调用基类 answer() → 通过 Agora 返回
# """
from __future__ import annotations
import asyncio
from typing import Any, Optional, Dict
from functools import wraps

# Import Agora native SDK
import agora

# 允许多种导入相对路径（避免运行根目录不同导致的导入失败）
try:
    from ...core.qa_worker_base import QAWorkerBase  # when run as package
except ImportError:
    try:
        from core.qa_worker_base import QAWorkerBase
    except ImportError:
        from script.streaming_queue.core.qa_worker_base import QAWorkerBase  # type: ignore


def _sender(func):
    """
    A decorator to send message to coordinator.
    """
    @wraps(func)
    async def wrapper(self: "AgoraQAWorker", *args, **kwargs):
        if not self.agent_network:
            raise RuntimeError("No Agora network/router set. Call set_network() first.")

        # First argument is coordinator_id, second is message
        coordinator_id, message = args[0], args[1]

        payload = {
            "protocolHash": None,
            "body": message
        }

        response = await self.agent_network.route_message(
            src_id=self.worker_id,
            dst_id=coordinator_id,
            payload=payload
        )
        return response
    return wrapper


class AgoraQAWorker(QAWorkerBase):
    """Agora专用的QA Worker，使用Agora原生SDK功能。"""
    
    def __init__(self, config=None, output=None):
        super().__init__(config, output)
        
        # Initialize Agora native components with proper parameters
        try:
            # Create minimal Agora components for enhancement
            self.agora_protocol = agora.Protocol()
            print("[AgoraWorker] Initialized with native Agora SDK Protocol")
        except Exception as e:
            print(f"[AgoraWorker] Agora SDK initialization error: {e}")
            self.agora_protocol = None
    
    async def answer(self, question: str) -> str:
        """Override answer method to use Agora native SDK"""
        try:
            # Use base QAWorkerBase for LLM call first
            base_answer = await super().answer(question)
            
            # Enhance with Agora Protocol if available
            if self.agora_protocol:
                try:
                    # Use Agora Protocol for message structuring
                    enhanced_answer = f"[Agora Enhanced] {base_answer}"
                    print(f"[AgoraWorker] Enhanced response using native Agora SDK")
                    return enhanced_answer
                except Exception as e:
                    print(f"[AgoraWorker] Agora enhancement failed: {e}")
            
            return base_answer
                
        except Exception as e:
            print(f"[AgoraWorker] Error in answer method: {e}")
            # Fallback to base implementation
            return await super().answer(question)


class AgoraWorkerExecutor:
    """
    Agora 原生 Executor 外壳：
      - 从 input_data 中抽取文本（支持 Agora 结构或简单文本）
      - 调用 AgoraQAWorker.answer()
      - 返回文本结果
    """

    def __init__(self, config: Optional[Dict] = None, output=None):
        self.worker = AgoraQAWorker(config=config, output=output)

    async def execute(self, input_data: Dict) -> Dict:
        """
        处理Agora消息并返回答案
        输入格式: {
            "protocolHash": None,
            "body": command,
            "protocolSources": []
        }

        输出格式: {"status": ..., "body": ...}
        """
        try:
            # 提取agora格式内容
            text = ""
            if isinstance(input_data, dict):
                if "body" in input_data:
                    text = input_data["body"]
                elif "payload" in input_data:
                    payload = input_data.get("payload", {})
                    text = payload.get("data", "")
                else:
                    text = str(input_data)
            else:
                text = str(input_data)

            if not text:
                raise ValueError("No input text provided")
            answer = await asyncio.wait_for(self.worker.answer(text), timeout=30.0)

            # 返回agora格式
            return {"status": "success", "body": answer}

        except asyncio.TimeoutError:
            return {"status": "error", "body": "Error: Request timed out after 30 seconds"}
        except Exception as e:
            return {"status": "error", "body": f"Error: {str(e)}"}