#!/usr/bin/env python3
"""
Agora Agent implementation for Fail-Storm Recovery scenario.

本模块参照 ACP 版本 (`acp/agent.py`) 的结构，实现 Agora 协议下的 Agent 封装，
使用 `SimpleBaseAgent` 作为底层服务能力，并提供一个 `AgoraExecutorWrapper` 来
适配 shard worker 的执行接口到（未来可扩展的）Agora 协议风格执行流程。

与 ACP 版本的差异：
1. 类/日志前缀/Docstring 全部替换为 Agora 语义。
2. 新增 `_extract_text_from_agora_response` 方法，用于按用户指定的多种可能字段
   提取文本：body / text / result(body) / data 等；失败回退为字符串化。
3. 产出的内容保持最小必要实现，便于后续真正接入官方 Agora SDK 时扩展。

使用场景：Fail-Storm Recovery 运行器会调用 `create_agora_agent` 生成一个基础的
`SimpleBaseAgent`（同 ACP 的 `create_acp_agent` 逻辑），其 executor 由上层传入。
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, Optional, List, AsyncGenerator
from pathlib import Path
import sys

# 将 fail_storm_recovery core 路径加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 引入 SimpleBaseAgent
from core.simple_base_agent import SimpleBaseAgent as BaseAgent

# （如果未来有官方 Agora SDK，可在此处尝试导入并做可选处理）


class AgoraExecutorWrapper:
	"""将 ShardWorkerExecutor 适配为 Agora 协议执行器接口的包装器。

	参考 ACPExecutorWrapper 结构；当前先简单处理文本消息并调用 worker.answer()。
	预期未来官方 Agora SDK 若需要特定的消息 / 流式 RunYield，可在此扩展。
	"""

	def __init__(self, shard_worker_executor: Any):
		self.shard_worker_executor = shard_worker_executor
		self.capabilities = ["text_processing", "async_generation", "agora_proto_simplified"]
		self.logger = logging.getLogger("AgoraExecutorWrapper")

	def _extract_text_from_agora_response(self, response: Dict[str, Any]) -> str:
		"""Extract text content from Agora response.

		规则（按优先级）：
		1. 若 response 本身是 str，直接返回。
		2. 若为 dict，依次尝试字段：body / text / result(body) / data。
		3. 若都不命中，转成字符串返回；异常时返回空串。
		"""
		try:
			if isinstance(response, str):
				return response
			elif isinstance(response, dict):
				if "body" in response:
					return response["body"]
				elif "text" in response:
					return response["text"]
				elif "result" in response:
					result = response["result"]
					if isinstance(result, str):
						return result
					elif isinstance(result, dict) and "body" in result:
						return result["body"]
				elif "data" in response:
					return str(response["data"])
				else:
					return str(response)
			return ""
		except Exception:
			return ""

	async def __call__(self, messages: List[Any], context: Optional[Any] = None) -> AsyncGenerator[Any, None]:
		"""简化的 Agora 执行入口。

		Args:
			messages: 传入的消息列表（当前假定为包含文本的最小结构）。
			context: 预留上下文（未使用）。

		Yields:
			单次字符串结果（模拟 RunYield）。
		"""
		self.logger.debug(f"[Agora] AgoraExecutorWrapper called with {len(messages)} messages")

		try:
			for i, message in enumerate(messages):
				run_id = str(uuid.uuid4())
				self.logger.debug(f"[Agora] Processing message {i+1}/{len(messages)} run_id={run_id}")

				# 提取文本内容（支持多种结构：dict / str / {parts:[{type:'text',text:'...'}]}）
				text_content = ""
				try:
					if isinstance(message, dict):
						# 常见结构：{"parts": [...]}
						if "parts" in message and isinstance(message["parts"], list):
							for part in message["parts"]:
								if isinstance(part, dict) and part.get("type") == "text":
									text_content += part.get("text") or part.get("content", "")
						elif "body" in message:
							text_content = message["body"]
						elif "text" in message:
							text_content = message["text"]
						else:
							text_content = json.dumps(message, ensure_ascii=False)
					else:
						text_content = str(message)
				except Exception as e:
					self.logger.warning(f"[Agora] Failed to parse message content: {e}")
					text_content = str(message)

				# 调用 shard worker 处理
				if self.shard_worker_executor and hasattr(self.shard_worker_executor, 'worker'):
					try:
						worker = self.shard_worker_executor.worker
						if hasattr(worker, 'answer'):
							raw_result = await worker.answer(text_content)
						else:
							raw_result = await worker.start_task(0)

						# 统一提取文本
						extracted = self._extract_text_from_agora_response(raw_result if isinstance(raw_result, dict) else raw_result)
						yield extracted or (str(raw_result) if raw_result is not None else "No result")
						self.logger.debug(f"[Agora] Result len={len((extracted or str(raw_result or '')))}")
					except Exception as e:
						self.logger.error(f"[Agora] Execution error: {e}")
						yield f"Execution error: {e}"
				else:
					yield "Shard worker executor not available"

		except Exception as e:
			error_msg = f"Agora processing error: {e}"
			self.logger.error(f"[Agora] Executor error: {e}", exc_info=True)
			yield error_msg


async def create_agora_agent(agent_id: str, host: str, port: int, executor: Any) -> BaseAgent:
	"""创建 Agora 协议 Agent（简化版）。

	与 ACP 版本一致：调用 `SimpleBaseAgent.create_agora` 返回一个基础 HTTP 服务，
	上层 runner 会使用该 agent 的 /message 接口。这里不直接耦合官方 Agora SDK，
	便于后续平滑替换。

	Args:
		agent_id: 唯一 Agent ID
		host: 绑定主机
		port: 端口
		executor: ShardWorkerExecutor 实例

	Returns:
		配置完成的 `SimpleBaseAgent` 实例
	"""
	agent = await BaseAgent.create_agora(
		agent_id=agent_id,
		host=host,
		port=port,
		executor=executor
	)
	return agent

