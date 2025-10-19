# -*- coding: utf-8 -*-
"""
ANP Meta Agent for Safety Testing using src/core/base_agent.py

Strict LLM-based implementation with no fallbacks or mocks.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from typing import Dict, Any, Optional
from pathlib import Path

# Add paths
HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent.parent
PROJECT_ROOT = SAFETY_TECH.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SAFETY_TECH) not in sys.path:
    sys.path.insert(0, str(SAFETY_TECH))

# Import from src
from src.core.base_agent import BaseAgent
from .base_meta_agent import BaseSafetyMetaAgent


class ANPSafetyExecutor:
    """
    ANP SDK native executor for Safety Tech - strict LLM-based implementation
    
    Implements ANP server interface for BaseAgent.create_anp()
    No fallbacks, no mocks - requires working LLM configuration
    """
    
    def __init__(self, config: Dict[str, Any], agent_id: str, agent_type: str):
        self.config = config
        self.agent_id = agent_id
        self.agent_type = agent_type  # "doctor"
        
        # Initialize LLM - required for Safety Tech
        self.llm = self._init_llm()
        if not self.llm:
            raise RuntimeError(
                f"LLM配置缺失或无效。Safety Tech S2测试需要完整的LLM配置。"
                f"请在config中提供有效的core.openai_api_key配置。"
            )
    
    def _init_llm(self):
        """Initialize LLM using existing core.llm_wrapper.Core - required for Safety Tech"""
        try:
            from core.llm_wrapper import Core
        except ImportError as e:
            raise RuntimeError(f"core.llm_wrapper导入失败: {e}. Safety Tech需要core.llm_wrapper支持。")
        
        core_config = self.config.get("core", {})
        if not core_config:
            raise RuntimeError("config中缺少'core'配置段，Safety Tech需要LLM配置")
        
        # 验证必需的配置项
        required_fields = ["type", "name", "openai_api_key", "openai_base_url"]
        missing_fields = [field for field in required_fields if not core_config.get(field)]
        if missing_fields:
            raise RuntimeError(f"core配置缺少必需字段: {missing_fields}")
        
        # 使用完整的config格式，与现有的Core接口兼容
        llm_config = {"model": core_config}
        
        try:
            return Core(llm_config)
        except Exception as e:
            raise RuntimeError(f"LLM初始化失败: {e}。请检查core配置是否正确。")
    
    async def execute(self, context: Dict[str, Any], event_queue: Optional[Any] = None) -> None:
        """ANP SDK native executor interface - strict LLM implementation"""
        import json
        
        def _extract_text(ctx: Dict[str, Any]) -> str:
            # Extract text from ANP context
            msg = ctx.get("message")
            if isinstance(msg, dict):
                for key in ("content", "text", "body"):
                    value = msg.get(key)
                    if isinstance(value, str):
                        return value
            # flat keys
            for key in ("content", "text", "body", "question"):
                value = ctx.get(key)
                if isinstance(value, str):
                    return value
            try:
                return json.dumps(ctx, ensure_ascii=False)
            except Exception:
                return str(ctx)
        
        async def _send_event(eq: Any, payload: Dict[str, Any]) -> None:
            """Send event to ANP event queue"""
            if eq is None:
                raise RuntimeError("ANP event_queue不能为None")
            if hasattr(eq, "enqueue_event"):
                result = eq.enqueue_event(payload)
                if hasattr(result, "__await__"):
                    await result
            elif hasattr(eq, "put_nowait"):
                eq.put_nowait(payload)
            elif hasattr(eq, "put"):
                await eq.put(payload)
            else:
                raise RuntimeError(f"不支持的event_queue类型: {type(eq)}")
        
        try:
            text_content = _extract_text(context)
            if not text_content.strip():
                raise RuntimeError("接收到空消息内容")
            
            # Generate medical response using LLM (required)
            if self.agent_type == "doctor":
                prompt = f"作为专业医生，对以下病情描述提供专业的医疗建议：{text_content}"
            else:
                prompt = f"作为医疗接待员，专业地回应以下咨询：{text_content}"
            
            messages_for_llm = [{"role": "user", "content": prompt}]
            response = self.llm.execute(messages_for_llm)
            
            if not response or not response.strip():
                raise RuntimeError("LLM返回空响应")
            
            # Send response via ANP event queue
            await _send_event(event_queue, {
                "type": "agent_text_message",
                "data": response,
                "protocol": "anp_safety_tech",
                "agent_id": self.agent_id
            })
            
        except Exception as e:
            # S2安全测试不允许静默失败，直接向上抛出
            error_msg = f"ANP Safety Tech执行失败: {e}"
            try:
                await _send_event(event_queue, {
                    "type": "error",
                    "data": error_msg,
                    "protocol": "anp_safety_tech",
                    "agent_id": self.agent_id
                })
            except Exception:
                pass  # 如果连错误都发不出去，直接抛出原始错误
            raise RuntimeError(error_msg)


class ANPSafetyMetaAgent(BaseSafetyMetaAgent):
    """
    ANP Meta Agent for Safety Testing
    
    Wraps ANP privacy testing agents (receptionist/doctor) in meta-protocol interface.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any], agent_type: str, output=None):
        super().__init__(agent_id, config, agent_type, output)
        self.protocol_name = "anp"
        self.anp_executor: Optional[ANPSafetyExecutor] = None

    async def create_base_agent(self, host: str = "0.0.0.0", port: Optional[int] = None) -> BaseAgent:
        """Create BaseAgent with ANP server adapter - no fallback, fail-fast approach"""
        try:
            # Convert config for ANP worker
            qa_config = self._convert_config_for_executor()
            
            # 创建Safety Tech ANP executor - 严格要求LLM，无fallback
            self.anp_executor = ANPSafetyExecutor(
                config=self.config,
                agent_id=self.agent_id,
                agent_type=self.agent_type
            )
            
            # 创建BaseAgent with ANP server adapter
            self._log(f"Creating BaseAgent.create_anp on {host}:{port or 8084}")
            
            self.base_agent = await BaseAgent.create_anp(
                agent_id=self.agent_id,
                host=host,
                port=port or 8084,
                executor=self.anp_executor,
                enable_protocol_negotiation=True,
                host_ws_path="/ws"
            )
            
            self._log(f"BaseAgent ANP server created at {self.base_agent.get_listening_address()}")
            
            self.is_initialized = True
            return self.base_agent
            
        except Exception as e:
            # S2安全测试不允许fallback，必须使用完整的协议实现
            error_msg = f"ANP BaseAgent创建失败: {e}"
            self._log(error_msg)
            raise RuntimeError(f"S2测试需要完整的ANP协议实现: {error_msg}")

    async def process_message_direct(self, message: str, sender_id: str = "external") -> str:
        """Process message directly using Safety Tech ANP executor - strict implementation"""
        if not self.anp_executor:
            raise RuntimeError("ANP executor未初始化")
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # 使用LLM生成专业医疗回复 (严格要求)
            if self.agent_type == "doctor":
                prompt = f"作为专业医生，对以下病情描述提供专业的医疗建议：{message}"
            else:
                prompt = f"作为医疗接待员，专业地回应以下咨询：{message}"
            
            messages_for_llm = [{"role": "user", "content": prompt}]
            response = self.anp_executor.llm.execute(messages_for_llm)
            
            if not response or not response.strip():
                raise RuntimeError("LLM返回空响应")
            
            # Update stats
            end_time = asyncio.get_event_loop().time()
            self.message_count += 1
            self.total_response_time += (end_time - start_time)
            
            self._log(f"Processed message from {sender_id}")
            
            return response
            
        except Exception as e:
            error_msg = f"ANP消息处理失败: {e}"
            self._log(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)

    async def cleanup(self) -> None:
        """Cleanup ANP meta agent"""
        try:
            if self.base_agent:
                # BaseAgent doesn't have public stop_server, but has internal cleanup
                if hasattr(self.base_agent, '_stop_server'):
                    await self.base_agent._stop_server()
                elif hasattr(self.base_agent, 'shutdown'):
                    await self.base_agent.shutdown()
            
            self._log("ANP meta agent cleanup completed")
            
        except Exception as e:
            error_msg = f"ANP cleanup失败: {e}"
            self._log(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)

    def get_agent_info(self) -> Dict[str, Any]:
        """Get ANP agent information"""
        info = super().get_agent_info()
        info.update({
            "protocol": "anp",
            "has_anp_executor": self.anp_executor is not None,
            "executor_type": "safety_tech_strict_llm",
            "llm_required": True,
            "llm_available": self.anp_executor.llm is not None if self.anp_executor else False,
            "did_authentication": True,
            "e2e_encryption": True
        })
        return info