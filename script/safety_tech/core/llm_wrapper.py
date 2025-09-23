# -*- coding: utf-8 -*-
"""
LLM Wrapper for Safety Tech

Bridges Safety Tech's expected Core interface to the shared src.utils.core.Core
"""

from __future__ import annotations

try:
    # Prefer absolute import from project root
    from src.utils.core import Core  # type: ignore
except Exception as e:  # pragma: no cover
    # Fallback: attempt relative import if path handling differs
    from ...src.utils.core import Core  # type: ignore

__all__ = ["Core", "generate_doctor_reply", "unified_llm_call"]

import time
import os


class Core:
    def __init__(self, config):
        self.config = config
        self.init_model()
    
    
    def init_model(self):
        self.model_name = self.config["model"]["name"]  # Save model name for later use
        
        if self.config["model"]["type"] == "local":
            base_url = self.config.get("base_url") or f"http://localhost:{self.config.get('port', 8000)}/v1"
            
            # Import OpenAI locally to avoid conflicts
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key="dummy", base_url=base_url)
                self.model = self.client
            except Exception as e:
                print(f"[Core] Failed to initialize local OpenAI client: {e}")
                raise

            # ---------- HARD-CODED name mapping ----------
            hard_map = {
                "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": "Llama-3.1-70B-Instruct",
                "Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL-72B-Instruct",
            }
            # if not in map, fall back to first model id served by vLLM
            try:
                self._local_model_id = hard_map.get(self.model_name) or self.client.models.list().data[0].id
            except Exception as e:
                print(f"[Core] Warning: Could not get model list, using default model name: {e}")
                self._local_model_id = self.model_name
            # ---------------------------------------------

            print(f"[Core] Local endpoint → {base_url} | served_model_id = {self._local_model_id}")

        elif self.config["model"]["type"] == "openai":
            # Import OpenAI locally to avoid conflicts
            try:
                from openai import OpenAI
                
                # Get configuration
                api_key = self.config["model"].get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
                base_url = (
                    self.config["model"].get("openai_base_url")
                    or os.environ.get("OPENAI_BASE_URL")
                    or "https://integrate.api.nvidia.com/v1"
                )
                
                if not api_key:
                    raise ValueError("OpenAI API key is required but not provided")
                
                # Create client with only supported parameters
                client_kwargs = {
                    "api_key": api_key,
                    "base_url": base_url
                }
                
                # Remove any None values
                client_kwargs = {k: v for k, v in client_kwargs.items() if v is not None}
                
                self.client = OpenAI(**client_kwargs)
                self.model = self.client
                
                print(f"[Core] OpenAI client initialized with base_url: {base_url}")
                
            except Exception as e:
                print(f"[Core] Failed to initialize OpenAI client: {e}")
                raise
    
    
    def execute(self, messages):
        if self.config["model"]["type"] == "local":
            try:
                response = self.client.chat.completions.create(
                    model=self._local_model_id,
                    messages=messages,
                    temperature=0.3, # self.config["model"]["temperature"],
                    max_tokens=8192,
                    n=1,
                )
                output = response.choices[0].message.content
                
                # Parse for Qwen
                if "</think>" in output:
                    output = output.split("</think>")[-1].strip()
                
                return output
            except Exception as e:
                print(f"[Core] Local chat generation error: {e}")
                return f"Error in local chat generation: {str(e)}"
        
        elif self.config["model"]["type"] == "openai":
            rounds = 0
            # S1快速失败：根据环境变量加速（默认light）
            s1_mode = os.environ.get('AGORA_S1_TEST_MODE', '').lower()
            fast_fail = s1_mode != 'skip' and s1_mode != ''
            threshold = 1 if fast_fail else 3
            # NVIDIA API需要更长的超时时间
            request_timeout = 10.0 if fast_fail else float(os.environ.get('NVIDIA_REQUEST_TIMEOUT', '30'))
            wait_on_error = 0.2 if fast_fail else 10
            while True:
                rounds += 1
                try:
                    # Build parameters
                    params = {
                        "model": self.config["model"]["name"],
                        "messages": messages,
                        "temperature": self.config["model"]["temperature"],
                        "n": 1,
                    }
                    
                    # Add max_tokens if specified
                    if "max_tokens" in self.config["model"]:
                        params["max_tokens"] = self.config["model"]["max_tokens"]
                    
                    response = self.client.chat.completions.create(timeout=request_timeout, **params)
                    content = response.choices[0].message.content
                    return content.strip()
                except Exception as e:
                    print(f"[Core] OpenAI chat generation error: {e}")
                    time.sleep(wait_on_error)
                    if rounds > threshold:
                        return f"Error in OpenAI chat generation: {str(e)}"
        
                        

    def function_call_execute(self, messages, functions, max_length=300000):
        """
        Execute LLM call with function calling capability
        
        Args:
            messages: List of messages
            functions: List of function definitions (will be wrapped as tools for new API format)
            max_length: Maximum character length limit, default 300000 chars
        
        Returns:
            Complete LLM response object
        """
        # Truncate long message content
        max_length = max_length if self.config["model"]["type"] != "local" else 16384
        HEAD_LEN = int(max_length / 2)
        TAIL_LEN = int(max_length / 2)

        truncated_messages = []
        for msg in messages:
            # Only process regular text content
            if (isinstance(msg, dict)
                    and 'content' in msg
                    and isinstance(msg['content'], str)
                    and len(msg['content']) > max_length):
                
                content = msg['content']
                truncated_msg = msg.copy()
                
                # If shorter than (HEAD+TAIL), just truncate to max_length
                if len(content) <= HEAD_LEN + TAIL_LEN:
                    truncated_msg['content'] = (
                        content[:max_length] + "\n...[Content truncated]..."
                    )
                else:
                    truncated_msg['content'] = (
                        content[:HEAD_LEN] +
                        "\n...[{} chars truncated]...\n".format(len(content) - HEAD_LEN - TAIL_LEN) +
                        content[-TAIL_LEN:]
                    )
                truncated_messages.append(truncated_msg)
                print(f"Message content truncated from {len(content)} to "
                    f"{len(truncated_msg['content'])} characters "
                    f"(head {HEAD_LEN}, tail {TAIL_LEN})")
            else:
                truncated_messages.append(msg)
        
        
        if self.config["model"]["type"] == "local":
            for attempt in range(2):
                try:
                    response = self.client.chat.completions.create(
                        model=self._local_model_id,
                        messages=truncated_messages,
                        tools=[                       
                            {
                                "type": "function",
                                "function": functions[0]
                            }
                        ],
                        tool_choice="auto",
                        temperature=self.config["model"]["temperature"],
                        n=1,
                    )
                    return response
                except Exception as e:
                    print(f"[Core] Local function call attempt {attempt + 1} failed: {e}")
                    if attempt == 1:  # Last attempt
                        raise RuntimeError(f"Local function_call failed twice: {str(e)}")

                            
        elif self.config["model"]["type"] == "openai":
            rounds = 0
            s1_mode = os.environ.get('AGORA_S1_TEST_MODE', '').lower()
            fast_fail = s1_mode != 'skip' and s1_mode != ''
            threshold = 1 if fast_fail else 3
            request_timeout = 1.5 if fast_fail else float(os.environ.get('OPENAI_REQUEST_TIMEOUT', '15'))
            wait_on_error = 0.2 if fast_fail else 10
            
            tools = []
            truncated_messages = self._sanitize_for_gemini(truncated_messages)
  
            first_is_tool = False
            if isinstance(functions, list) and len(functions) > 0:
                first_item = functions[0]
                if isinstance(first_item, dict) and first_item.get('type') == 'function' and 'function' in first_item:
                    first_is_tool = True
                    
            if first_is_tool:
                tools = functions 
            else:
                for func in functions:
                    if isinstance(func, dict) and 'name' in func:
                        tools.append({"type": "function", "function": func})
            
            while True:
                rounds += 1
                try:
                    try:
                        response = self.client.chat.completions.create(
                            model=self.config["model"]["name"],
                            messages=truncated_messages,
                            tools=tools,
                            tool_choice="required",
                            temperature=self.config["model"]["temperature"],
                            n=1,
                            timeout=request_timeout,
                        )
                    except Exception:
                        response = self.client.chat.completions.create(
                            model=self.config["model"]["name"],
                            messages=truncated_messages,
                            tools=tools,
                            tool_choice="auto",  # 回退到自动模式
                            temperature=self.config["model"]["temperature"],
                            n=1,
                            timeout=request_timeout,
                        )
                    
                    choice = response.choices[0] if response.choices else None
                    message = choice.message if choice else None
                    has_function_call = hasattr(message, 'function_call') and message.function_call
                    has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls
                    
                    if not has_function_call and not has_tool_calls and message and message.content and rounds == 1:

                        new_messages = truncated_messages.copy()
                        new_messages.append({
                            "role": "system",
                            "content": "You MUST call one of the provided functions. Respond using function_call only, not with text."
                        })
                        
                        try:
                            retry_response = self.client.chat.completions.create(
                                model=self.config["model"]["name"],
                                messages=new_messages,
                                tools=tools,
                                tool_choice="auto",
                                temperature=self.config["model"]["temperature"],
                                n=1,
                                timeout=request_timeout,
                            )
                            
                            retry_message = retry_response.choices[0].message
                            if hasattr(retry_message, 'tool_calls') and retry_message.tool_calls:
                                return retry_response
                        except:
                            pass
                    
                    return response
                except Exception as e:
                    print(f"[Core] OpenAI function call error (round {rounds}): {e}")
                    time.sleep(wait_on_error)
                    if rounds > threshold:
                        raise Exception(f"Function call generation failed too many times. Last error: {str(e)}")


# ----------------- High-level helper for doctors -----------------
def _build_doctor_context(role: str) -> str:
    role = (role or '').lower()
    if role in ("doctor_a", "a", "receptionist", "primary"):
        return (
            "You are Doctor A, a primary care physician consulting with a specialist colleague about a medical case. "
            "Focus exclusively on clinical aspects: patient symptoms, medical history, physical examination findings, and diagnostic considerations. "
            "Discuss differential diagnoses, treatment protocols, medication management, and follow-up care planning. "
            "Share relevant medical knowledge, evidence-based guidelines, and clinical experience. "
            "Ask about laboratory results, imaging studies, vital signs, and other objective medical data. "
            "Provide professional medical opinions on diagnosis, prognosis, and treatment recommendations. "
            "Maintain a collaborative, educational tone focused purely on advancing patient care through medical expertise exchange."
        )
    else:
        return (
            "You are Doctor B, a specialist physician providing expert consultation on a medical case. "
            "Focus on comprehensive medical analysis: review symptoms, medical history, examination findings, and diagnostic results. "
            "Ask specific clinical questions about disease progression, symptom patterns, response to previous treatments, and comorbidities. "
            "Provide specialist insights on complex diagnoses, advanced treatment options, and management strategies. "
            "Discuss relevant medical literature, clinical guidelines, and best practices. "
            "Evaluate risk factors, contraindications, and potential complications. "
            "Collaborate professionally to develop optimal patient care plans through detailed medical discussion and knowledge sharing."
        )


def _read_env_model_config() -> dict:
    """统一的NVIDIA LLaMA配置读取"""
    import os as _os
    
    # NVIDIA LLaMA配置 - 统一默认值
    api_key = _os.environ.get('NVIDIA_API_KEY', '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')
    base_url = _os.environ.get('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1')
    model_name = _os.environ.get('NVIDIA_MODEL', 'meta/llama-3.1-8b-instruct')
    temperature = float(_os.environ.get('NVIDIA_TEMPERATURE', '0.3'))
    
    print(f"[LLM-CONFIG] Using NVIDIA LLaMA: {model_name}")
    print(f"[LLM-CONFIG] Base URL: {base_url}")
    print(f"[LLM-CONFIG] API Key: {api_key[:20]}...")
    
    return {
        'model': {
            'type': 'openai',  # 使用OpenAI兼容库连接NVIDIA
            'name': model_name,
            'temperature': temperature,
            'openai_api_key': api_key,
            'openai_base_url': base_url,
        }
    }


def generate_doctor_reply(role: str, text: str, model_config: dict | None = None) -> str:
    """Generate a medical reply for Doctor A/B using unified LLM interface.

    Args:
        role: 'doctor_a' or 'doctor_b' (case-insensitive). Others treated as B.
        text: user input text
        model_config: deprecated, now uses unified config

    Returns:
        str reply content
    """
    # 使用统一的LLM调用接口
    context = _build_doctor_context(role)
    messages = [{"role": "user", "content": text or ""}]
    
    result = unified_llm_call(messages, system_prompt=context)
    
    # 如果统一调用失败，返回友好提示
    if result in ("[LLM暂不可用]", "[LLM调用失败]"):
        return "I apologize, but I'm unable to provide a response at this time."
    
    return result


# ===================== 统一LLM调用接口 =====================

def unified_llm_call(messages: list, system_prompt: str = None, temperature: float = None) -> str:
    """统一的LLM调用接口 - 所有Safety Tech模块都应使用此函数
    
    Args:
        messages: 消息列表，格式: [{"role": "user", "content": "..."}]
        system_prompt: 可选的系统提示词，会自动插入到消息开头
        temperature: 可选的温度参数，覆盖默认值
    
    Returns:
        LLM回复的文本内容
        
    Example:
        reply = unified_llm_call([{"role": "user", "content": "Hello"}], 
                               system_prompt="You are a helpful assistant")
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # 获取统一配置
        config = _read_env_model_config()
        
        # 覆盖温度参数
        if temperature is not None:
            config['model']['temperature'] = temperature
        
        # 构造完整消息
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        
        # 创建Core实例并调用，带重试机制
        core = Core(config)
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                result = core.execute(full_messages)
                
                if result and not any(err in result for err in ["Error in", "医生回复暂不可用", "Request timed out"]):
                    return result
                else:
                    if attempt < max_retries - 1:
                        logger.warning(f"[LLM] Attempt {attempt + 1} failed: {result}, retrying...")
                        continue
                    else:
                        logger.warning(f"[LLM] All attempts failed, last result: {result}")
                        return "[LLM暂不可用]"
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"[LLM] Attempt {attempt + 1} exception: {e}, retrying...")
                    continue
                else:
                    logger.error(f"[LLM] Final attempt failed: {e}")
                    return "[LLM调用失败]"
            
    except Exception as e:
        logger.error(f"[LLM] Unified call failed: {e}")
        return "[LLM调用失败]"