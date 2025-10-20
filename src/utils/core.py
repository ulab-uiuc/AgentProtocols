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
                import httpx
                from openai import OpenAI

                # Create a dedicated httpx client with a longer read timeout for local/back-end models
                # Local models can take longer to generate; increase read timeout to avoid premature failures.
                http_client = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))

                self.client = OpenAI(api_key="dummy", base_url=base_url, http_client=http_client)
                self.model = self.client
            except Exception as e:
                print(f"[Core] Failed to initialize local OpenAI client: {e}")
                raise

            # ---------- HARD-CODED name mapping ----------
            hard_map = {
                "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": "Llama-3.1-70B-Instruct"
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
            import httpx
            from openai import OpenAI
            
            # Create a clean httpx client without passing any proxy configuration
            http_client = httpx.Client()
            
            try:
                client = OpenAI(
                    api_key=self.config["model"]["openai_api_key"],
                    base_url=self.config["model"].get("openai_base_url", "https://api.openai.com/v1"),
                    http_client=http_client
                )
                print(f"[Core] OpenAI client initialized with custom httpx client")
            except Exception as e:
                print(f"[Core] Failed with custom httpx client: {e}")
                # Try to avoid httpx client customization entirely
                try:
                    import os
                    # Temporarily clear possible proxy environment variables
                    old_env = {}
                    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
                    for var in proxy_vars:
                        if var in os.environ:
                            old_env[var] = os.environ.pop(var)
                    
                    client = OpenAI(
                        api_key=self.config["model"]["openai_api_key"],
                        base_url=self.config["model"].get("openai_base_url", "https://api.openai.com/v1"),
                    )
                    
                    # Restore environment variables
                    for var, value in old_env.items():
                        os.environ[var] = value
                        
                    print(f"[Core] OpenAI client initialized after clearing proxy env vars")
                except Exception as e2:
                    raise RuntimeError(f"All OpenAI client initialization attempts failed. Last error: {e2}")
                
            self.model = client
            self.client = client
    
    
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
            threshold = 3
            while True:
                rounds += 1
                try:
                    response = self.client.chat.completions.create(
                        model=self.config["model"]["name"],
                        messages=messages,
                        temperature=self.config["model"]["temperature"],
                        n=1,
                    )
                    content = response.choices[0].message.content
                    return content.strip()
                except Exception as e:
                    print(f"[Core] OpenAI chat generation error: {e}")
                    time.sleep(10)
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
                    # Build tools properly: accept either full tool schema or raw functions
                    tools = []
                    if isinstance(functions, list) and len(functions) > 0:
                        first = functions[0]
                        if isinstance(first, dict) and first.get("type") == "function" and "function" in first:
                            tools = functions  # already tool schema
                        else:
                            # wrap raw function defs into tool schema
                            for fn in functions:
                                if isinstance(fn, dict) and "name" in fn:
                                    tools.append({"type": "function", "function": fn})
                    
                    response = self.client.chat.completions.create(
                        model=self._local_model_id,
                        messages=truncated_messages,
                        tools=tools,
                        tool_choice="required",
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
            threshold = 3
            
            tools = []
            
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
                        )
                    except Exception:
                        response = self.client.chat.completions.create(
                            model=self.config["model"]["name"],
                            messages=truncated_messages,
                            tools=tools,
                            tool_choice="auto",  # Fallback to auto mode
                            temperature=self.config["model"]["temperature"],
                            n=1,
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
                            )
                            
                            retry_message = retry_response.choices[0].message
                            if hasattr(retry_message, 'tool_calls') and retry_message.tool_calls:
                                return retry_response
                        except:
                            pass
                    
                    return response
                except Exception as e:
                    print(f"[Core] OpenAI function call error (round {rounds}): {e}")
                    time.sleep(10)
                    if rounds > threshold:
                        raise Exception(f"Function call generation failed too many times. Last error: {str(e)}")