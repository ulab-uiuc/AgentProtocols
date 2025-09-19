import pytest
import asyncio
from unittest.mock import patch, MagicMock

# 假设您的工具类保存在名为 sandbox_tool.py 的文件中
from sandbox_tool import SandboxPythonExecute 

# 使用 pytest-asyncio 来测试异步函数
@pytest.mark.asyncio
async def test_llm_is_called_on_execution_error_and_fixes_code():
    """
    测试当代码执行失败时，LLM是否被调用以修复代码，并最终成功执行。
    """
    # 1. 准备阶段
    # ----------------------------------------------------------------------
    # 创建工具实例
    tool = SandboxPythonExecute()
    # 确保沙箱客户端被模拟，避免真实创建 Docker 容器
    tool._sandbox_client = MagicMock()

    # 故意写错的代码 (NameError: name 'my_variable' is not defined)
    bad_code = "result = 1 + my_variable\nprint(result)"

    # 预期的、由 LLM "修复" 后的代码
    fixed_code = "my_variable = 10\nresult = 1 + my_variable\nprint(result)"
    
    # 模拟沙箱的两次执行：第一次失败，第二次成功
    # 第一次执行 bad_code 会返回一个错误
    error_output = """
    EXECUTION_ERROR
    ERROR_START
    NameError: name 'my_variable' is not defined
    TRACEBACK_START
    Traceback (most recent call last):
      File "execute_code.py", line 100, in <module>
    NameError: name 'my_variable' is not defined
    TRACEBACK_END
    ERROR_END
    """
    # 第二次执行 fixed_code 会返回成功结果
    success_output = """
    EXECUTION_SUCCESS
    OUTPUT_START
    11
    OUTPUT_END
    """
    
    # 配置沙箱模拟对象的返回值，让它按顺序返回失败和成功的结果
    tool._sandbox_client.run_command.side_effect = [
        error_output,
        success_output
    ]
    # 模拟文件写入，因为代码会尝试写入 execute_code.py
    tool._sandbox_client.write_file = MagicMock(return_value=None)


    # 2. Mock LLM 的响应
    # ----------------------------------------------------------------------
    # 这是测试的核心：我们"拦截"对 llm.LLM.call_llm 的真实调用
    # 你需要根据你的项目结构调整 'sandbox_tool.LLM.call_llm' 这个路径
    with patch('sandbox_tool.LLM.call_llm') as mock_call_llm:
        # 准备一个模拟的 LLM API 响应
        # 这个结构必须与你的 _fix_code_with_llm 方法中解析的结构一致
        mock_llm_response_content = {
            "analysis": "The code failed because 'my_variable' was not defined. I will define it before use.",
            "fixed_code": fixed_code,
            "packages": [],
            "confidence": "high"
        }
        
        # 将其包装成类似真实 API 返回的格式
        mock_response = {
            'choices': [{
                'message': {
                    'content': f"```json\n{json.dumps(mock_llm_response_content)}\n```"
                }
            }]
        }
        
        # 设置当 mock_call_llm 被调用时，返回我们准备好的模拟响应
        mock_call_llm.return_value = mock_response

        # 3. 执行并验证
        # ----------------------------------------------------------------------
        # 执行带有错误代码的工具
        result = await tool.execute(code=bad_code, max_retries=1)

        # 验证 a: LLM 的 mock 函数是否被调用了，并且只调用了一次
        mock_call_llm.assert_called_once()
        print("\n✅ LLM call was successfully mocked and called.")

        # 验证 b: 最终结果是否是成功状态
        assert result.error is None
        assert result.output is not None
        print(f"✅ Tool execution succeeded with output:\n---\n{result.output}\n---")

        # 验证 c: 最终的输出是否来自"修复后"的代码
        # 这里的 "11" 是 fixed_code (1 + 10) 的运行结果
        assert "Code Execution Output:\n11" in result.output
        print("✅ The output is from the LLM-fixed code.")
        
        # 验证 d: 沙箱的 run_command 被调用了两次（一次失败，一次成功）
        assert tool._sandbox_client.run_command.call_count == 2
        print("✅ Sandbox execution was attempted twice as expected.")

    # 4. 清理
    await tool.cleanup()