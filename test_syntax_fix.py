#!/usr/bin/env python3

"""
Test the improved syntax error fixing.
"""

import asyncio
import sys
import os

# Add gaia tools to path
sys.path.append('/root/Multiagent-Protocol/script/gaia')

from tools.sandbox_python_execute import SandboxPythonExecute

async def test_syntax_fix():
    """Test syntax error fixing specifically."""
    
    tool = SandboxPythonExecute()
    
    print("🧪 Testing Improved Syntax Error Fixing\n")
    
    # Test syntax error with comment
    print("=" * 50)
    print("Test: Syntax error with comment")
    print("=" * 50)
    
    code = """
# This code has syntax errors
for i in range(5)  # Missing colon
    print(f"Number: {i}")

if True  # Another missing colon
    print("This should work after fix")
"""
    
    try:
        result = await tool.execute(code, packages=[], max_retries=2, timeout=30)
        print(f"Result: {result.output if result.output else f'Error: {result.error}'}")
    except Exception as e:
        print(f"Test error: {e}")
    
    # Cleanup
    await tool.cleanup()
    print("\n🎉 Syntax fix test completed!")

if __name__ == "__main__":
    asyncio.run(test_syntax_fix())
