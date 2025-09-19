#!/usr/bin/env python3

"""
Test script for intelligent retry mechanism in SandboxPythonExecute.
"""

import asyncio
import sys
import os

# Add gaia tools to path
sys.path.append('/root/Multiagent-Protocol/script/gaia')

from tools.sandbox_python_execute import SandboxPythonExecute

async def test_intelligent_retry():
    """Test the intelligent retry mechanism with various error scenarios."""
    
    tool = SandboxPythonExecute()
    
    print("🧪 Testing Intelligent Retry Mechanism\n")
    
    # Test 1: ModuleNotFoundError - should be auto-fixed by adding packages
    print("=" * 50)
    print("Test 1: ModuleNotFoundError (should auto-fix with packages)")
    print("=" * 50)
    
    code1 = """
import pandas as pd
import numpy as np

# Create some sample data
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
print("DataFrame created successfully:")
print(df)
print(f"DataFrame shape: {df.shape}")
"""
    
    try:
        result = await tool.execute(code1, packages=[], max_retries=2)
        print(f"Result: {result.output if result.output else f'Error: {result.error}'}")
    except Exception as e:
        print(f"Test error: {e}")
    
    print("\n" + "=" * 50)
    print("Test 2: Logic error (should be fixed by LLM)")
    print("=" * 50)
    
    # Test 2: Logic error - accessing None object
    code2 = """
# This code has a logical error - trying to access attributes of None
data = None
result = data.get('key')  # This will fail
print(f"Result: {result}")
"""
    
    try:
        result = await tool.execute(code2, packages=[], max_retries=2)
        print(f"Result: {result.output if result.output else f'Error: {result.error}'}")
    except Exception as e:
        print(f"Test error: {e}")
    
    print("\n" + "=" * 50)
    print("Test 3: Syntax error (should be fixed by LLM)")  
    print("=" * 50)
    
    # Test 3: Syntax error
    code3 = """
# This code has syntax errors
for i in range(5)  # Missing colon
    print(f"Number: {i}")
"""
    
    try:
        result = await tool.execute(code3, packages=[], max_retries=2)
        print(f"Result: {result.output if result.output else f'Error: {result.error}'}")
    except Exception as e:
        print(f"Test error: {e}")
    
    print("\n" + "=" * 50)
    print("Test 4: Successful code (no retries needed)")
    print("=" * 50)
    
    # Test 4: Working code
    code4 = """
print("Hello, World!")
for i in range(3):
    print(f"Count: {i}")
"""
    
    try:
        result = await tool.execute(code4, packages=[], max_retries=2)
        print(f"Result: {result.output if result.output else f'Error: {result.error}'}")
    except Exception as e:
        print(f"Test error: {e}")
    
    # Cleanup
    await tool.cleanup()
    print("\n🎉 Tests completed!")

if __name__ == "__main__":
    asyncio.run(test_intelligent_retry())
