#!/usr/bin/env python3
"""
测试Core导入 - 模拟ShardWorkerExecutor的导入环境
"""

import sys
from pathlib import Path

def test_core_import():
    """测试Core类导入"""
    print("🧪 Testing Core import in fail_storm environment")
    
    # 添加路径（与agent_executor.py中相同的设置）
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root))
    
    print(f"Project root: {project_root}")
    print("Python paths:")
    for i, p in enumerate(sys.path[:5]):
        print(f"  {i}: {p}")
    
    print()
    
    try:
        from utils.core import Core
        print("✅ Successfully imported Core from utils.core")
        
        # 测试Core初始化
        test_config = {
            "model": {
                "type": "openai",
                "name": "gpt-4o",
                "openai_api_key": "test_key"
            }
        }
        
        try:
            core = Core(test_config)
            print("✅ Core initialization successful")
            print(f"Model name: {core.model_name}")
            return True
        except Exception as e:
            print(f"❌ Core initialization failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to import Core: {e}")
        
        # 调试信息
        print("\nDebugging import issue...")
        import os
        
        utils_path = project_root / "src" / "utils"
        core_path = project_root / "src" / "utils" / "core.py"
        
        print(f"utils directory exists: {utils_path.exists()}")
        print(f"core.py exists: {core_path.exists()}")
        
        if utils_path.exists():
            print("Contents of utils directory:")
            for item in utils_path.iterdir():
                print(f"  {item.name}")
        
        return False

if __name__ == "__main__":
    success = test_core_import()
    sys.exit(0 if success else 1)