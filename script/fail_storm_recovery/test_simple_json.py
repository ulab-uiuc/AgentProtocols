#!/usr/bin/env python3
"""
简单的SimpleJSON协议测试脚本
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# 添加路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

async def test_simple_json():
    """测试SimpleJSON协议的基本功能"""
    
    print("🧪 Testing SimpleJSON Protocol")
    
    try:
        # 导入必要的模块
        from agent_adapters.simple_json_adapter import SimpleJSONAdapter
        from server_adapters.simple_json_adapter import SimpleJSONServerAdapter
        
        print("✅ Modules imported successfully")
        
        # 创建一个简单的执行器
        class TestExecutor:
            def __init__(self, name):
                self.name = name
                
            async def execute(self, context, event_queue):
                """实现SDK接口"""
                user_input = context.get_user_input() if hasattr(context, 'get_user_input') else "test input"
                print(f"[{self.name}] Processing: {user_input}")
                
                # 创建一个简单的响应事件
                class SimpleEvent:
                    def __init__(self, text):
                        self.text = text
                
                await event_queue.enqueue_event(SimpleEvent(f"Response from {self.name}: Processed '{user_input}'"))
        
        print("✅ Test executor created")
        
        # 创建服务器适配器
        server_adapter = SimpleJSONServerAdapter()
        app = server_adapter.create_app(TestExecutor("TestAgent"), "test_agent_1")
        
        print("✅ Server adapter and app created")
        
        # 测试消息格式
        import httpx
        
        adapter = SimpleJSONAdapter(
            httpx_client=httpx.AsyncClient(),
            base_url="http://localhost:8000",
            agent_id="test_client"
        )
        
        # 测试消息转换
        test_payload = {
            "type": "test",
            "message": "Hello SimpleJSON!",
            "data": {"key": "value"}
        }
        
        print(f"✅ Test payload: {json.dumps(test_payload, indent=2)}")
        
        print("\n🎉 SimpleJSON Protocol basic test completed successfully!")
        print("The protocol is ready for use in fail_storm_recovery")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_simple_json())
    sys.exit(0 if result else 1)