import requests
import json
import time
from typing import Optional


class QAAgentClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def create_task(self, question: str) -> Optional[str]:
        """创建新的问答任务"""
        try:
            response = requests.post(
                f"{self.base_url}/ap/v1/agent/tasks",
                json={"input": question}
            )
            response.raise_for_status()
            task_data = response.json()
            return task_data.get("task_id")
        except Exception as e:
            print(f"创建任务失败: {e}")
            return None
    
    def get_task_status(self, task_id: str) -> Optional[dict]:
        """获取任务状态"""
        try:
            response = requests.get(f"{self.base_url}/ap/v1/agent/tasks/{task_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取任务状态失败: {e}")
            return None
    
    def list_steps(self, task_id: str) -> Optional[list]:
        """获取任务的所有步骤"""
        try:
            response = requests.get(f"{self.base_url}/ap/v1/agent/tasks/{task_id}/steps")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取步骤列表失败: {e}")
            return None
    
    def execute_step(self, task_id: str) -> Optional[dict]:
        """执行下一个步骤"""
        try:
            response = requests.post(f"{self.base_url}/ap/v1/agent/tasks/{task_id}/steps")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"执行步骤失败: {e}")
            return None
    
    def ask_question(self, question: str, max_wait_time: int = 30) -> str:
        """提问并等待完整回答"""
        print(f"🤖 提问: {question}")
        
        # 创建任务
        task_id = self.create_task(question)
        if not task_id:
            return "创建任务失败"
        
        print(f"📋 任务ID: {task_id}")
        
        # 持续执行步骤直到完成
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            # 执行下一个步骤
            step_result = self.execute_step(task_id)
            if not step_result:
                time.sleep(1)
                continue
            
            step_name = step_result.get('name', '未知步骤')
            step_output = step_result.get('output', '')
            is_last = step_result.get('is_last', False)
            
            print(f"⚡ 执行步骤: {step_name}")
            if step_output:
                print(f"📄 输出: {step_output[:100]}...")  # 只显示前100个字符
            
            # 如果是最后一步，返回完整结果
            if is_last:
                return step_output
            
            time.sleep(1)  # 避免过于频繁的请求
        
        return "处理超时"
    
    def interactive_chat(self):
        """交互式聊天模式"""
        print("🎯 问答Agent客户端启动！")
        print("📝 输入问题开始对话，输入 'quit' 退出\n")
        
        while True:
            try:
                question = input("❓ 您的问题: ").strip()
                if question.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                if not question:
                    continue
                
                answer = self.ask_question(question)
                print(f"\n💬 回答:\n{answer}\n")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 出现错误: {e}")


def test_sample_questions():
    """测试一些示例问题"""
    client = QAAgentClient()
    
    sample_questions = [
        "what is java for",
        "where is the graphic card located in the cpu",
        "what is the nutritional value of oatmeal",
        "how to become a teacher assistant",
        "what foods are good if you have gout"
    ]
    
    print("🧪 测试示例问题...\n")
    
    for question in sample_questions:
        print(f"测试问题: {question}")
        answer = client.ask_question(question)
        print(f"答案: {answer[:200]}...\n")  # 只显示前200个字符
        time.sleep(2)  # 等待2秒再测试下一个问题


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 运行测试模式
        test_sample_questions()
    else:
        # 运行交互模式
        client = QAAgentClient()
        client.interactive_chat()