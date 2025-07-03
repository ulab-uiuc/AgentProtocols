import json
import asyncio
from agent_protocol import Agent, Step, Task
from typing import Dict, Any, List
import random
import requests


class LLMCore:
    """Core LLM client for calling local LLM on port 8001"""
    def __init__(self, llm_url: str = "http://db93:8001"):
        self.llm_url = llm_url
        self.session = requests.Session()
    
    def execute(self, messages: List[Dict]) -> str:
        """Execute LLM request with messages"""
        try:
            # Format request for local LLM API
            payload = {
                "model": "Qwen2.5-7B-Instruct",
                "messages": messages,
                "max_tokens": 150,
                "temperature": 0.7
            }
            
            response = self.session.post(
                f"{self.llm_url}/v1/chat/completions",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "No response from LLM"
                
        except Exception as e:
            print(f"LLM execution error: {e}")
            print(f"Request URL: {self.llm_url}/v1/chat/completions")
            print(f"Request payload: {payload}")
            return f"LLM error: {str(e)[:100]}..."


# QA Data Loader with LLM integration
class QADataLoader:
    def __init__(self, data_path: str, use_mock: bool = False, core_llm: LLMCore = None):
        self.data_path = data_path
        self.qa_pairs = []
        self.use_mock = use_mock
        self.core = core_llm
        self.load_data()
    
    def load_data(self):
        """Load Q&A data"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        # Simplified data format: {"id": 1, "q": "question"}
                        if "id" in data and "q" in data:
                            self.qa_pairs.append({
                                'id': data['id'],
                                'question': data['q'].strip()
                            })
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
    
    async def invoke(self, question: str) -> str:
        """Answer a question using Core LLM."""
        try:
            if self.use_mock or self.core is None:
                # Mock response for testing
                await asyncio.sleep(0.1)
                return f"Mock answer for: {question[:50]}{'...' if len(question) > 50 else ''}"
            
            # Input validation
            if not question or question.strip() == "":
                return "Please provide a valid question."
            
            # Prepare messages for Core LLM
            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant. Provide concise, accurate answers to questions. Keep responses under 150 words."
                },
                {
                    "role": "user", 
                    "content": question
                }
            ]
            
            # Call Core.execute() in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(None, self.core.execute, messages)
            
            return answer.strip() if answer else "Unable to generate response"
            
        except Exception as e:
            print(f"[QADataLoader] Error processing question: {e}")
            return f"Error: {str(e)[:100]}..."
    
    def find_answer(self, query: str) -> str:
        """Find answer based on query - fallback method"""
        if not query or query.strip() == "":
            return "Please provide a valid question."
            
        # Simple fallback when LLM is not available
        return f"Processing question: {query[:100]}{'...' if len(query) > 100 else ''}"
    
    def get_random_questions(self, count: int = 5) -> List[str]:
        """Get random question samples"""
        if len(self.qa_pairs) >= count:
            samples = random.sample(self.qa_pairs, count)
            return [qa['question'] for qa in samples]
        return [qa['question'] for qa in self.qa_pairs[:count]]


# Initialize Core LLM and QA data loader
core_llm = LLMCore()
qa_loader = QADataLoader(
    '/GPFS/data/sujiaqi/gui/Multiagent-Protocol/ANP/streaming_queue/data/top1000_simplified.jsonl',
    core_llm=core_llm
)


async def plan_step(step: Step) -> Step:
    """Planning step: Analyze user question and create response plan"""
    task = await Agent.db.get_task(step.task_id)
    user_question = task.input
    
    # Planning steps
    steps = [
        "Analyze user question",
        "Generate LLM response", 
        "Format answer",
        "Provide final answer"
    ]
    
    # Create subsequent steps
    for i, step_name in enumerate(steps[:-1]):
        await Agent.db.create_step(
            task_id=task.task_id, 
            name=step_name,
            input=user_question
        )
    
    # Create final step
    await Agent.db.create_step(
        task_id=task.task_id, 
        name=steps[-1],
        input=user_question,
        is_last=True
    )
    
    step.output = f"Response workflow planned for question: '{user_question}'\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])
    return step


async def analyze_question_step(step: Step) -> Step:
    """Question analysis step"""
    user_question = step.input
    
    # Add null check
    if not user_question:
        task = await Agent.db.get_task(step.task_id)
        user_question = task.input
    
    if not user_question:
        step.output = "Invalid question input"
        return step
    
    # Analyze question type and keywords
    question_words = user_question.lower().split()
    
    analysis = {
        "question": user_question,
        "keywords": question_words[:10],  # First 10 words as keywords
        "question_length": len(user_question),
        "word_count": len(question_words)
    }
    
    step.output = f"Question analysis completed:\n{json.dumps(analysis, indent=2, ensure_ascii=False)}"
    return step


async def generate_llm_response_step(step: Step) -> Step:
    """LLM response generation step"""
    user_question = step.input
    
    # Add null check
    if not user_question:
        task = await Agent.db.get_task(step.task_id)
        user_question = task.input
    
    if not user_question:
        step.output = "Generated answer: Unable to get question content"
        return step
    
    # Use QA loader's invoke method to get LLM response
    answer = await qa_loader.invoke(user_question)
    
    step.output = f"Generated answer: {answer}"
    return step


async def format_answer_step(step: Step) -> Step:
    """Answer formatting step"""
    # Get user question
    user_question = step.input
    if not user_question:
        task = await Agent.db.get_task(step.task_id)
        user_question = task.input
    
    # Get answer from previous step
    task = await Agent.db.get_task(step.task_id)
    steps = await Agent.db.list_steps(step.task_id)
    
    answer = "No answer found"
    for prev_step in steps:
        if prev_step.name == "Generate LLM response" and prev_step.output:
            answer = prev_step.output.replace("Generated answer: ", "")
            break
    
    # Format answer
    if answer == "No answer found" or "Bad Request" in answer:
        formatted_answer = " ❌ No answer found for the question."

    else:
        formatted_answer = f"""✅{answer} """
    
    step.output = formatted_answer.strip()
    return step


async def provide_answer_step(step: Step) -> Step:
    """Provide final answer step"""
    # Get final answer from formatting step
    task = await Agent.db.get_task(step.task_id)
    steps = await Agent.db.list_steps(step.task_id)
    
    final_answer = "Processing error"
    for prev_step in steps:
        if prev_step.name == "Format answer" and prev_step.output:
            final_answer = prev_step.output
            break
    
    step.output = final_answer
    
    # # Add some related question suggestions
    # sample_questions = qa_loader.get_random_questions(3)
    # suggestions = "\n\n🔍 You might also be interested in:\n" + "\n".join([f"• {q}" for q in sample_questions])
    # step.output += suggestions
    
    return step


async def task_handler(task: Task) -> None:
    """Task handler: Create initial planning step for new task"""
    await Agent.db.create_step(
        task_id=task.task_id, 
        name="Plan response workflow",
        input=task.input
    )


async def step_handler(step: Step) -> Step:
    """Step handler: Execute corresponding processing logic based on step name"""
    
    if step.name == "Plan response workflow":
        return await plan_step(step)
    elif step.name == "Analyze user question":
        return await analyze_question_step(step)
    elif step.name == "Generate LLM response":
        return await generate_llm_response_step(step)
    elif step.name == "Format answer":
        return await format_answer_step(step)
    elif step.name == "Provide final answer":
        return await provide_answer_step(step)
    else:
        step.output = f"Unknown step type: {step.name}"
        return step


if __name__ == "__main__":
    print("🚀 Q&A Agent starting...")
    print(f"📚 Loaded {len(qa_loader.qa_pairs)} Q&A data entries")
    print("💬 Submit questions via API or frontend interface to start Q&A")
    print("🌐 Agent Protocol service will run on http://localhost:8000")
    print("🧠 Using local LLM core for intelligent responses")
    
    # Start Agent
    Agent.setup_agent(task_handler, step_handler).start()