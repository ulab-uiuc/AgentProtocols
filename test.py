#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 1000 QA data execution results on Agent Protocol service
Using agent mounted on http://localhost:8000 for QA tasks
"""

import json
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
import requests
from tqdm import tqdm

class QATestClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def create_task(self, question: str) -> Optional[str]:
        """Create a new Q&A task"""
        try:
            response = self.session.post(
                f"{self.base_url}/ap/v1/agent/tasks",
                json={"input": question},
                timeout=30
            )
            response.raise_for_status()
            task_data = response.json()
            return task_data.get("task_id")
        except Exception as e:
            print(f"Task creation failed: {e}")
            return None
    
    def execute_step(self, task_id: str) -> Optional[dict]:
        """Execute next step"""
        try:
            response = self.session.post(
                f"{self.base_url}/ap/v1/agent/tasks/{task_id}/steps",
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Step execution failed: {e}")
            return None
    
    def ask_question_with_timeout(self, question: str, max_wait_time: int = 60) -> Dict:
        """Ask question and wait for complete answer, return result statistics"""
        result = {
            "question": question,
            "answer": "",
            "status": "fail",
            "duration": 0,
            "steps_count": 0,
            "error_message": ""
        }
        
        start_time = time.time()
        
        try:
            # Create task
            task_id = self.create_task(question)
            if not task_id:
                result["error_message"] = "Task creation failed"
                result["duration"] = time.time() - start_time
                return result
            
            # Keep executing steps until completion or timeout
            while time.time() - start_time < max_wait_time:
                step_result = self.execute_step(task_id)
                if not step_result:
                    time.sleep(1)
                    continue
                
                result["steps_count"] += 1
                step_output = step_result.get('output', '')
                is_last = step_result.get('is_last', False)
                
                # If it's the last step, return result
                if is_last:
                    result["answer"] = step_output
                    result["status"] = "success"
                    result["duration"] = time.time() - start_time
                    return result
                
                time.sleep(0.5)  # Brief wait to avoid too frequent requests
            
            # Timeout case
            result["error_message"] = "Processing timeout"
            result["duration"] = time.time() - start_time
            
        except Exception as e:
            result["error_message"] = str(e)
            result["duration"] = time.time() - start_time
        
        return result


def load_qa_data(file_path: str) -> List[Dict]:
    """Load simplified JSONL format QA data"""
    qa_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    
                    # Extract question from simplified format {"id": 1, "q": "question"}
                    if "id" in data and "q" in data:
                        qa_data.append({
                            "question_id": data["id"],
                            "question": data["q"]
                        })
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parsing failed at line {line_num}: {e}")
                except Exception as e:
                    print(f"Processing failed at line {line_num}: {e}")
    
    except FileNotFoundError:
        print(f"Error: Data file not found {file_path}")
        return []
    except Exception as e:
        print(f"Failed to read data file: {e}")
        return []
    
    return qa_data[:50]


def save_results_to_json(results: List[Dict], metadata: Dict, output_file: str):
    """Save results to JSON file in qa_results.json format"""
    try:
        output_data = {
            "metadata": metadata,
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
                
        print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"Failed to save results: {e}")


def generate_test_report(results: List[Dict]) -> Dict:
    """Generate test report statistics"""
    total_count = len(results)
    success_count = sum(1 for r in results if r['status'] == 'success')
    fail_count = total_count - success_count
    
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    
    durations = [r['response_time'] for r in results if r.get('response_time', 0) > 0]
    avg_duration = sum(durations) / len(durations) if durations else 0
    
    steps_counts = [r.get('steps_count', 0) for r in results if r.get('steps_count', 0) > 0]
    avg_steps = sum(steps_counts) / len(steps_counts) if steps_counts else 0
    
    # Error statistics
    error_types = {}
    for result in results:
        if result['status'] == 'fail' and result.get('error_message'):
            error_msg = result['error_message']
            if 'timeout' in error_msg.lower():
                error_types['timeout'] = error_types.get('timeout', 0) + 1
            elif 'creation failed' in error_msg.lower():
                error_types['task_creation_failed'] = error_types.get('task_creation_failed', 0) + 1
            elif 'connection' in error_msg.lower():
                error_types['connection_error'] = error_types.get('connection_error', 0) + 1
            else:
                error_types['other'] = error_types.get('other', 0) + 1
    
    return {
        'total_questions': total_count,
        'successful_questions': success_count,
        'failed_questions': fail_count,
        'success_rate': round(success_rate, 2),
        'average_response_time': round(avg_duration, 2),
        'average_steps': round(avg_steps, 2),
        'error_breakdown': error_types,
        'timestamp': time.time(),
        'network_type': 'local_llm_agent'
    }


def test_agent_connectivity(client: QATestClient) -> bool:
    """Test Agent connectivity"""
    print("🔍 Testing Agent connectivity...")
    try:
        # Test with a simple question
        test_question = "Hello, are you working?"
        result = client.ask_question_with_timeout(test_question, max_wait_time=30)
        
        if result['status'] == 'success':
            print("✅ Agent connection successful")
            return True
        else:
            print(f"❌ Agent connection test failed: {result['error_message']}")
            return False
            
    except Exception as e:
        print(f"❌ Agent connection test exception: {e}")
        return False


def main():
    """Main function"""
    print("=" * 80)
    print("🤖 QA Agent 1000 Data Test Started")
    print("=" * 80)
    
    # Configuration parameters
    data_file = "/GPFS/data/sujiaqi/gui/Multiagent-Protocol/ANP/streaming_queue/data/top1000_simplified.jsonl"
    output_file = f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    agent_url = "http://localhost:8000"  # Agent Protocol runs on 8000, LLM on db93:8001
    
    # Initialize client
    client = QATestClient(agent_url)
    
    # Test connectivity
    if not test_agent_connectivity(client):
        print("🚫 Agent connection failed, exiting test")
        return "fail"
    
    # Load data
    print(f"📁 Loading data file: {data_file}")
    qa_data = load_qa_data(data_file)
    
    if not qa_data:
        print("❌ No valid data loaded")
        return "fail"
    
    print(f"📊 Successfully loaded {len(qa_data)} questions")
    
    # Start testing
    print(f"🚀 Starting test ...")
    results = []
    
    pbar = tqdm(total=len(qa_data), desc="Processing Questions", unit="question")
    for i, qa_item in enumerate(qa_data, 1):
        question_id = qa_item['question_id']
        question = qa_item['question']
        
        # Execute QA test
        test_result = client.ask_question_with_timeout(question, max_wait_time=90)
        
        # Format result according to qa_results.json structure
        result = {
            "question_id": question_id,
            "question": question,
            "worker": "Local-LLM-Agent",
            "answer": test_result["answer"],
            "response_time": test_result["duration"],
            "timestamp": time.time(),
            "status": test_result["status"],
            "steps_count": test_result["steps_count"]
        }
        
        if test_result["status"] == "fail":
            result["error_message"] = test_result["error_message"]
        
        results.append(result)
        
        # Show progress
        # if i % 10 == 0 or i == len(qa_data):
        #     success_so_far = sum(1 for r in results if r['status'] == 'success')
        #     print(f"📈 Progress: {i}/{len(qa_data)} ({i/len(qa_data)*100:.1f}%) - "
        #           f"Success: {success_so_far}/{i} ({success_so_far/i*100:.1f}%)")
        
        # Avoid too frequent requests
        time.sleep(0.1)
        pbar.update(1)
    
    # Generate report
    print("\n📋 Generating test report...")
    metadata = generate_test_report(results)
    
    # Save results
    save_results_to_json(results, metadata, output_file)
    
    # Print report
    print("\n" + "=" * 80)
    print("📊 QA Agent Test Report")
    print("=" * 80)
    print(f"Total questions: {metadata['total_questions']}")
    print(f"Successful answers: {metadata['successful_questions']}")
    print(f"Failed answers: {metadata['failed_questions']}")
    print(f"Success rate: {metadata['success_rate']}%")
    print(f"Average response time: {metadata['average_response_time']} seconds")
    print(f"Average steps: {metadata['average_steps']}")
    
    if metadata['error_breakdown']:
        print("\nError type statistics:")
        for error_type, count in metadata['error_breakdown'].items():
            print(f"  - {error_type}: {count}")
    
    print(f"\nDetailed results saved to: {output_file}")
    print("=" * 80)
    
    # Determine overall result
    if metadata['success_rate'] >= 80:
        print("🎉 Test result: SUCCESS (Success rate >= 80%)")
        return "success"
    elif metadata['success_rate'] >= 60:
        print("⚠️ Test result: PARTIAL SUCCESS (Success rate >= 60%)")
        return "partial_success"
    else:
        print("❌ Test result: FAIL (Success rate < 60%)")
        return "fail"


if __name__ == "__main__":
    try:
        result = main()
        # Set exit code based on result
        if result == "success":
            sys.exit(0)
        elif result == "partial_success":
            sys.exit(1)
        else:
            sys.exit(2)
    except KeyboardInterrupt:
        print("\n\n⏹️ User interrupted test")
        sys.exit(3)
    except Exception as e:
        print(f"\n\n💥 Error during test process: {e}")
        sys.exit(4)