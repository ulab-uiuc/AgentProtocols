#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real AgentNetwork Demo - Using real AgentNetwork class and BaseAgent with Agent Protocol
"""
import asyncio
import json
import yaml
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
import httpx

# Add colorama for colored output
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    # Fallback if colorama is not available
    COLORS_AVAILABLE = False
    class Fore:
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        CYAN = ""
        WHITE = ""
    class Style:
        BRIGHT = ""
        RESET_ALL = ""

import sys
from pathlib import Path

# 获取当前文件的路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
src_path = project_root / "src"
script_path = project_root / "script"

# 添加必要的路径
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(script_path))

# 导入基础模块
from network import AgentNetwork
from base_agent import BaseAgent

# 如果文件存在，尝试导入
try:
    from qa_coordinator.agent_executor import QACoordinatorExecutor
except ImportError as e:
    print(f"导入 QACoordinatorExecutor 失败: {e}")

try:
    from qa_worker.agent_executor import QAAgentExecutor
except ImportError as e:
    print(f"导入 QAAgentExecutor 失败: {e}")

class ColoredOutput:
    """Helper class for colored console output"""
    
    @staticmethod
    def info(message: str) -> None:
        """Print info message in blue"""
        print(f"{Fore.BLUE}{Style.BRIGHT}ℹ️  {message}{Style.RESET_ALL}")
    
    @staticmethod
    def success(message: str) -> None:
        """Print success message in green"""
        print(f"{Fore.GREEN}{Style.BRIGHT}✅ {message}{Style.RESET_ALL}")
    
    @staticmethod
    def warning(message: str) -> None:
        """Print warning message in yellow"""
        print(f"{Fore.YELLOW}{Style.BRIGHT}⚠️  {message}{Style.RESET_ALL}")
    
    @staticmethod
    def error(message: str) -> None:
        """Print error message in red"""
        print(f"{Fore.RED}{Style.BRIGHT}❌ {message}{Style.RESET_ALL}")
    
    @staticmethod
    def system(message: str) -> None:
        """Print system status in cyan"""
        print(f"{Fore.CYAN}{Style.BRIGHT}🔧 {message}{Style.RESET_ALL}")
    
    @staticmethod
    def progress(message: str) -> None:
        """Print progress message in white"""
        print(f"{Fore.WHITE}   {message}{Style.RESET_ALL}")


class AgentProtocolExecutor:
    """Agent Protocol 执行器适配器"""
    
    def __init__(self, qa_executor, agent_type="worker"):
        self.qa_executor = qa_executor
        self.agent_type = agent_type
        self.current_tasks = {}
    
    async def handle_task_creation(self, task):
        """处理任务创建"""
        self.current_tasks[task.task_id] = task
        task.status = "ready"
    
    async def execute_step(self, step):
        """执行步骤"""
        try:
            # 根据智能体类型处理不同的逻辑
            if self.agent_type == "coordinator":
                if hasattr(self.qa_executor, 'coordinator'):
                    result = await self._execute_coordinator_step(step)
                else:
                    result = {"output": "Coordinator not ready", "status": "failed"}
            else:
                result = await self._execute_worker_step(step)
            
            return result
            
        except Exception as e:
            return {
                "output": f"Execution error: {str(e)}",
                "status": "failed",
                "is_last": True
            }
    
    async def _execute_coordinator_step(self, step):
        """执行协调器步骤"""
        input_text = step.input.lower()
        
        if "status" in input_text:
            # 获取状态信息
            try:
                network_status = "Connected" if self.qa_executor.coordinator.agent_network else "Not connected"
                worker_count = len(self.qa_executor.coordinator.worker_ids)
                
                status_info = (
                    f"QA Coordinator Status:\n"
                    f"Configuration: batch_size={self.qa_executor.coordinator.batch_size}, "
                    f"first_50={self.qa_executor.coordinator.first_50}\n"
                    f"Data path: {self.qa_executor.coordinator.data_path}\n"
                    f"Network status: {network_status}\n"
                    f"Worker count: {worker_count}\n"
                    f"Available commands: dispatch, status, load_questions"
                )
                
                return {
                    "output": status_info,
                    "status": "completed",
                    "is_last": True
                }
            except Exception as e:
                return {
                    "output": f"Status check failed: {str(e)}",
                    "status": "failed",
                    "is_last": True
                }
                
        elif "load_questions" in input_text:
            # 加载问题数据
            try:
                questions = await self.qa_executor.coordinator.load_questions()
                return {
                    "output": f"Successfully loaded {len(questions)} questions from data file",
                    "status": "completed",
                    "is_last": True,
                    "additional_output": {
                        "questions_count": len(questions),
                        "questions_preview": questions[:3] if questions else []
                    }
                }
            except Exception as e:
                return {
                    "output": f"Failed to load questions: {str(e)}",
                    "status": "failed",
                    "is_last": True
                }
                
        elif "dispatch" in input_text:
            # 执行分发逻辑 - 修复方法名从 run_dispatch 到 dispatch_round
            if hasattr(self.qa_executor.coordinator, 'dispatch_round'):
                try:
                    result = await self.qa_executor.coordinator.dispatch_round()
                    return {
                        "output": f"Dispatch completed: {result}",
                        "status": "completed",
                        "is_last": True
                    }
                except Exception as e:
                    return {
                        "output": f"Dispatch failed: {str(e)}",
                        "status": "failed",
                        "is_last": True
                    }
            else:
                return {
                    "output": "Dispatch functionality not available",
                    "status": "failed",
                    "is_last": True
                }
        else:
            # 对于其他输入，使用 LLM 进行回答
            if hasattr(self.qa_executor, 'coordinator') and hasattr(self.qa_executor.coordinator, 'agent'):
                try:
                    # 使用协调器的 LLM 进行回答
                    result = await self.qa_executor.coordinator.agent.invoke(step.input)
                    return {
                        "output": result,
                        "status": "completed",
                        "is_last": True
                    }
                except Exception as e:
                    return {
                        "output": f"LLM response failed: {str(e)}",
                        "status": "failed",
                        "is_last": True
                    }
            else:
                return {
                    "output": f"Processed coordinator command: {step.input}",
                    "status": "completed",
                    "is_last": True
                }
    
    async def _execute_worker_step(self, step):
        """执行工作器步骤"""
        # 检查是否有可用的 QA Agent
        if hasattr(self.qa_executor, 'agent'):
            try:
                # 使用 QA Agent 的 LLM 进行回答
                result = await self.qa_executor.agent.invoke(step.input)
                return {
                    "output": result,
                    "status": "completed",
                    "is_last": True
                }
            except Exception as e:
                return {
                    "output": f"LLM response failed: {str(e)}",
                    "status": "failed",
                    "is_last": True
                }
        else:
            # 回退到简单响应
            return {
                "output": f"Worker processed: {step.input}",
                "status": "completed",
                "is_last": True
            }


class RealAgentNetworkDemo:
    """Real AgentNetwork Demo Class using Agent Protocol"""
    
    def __init__(self, config_path="config.yaml"):
        self.config = self.load_config(config_path)
        self.network = AgentNetwork()  # Use real AgentNetwork
        self.coordinator = None
        self.workers = []
        self.httpx_client = httpx.AsyncClient(timeout=30.0)
        self.output = ColoredOutput()
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        config_file = Path(__file__).parent / config_path
        with open(config_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def _convert_config_for_qa_agent(self, config):
        """Convert configuration format to adapt QA Agent"""
        if not config:
            return None
            
        core_config = config.get('core', {})
        if core_config.get('type') == 'openai':
            return {
                "model": {
                    "type": "openai",
                    "name": core_config.get('name', 'gpt-4o'),
                    "openai_api_key": core_config.get('openai_api_key'),
                    "openai_base_url": core_config.get('openai_base_url', 'https://api.openai.com/v1'),
                    "temperature": core_config.get('temperature', 0.0)
                }
            }
        elif core_config.get('type') == 'local':
            return {
                "model": {
                    "type": "local",
                    "name": core_config.get('name', 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'),
                    "temperature": core_config.get('temperature', 0.0)
                },
                "base_url": core_config.get('base_url', 'http://localhost:8000/v1'),
                "port": core_config.get('port', 8000)
            }
        return None
    
    async def setup_agents(self):
        """Setup real Agent Protocol Agents"""
        self.output.info("Initializing real AgentNetwork and Agent Protocol Agents...")
        
        qa_config = self._convert_config_for_qa_agent(self.config)
        
        # Create Coordinator Agent Protocol Agent - 传递完整配置而不是只传递qa_config
        coordinator_executor = QACoordinatorExecutor(self.config, self.output)
        # 包装为 Agent Protocol 执行器
        ap_coordinator_executor = AgentProtocolExecutor(coordinator_executor, "coordinator")
        
        self.coordinator = await BaseAgent.create_ap(
            agent_id="Coordinator-1",
            host="localhost",
            port=9998,
            executor=ap_coordinator_executor,
            httpx_client=self.httpx_client
        )
        await self.network.register_agent(self.coordinator)
        self.output.success("Coordinator-1 created and registered to AgentNetwork (Agent Protocol)")
        
        # Store coordinator's executor for easy access
        self.coordinator_executor = coordinator_executor
        
        # Create Worker Agent Protocol Agents
        worker_count = self.config['qa']['worker']['count']
        start_port = self.config['qa']['worker']['start_port']
        worker_ids = []
        
        for i in range(worker_count):
            worker_id = f"Worker-{i+1}"
            port = start_port + i
            
            # Create Worker executor - 传递完整配置给Worker
            worker_executor = QAAgentExecutor(qa_config)
            # 包装为 Agent Protocol 执行器
            ap_worker_executor = AgentProtocolExecutor(worker_executor, "worker")
            
            # Create Worker Agent Protocol Agent
            worker = await BaseAgent.create_ap(
                agent_id=worker_id,
                host="localhost",
                port=port,
                executor=ap_worker_executor,
                httpx_client=self.httpx_client
            )
            
            await self.network.register_agent(worker)
            self.workers.append(worker)
            worker_ids.append(worker_id)
            self.output.success(f"{worker_id} created and registered to AgentNetwork (Agent Protocol, port: {port})")
        
        # Set up coordinator with network and worker information
        self.coordinator_executor.coordinator.set_network(self.network, worker_ids)
        
        return worker_ids
    
    async def setup_topology(self):
        """Setup network topology"""
        self.output.info("=== Setting up Network Topology ===")
        
        topology = self.config['qa']['network']['topology']
        
        if topology == "star":
            self.network.setup_star_topology("Coordinator-1")
            self.output.success("Setup star topology with center node: Coordinator-1")
        elif topology == "mesh":
            self.network.setup_mesh_topology()
            self.output.success("Setup mesh topology")
        else:
            self.output.error("Unknown topology type")
            return
        
        # Wait for topology setup to complete
        await asyncio.sleep(1)
        
        # Display topology information
        topology_info = self.network.get_topology()
        edge_count = sum(len(edges) for edges in topology_info.values())
        self.output.system(f"Current topology connection count: {edge_count}")
        
        self.output.info("Detailed connection information:")
        for agent_id, connections in topology_info.items():
            if connections:
                self.output.progress(f"{agent_id} → {list(connections)}")
    
    async def send_message_to_coordinator(self, command: str):
        """Send Agent Protocol message directly to coordinator"""
        coordinator_url = "http://localhost:9998"
        
        try:
            # Step 1: Create a task using Agent Protocol
            task_payload = {
                "input": command,
                "additional_input": {
                    "source": "demo_client",
                    "timestamp": time.time()
                }
            }
            
            self.output.info(f"Creating task with payload: {task_payload}")
            response = await self.httpx_client.post(
                f"{coordinator_url}/ap/v1/agent/tasks",
                json=task_payload,
                timeout=60.0
            )
            response.raise_for_status()
            task_result = response.json()
            task_id = task_result.get("task_id")
            
            self.output.info(f"Task created successfully: {task_id}")
            
            if not task_id:
                return {"result": "Failed to create task"}
            
            # Step 2: Execute a step for the task
            step_payload = {
                "name": f"execute_{command}",
                "input": command,
                "additional_input": {}
            }
            
            self.output.info(f"Executing step with payload: {step_payload}")
            step_response = await self.httpx_client.post(
                f"{coordinator_url}/ap/v1/agent/tasks/{task_id}/steps",
                json=step_payload,
                timeout=60.0
            )
            step_response.raise_for_status()
            step_result = step_response.json()
            
            self.output.info(f"Step executed successfully: {step_result.get('status')}")
            
            return {"result": step_result.get("output", "Command processed")}
            
        except httpx.HTTPStatusError as e:
            self.output.error(f"HTTP {e.response.status_code} error: {e.response.text}")
            return None
        except httpx.TimeoutException as e:
            self.output.error(f"Request timeout: {e}")
            return None
        except Exception as e:
            self.output.error(f"Agent Protocol request to coordinator failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def load_questions(self):
        """Check coordinator status"""
        self.output.info("Checking coordinator status...")
        response = await self.send_message_to_coordinator("status")
        if response:
            self.output.success("Coordinator status checked")
        return []
    
    async def dispatch_questions_dynamically(self, questions: List[Dict]):
        """Start dispatch process via Agent Protocol"""
        self.output.info("Starting dispatch process via Agent Protocol...")
        
        # Send dispatch command to coordinator
        response = await self.send_message_to_coordinator("dispatch")
        if response and "result" in response:
            self.output.success("Dispatch completed!")
            self.output.system(response["result"])
            # Return empty list since results are handled internally by coordinator
            return []
        else:
            self.output.error("Failed to start dispatch")
            return []
    
    async def save_results(self, results):
        """Save results (coordinator handles result saving internally)"""
        self.output.info("Results are handled and saved by coordinator internally")
    
    async def run_health_check(self):
        """Run health check"""
        self.output.info("=== Health Check ===")
        
        health_status = await self.network.health_check()
        healthy_count = sum(1 for status in health_status.values() if status)
        total_count = len(health_status)
        
        self.output.system(f"Health check results ({healthy_count}/{total_count} healthy):")
        for agent_id, status in health_status.items():
            if status:
                self.output.success(f"{agent_id}: Healthy")
            else:
                self.output.error(f"{agent_id}: Failed")
    
    async def run_demo(self):
        """Run complete demo"""
        self.output.info("Real Agent Protocol AgentNetwork QA System Demo")
        print("=" * 60)
        
        try:
            # 1. Setup Agent Protocol Agents
            worker_ids = await self.setup_agents()
            
            # 2. Setup network topology
            await self.setup_topology()
            
            # 3. Health check
            await self.run_health_check()
            
            # 4. Check coordinator status via Agent Protocol
            questions = await self.load_questions()
            
            # 5. Start dispatch process via Agent Protocol
            self.output.info("=== Starting Q&A Processing via Agent Protocol ===")
            start_time = time.time()
            results = await self.dispatch_questions_dynamically(questions)
            end_time = time.time()
            
            # 6. Save results  
            await self.save_results(results)
            
            # 7. Display completion
            self.output.success("Demo completed!")
            self.output.system(f"Total time: {end_time - start_time:.2f} seconds")
            
            # 8. Final health check
            await self.run_health_check()
            
            # 9. Display network metrics
            metrics = self.network.snapshot_metrics()
            self.output.info("Network metrics:")
            self.output.progress(f"Agent count: {metrics['agent_count']}")
            self.output.progress(f"Connection count: {metrics['edge_count']}")
            
        except Exception as e:
            self.output.error(f"Error during demo: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup resources
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        self.output.system("Cleaning up resources...")
        
        # Stop all Agents
        if self.coordinator:
            await self.coordinator.stop()
        
        for worker in self.workers:
            await worker.stop()
        
        # Close HTTP client
        await self.httpx_client.aclose()
        
        self.output.success("Resource cleanup completed")

async def main():
    demo = RealAgentNetworkDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())