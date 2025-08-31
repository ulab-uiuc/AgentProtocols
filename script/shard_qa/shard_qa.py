#!/usr/bin/env python3
"""
8 Agent Ring Collaborative Retrieval Benchmark v1.1
Multi-Question Multi-Fragment with Shuffled Fragments and OpenAI Function-Calling Tools
"""
import asyncio
import json
import yaml
import time
import sys
import signal
from pathlib import Path
from typing import Dict, List, Any
import httpx

# Add colorama for colored output
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
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

# Add paths to import AgentNetwork and protocols
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from network import AgentNetwork
from base_agent import BaseAgent

# Import Shard Agent Executors
sys.path.insert(0, str(Path(__file__).parent / "shard_worker"))
sys.path.insert(0, str(Path(__file__).parent / "shard_coordinator"))
from shard_worker.agent_executor import ShardWorkerExecutor
from shard_coordinator.agent_executor import ShardCoordinatorExecutor

# Global shutdown event
shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    """Handle Ctrl+C signal"""
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}🛑 Received interrupt signal. Shutting down gracefully...{Style.RESET_ALL}")
    shutdown_event.set()

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class ColoredOutput:
    """Helper class for colored console output with logging"""
    
    def __init__(self):
        """Initialize with timestamped log file"""
        import datetime
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = Path(__file__).parent / "logs" / f"shard_qa_{self.timestamp}.log"
        
        # Ensure logs directory exists
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Open log file
        self.log_handle = open(self.log_file, 'w', encoding='utf-8')
        self._log_plain(f"=== Shard QA Log Started at {datetime.datetime.now()} ===")
    
    def _log_plain(self, message: str) -> None:
        """Write plain message to log file"""
        if not hasattr(self, 'log_handle') or not self.log_handle or self.log_handle.closed:
            return  # 日志文件已关闭，静默跳过
        
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]  # 毫秒精度
            self.log_handle.write(f"[{timestamp}] {message}\n")
            self.log_handle.flush()  # 立即写入文件
        except (ValueError, OSError):
            # 文件已关闭或其他I/O错误，静默跳过
            pass
    
    def info(self, message: str) -> None:
        """Print info message in blue and log it"""
        colored_msg = f"{Fore.BLUE}{Style.BRIGHT}[INFO]  {message}{Style.RESET_ALL}"
        plain_msg = f"[INFO]  {message}"
        print(colored_msg)
        self._log_plain(plain_msg)
    
    def success(self, message: str) -> None:
        """Print success message in green and log it"""
        colored_msg = f"{Fore.GREEN}{Style.BRIGHT}[OK] {message}{Style.RESET_ALL}"
        plain_msg = f"[OK] {message}"
        print(colored_msg)
        self._log_plain(plain_msg)
    
    def warning(self, message: str) -> None:
        """Print warning message in yellow and log it"""
        colored_msg = f"{Fore.YELLOW}{Style.BRIGHT}⚠️  {message}{Style.RESET_ALL}"
        plain_msg = f"⚠️  {message}"
        print(colored_msg)
        self._log_plain(plain_msg)
    
    def error(self, message: str) -> None:
        """Print error message in red and log it"""
        colored_msg = f"{Fore.RED}{Style.BRIGHT}[ERROR] {message}{Style.RESET_ALL}"
        plain_msg = f"[ERROR] {message}"
        print(colored_msg)
        self._log_plain(plain_msg)
    
    def system(self, message: str) -> None:
        """Print system status in cyan and log it"""
        colored_msg = f"{Fore.CYAN}{Style.BRIGHT}[CONFIG] {message}{Style.RESET_ALL}"
        plain_msg = f"[CONFIG] {message}"
        print(colored_msg)
        self._log_plain(plain_msg)
    
    def progress(self, message: str) -> None:
        """Print progress message in white and log it"""
        colored_msg = f"{Fore.WHITE}   {message}{Style.RESET_ALL}"
        plain_msg = f"   {message}"
        print(colored_msg)
        self._log_plain(plain_msg)
    
    def close(self) -> None:
        """Close log file"""
        if hasattr(self, 'log_handle') and self.log_handle and not self.log_handle.closed:
            try:
                import datetime
                self._log_plain(f"=== Shard QA Log Ended at {datetime.datetime.now()} ===")
                self.log_handle.close()
                print(f"{Fore.CYAN}{Style.BRIGHT}📁 Log saved to: {self.log_file}{Style.RESET_ALL}")
            except (ValueError, OSError):
                # 文件已关闭或其他I/O错误，静默跳过
                pass
    
    def __del__(self):
        """Ensure log file is closed on destruction"""
        try:
            self.close()
        except:
            # 忽略析构函数中的任何错误
            pass

class ShardQADemo:
    """8 Agent Ring Collaborative Retrieval System Demo"""
    
    def __init__(self, config_path="config.yaml"):
        self.config = self.load_config(config_path)
        self.network = AgentNetwork()
        self.coordinator = None
        self.workers = []  # List of 8 shard workers
        self.httpx_client = httpx.AsyncClient(timeout=5.0)  # 与业务逻辑匹配的合理超时
        self.output = ColoredOutput()  # 现在是实例而不是静态类
        self.current_group_id = 0
        self.independent_mode = False  # Flag for independent processing mode
        
        # 设置全局异常处理器，防止未捕获的异常导致崩溃
        self._setup_exception_handlers()
    
    def _setup_exception_handlers(self):
        """设置全局异常处理器，防止未捕获异常导致agent崩溃"""
        import asyncio
        import sys
        
        def handle_exception(loop, context):
            """处理asyncio事件循环中的未捕获异常"""
            exception = context.get('exception')
            if exception:
                if isinstance(exception, asyncio.CancelledError):
                    # CancelledError是正常的取消操作，只记录但不报错
                    if hasattr(self, 'output') and self.output:
                        self.output.progress(f"Task cancelled: {context.get('message', 'Unknown')}")
                else:
                    # 其他异常需要记录
                    if hasattr(self, 'output') and self.output:
                        self.output.error(f"Unhandled exception in event loop: {exception}")
                        self.output.error(f"Context: {context}")
            else:
                if hasattr(self, 'output') and self.output:
                    self.output.warning(f"Event loop error: {context}")
        
        # 设置asyncio异常处理器
        try:
            loop = asyncio.get_event_loop()
            loop.set_exception_handler(handle_exception)
        except RuntimeError:
            # 如果没有运行中的event loop，稍后再设置
            pass
        
        # 设置系统级异常处理器
        def sys_exception_handler(exc_type, exc_value, exc_traceback):
            if hasattr(self, 'output') and self.output:
                self.output.error(f"Unhandled system exception: {exc_type.__name__}: {exc_value}")
            # 调用默认的异常处理器
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        
        sys.excepthook = sys_exception_handler
    
    def _check_port_available(self, host: str, port: int) -> bool:
        """检查端口是否可用"""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result != 0  # 0表示连接成功（端口被占用）
        except Exception:
            return False
    
    async def _wait_for_port_release(self, host: str, port: int, max_wait: int = 10) -> bool:
        """等待端口释放"""
        for i in range(max_wait):
            if self._check_port_available(host, port):
                return True
            self.output.progress(f"Waiting for port {port} to be released... ({i+1}/{max_wait})")
            await asyncio.sleep(1)
        return False
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        config_file = Path(__file__).parent / config_path
        with open(config_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def _convert_config_for_shard_agent(self, config):
        """Convert configuration format to adapt Shard Agent"""
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
                    "temperature": core_config.get('temperature', 0.0),
                    "max_tokens": core_config.get('max_tokens', 4096)
                }
            }
        elif core_config.get('type') == 'local':
            return {
                "model": {
                    "type": "local",
                    "name": core_config.get('name', 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'),
                    "temperature": core_config.get('temperature', 0.0),
                    "max_tokens": core_config.get('max_tokens', 4096)
                },
                "base_url": core_config.get('base_url', 'http://localhost:8000/v1'),
                "port": core_config.get('port', 8000)
            }
        return None
    
    async def setup_agents(self):
        """Setup 8 Shard Worker Agents and 1 Coordinator"""
        self.output.info("Initializing 8 Agent Ring Collaborative Retrieval System...")
        
        # Check for shutdown signal
        if shutdown_event.is_set():
            return []
        
        shard_config = self._convert_config_for_shard_agent(self.config)
        worker_config = self.config['shard_qa']['workers']
        coordinator_config = self.config['shard_qa']['coordinator']
        
        # Create Coordinator Agent
        coordinator_executor = ShardCoordinatorExecutor(shard_config, self.config, self.output)
        self.coordinator = await BaseAgent.create_a2a(
            agent_id="coordinator",
            host="localhost",
            port=coordinator_config['start_port'],
            executor=coordinator_executor,
            httpx_client=self.httpx_client
        )
        await self.network.register_agent(self.coordinator)
        self.output.success("Coordinator created and registered to AgentNetwork")
        
        # Store coordinator's executor for easy access
        self.coordinator_executor = coordinator_executor
        
        # Create 8 Shard Worker Agents (shard0 - shard7)
        worker_count = worker_config['count']
        start_port = worker_config['start_port']
        data_files = worker_config['data_files']
        ring_config = self.config['shard_qa']['ring_config']
        
        worker_ids = []
        
        for i in range(worker_count):
            # Check for shutdown signal
            if shutdown_event.is_set():
                break
                
            shard_id = f"shard{i}"
            port = start_port + i
            data_file = data_files[i]
            
            # Get ring neighbors
            neighbors = ring_config[shard_id]
            
            # Create Shard Worker executor
            worker_executor = ShardWorkerExecutor(
                shard_config, 
                self.config,
                shard_id=shard_id,
                data_file=data_file,
                neighbors=neighbors,
                output=self.output
            )
            
            # Store executor reference for later network setup
            self.worker_executors = getattr(self, 'worker_executors', [])
            self.worker_executors.append(worker_executor)
            
            # Create Shard Worker A2A Agent with robust port conflict handling
            original_port = port
            max_retries = 3
            
            # 预检查端口可用性
            if not self._check_port_available("localhost", port):
                self.output.warning(f"Port {port} for {shard_id} is already occupied")
                if await self._wait_for_port_release("localhost", port, max_wait=5):
                    self.output.success(f"Port {port} is now available for {shard_id}")
                else:
                    self.output.warning(f"Port {port} still occupied, trying alternative ports")
                    port = original_port + 100  # 使用备用端口
            
            for retry in range(max_retries):
                try:
                    # 再次检查端口（避免时间窗口问题）
                    if not self._check_port_available("localhost", port):
                        self.output.warning(f"Port {port} occupied just before creation, finding alternative...")
                        port = original_port + 100 + retry * 10
                        if not self._check_port_available("localhost", port):
                            raise OSError(f"No available ports found for {shard_id}")
                    
                    worker = await BaseAgent.create_a2a(
                        agent_id=shard_id,
                        host="localhost",
                        port=port,
                        executor=worker_executor,
                        httpx_client=self.httpx_client
                    )
                    
                    await self.network.register_agent(worker)
                    self.workers.append(worker)
                    worker_ids.append(shard_id)
                    
                    if port != original_port:
                        self.output.warning(f"{shard_id} created on alternative port {port} (original: {original_port})")
                    else:
                        self.output.success(f"{shard_id} created and registered to AgentNetwork (port: {port})")
                    break  # 成功创建，跳出重试循环
                    
                except OSError as e:
                    if "10048" in str(e) or "address already in use" in str(e).lower():
                        self.output.warning(f"Port {port} for {shard_id} is occupied, attempt {retry+1}/{max_retries}")
                        if retry < max_retries - 1:
                            port = original_port + 100 + (retry + 1) * 10  # 尝试使用更高的端口
                            await asyncio.sleep(2)  # 等待2秒再重试
                        else:
                            self.output.error(f"Failed to create {shard_id} after {max_retries} attempts - all ports occupied")
                            raise
                    else:
                        self.output.error(f"Unexpected OSError creating {shard_id}: {e}")
                        raise
                except Exception as e:
                    self.output.error(f"Failed to create {shard_id}: {e}")
                    if retry < max_retries - 1:
                        await asyncio.sleep(1)
                    else:
                        raise
        
        # Set network reference for all workers
        for worker_executor in self.worker_executors:
            if hasattr(worker_executor, 'worker'):
                worker_executor.worker.set_network(self.network)
        
        # Set up coordinator with network and worker information
        self.coordinator_executor.coordinator.set_network(self.network, worker_ids)
        
        # Register coordinator message handler for A2A messages from workers
        await self._register_coordinator_handler()
        
        return worker_ids
    
    async def _register_coordinator_handler(self):
        """Register coordinator to handle A2A messages from workers"""
        # 暂时禁用自定义handler，让所有消息通过execute()方法处理
        # 这样可以避免A2A框架中的NoneType错误
        if self.output:
            self.output.system("Using execute() method for message handling (handler registration disabled)")
    
    async def setup_topology(self):
        """Setup ring network topology"""
        self.output.info("=== Setting up Ring Network Topology ===")
        
        # Check for shutdown signal
        if shutdown_event.is_set():
            return
        
        # Setup ring topology: each agent connects to prev and next
        ring_config = self.config['shard_qa']['ring_config']
        
        for shard_id, neighbors in ring_config.items():
            # Check for shutdown signal
            if shutdown_event.is_set():
                break
                
            prev_id = neighbors['prev_id']
            next_id = neighbors['next_id']
            
            # Connect to previous and next neighbors
            await self.network.connect_agents(shard_id, prev_id)
            await self.network.connect_agents(shard_id, next_id)
            
            # Bidirectional connection with coordinator
            await self.network.connect_agents(shard_id, "coordinator")
            await self.network.connect_agents("coordinator", shard_id)
        
        # Wait for topology setup to complete
        await asyncio.sleep(2)
        
        # Display topology information
        topology_info = self.network.get_topology()
        edge_count = sum(len(edges) for edges in topology_info.values())
        self.output.system(f"Ring topology connection count: {edge_count}")
        
        self.output.info("Ring connection details:")
        for agent_id, connections in topology_info.items():
            if connections:
                self.output.progress(f"{agent_id} <-> {list(connections)}")
    
    async def run_benchmark_group(self, group_id: int):
        """Run benchmark for a specific group - each agent processes their own question but can communicate"""
        self.output.info(f"=== Running 8 Agent Ring Benchmark for Group {group_id} ===")
        self.output.system("Each agent gets their own question but can communicate with neighbors")
        
        # Check for shutdown signal
        if shutdown_event.is_set():
            return None
        
        # Use ring communication mode
        broadcast_message = f"Process GROUP_ID = {group_id} independently. You have your own question to answer, but you can communicate with neighbors if needed."
        
        start_time = time.time()
        
        # Send message to coordinator to start the benchmark
        response = await self.send_message_to_coordinator(broadcast_message)
        
        if response and "result" in response:
            end_time = time.time()
            self.output.success(f"Group {group_id} completed!")
            self.output.system(f"Time: {end_time - start_time:.2f} seconds")
            self.output.system(response["result"])
            return response
        else:
            self.output.error(f"Failed to complete Group {group_id}")
            return None
    
    async def send_message_to_coordinator(self, command: str):
        """Send HTTP message directly to coordinator"""
        coordinator_url = "http://localhost:9998"
        
        # Use A2A message format - ensure all data is JSON serializable
        message_payload = {
            "id": str(time.time_ns()),
            "params": {
                "message": {
                    "role": "user",  # Use string instead of Role object
                    "parts": [
                        {
                            "kind": "text",
                            "text": str(command)
                        }
                    ],
                    "messageId": str(time.time_ns())
                }
            }
        }
        
        try:
            response = await self.httpx_client.post(
                f"{coordinator_url}/message",
                json=message_payload,
                timeout=60.0
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract text response from events array
            if "events" in result and result["events"]:
                for event in result["events"]:
                    if event.get("type") == "agent_text_message":
                        return {"result": event.get("data", event.get("text", str(event)))}
            
            return {"result": "Command processed"}
            
        except Exception as e:
            self.output.error(f"HTTP request to coordinator failed: {e}")
            return None
    
    async def run_health_check(self):
        """Run health check"""
        self.output.info("=== Health Check ===")
        
        # Check for shutdown signal
        if shutdown_event.is_set():
            return
        
        health_status = await self.network.health_check()
        healthy_count = sum(1 for status in health_status.values() if status)
        total_count = len(health_status)
        
        self.output.system(f"Health check results ({healthy_count}/{total_count} healthy):")
        for agent_id, status in health_status.items():
            if status:
                self.output.success(f"{agent_id}: Healthy")
            else:
                self.output.error(f"{agent_id}: Failed")
    
    async def run_demo(self, target_groups: List[int] = None):
        """Run complete demo"""
        self.output.info("8 Agent Ring Collaborative Retrieval Benchmark v1.1")
        print("=" * 70)
        
        try:
            # 1. Setup Shard Agents
            worker_ids = await self.setup_agents()
            
            if shutdown_event.is_set():
                self.output.warning("Shutdown requested during agent setup")
                return
            
            # 2. Setup ring topology
            await self.setup_topology()
            
            if shutdown_event.is_set():
                self.output.warning("Shutdown requested during topology setup")
                return
            
            # 3. Health check
            await self.run_health_check()
            
            if shutdown_event.is_set():
                self.output.warning("Shutdown requested during health check")
                return
            
            # 4. Run benchmark groups
            if target_groups is None:
                target_groups = list(range(min(5, self.config['shard_qa']['coordinator']['total_groups'])))
            
            self.output.info(f"=== Running Benchmark for Groups: {target_groups} ===")
            
            all_results = []
            total_start_time = time.time()
            
            for group_id in target_groups:
                if shutdown_event.is_set():
                    self.output.warning("Shutdown requested during benchmark execution")
                    break
                    
                result = await self.run_benchmark_group(group_id)
                if result:
                    all_results.append(result)
                await asyncio.sleep(1)  # Brief pause between groups
            
            total_end_time = time.time()
            
            # 5. Summary statistics
            if not shutdown_event.is_set():
                self.output.success("Benchmark completed!")
                self.output.system(f"Total groups processed: {len(all_results)}")
                self.output.system(f"Total time: {total_end_time - total_start_time:.2f} seconds")
                
                # 6. Final health check
                await self.run_health_check()
                
                # 7. Display network metrics
                metrics = self.network.snapshot_metrics()
                self.output.info("Final network metrics:")
                self.output.progress(f"Agent count: {metrics['agent_count']}")
                self.output.progress(f"Connection count: {metrics['edge_count']}")
            
        except Exception as e:
            if not shutdown_event.is_set():
                self.output.error(f"Error during demo: {e}")
                import traceback
                traceback.print_exc()
        
        finally:
            # Cleanup resources
            await self.cleanup()
    
    async def cleanup(self):
        """强化的资源清理，防止资源泄漏导致的崩溃"""
        self.output.system("Cleaning up resources...")
        
        cleanup_errors = []
        
        try:
            # 1. 停止所有Workers（优先清理，避免连接残留）
            if self.workers:
                self.output.progress("Stopping worker agents...")
                for i, worker in enumerate(self.workers):
                    try:
                        if worker:
                            await asyncio.wait_for(worker.stop(), timeout=3.0)
                            self.output.progress(f"Worker {i} stopped")
                    except Exception as e:
                        cleanup_errors.append(f"Worker {i} stop failed: {e}")
                        
            # 2. 停止Coordinator
            if self.coordinator:
                try:
                    self.output.progress("Stopping coordinator...")
                    await asyncio.wait_for(self.coordinator.stop(), timeout=3.0)
                    self.output.progress("Coordinator stopped")
                except Exception as e:
                    cleanup_errors.append(f"Coordinator stop failed: {e}")
            
            # 3. 清理网络连接
            try:
                if hasattr(self, 'network') and self.network:
                    self.output.progress("Cleaning up network connections...")
                    # 这里可以添加网络清理逻辑
            except Exception as e:
                cleanup_errors.append(f"Network cleanup failed: {e}")
            
            # 4. 关闭HTTP客户端
            try:
                if hasattr(self, 'httpx_client') and self.httpx_client:
                    self.output.progress("Closing HTTP client...")
                    await asyncio.wait_for(self.httpx_client.aclose(), timeout=2.0)
            except Exception as e:
                cleanup_errors.append(f"HTTP client close failed: {e}")
            
            # 5. 强制垃圾回收
            import gc
            gc.collect()
            
            if cleanup_errors:
                self.output.warning(f"Cleanup completed with {len(cleanup_errors)} errors:")
                for error in cleanup_errors:
                    self.output.warning(f"  - {error}")
            else:
                self.output.success("Resource cleanup completed successfully")
                
        except Exception as e:
            self.output.error(f"Critical error during cleanup: {e}")
        finally:
            # 6. 确保日志文件正确关闭（最优先）
            try:
                if hasattr(self, 'output') and self.output:
                    self.output.close()
            except:
                pass  # 日志关闭失败也不要影响程序退出

async def run_quick_demo():
    """Run a quick demonstration with test data"""
    print("🔄 8 Agent Ring Collaborative Retrieval Demo")
    print("=" * 50)
    
    # Initialize demo
    demo = ShardQADemo("config.yaml")
    
    # Run with first 3 groups for quick demo
    target_groups = [0, 1, 2]
    
    await demo.run_demo(target_groups)

async def run_full_demo():
    """Run full demonstration with all test data"""
    print("🔄 8 Agent Ring Collaborative Retrieval - Full Demo")
    print("=" * 60)
    
    # Initialize demo
    demo = ShardQADemo("config.yaml")
    
    # Run with all 5 test groups
    target_groups = [0, 1, 2, 3, 4]
    
    await demo.run_demo(target_groups)

async def main():
    """Main entry point with signal handling"""
    import argparse
    
    parser = argparse.ArgumentParser(description='8 Agent Ring Collaborative Retrieval Benchmark')
    parser.add_argument('--groups', type=str, help='Comma-separated group IDs to run (e.g., "0,1,2")')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--mode', type=str, choices=['quick', 'full', 'custom'], 
                       default='custom', help='Demo mode: quick (3 groups), full (5 groups), or custom')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'quick':
            print("Running quick demo (3 groups)...")
            await run_quick_demo()
        elif args.mode == 'full':
            print("Running full demo (5 groups)...")
            await run_full_demo()
        else:
            # Custom mode - now uses independent mode by default
            target_groups = None
            if args.groups:
                target_groups = [int(x.strip()) for x in args.groups.split(',')]
            else:
                # Default to first 2 groups
                target_groups = [0, 1]
            
            demo = ShardQADemo(args.config)
            demo.independent_mode = True  # All modes now use independent mode
            await demo.run_demo(target_groups)
    
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}[STOP] Keyboard interrupt received. Shutting down...{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}{Style.BRIGHT}[ERROR] Unexpected error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"{Fore.GREEN}{Style.BRIGHT}👋 Goodbye!{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}🛑 Program interrupted by user{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}{Style.BRIGHT}[FATAL] Fatal error: {e}{Style.RESET_ALL}")
        sys.exit(1) 