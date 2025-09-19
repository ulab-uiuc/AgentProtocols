"""Main runner base for GAIA benchmark using multi-agent framework."""
import asyncio
import sys
import abc
from pathlib import Path
import json
import time
import yaml
import os
import shutil
from typing import Dict, Any, List, Optional
from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf

# GAIA 根目录 (script/gaia)
GAIA_ROOT = Path(__file__).resolve().parent.parent

# Add parent directory to path for imports
sys.path.append(str(GAIA_ROOT))

from core.planner import TaskPlanner


class Tee:
    """Redirect output to both console and file."""
    def __init__(self, console, file):
        self.console = console
        self.file = file
    
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure immediate write
    
    def flush(self):
        self.console.flush()
        self.file.flush()
    
    def isatty(self):
        """Return whether the console is a TTY. Required for uvicorn compatibility."""
        return hasattr(self.console, 'isatty') and self.console.isatty()


class RunnerBase(abc.ABC):
    """Abstract GAIA runner with stepwise hooks and health check."""

    def __init__(self, config_path: str, protocol_name: str = "base") -> None:
        self.protocol_name = protocol_name
        self.config = self._load_config(config_path)
        
        # Initialize runtime configuration
        self.mode = self.config.get("mode", "normal")  # normal | debug | mm
        runtime_config = self.config.get('runtime', {})
        task_file = runtime_config.get('task_file', './script/gaia/dataset/2023/validation/metadata.jsonl')
        # Special multimodal mode: override task file to multimodal.jsonl
        if self.mode == "mm":
            task_file = '/root/Multiagent-Protocol/script/gaia/dataset/2023/validation/multimodal.jsonl'
        # Store resolved task file path for later env wiring
        self._task_file_path = str(Path(task_file).resolve())
        self.tasks = self.load_tasks(task_file)
        self.timeout = runtime_config.get('timeout', 300)
        self.output_file = self._resolve_output_file(runtime_config)

        # Initialize logging
        self._log_file = None
        self._original_stdout = None
        self._original_stderr = None
        self.setup_logging()

    def _resolve_output_file(self, runtime_config: Dict[str, Any]) -> str:
        """解析结果输出文件路径。
        规则:
          - 默认: GAIA_ROOT/workspaces/<protocol_name>/gaia_<protocol_name>_results.json
          - 若 runtime.output_file 提供:
              * 绝对路径: 原样使用
              * 相对路径:
                  - 以 workspaces 开头: GAIA_ROOT/<path>
                  - 否则: GAIA_ROOT/workspaces/<protocol_name>/<path>
        """
        default_output = GAIA_ROOT / 'workspaces' / self.protocol_name / f'gaia_{self.protocol_name}_results.json'
        cfg_output = runtime_config.get('output_file')
        if cfg_output:
            cfg_path = Path(cfg_output)
            if cfg_path.is_absolute():
                return str(cfg_path)
            first_part = cfg_path.parts[0] if cfg_path.parts else ''
            if first_part == 'workspaces':
                return str(GAIA_ROOT / cfg_path)  # 避免重复 workspaces/workspaces
            return str(GAIA_ROOT / 'workspaces' / self.protocol_name / cfg_path)
        return str(default_output)

    def _setup_task_workspace(self, task_id: str, task: Dict[str, Any]) -> Path:
        """
        创建并设置任务专用的工作区目录，复制所需文件到工作区。
        
        Args:
            task_id: 任务ID
            task: 任务数据字典
            
        Returns:
            Path: 任务工作区路径 (workspaces/<protocol_name>/<task_id>)
        """
        # 创建任务工作区目录
        workspace_dir = GAIA_ROOT / 'workspaces' / self.protocol_name / task_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制任务所需文件到工作区
        file_name = task.get("file_name")
        if file_name and isinstance(file_name, str) and file_name.strip():
            # 解析源文件路径 (相对于数据集目录)
            dataset_dir = Path(self._task_file_path).parent
            source_file = dataset_dir / file_name
            
            if source_file.exists():
                # 保持文件名，复制到工作区
                dest_file = workspace_dir / file_name
                dest_file.parent.mkdir(parents=True, exist_ok=True)  # 创建子目录如果需要
                
                try:
                    shutil.copy2(source_file, dest_file)
                    print(f"📄 Copied task file: {file_name} -> {workspace_dir}")
                except Exception as e:
                    print(f"⚠️ Failed to copy task file {file_name}: {e}")
            else:
                print(f"⚠️ Task file not found: {source_file}")
        
        # 复制数据集通用文件（如果存在）
        dataset_dir = Path(self._task_file_path).parent
        common_files = ['multimodal.jsonl', 'metadata.jsonl']
        for common_file in common_files:
            source = dataset_dir / common_file
            if source.exists():
                dest = workspace_dir / common_file
                try:
                    shutil.copy2(source, dest)
                except Exception as e:
                    print(f"⚠️ Failed to copy {common_file}: {e}")
        
        return workspace_dir

    # -------------------- Abstract Methods --------------------
    @abc.abstractmethod
    def create_network(self, general_config: Dict[str, Any]) -> Any:
        """Return a protocol-specific network implementation."""
        raise NotImplementedError

    # -------------------- Logging Management --------------------
    def setup_logging(self, file_name: Optional[str] = None) -> None:
        """
        Setup logging redirection to both console and file.
        
        Args:
            file_name: Optional custom log file name. If None, uses protocol_name with timestamp.
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{self.protocol_name}_runner_{timestamp}.log"
        
        # Create protocol-specific logs directory
        protocol_log_dir = Path(__file__).parent.parent / "logs" / self.protocol_name
        protocol_log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = protocol_log_dir / file_name
        
        try:
            # Store original streams
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            
            # Open log file
            self._log_file = open(log_file_path, 'w', encoding='utf-8')
            
            # Setup tee redirection
            tee_stdout = Tee(self._original_stdout, self._log_file)
            tee_stderr = Tee(self._original_stderr, self._log_file)
            
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr
            
            print(f"🎯 {self.protocol_name.upper()} Runner 启动 - {timestamp}")
            print(f"📝 日志文件: {log_file_path}")
            
        except Exception as e:
            print(f"❌ 无法设置日志重定向: {e}")
            self.restore_logging()
            raise
    
    def restore_logging(self) -> None:
        """Restore original stdout/stderr streams."""
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
            self._original_stdout = None
        
        if self._original_stderr is not None:
            sys.stderr = self._original_stderr
            self._original_stderr = None
        
        if self._log_file is not None:
            try:
                self._log_file.close()
            except:
                pass
            self._log_file = None

    # -------------------- Config Loading Helpers --------------------
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Internal helper to load a YAML config with flexible relative resolution.
        Search order (first hit wins):
          1. Absolute path (if given)
          2. <gaia_root>/config/<config_path>
          3. <gaia_root>/<config_path>
          4. <runners_dir>/<config_path>
        Where:
          - runners_dir = .../script/gaia/runners
          - gaia_root   = runners_dir.parent
        Args:
            config_path: relative filename or path (e.g. 'a2a.yaml' / 'config/a2a.yaml').
        """
        p = Path(config_path)
        if p.is_absolute():
            candidate_list = [p]
        else:
            runners_dir = Path(__file__).parent
            gaia_root = runners_dir.parent
            # If user already passes 'config/xxx.yaml' keep it; still try with explicit config dir
            candidate_list = [
                gaia_root / 'config' / p.name if p.parent == Path('.') else gaia_root / 'config' / p,  # ensure config/<file>
                gaia_root / p,               # gaia_root/<path or file>
                runners_dir / p,             # runners/<path or file>
            ]
            # Avoid duplicates while preserving order
            seen = set()
            uniq = []
            for c in candidate_list:
                if c not in seen:
                    uniq.append(c)
                    seen.add(c)
            candidate_list = uniq
        target = None
        for cand in candidate_list:
            if cand.exists():
                target = cand
                break
        if target is None:
            raise FileNotFoundError("Config file not found; tried:\n" + "\n".join(str(c) for c in candidate_list))
        with target.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
        return data

    def load_tasks(self, task_file: str) -> List[Dict[str, Any]]:
        path = Path(task_file)
        if not path.exists():
            raise FileNotFoundError(f"Task file '{task_file}' not found")
        with open(path, "r", encoding="utf-8") as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        # For quick smoke test, limit if needed (adjust/remove in concrete runners)
        return tasks

    def load_planned_config(self, config_path: Path) -> Dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------- Result Display Helper --------------------
    def display_results(self, total_tasks: int, success_count: int, timeout_count: int, error_count: int) -> None:
        """展示运行结果汇总信息。"""
        print("\n" + "=" * 60)
        print("📊 GAIA Benchmark Results Summary")
        print("=" * 60)
        print(f"📋 Total tasks: {total_tasks}")
        print(f"✅ Successful: {success_count} ({(success_count/total_tasks*100 if total_tasks else 0):.1f}%)")
        print(f"⏰ Timeout: {timeout_count} ({(timeout_count/total_tasks*100 if total_tasks else 0):.1f}%)")
        print(f"❌ Errors: {error_count} ({(error_count/total_tasks*100 if total_tasks else 0):.1f}%)")
        print(f"💾 Results saved to: {self.output_file}")
        print("=" * 60)

    # -------------------- Stepwise hooks --------------------
    async def plan(self, task_id: str, level: int, task_doc: str, workspace_dir: Path) -> Path:
        """
        Plan agents for the task using the dedicated workspace directory.
        
        Args:
            task_id: Task ID
            level: Task level
            task_doc: Task description
            workspace_dir: Task-specific workspace directory
            
        Returns:
            Path: Agent configuration file path
        """
        # 将当前 Runner 的协议名和工作区路径传入 Planner
        planner = TaskPlanner(task_id=task_id, level=level, protocol_name=self.protocol_name)
        # 传入工作区路径，让 planner 在该目录下工作
        config_path = await planner.plan_agents(task_doc, workspace_dir=workspace_dir)
        return Path(config_path)

    async def check_health(self, network: Any) -> None:
        ok = await network.monitor_agent_health()
        if not ok:
            raise RuntimeError("Network health check failed")

    async def start_network(self, network: Any) -> None:
        await network.start()

    async def run_workflow(self, network: Any, general_config: Dict[str, Any], task_doc: str, timeout: int) -> str:
        return await asyncio.wait_for(network.execute_workflow(general_config, task_doc), timeout=timeout)

    async def stop_network(self, network: Any) -> None:
        await network.stop()

    # -------------------- Single task --------------------
    async def run_single_task(self, task: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        start_time = time.time()
        task_id = task.get("task_id", f"task_{int(start_time)}")
        task_doc = task.get("Question")
        ground_truth = task.get("Final answer", "")
        level = task.get("Level", 1)
        # Ensure general_config is always defined for error/timeout paths
        general_config: Dict[str, Any] = {}

        if not task_doc:
            return {
                "task_id": task_id,
                "question": "",
                "ground_truth": "",
                "predicted_answer": "ERROR: No question found in task",
                "execution_time": 0.0,
                "status": "error",
                "level": level,
            }

        try:
            # 设置任务专用工作区并复制所需文件
            workspace_dir = self._setup_task_workspace(task_id, task)
            
            # Plan -> Config (使用工作区路径)
            config_path = await self.plan(task_id, level, task_doc, workspace_dir)
            general_config = self.load_planned_config(config_path)

            # Merge network configuration from runner config into planned config
            network_config = self.config.get("network", {})
            if network_config:
                # Convert to OmegaConf for easier merging
                planned_conf = OmegaConf.create(general_config)
                network_conf = OmegaConf.create({"network": network_config})
                
                # Merge network config into planned config
                merged_config = OmegaConf.merge(planned_conf, network_conf)
                general_config = OmegaConf.to_container(merged_config, resolve=True)
                
                print(f"🔗 Merged network config: timeout={network_config.get('timeout_seconds', 'default')}")

            # Expose workspace and dataset dirs to tools via environment variables
            try:
                # 使用任务专用的工作区目录
                os.environ["GAIA_AGENT_WORKSPACE_DIR"] = str(workspace_dir)
                os.environ["GAIA_WORKSPACE_DIR"] = str(workspace_dir)  # 向后兼容
                print(f"📁 Task workspace created at {workspace_dir}")
            except Exception:
                pass
            try:
                dataset_dir = str(Path(self._task_file_path).parent)
                os.environ["GAIA_DATASET_DIR"] = dataset_dir
            except Exception:
                pass
            try:
                os.environ["GAIA_ROOT"] = str(GAIA_ROOT)
            except Exception:
                pass

            # Create network (abstract)
            network = self.create_network(general_config)

            # Start -> Health check -> Execute -> Stop
            await self.start_network(network)
            await self.check_health(network)
            try:
                result = await self.run_workflow(network, general_config, task_doc, timeout)
                execution_time = time.time() - start_time
                
                # Use enhanced LLM as judge to evaluate answer quality
                from core.llm_judge import create_llm_judge
                
                # Get judge timeout from config
                judge_timeout = general_config.get("evaluation", {}).get("judge_timeout", 30)
                judge = create_llm_judge(judge_timeout=judge_timeout)
                
                # Judge the answer with comprehensive handling
                judgment = await judge.judge_answer(
                    question=task_doc,
                    ground_truth=ground_truth,
                    predicted_answer=result,
                    execution_time=execution_time,
                    status="success"
                )
                
                # Format result using the enhanced judge
                return judge.format_judgment_for_result(
                    judgment=judgment,
                    task_id=task_id,
                    question=task_doc,
                    ground_truth=ground_truth,
                    predicted_answer=result,
                    execution_time=execution_time,
                    level=level
                )
            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                
                # Use enhanced LLM judge for timeout case
                from core.llm_judge import create_llm_judge
                judge_timeout = general_config.get("evaluation", {}).get("judge_timeout", 30)
                judge = create_llm_judge(judge_timeout=judge_timeout)
                
                # Judge the timeout case
                judgment = await judge.judge_answer(
                    question=task_doc,
                    ground_truth=ground_truth,
                    predicted_answer="TIMEOUT: Framework execution exceeded time limit",
                    execution_time=execution_time,
                    status="timeout"
                )
                
                # Format result using the enhanced judge
                return judge.format_judgment_for_result(
                    judgment=judgment,
                    task_id=task_id,
                    question=task_doc,
                    ground_truth=ground_truth,
                    predicted_answer="TIMEOUT: Framework execution exceeded time limit",
                    execution_time=execution_time,
                    level=level
                )
            finally:
                await self.stop_network(network)
        except NotImplementedError:
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Use enhanced LLM judge for error case
            from core.llm_judge import create_llm_judge
            judge_timeout = general_config.get("evaluation", {}).get("judge_timeout", 30)
            judge = create_llm_judge(judge_timeout=judge_timeout)
            
            # Judge the error case
            judgment = await judge.judge_answer(
                question=task_doc,
                ground_truth=ground_truth,
                predicted_answer=f"ERROR: {e}",
                execution_time=execution_time,
                status="error"
            )
            
            # Format result using the enhanced judge
            return judge.format_judgment_for_result(
                judgment=judgment,
                task_id=task_id,
                question=task_doc,
                ground_truth=ground_truth,
                predicted_answer=f"ERROR: {e}",
                execution_time=execution_time,
                level=level
            )

    # -------------------- Common Workflow --------------------
    async def run(self) -> None:
        """Main runner orchestration (load tasks, iterate, save results)."""
        tasks = self.tasks
        # For fast iteration/dev, limit here if needed
        runtime_config = self.config.get('runtime', {})
        max_tasks = runtime_config.get('max_tasks', None)
        
        if self.mode == "debug":
            # Debug mode: limit to 1 task
            if isinstance(tasks, list) and tasks:
                tasks = tasks[:3]
            else:
                print(f"[WARN] Debug 模式下任务结构异常，tasks 类型: {type(tasks).__name__}")
                tasks = [] if tasks is None else ([tasks] if isinstance(tasks, dict) else [])
        elif self.mode == "mm":
            # Multimodal mode: only run first 3 tasks
            if isinstance(tasks, list) and tasks:
                tasks = tasks[:3]
                print("🎯 MM 模式：仅运行前 3 条任务 (multimodal.jsonl)")
            else:
                print(f"[WARN] MM 模式下任务结构异常，tasks 类型: {type(tasks).__name__}")
                tasks = [] if tasks is None else ([tasks] if isinstance(tasks, dict) else [])
        elif max_tasks and isinstance(max_tasks, int) and max_tasks > 0:
            # Normal mode with max_tasks limit
            if isinstance(tasks, list) and tasks:
                tasks = tasks[:max_tasks]
                print(f"📊 Limited to {max_tasks} tasks for batch testing")

        # 额外类型校验，确保后续逻辑安全
        if not isinstance(tasks, list):
            raise TypeError(f"Tasks must be a list of dicts, got {type(tasks).__name__}")

        print(f"📄 Loaded {len(tasks)} GAIA tasks")
        print(f"⏰ Timeout per task: {self.timeout}s")
        print(f"💾 Results will be saved to: {self.output_file}")
        print("-" * 60)

        results: List[Dict[str, Any]] = []
        failed_tasks: List[Dict[str, Any]] = []
        success_count = 0
        timeout_count = 0
        error_count = 0

        with tqdm(total=len(tasks), desc="Processing GAIA tasks", unit="task") as pbar:
            for task in tasks:
                try:
                    result = await self.run_single_task(task, self.timeout)
                    results.append(result)

                    if result["status"] == "success":
                        success_count += 1
                    elif result["status"] == "timeout":
                        timeout_count += 1
                    else:
                        error_count += 1
                        failed_tasks.append(result)

                    pbar.set_postfix({
                        "Success": success_count,
                        "Timeout": timeout_count,
                        "Error": error_count,
                    })
                except NotImplementedError:
                    raise
                except Exception as e:
                    error_count += 1
                    error_result = {
                        "task_id": task.get("task_id", "unknown") if isinstance(task, dict) else "unknown",
                        "question": task.get("Question", "") if isinstance(task, dict) else str(task),
                        "ground_truth": task.get("Final answer", "") if isinstance(task, dict) else "",
                        "predicted_answer": f"EXECUTION_ERROR: {e}",
                        "execution_time": 0.0,
                        "status": "execution_error",
                        "level": task.get("Level", 1) if isinstance(task, dict) else 1,
                    }
                    results.append(error_result)
                    failed_tasks.append(error_result)
                    pbar.set_postfix({
                        "Success": success_count,
                        "Timeout": timeout_count,
                        "Error": error_count,
                    })
                pbar.update(1)

        output_data = {
            "metadata": {
                "total_tasks": len(tasks),
                "successful_tasks": success_count,
                "timeout_tasks": timeout_count,
                "error_tasks": error_count,
                "success_rate": (success_count / len(tasks) * 100) if tasks else 0.0,
                "timeout_per_task": self.timeout,
                "execution_timestamp": time.time(),
            },
            "results": results,
            "failed_tasks": failed_tasks,
        }

        out_path = Path(self.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # 调用封装的结果展示函数
        self.display_results(len(tasks), success_count, timeout_count, error_count)


async def _main():
    runner = RunnerBase()
    try:
        await runner.run()
    except NotImplementedError:
        print("RunnerBase is abstract. Please run a concrete protocol runner instead.")


if __name__ == "__main__":
    asyncio.run(_main())
