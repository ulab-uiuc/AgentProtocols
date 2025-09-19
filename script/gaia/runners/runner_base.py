"""Main runner base for GAIA benchmark using multi-agent framework."""
import asyncio
import sys
import abc
from pathlib import Path
import json
import time
import yaml
import os
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

    # -------------------- Helper: resolve used file from task --------------------
    def _resolve_used_file(self, task: Dict[str, Any]) -> Optional[str]:
        """从 task 中解析 file_name，并解析为绝对路径。
        - 若 file_name 为空或无效，返回 None
        - 若为相对路径，则相对当前任务文件所在目录进行解析
        """
        try:
            if not isinstance(task, dict):
                return None
            fn = task.get("file_name")
            if not isinstance(fn, str) or not fn.strip():
                return None
            p = Path(fn)
            if not p.is_absolute():
                base = Path(self._task_file_path).parent
                p = (base / fn).resolve()
            return str(p)
        except Exception:
            return None

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
    async def plan(self, task_id: str, level: int, task_doc: str, used_file: Optional[str] = None) -> Path:
        # 将当前 Runner 的协议名传入 Planner，确保工作区路径为 workspaces/<protocol>/<task_id>
        # 以及按对应协议的资源/配置做规划
        planner = TaskPlanner(task_id=task_id, level=level, protocol_name=self.protocol_name)
        config_path = await planner.plan_agents(task_doc, used_file=used_file)
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
            # 解析可选附件文件路径（绝对路径）
            used_file: Optional[str] = self._resolve_used_file(task)

            # Plan -> Config
            config_path = await self.plan(task_id, level, task_doc, used_file=used_file)
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
                config_dir = str(Path(config_path).parent)
                os.environ["GAIA_WORKSPACE_DIR"] = config_dir
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
                tasks = tasks[:1]
            else:
                print(f"[WARN] Debug 模式下任务结构异常，tasks 类型: {type(tasks).__name__}")
                tasks = [] if tasks is None else ([tasks] if isinstance(tasks, dict) else [])
        elif self.mode == "mm":
            # Multimodal mode: only run first 3 tasks
            if isinstance(tasks, list) and tasks:
                tasks = tasks[1:3]
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
