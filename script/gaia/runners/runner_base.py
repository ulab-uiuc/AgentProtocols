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
        # Preserve the original config path so downstream components (TaskPlanner)
        # can load the same file if they need to. Keep as Path for easy resolution.
        try:
            self._config_path = Path(config_path) if config_path else None
        except Exception:
            self._config_path = None
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

        def _append_mode_suffix(p: Path) -> Path:
            """If runner is in debug/mm mode, append _{mode} before file suffix."""
            try:
                mode = getattr(self, 'mode', None)
            except Exception:
                mode = None
            if mode in ('debug', 'mm'):
                stem = p.stem
                suffix = p.suffix
                new_name = f"{stem}_{mode}{suffix}"
                return p.with_name(new_name)
            return p

        if cfg_output:
            cfg_path = Path(cfg_output)
            if cfg_path.is_absolute():
                return str(_append_mode_suffix(cfg_path))
            first_part = cfg_path.parts[0] if cfg_path.parts else ''
            if first_part == 'workspaces':
                return str(_append_mode_suffix(GAIA_ROOT / cfg_path))  # 避免重复 workspaces/workspaces
            return str(_append_mode_suffix(GAIA_ROOT / 'workspaces' / self.protocol_name / cfg_path))
        return str(_append_mode_suffix(default_output))

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
        
        # 复制任务所需文件到工作区并设置环境变量
        file_name = task.get("file_name")
        copied_files = []
        
        if file_name and isinstance(file_name, str) and file_name.strip():
            # 排除不需要复制的文件
            if file_name not in ['multimodal.jsonl', 'metadata.jsonl']:
                # 解析源文件路径 (相对于数据集目录)
                dataset_dir = Path(self._task_file_path).parent
                source_file = dataset_dir / file_name
                
                if source_file.exists():
                    # 保持文件名，复制到工作区
                    dest_file = workspace_dir / file_name
                    dest_file.parent.mkdir(parents=True, exist_ok=True)  # 创建子目录如果需要
                    
                    try:
                        shutil.copy2(source_file, dest_file)
                        copied_files.append(file_name)
                        print(f"📄 Copied task file: {file_name} -> {workspace_dir}")
                    except Exception as e:
                        print(f"⚠️ Failed to copy task file {file_name}: {e}")
                else:
                    print(f"⚠️ Task file not found: {source_file}")
            else:
                print(f"📋 Skipping system file: {file_name} (not copied to workspace)")
        
        # 设置文件名环境变量，供agent prompt使用
        if copied_files:
            os.environ["GAIA_TASK_FILE_NAMES"] = ",".join(copied_files)
            os.environ["GAIA_PRIMARY_FILE_NAME"] = copied_files[0]  # 主要文件名
        else:
            os.environ.pop("GAIA_TASK_FILE_NAMES", None)
            os.environ.pop("GAIA_PRIMARY_FILE_NAME", None)
        
        # 不再复制数据集通用文件（multimodal.jsonl, metadata.jsonl）
        # 这些文件保留在原始数据集目录中，通过环境变量访问
        
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
    def display_results(self, total_tasks: int, success_count: int, timeout_count: int, error_count: int, avg_quality: Optional[float] = None, total_execution_time: Optional[float] = None, communication_overhead: Optional[float] = None) -> None:
        """Show run summary. Display average quality, total execution time and communication overhead."""
        print("\n" + "=" * 60)
        print("📊 GAIA Benchmark Results Summary")
        print("=" * 60)
        print(f"📋 Total tasks: {total_tasks}")
        print(f"✅ Successful: {success_count} ({(success_count/total_tasks*100 if total_tasks else 0):.1f}%)")
        print(f"⏰ Timeout: {timeout_count} ({(timeout_count/total_tasks*100 if total_tasks else 0):.1f}%)")
        print(f"❌ Errors: {error_count} ({(error_count/total_tasks*100 if total_tasks else 0):.1f}%)")
        print()

        # Print average quality score if provided
        if avg_quality is not None:
            try:
                print(f"⭐ Average quality_score (1-5): {avg_quality:.2f}")
            except Exception:
                print(f"⭐ Average quality_score (1-5): {avg_quality}")
        else:
            print("⭐ Average quality_score (1-5): N/A")

        # Display only the requested timing metrics
        if total_execution_time is not None:
            try:
                print(f"⏱️ Total execution time: {total_execution_time:.2f} seconds")
            except Exception:
                print(f"⏱️ Total execution time: {total_execution_time}")
        else:
            print("⏱️ Total execution time: N/A")

        if communication_overhead is not None:
            try:
                print(f"🔁 Communication time: {communication_overhead:.2f} seconds")
            except Exception:
                print(f"🔁 Communication time: {communication_overhead}")
        else:
            print("🔁 Communication time: N/A")

        print(f"💾 Results saved to: {self.output_file}")
        print("=" * 60)

    def compute_avg_quality(self, results: List[Dict[str, Any]]) -> Optional[float]:
        """Compute average enhanced LLM judge quality_score (1-5) from results.

        Args:
            results: List of result dicts as saved to output file.

        Returns:
            Average quality score as float, or None if no valid scores found.
        """
        if not results:
            return None
        scores = []
        for e in results:
            try:
                if isinstance(e, dict):
                    # Prefer enhanced_llm_judge.quality_score
                    judge = e.get('enhanced_llm_judge') or {}
                    score = judge.get('quality_score')
                    if score is None:
                        # Some historical outputs may use 'quality' or 'quality_score' at top level
                        score = e.get('quality_score') or e.get('quality')
                    if score is None and isinstance(e, (int, float)):
                        score = float(e)
                    if score is not None:
                        scores.append(float(score))
            except Exception:
                continue
        if not scores:
            return None
        return sum(scores) / len(scores)

    def compute_task_toolcall_metrics(self, network: Any) -> Dict[str, Any]:
        """Compute total toolcall time and counts for a running/just-stopped network.

        This inspects agents attached to the provided network and sums the
        per-agent attributes `._toolcall_total` and `._toolcall_count` that
        agents are expected to maintain in-memory.

        Returns a dict with:
          - task_toolcall_time: float (seconds)
          - task_toolcall_count: int
          - agent_tool_stats: list of {agent_id, agent_name, total, count}
        """
        total_time = 0.0
        total_count = 0
        agent_stats = []
        try:
            agents = getattr(network, 'agents', None) or []
            for a in agents:
                try:
                    t = float(getattr(a, '_toolcall_total', 0.0) or 0.0)
                except Exception:
                    t = 0.0
                try:
                    c = int(getattr(a, '_toolcall_count', 0) or 0)
                except Exception:
                    c = 0
                total_time += t
                total_count += c
                agent_stats.append({
                    'agent_id': getattr(a, 'id', None),
                    'agent_name': getattr(a, 'name', None),
                    'toolcall_total': round(t, 6),
                    'toolcall_count': c,
                })
        except Exception:
            # Defensive: if network shape is unexpected just return zeros
            return {
                'task_toolcall_time': 0.0,
                'task_toolcall_count': 0,
                'agent_tool_stats': []
            }
        return {
            'task_toolcall_time': round(total_time, 6),
            'task_toolcall_count': total_count,
            'agent_tool_stats': agent_stats,
        }

    def compute_task_llm_metrics(self, network: Any) -> Dict[str, Any]:
        """Compute total LLM call time and counts for a running/just-stopped network.

        This inspects agents attached to the provided network and sums the
        per-agent attributes `._llm_call_total` and `._llm_call_count` that
        agents are expected to maintain in-memory (same pattern as toolcalls).

        Returns a dict with:
          - task_llm_time: float (seconds)
          - task_llm_count: int
          - agent_llm_stats: list of {agent_id, agent_name, llm_call_total, llm_call_count}
        """
        total_time = 0.0
        total_count = 0
        agent_stats = []
        try:
            agents = getattr(network, 'agents', None) or []
            for a in agents:
                try:
                    t = float(getattr(a, '_llm_call_total', 0.0) or 0.0)
                except Exception:
                    t = 0.0
                try:
                    c = int(getattr(a, '_llm_call_count', 0) or 0)
                except Exception:
                    c = 0
                total_time += t
                total_count += c
                agent_stats.append({
                    'agent_id': getattr(a, 'id', None),
                    'agent_name': getattr(a, 'name', None),
                    'llm_call_total': round(t, 6),
                    'llm_call_count': c,
                })
        except Exception:
            return {
                'task_llm_time': 0.0,
                'task_llm_count': 0,
                'agent_llm_stats': []
            }
        return {
            'task_llm_time': round(total_time, 6),
            'task_llm_count': total_count,
            'agent_llm_stats': agent_stats,
        }

    # -------------------- Stepwise hooks --------------------
    async def plan(self, task_id: str, level: int, task_doc: str, workspace_dir: Path) -> Path:
        """
        Plan agents for the task using the dedicated workspace directory.

        This implementation normalizes planner output so that all protocols produce
        a single config file at: <GAIA_ROOT>/workspaces/agent_config.<task_id>.json
        which ensures fair comparison across protocols.

        Args:
            task_id: Task ID
            level: Task level
            task_doc: Task description
            workspace_dir: Task-specific workspace directory

        Returns:
            Path: Agent configuration file path (GAIA_ROOT/workspaces/agent_config.<task_id>.json)
        """
        # 将当前 Runner 的协议名和工作区路径传入 Planner
        # Pass through the runner's actual config path so the planner loads the
        # exact same configuration file instead of falling back to defaults.
        planner_cfg_path = str(self._config_path) if getattr(self, '_config_path', None) else None
        planner = TaskPlanner(config_path=planner_cfg_path, task_id=task_id, level=level, protocol_name=self.protocol_name)
        # 让 planner 在该目录下工作并返回其生成的配置（路径或内容）
        planner_result = await planner.plan_agents(task_doc, workspace_dir=workspace_dir)

        # Ensure top-level workspaces dir exists and write unified config there
        top_workspaces_dir = GAIA_ROOT / 'workspaces'
        top_workspaces_dir.mkdir(parents=True, exist_ok=True)

        # Use a dedicated agent_config directory under top-level workspaces
        agent_config_dir = top_workspaces_dir / 'agent_config'
        agent_config_dir.mkdir(parents=True, exist_ok=True)

        dest_path = agent_config_dir / f"{task_id}.json"

        # Normalize planner_result to JSON-serializable dict
        config_data = None
        try:
            # If planner returned a Path-like string that exists on disk, try several locations
            candidate = Path(planner_result)
            if candidate.is_absolute() and candidate.exists():
                with candidate.open('r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                # Try relative to workspace_dir
                rel_candidate = Path(workspace_dir) / candidate
                if rel_candidate.exists():
                    with rel_candidate.open('r', encoding='utf-8') as f:
                        config_data = json.load(f)
                else:
                    # Maybe planner returned a JSON string
                    config_data = json.loads(str(planner_result))
        except Exception:
            # If planner returned a dict-like object, use it directly
            if isinstance(planner_result, dict):
                config_data = planner_result
            else:
                # As a fallback, wrap the string representation
                config_data = {"planner_output": str(planner_result)}

        # Write unified config file at GAIA_ROOT/workspaces/agent_config.<task_id>.json
        with dest_path.open('w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        return dest_path

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

            # Ensure network has protocol and workspace info so network loggers use correct path
            try:
                if isinstance(general_config, dict):
                    # prefer explicit protocol name from runner (avoids defaulting to 'general')
                    general_config.setdefault("protocol", self.protocol_name)
                    # include absolute workspace path for components that prefer it
                    general_config.setdefault("workspace_dir", str(workspace_dir))
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

                # Determine network log path for this task and pass to judge
                try:
                    network_log_path = str(Path(workspace_dir) / 'network_execution_log.json')
                except Exception:
                    network_log_path = None

                # Judge the answer with comprehensive handling
                judgment = await judge.judge_answer(
                    question=task_doc,
                    ground_truth=ground_truth,
                    predicted_answer=result,
                    execution_time=execution_time,
                    status="success",
                    network_log_path=network_log_path
                )

                # Format result using the enhanced judge
                formatted = judge.format_judgment_for_result(
                    judgment=judgment,
                    task_id=task_id,
                    question=task_doc,
                    ground_truth=ground_truth,
                    predicted_answer=result,
                    execution_time=execution_time,
                    level=level
                )
                try:
                    metrics = self.compute_task_toolcall_metrics(network)
                    llm_metrics = self.compute_task_llm_metrics(network)
                    formatted['task_toolcall_time'] = metrics.get('task_toolcall_time', 0.0)
                    formatted['task_toolcall_count'] = metrics.get('task_toolcall_count', 0)
                    formatted['agent_tool_stats'] = metrics.get('agent_tool_stats', [])
                    formatted['task_llm_call_time'] = llm_metrics.get('task_llm_time', 0.0)
                    formatted['task_llm_call_count'] = llm_metrics.get('task_llm_count', 0)
                    formatted['agent_llm_stats'] = llm_metrics.get('agent_llm_stats', [])
                except Exception:
                    formatted['task_toolcall_time'] = 0.0
                    formatted['task_toolcall_count'] = 0
                    formatted['agent_tool_stats'] = []
                    formatted['task_llm_call_time'] = 0.0
                    formatted['task_llm_call_count'] = 0
                    formatted['agent_llm_stats'] = []
                return formatted
            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                
                # Use enhanced LLM judge for timeout case
                from core.llm_judge import create_llm_judge
                judge_timeout = general_config.get("evaluation", {}).get("judge_timeout", 30)
                judge = create_llm_judge(judge_timeout=judge_timeout)

                # Determine network log path for this task and pass to judge
                try:
                    network_log_path = str(Path(workspace_dir) / 'network_execution_log.json')
                except Exception:
                    network_log_path = None

                # Judge the timeout case
                judgment = await judge.judge_answer(
                    question=task_doc,
                    ground_truth=ground_truth,
                    predicted_answer="TIMEOUT: Framework execution exceeded time limit",
                    execution_time=execution_time,
                    status="timeout",
                    network_log_path=network_log_path
                )

                # Format result using the enhanced judge
                formatted = judge.format_judgment_for_result(
                    judgment=judgment,
                    task_id=task_id,
                    question=task_doc,
                    ground_truth=ground_truth,
                    predicted_answer="TIMEOUT: Framework execution exceeded time limit",
                    execution_time=execution_time,
                    level=level
                )
                try:
                    metrics = self.compute_task_toolcall_metrics(network)
                    llm_metrics = self.compute_task_llm_metrics(network)
                    formatted['task_toolcall_time'] = metrics.get('task_toolcall_time', 0.0)
                    formatted['task_toolcall_count'] = metrics.get('task_toolcall_count', 0)
                    formatted['agent_tool_stats'] = metrics.get('agent_tool_stats', [])
                    formatted['task_llm_call_time'] = llm_metrics.get('task_llm_time', 0.0)
                    formatted['task_llm_call_count'] = llm_metrics.get('task_llm_count', 0)
                    formatted['agent_llm_stats'] = llm_metrics.get('agent_llm_stats', [])
                except Exception:
                    formatted['task_toolcall_time'] = 0.0
                    formatted['task_toolcall_count'] = 0
                    formatted['agent_tool_stats'] = []
                    formatted['task_llm_call_time'] = 0.0
                    formatted['task_llm_call_count'] = 0
                    formatted['agent_llm_stats'] = []
                return formatted
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

            # Determine network log path for this task and pass to judge
            try:
                network_log_path = str(Path(workspace_dir) / 'network_execution_log.json')
            except Exception:
                network_log_path = None

            # Judge the error case
            judgment = await judge.judge_answer(
                question=task_doc,
                ground_truth=ground_truth,
                predicted_answer=f"ERROR: {e}",
                execution_time=execution_time,
                status="error",
                network_log_path=network_log_path
            )

            # Format result using the enhanced judge
            formatted = judge.format_judgment_for_result(
                judgment=judgment,
                task_id=task_id,
                question=task_doc,
                ground_truth=ground_truth,
                predicted_answer=f"ERROR: {e}",
                execution_time=execution_time,
                level=level
            )
            try:
                metrics = self.compute_task_toolcall_metrics(network)
                formatted['task_toolcall_time'] = metrics.get('task_toolcall_time', 0.0)
                formatted['task_toolcall_count'] = metrics.get('task_toolcall_count', 0)
                formatted['agent_tool_stats'] = metrics.get('agent_tool_stats', [])
            except Exception:
                formatted['task_toolcall_time'] = 0.0
                formatted['task_toolcall_count'] = 0
                formatted['agent_tool_stats'] = []
            try:
                llm_metrics = self.compute_task_llm_metrics(network)
                formatted['task_llm_time'] = llm_metrics.get('task_llm_time', 0.0)
                formatted['task_llm_count'] = llm_metrics.get('task_llm_count', 0)
                formatted['agent_llm_stats'] = llm_metrics.get('agent_llm_stats', [])
            except Exception:
                formatted['task_llm_time'] = 0.0
                formatted['task_llm_count'] = 0
                formatted['agent_llm_stats'] = []
            return formatted

    # -------------------- Common Workflow --------------------
    async def run(self) -> None:
        """Main runner orchestration (load tasks, iterate, save results)."""
        tasks = self.tasks
        run_start_time = time.time()
        # For fast iteration/dev, limit here if needed
        runtime_config = self.config.get('runtime', {})
        max_tasks = runtime_config.get('max_tasks', None)
        
        if self.mode == "debug":
            # Debug mode: limit to 2 tasks
            if isinstance(tasks, list) and tasks:
                tasks = tasks[:2]
            else:
                print(f"[WARN] Debug 模式下任务结构异常，tasks 类型: {type(tasks).__name__}")
                tasks = [] if tasks is None else ([tasks] if isinstance(tasks, dict) else [])
        elif self.mode == "mm":
            # Multimodal mode: only run first 2 tasks
            if isinstance(tasks, list) and tasks:
                tasks = tasks[:2]
                print("🎯 MM 模式：仅运行前 2 条任务 (multimodal.jsonl)")
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

        # Aggregate total toolcall and llm call time across all task results
        total_toolcall_time = 0.0
        total_llm_call_time = 0.0
        for r in results:
            try:
                total_toolcall_time += float(r.get('task_toolcall_time', 0.0) or 0.0)
            except Exception:
                pass
            try:
                total_llm_call_time += float(r.get('task_llm_call_time', 0.0) or 0.0)
            except Exception:
                pass

        total_execution_time = time.time() - run_start_time
        # Communication overhead = total execution time minus toolcall and llm call time
        comm_overhead = max(0.0, total_execution_time - total_toolcall_time - total_llm_call_time)

        output_data = {
            "metadata": {
                "total_tasks": len(tasks),
                "successful_tasks": success_count,
                "timeout_tasks": timeout_count,
                "error_tasks": error_count,
                "success_rate": (success_count / len(tasks) * 100) if tasks else 0.0,
                "timeout_per_task": self.timeout,
                "execution_timestamp": time.time(),
                # Average quality score (1-5) across evaluated tasks, null if unavailable
                "avg_quality_score": self.compute_avg_quality(results),
                # Total execution time for the whole run in seconds
                "total_execution_time": total_execution_time,
                # Aggregate toolcall time across all agents/tasks
                "total_toolcall_time": round(total_toolcall_time, 6),
                # Aggregate LLM call time across all agents/tasks
                "total_llm_call_time": round(total_llm_call_time, 6),
                # Communication overhead = total_execution_time - total_toolcall_time - total_llm_call_time
                "communication_overhead": round(comm_overhead, 6),
            },
            "results": results,
            "failed_tasks": failed_tasks,
        }

        out_path = Path(self.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # 调用封装的结果展示函数
        avg_quality = output_data.get('metadata', {}).get('avg_quality_score')
        total_tool_time = output_data.get('metadata', {}).get('total_toolcall_time')
        # Pass average quality, total execution time and communication overhead for display
        self.display_results(len(tasks), success_count, timeout_count, error_count, avg_quality, total_execution_time, comm_overhead)


async def _main():
    runner = RunnerBase()
    try:
        await runner.run()
    except NotImplementedError:
        print("RunnerBase is abstract. Please run a concrete protocol runner instead.")


if __name__ == "__main__":
    asyncio.run(_main())
