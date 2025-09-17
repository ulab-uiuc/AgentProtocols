# **基于 Gaia 数据集的多智能体框架技术设计文档**

## **0. 设计目标与硬约束**

本文档概述了一个专门针对 **Gaia 基准测试数据集**的多智能体执行框架。该框架旨在处理 Gaia 数据集中的复杂任务，并提供本地执行和评估能力。系统遵循以下核心原则和约束。

| # | 约束 | 描述 |
| --- | --- | --- |
| 1 | **智能规划器控制** | `Planner` 组件负责分析任务并动态生成智能体配置，配置保存在 `agent_config.json` 中。 |
| 2 | **初始文档广播** | `MeshNetwork` 进程必须在 `t=0` 时通过 `MeshNetwork.broadcast_init()` 向所有智能体广播 Gaia 数据集中的任务文档。 |
| 3 | **异步点对点通信** | 每个智能体作为独立的异步进程运行。所有通信都是点对点的，由中央 `MeshNetwork` 中继。 |
| 4 | **每个智能体一个工具** | 每个智能体在启动时绑定到单一的特定工具（如 `search`、`extract`）。 |
| 5 | **仅本地执行** | 所有组件在本地运行（`127.0.0.1`），使用专用端口范围（如 9000-900N）。 |
| 6 | **可插拔通信协议** | 底层通信协议（如 JSON、ANP、A2A）必须通过 `ProtocolAdapter` 抽象层可互换。 |
| 7 | **私有智能体工作空间** | 每个智能体维护一个私有的、隔离的工作空间，用于日志、内存和其他工件，位于 `workspaces/<node_id>/`。 |
| 8 | **集成评估与日志记录** | `MeshNetwork` 负责最终评估、指标聚合（`metrics.json`）和归档所有运行工件（`run_artifacts.tar.gz`）。 |

---

## **1. 整体架构**

该多智能体框架采用 **分层架构** 设计，包含智能规划器（Planner）、网格网络（MeshNetwork）和动态生成的智能体。框架专门设计用于处理 Gaia 基准测试数据集中的复杂任务。Planner 首先分析 Gaia 任务并生成最优的智能体配置，然后 MeshNetwork 根据配置创建智能体并管理它们之间的通信。

```
┌─────────────────────── 智能规划层 ───────────────────────┐
│                      Planner                           │
│  ┌─────────────┐ analyze  ┌─────────────────────────┐    │
│  │ Gaia Task   │ ──────→  │ Agent Configuration     │    │
│  │ Document    │          │ Generator & Optimizer   │    │
│  └─────────────┘          └─────────────────────────┘    │
│                                     │                    │
│                                     ▼                    │
│                          ┌─────────────────────────┐    │
│                          │   agent_config.json     │    │
│                          │ {agents, tools, ports}  │    │
│                          └─────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                                     │
                                     ▼ 配置驱动创建
┌─────────────────────── 网络通信层 ───────────────────────┐
│                 MeshNetwork (传输 + 指标)                │
│  (1) load_config(agent_config.json)                     │
│  (2) create_agents_dynamically()                        │
│  (3) TCP relay ⤷ count bytes / tokens / pkt             │
┌─────────────┐  │  (4) detect tag="final_answer"                   │
│  智能体#0   │◀─┼─────────┐                                        │
│ tool=dynamic│  │         ▼                                        │
│ port=dynamic│  │  eval_runner → metrics.json & artifacts archive  │
└─────────────┘  └───────────────────────────────────────────────────┘
      ▲                                ▲
      ▼ 动态消息流                     ▼ "events"
┌─────────────┐                ┌─────────────┐
│  智能体#1   │                │  智能体#N    │
│ tool=dynamic│                │ tool=dynamic│
│ port=dynamic│                │ port=dynamic│
└─────────────┘  ▲             └─────────────┘  ▲
      ▼ 动态消息流│                    ▼ "final_answer"
      (根据 planner 配置的工作流)
```

**工作流序列：**

1. **任务分析**：`Planner` 分析输入的 Gaia 任务文档，确定所需的智能体类型、数量和工作流。
2. **配置生成**：`Planner` 生成最优的智能体配置并保存到 `agent_config.json`。
3. **动态创建**：`MeshNetwork` 读取配置文件，动态创建指定的智能体。
4. **初始化**：`MeshNetwork` 启动并向所有创建的智能体广播初始 Gaia 任务文档。
5. **任务处理**：智能体根据 Planner 配置的工作流开始处理 Gaia 任务。
6. **消息中继**：`MeshNetwork` 按照配置的消息路由规则中继消息。
7. **流水线执行**：智能体按照 Planner 设计的序列执行其工具并发布结果。
8. **最终答案**：指定的最终智能体产生最终答案并标记。
9. **评估**：`MeshNetwork` 检测到 `final_answer` 标签，运行评估并归档结果。

---

## **2. 目录结构**

为了维护一个清洁和可扩展的代码库，项目将遵循以下目录结构，分离核心逻辑、协议和执行脚本之间的关注点。

```
gaia_benchmark_framework/        # 多智能体框架根目录
├── core/                       # ★ 核心系统逻辑
│   ├── planner.py             # 智能规划器实现
│   ├── network.py             # MeshNetwork 实现
│   └── agent.py               # MeshAgent 模板
├── protocols/                # 🔌 可插拔协议适配器
│   ├── base_adapter.py       # ProtocolAdapter 抽象基类
│   └── json_adapter.py       # 默认 JSON 实现
├── tools/                    # 🛠️ 智能体工具定义
│   └── registry.py           # 工具注册逻辑
├── workspaces/               # 📂 智能体特定的运行时目录（自动生成）
│   ├── 0/
│   └── 1/
├── scripts/                  # 🚀 入口点和评估
│   ├── run_benchmark.py      # 启动系统的主脚本
│   └── evaluate.py           # 评估逻辑（eval_runner）
├── config/                   # ⚙️ 配置文件
│   ├── default.yaml          # 默认系统配置
│   └── agent_config.json     # Planner 生成的智能体配置
└── docs/                     # 📖 系统文档
    └── README.md
```

---

## **3. 智能规划器（Planner）**

`Planner` 是框架的大脑，负责分析 Gaia 数据集中的任务并生成最优的智能体配置。它通过分析任务复杂度、领域特征和性能要求来决定需要哪些智能体以及它们之间的协作模式。

### **3.1. Planner 核心功能**

```python
# core/planner.py 的核心逻辑
class TaskPlanner:
    def __init__(self, strategy_type: str = "adaptive"):
        self.strategy = self._load_strategy(strategy_type)
        self.tool_registry = ToolRegistry()
        self.port_manager = PortManager(start_port=9000)
    
    async def analyze_and_plan(self, gaia_task_document: str) -> Dict[str, Any]:
        """分析 Gaia 任务并生成智能体配置"""
        # 1. 任务分析
        task_analysis = await self._analyze_task(gaia_task_document)
        
        # 2. 生成智能体配置
        agent_config = await self._generate_agent_config(task_analysis)
        
        # 3. 优化配置
        optimized_config = await self._optimize_config(agent_config)
        
        # 4. 保存配置
        await self._save_config(optimized_config)
        
        return optimized_config
    
    async def _analyze_task(self, document: str) -> Dict[str, Any]:
        """分析 Gaia 任务特征"""
        return {
            "task_type": self._detect_task_type(document),
            "complexity": self._assess_complexity(document),
            "required_capabilities": self._identify_capabilities(document),
            "estimated_steps": self._estimate_workflow_steps(document)
        }
    
    async def _generate_agent_config(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """根据分析结果生成智能体配置"""
        agents = []
        workflow = []
        
        # 根据任务类型选择智能体模板
        if analysis["task_type"] == "qa_with_reasoning":
            agents = self._create_qa_reasoning_agents(analysis)
            workflow = self._create_qa_reasoning_workflow(agents)
        elif analysis["task_type"] == "multi_step_analysis":
            agents = self._create_analysis_agents(analysis)
            workflow = self._create_analysis_workflow(agents)
        
        return {
            "task_id": f"task_{int(time.time())}",
            "agents": agents,
            "workflow": workflow,
            "communication_rules": self._generate_communication_rules(agents),
            "performance_targets": self._set_performance_targets(analysis)
        }
```

### **3.2. 智能体配置格式（agent_config.json）**

```json
{
  "task_id": "task_1690123456",
  "generated_at": "2025-07-28T10:30:00Z",
  "task_analysis": {
    "task_type": "qa_with_reasoning",
    "complexity": "medium",
    "estimated_duration": 30000,
    "required_capabilities": ["search", "extract", "reason"]
  },
  "agents": [
    {
      "id": 0,
      "name": "DocumentSearcher",
      "tool": "search",
      "port": 9000,
      "priority": 1,
      "max_tokens": 500,
      "specialization": "document_analysis"
    },
    {
      "id": 1,
      "name": "InformationExtractor", 
      "tool": "extract",
      "port": 9001,
      "priority": 2,
      "max_tokens": 300,
      "specialization": "entity_extraction"
    },
    {
      "id": 2,
      "name": "KnowledgeReasoner",
      "tool": "reason",
      "port": 9002,
      "priority": 3,
      "max_tokens": 800,
      "specialization": "logical_reasoning"
    }
  ],
  "workflow": {
    "start_agent": 0,
    "message_flow": [
      {"from": 0, "to": [1], "message_type": "search_results"},
      {"from": 1, "to": [2], "message_type": "extracted_info"},
      {"from": 2, "to": "final", "message_type": "final_answer"}
    ],
    "parallel_execution": [],
    "fallback_agents": []
  },
  "communication_rules": {
    "broadcast_types": ["doc_init", "system_status"],
    "direct_routing": {
      "search_results": [1],
      "extracted_info": [2]
    },
    "timeout_seconds": 30
  },
  "performance_targets": {
    "max_execution_time": 60000,
    "target_accuracy": 0.85,
    "max_total_tokens": 2000
  }
}
```

---

## **4. ProtocolAdapter 接口**

为了满足可插拔协议要求，所有通信编码和解码将通过 `ProtocolAdapter` 处理。这允许系统在不更改核心智能体或网络逻辑的情况下，从简单的 JSON 协议切换到更高效或更安全的二进制协议。

```python
# 位于 protocols/base_adapter.py
import abc

class ProtocolAdapter(abc.ABC):
    """编码和解码网络数据包的抽象接口。"""

    @abc.abstractmethod
    def encode(self, packet: dict) -> bytes:
        """将字典数据包编码为字节。"""
        ...

    @abc.abstractmethod
    def decode(self, blob: bytes) -> dict:
        """将字节解码为字典数据包。"""
        ...

    @abc.abstractmethod
    def header_size(self, packet: dict) -> int:
        """计算协议头的大小用于指标。"""
        ...
```

#### **基线 JSON 实现**

一个简单的、人类可读的 JSON 实现将作为基线。

```python
# 位于 protocols/json_adapter.py
import json
from .base_adapter import ProtocolAdapter

class JsonAdapter(ProtocolAdapter):
    def encode(self, p: dict) -> bytes:
        return json.dumps(p, ensure_ascii=False).encode('utf-8')

    def decode(self, b: bytes) -> dict:
        return json.loads(b)

    def header_size(self, p: dict) -> int:
        # JSON 没有单独的头部，所以开销为零。
        return 0
```

---

## **5. 核心组件逻辑**

框架由三个主要组件组成：`TaskPlanner`、`MeshNetwork` 和 `MeshAgent`。

### **5.1. TaskPlanner（智能规划器）**

`TaskPlanner` 是框架的决策中心，负责分析 Gaia 任务并生成最优的智能体配置。

```python
# core/planner.py 的完整实现示例
class TaskPlanner:
    def __init__(self):
        self.strategies = {
            "simple": SimpleStrategy(),
            "adaptive": AdaptiveStrategy(), 
            "performance": PerformanceOptimizedStrategy()
        }
    
    async def plan_agents(self, gaia_task_doc: str, strategy: str = "adaptive") -> str:
        """规划智能体并返回配置文件路径"""
        planner_strategy = self.strategies.get(strategy, self.strategies["adaptive"])
        
        # 分析 Gaia 任务
        analysis = await planner_strategy.analyze_task(gaia_task_doc)
        
        # 生成配置
        config = await planner_strategy.generate_config(analysis)
        
        # 保存到 JSON 文件
        config_path = "config/agent_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return config_path
```

### **5.2. MeshNetwork（动态网络管理器）**

`MeshNetwork` 现在增强为能够根据 Planner 的配置动态创建和管理智能体。

```python
# core/network.py 的增强实现
class MeshNetwork:
    def __init__(self, adapter: ProtocolAdapter):
        self.adapter = adapter
        self.agents: List[MeshAgent] = []
        self.conns: List[asyncio.StreamWriter] = []
        self.config: Dict[str, Any] = {}
        # 指标
        self.bytes_tx = self.bytes_rx = self.header_overhead = 0
        self.pkt_cnt = self.token_sum = 0
        self.start_ts = time.time() * 1000
        self.done_ts = None
        self.done_payload = None

    async def load_and_create_agents(self, config_path: str):
        """加载配置并动态创建智能体"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 根据配置创建智能体
        for agent_config in self.config["agents"]:
            agent = MeshAgent(
                node_id=agent_config["id"],
                name=agent_config["name"],
                tool=agent_config["tool"],
                adapter=self.adapter,
                port=agent_config["port"],
                config=agent_config
            )
            self.agents.append(agent)
            
            # 启动智能体服务器
            asyncio.create_task(agent.serve())
        
        # 等待智能体启动
        await asyncio.sleep(1.0)

    async def start(self):
        """连接到配置的智能体端口并启动中继任务"""
        ports = [agent.port for agent in self.agents]
        
        for port in ports:
            r, w = await asyncio.open_connection("127.0.0.1", port)
            self.conns.append(w)
            asyncio.create_task(self._relay(r, port))
        
        asyncio.create_task(self._monitor_done())

    async def broadcast_init(self, doc: str):
        """Broadcasts the initial document to all agents."""
        chunks = [doc[i:i+400] for i in range(0, len(doc), 400)]
        pkt = {"type": "doc_init", "chunks": chunks}
        blob = self.adapter.encode(pkt)
        for w in self.conns:
            w.write(len(blob).to_bytes(4, "big") + blob)
            await w.drain()
            
    async def _relay(self, reader, src_port):
        """根据配置的路由规则中继消息"""
        while True:
            size = int.from_bytes(await reader.readexactly(4), "big")
            data = await reader.readexactly(size)
            
            # 更新指标
            self.bytes_rx += 4 + size
            pkt = self.adapter.decode(data)
            self.pkt_cnt += 1
            self.header_overhead += self.adapter.header_size(pkt)
            self.token_sum += pkt.get("token_used", 0)

            # 检查最终答案
            if pkt.get("tag") == "final_answer" and not self.done_ts:
                self.done_ts = time.time() * 1000
                self.done_payload = pkt["payload"]

            # 根据配置的通信规则路由消息
            await self._route_message(pkt, src_port, data)

    async def _route_message(self, pkt: dict, src_port: int, data: bytes):
        """根据配置路由消息"""
        message_type = pkt.get("type", "unknown")
        
        # 检查是否为广播类型
        if message_type in self.config.get("communication_rules", {}).get("broadcast_types", []):
            # 广播给所有智能体
            for w in self.conns:
                if w.get_extra_info("peername")[1] != src_port:
                    w.write(len(data).to_bytes(4, "big") + data)
                    await w.drain()
                    self.bytes_tx += 4 + len(data)
        else:
            # 根据直接路由规则发送
            routing_rules = self.config.get("communication_rules", {}).get("direct_routing", {})
            target_agents = routing_rules.get(pkt.get("tag", ""), [])
            
            for target_id in target_agents:
                target_port = self._get_agent_port(target_id)
                target_writer = self._get_writer_by_port(target_port)
                if target_writer:
                    target_writer.write(len(data).to_bytes(4, "big") + data)
                    await target_writer.drain()
                    self.bytes_tx += 4 + len(data)

    async def _monitor_done(self):
        """等待 'final_answer' 标签然后触发评估"""
        while self.done_ts is None:
            await asyncio.sleep(1)
        await self._evaluate()

    async def _evaluate(self):
        """运行最终评估并归档工件"""
        quality = await eval_runner(self.done_payload, "ground_truth.json")
        report = dict(bytes_tx=self.bytes_tx,
                      bytes_rx=self.bytes_rx,
                      pkt_cnt=self.pkt_cnt,
                      header_overhead=self.header_overhead,
                      token_sum=self.token_sum,
                      elapsed_ms=self.done_ts - self.start_ts,
                      **quality)
        with open("metrics.json", "w") as f:
            json.dump(report, f, indent=2)

        # 归档工作空间
        with tarfile.open("run_artifacts.tar.gz", "w:gz") as tar:
            for d in Path("workspaces").iterdir():
                tar.add(d, arcname=d.name)
```

### **5.3. MeshAgent（增强型智能体）**

`MeshAgent` 现在支持动态配置和个性化设置。增强功能包括：智能体命名、配置驱动的工具初始化、专门化处理逻辑、令牌限制管理、增强的日志记录。

```python
# core/agent.py 的增强实现
class MeshAgent:
    """
    增强的多智能体网络节点，支持配置驱动的动态创建和个性化设置。
    
    主要增强功能：
    1. 配置驱动初始化：通过 JSON 配置文件动态设置智能体参数
    2. 智能体命名：支持有意义的智能体名称，便于调试和监控
    3. 工作空间命名：使用 "id_name" 格式的工作空间，提高可读性
    4. 专门化处理：根据 specialization 参数调整消息处理逻辑
    5. 令牌限制管理：支持配置化的令牌使用限制和超限告警
    6. 优先级支持：支持智能体优先级设置，用于任务调度
    7. 增强日志：详细的执行日志，包含智能体元信息
    8. 工具配置：支持工具的动态配置和参数传递
    """
    
    def __init__(self, node_id: int, name: str, tool: str, adapter, port: int, config: Dict[str, Any]):
        """
        初始化增强型智能体
        
        Args:
            node_id: 智能体唯一标识符
            name: 智能体可读名称（新增）
            tool: 工具名称
            adapter: 协议适配器
            port: 监听端口
            config: 配置字典，包含个性化参数（新增）
        """
        # 基础属性
        self.id = node_id
        self.name = name  # 新增：智能体名称
        self.tool_name = tool
        self.adapter = adapter
        self.port = port
        self.config = config  # 新增：配置对象
        
        # 根据配置设置个性化参数（新增功能）
        self.max_tokens = config.get("max_tokens", 500)
        self.priority = config.get("priority", 1)
        self.specialization = config.get("specialization", "general")
        
        # 增强的工作空间设置：使用 "id_name" 格式
        self.ws = f"workspaces/{self.id}_{self.name}"
        Path(self.ws).mkdir(parents=True, exist_ok=True)
        
        # 初始化 LLM 引擎（保持原有逻辑）
        self.engine = Engine(
            llm=LLM(model="gemma-7b-it"),
            tools=[self._register_tool(tool)],
            memory_path=f"{self.ws}/memory.json",
            on_token=self._count_token
        )
        self.token_used = 0

    def _count_token(self, _): 
        """令牌计数器（保持原有逻辑）"""
        self.token_used += 1

    def _register_tool(self, name: str):
        """
        注册工具（保持原有逻辑，但支持配置增强）
        
        增强功能：可以根据 self.config 为工具设置特定参数
        """
        if name == "search":
            @tool(name="search")
            def _t(chunks: list[str]) -> list[str]: 
                # 可以在这里根据 self.specialization 调整搜索策略
                return chunks  # 简化实现
            return _t
        if name == "extract":
            @tool(name="extract_event")
            def _t(docs: list[str]) -> list[dict]: 
                # 可以根据 self.specialization 调整提取策略
                return [{"event": doc} for doc in docs]  # 简化实现
            return _t
        if name == "triple":
            @tool(name="to_triple")
            def _t(events: list[dict]) -> list[list[str]]: 
                return [["subject", "predicate", "object"]]  # 简化实现
            return _t
        if name == "reason":
            @tool(name="reason")
            def _t(triples: list[list[str]]) -> str: 
                return "Final reasoning result"  # 简化实现
            return _t
        raise ValueError(f"Invalid tool: {name}")

    async def serve(self):
        """启动智能体服务器（保持原有逻辑）"""
        srv = await asyncio.start_server(self._handle, "127.0.0.1", self.port)
        async with srv: 
            await srv.serve_forever()

    async def _handle(self, r, w):
        """处理网络连接（保持原有逻辑）"""
        while True:
            sz = int.from_bytes(await r.readexactly(4), "big")
            pkt = self.adapter.decode(await r.readexactly(sz))
            asyncio.create_task(self._process(pkt, w))

    async def _process(self, pkt: dict, writer):
        """
        处理消息包（增强版本）
        
        增强功能：
        1. 支持配置驱动的消息处理逻辑
        2. 根据专门化决定是否处理特定消息
        3. 令牌使用限制检查
        4. 增强的日志记录
        """
        self.token_used = 0
        log_path = f"{self.ws}/agent.log"
        
        def log(x): 
            """增强的日志函数，包含智能体元信息"""
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] Agent-{self.id}({self.name}): {x}"
            Path(log_path).write_text(Path(log_path).read_text("") + log_entry + "\n")

        # 保持原有的消息处理逻辑，但添加配置检查
        if pkt["type"] == "doc_init" and self.tool_name == "search":
            if self._should_process_message("doc_init"):
                docs = await self.engine.run("search doc", chunks=pkt["chunks"])
                await self._emit(writer, "search_docs", {"docs": docs})
                log(f"Processed doc_init, found {len(docs)} docs")

        elif pkt["type"] == "search_docs" and self.tool_name == "extract":
            if self._should_process_message("search_docs"):
                evt = await self.engine.run("extract events", docs=pkt["docs"])
                await self._emit(writer, "events", {"events": evt})
                log(f"Extracted {len(evt)} events from docs")

        elif pkt["type"] == "events" and self.tool_name == "triple":
            if self._should_process_message("events"):
                tri = await self.engine.run("to triple", events=pkt["events"])
                await self._emit(writer, "triples", {"triples": tri})
                log(f"Generated {len(tri)} triples from events")

        elif pkt["type"] == "triples" and self.tool_name == "reason":
            if self._should_process_message("triples"):
                ans = await self.engine.run("reason final", triples=pkt["triples"])
                await self._emit(writer, "data_event", {
                    "tag": "final_answer", 
                    "payload": ans,
                    "log_uri": f"{self.ws}/reason.log",
                    "agent_id": self.id,  # 新增：智能体标识
                    "agent_name": self.name,  # 新增：智能体名称
                    "specialization": self.specialization  # 新增：专门化信息
                })
                log(f"Generated final answer: {ans[:50]}...")

        # 新增：令牌使用限制检查
        if self.token_used > self.max_tokens:
            await self._emit_warning(writer, "token_limit_exceeded")
            log(f"WARNING: Token limit exceeded ({self.token_used}/{self.max_tokens})")

    def _should_process_message(self, message_type: str) -> bool:
        """
        根据智能体专门化决定是否处理消息（新增功能）
        
        Returns:
            bool: 是否应该处理该类型的消息
        """
        processing_rules = {
            "document_analysis": ["doc_init"],
            "entity_extraction": ["search_docs", "raw_text"],
            "logical_reasoning": ["events", "triples"],
            "general": ["doc_init", "search_docs", "events", "triples"]
        }
        
        allowed_types = processing_rules.get(self.specialization, ["doc_init", "search_docs", "events", "triples"])
        return message_type in allowed_types

    async def _emit(self, w, pkt_type: str, extra: dict):
        """
        发送消息包（增强版本）
        
        增强功能：
        1. 添加智能体元信息到消息包
        2. 增强的日志记录格式
        """
        # 添加智能体元信息到消息包
        pkt = {
            "type": pkt_type, 
            "token_used": self.token_used,
            "agent_id": self.id,  # 新增
            "agent_name": self.name,  # 新增
            "priority": self.priority,  # 新增
            **extra
        }
        
        blob = self.adapter.encode(pkt)
        w.write(len(blob).to_bytes(4, "big") + blob)
        await w.drain()
        
        # 增强的日志记录
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agent_id": self.id,
            "agent_name": self.name,
            "specialization": self.specialization,
            "packet": pkt
        }
        Path(f"{self.ws}/agent.log").write_text(
            Path(f"{self.ws}/agent.log").read_text("") + 
            json.dumps(log_entry, ensure_ascii=False) + "\n"
        )

    async def _emit_warning(self, writer, warning_type: str):
        """
        发送警告消息（新增功能）
        
        Args:
            writer: 网络写入器
            warning_type: 警告类型
        """
        await self._emit(writer, "warning", {
            "warning_type": warning_type,
            "message": f"Agent {self.name} encountered {warning_type}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
```

---

## **6. 评估和指标**

当 `MeshNetwork` 接收到带有 `tag: "final_answer"` 的数据包时，会自动触发评估。

#### **评估运行器**

默认评估脚本基于与真实值文件的精确匹配（EM）和 ROUGE-L 分数的组合来测量质量。此功能设计为模块化，可以用其他评分机制（如 BLEU、GAIA 的官方评分器）替换。

```python
# scripts/evaluate.py 的逻辑
async def eval_runner(pred: str, truth_path: str) -> dict:
    """计算最终答案的质量分数。"""
    with open(truth_path, encoding='utf-8') as f:
        truth = json.load(f)
    em = int(pred.strip() == truth["answer"].strip())
    rouge = rouge_l(pred, truth["answer"])
    return {
        "quality_score": (em + rouge) / 2,
        "exact_match": em,
        "rouge_l": rouge
    }
```

#### **指标报告**

最终输出包括 `metrics.json`，包含性能和质量指标：
* **性能**: `bytes_tx`、`bytes_rx`、`pkt_cnt`、`header_overhead`、`token_sum`、`elapsed_ms`。
* **质量**: `quality_score`、`exact_match`、`rouge_l`。

---

## **7. 启动和使用**

框架现在通过智能规划器驱动的流程启动，包括 Gaia 任务分析、智能体配置生成和动态创建。

```python
# scripts/run_benchmark.py 的新实现示例
async def main():
    # 1. 初始化协议适配器
    adapter = JsonAdapter()
    
    # 2. 加载 Gaia 任务文档
    with open("gaia_task.txt", "r", encoding='utf-8') as f:
        gaia_task_document = f.read()
    
    # 3. 初始化智能规划器
    planner = TaskPlanner(strategy_type="adaptive")
    
    # 4. 分析 Gaia 任务并生成智能体配置
    print("🧠 分析 Gaia 任务并规划智能体...")
    config_path = await planner.plan_agents(gaia_task_document)
    print(f"📋 智能体配置已生成: {config_path}")
    
    # 5. 创建动态网络管理器
    network = MeshNetwork(adapter)
    
    # 6. 根据配置动态创建智能体
    print("🤖 根据配置创建智能体...")
    await network.load_and_create_agents(config_path)
    
    # 7. 启动网络
    print("🚀 启动网络通信...")
    await network.start()
    
    # 8. 开始任务执行
    print("📄 广播 Gaia 任务文档...")
    await network.broadcast_init(gaia_task_document)
    
    # 9. 监控执行
    print("⏳ 监控任务执行...")
    await network._monitor_done()
    
    print("✅ 任务完成 → 查看 metrics.json 和 run_artifacts.tar.gz")
```

### **7.1. 配置文件示例**

框架启动时会根据 Gaia 任务生成类似以下的配置文件：

```bash
# 查看生成的智能体配置
cat config/agent_config.json

# 启动框架处理 Gaia 任务
python scripts/run_benchmark.py --strategy adaptive

# 查看结果
cat metrics.json
```

---

## **8. 扩展点**

增强后的架构在以下关键领域具有更强的可扩展性：

| 方向 | 钩子/文件 | 描述 |
| --- | --- | --- |
| **新规划策略** | 实现新的 `PlanningStrategy` 子类。 | 添加基于机器学习的规划、多目标优化或特定领域的规划策略。 |
| **智能体模板** | 在 `planning/templates.py` 中添加模板。 | 为特定任务类型（如数学推理、代码分析）定义专门的智能体组合。 |
| **动态路由** | 修改 `MeshNetwork._route_message()`。 | 实现基于消息内容的智能路由、负载均衡或自适应通信模式。 |
| **性能优化** | 增强 `TaskPlanner` 的优化算法。 | 添加基于历史性能的智能体选择、资源分配优化或并行执行策略。 |
| **新协议** | 实现新的 `ProtocolAdapter` 子类。 | 添加头部压缩、CRC 校验和或签名验证。 |
| **LLM/工具运行时** | 替换智能体中的工具实现。 | 支持不同的 LLM 模型、专门化工具或外部 API 集成。 |
| **监控和调试** | 扩展指标收集和日志系统。 | 添加实时监控、性能分析或可视化调试工具。 |
| **容错性** | 在网络层添加故障检测和恢复。 | 实现智能体故障检测、自动重启或备用智能体切换。 |

### **8.1. 规划策略扩展示例**

```python
# planning/strategies.py 的扩展示例
class MachineLearningStrategy(PlanningStrategy):
    """基于机器学习的智能规划策略"""
    
    def __init__(self):
        self.model = self._load_planning_model()
        self.history = PlanningHistory()
    
    async def analyze_task(self, document: str) -> Dict[str, Any]:
        # 使用训练的模型分析任务
        features = self._extract_features(document)
        prediction = self.model.predict(features)
        
        return {
            "predicted_complexity": prediction.complexity,
            "recommended_agents": prediction.agent_types,
            "estimated_performance": prediction.performance,
            "confidence": prediction.confidence
        }
    
    async def generate_config(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        # 基于历史成功案例生成配置
        similar_cases = self.history.find_similar_tasks(analysis)
        optimized_config = self._optimize_from_history(similar_cases)
        
        return optimized_config
```

---

## **9. 框架优势与创新点**

### **9.1. 智能规划驱动**
- **自适应架构**：框架能够根据 Gaia 任务特征自动调整智能体配置
- **性能优化**：基于 Gaia 任务分析选择最优的智能体组合和工作流
- **可配置性**：通过 JSON 配置文件实现灵活的框架定制

### **9.2. 动态网格网络**
- **保持网格优势**：继承原有的去中心化通信和容错能力
- **智能路由**：基于配置的消息路由，提高通信效率
- **可扩展性**：支持任意数量和类型的智能体动态创建

### **9.3. 配置驱动开发**
- **声明式配置**：通过 JSON 文件描述整个框架行为
- **版本控制**：配置文件可以版本化管理和复用
- **调试友好**：配置和执行分离，便于问题诊断和优化

### **9.4. 企业级特性**
- **监控和指标**：全面的性能监控和度量收集
- **容错和恢复**：内置的错误处理和恢复机制
- **安全性**：支持协议级别的安全增强
