AgentNetwork Streaming Queue — 场景架构与「新增协议后端」开发指南

本文聚焦两件事：

先把场景架构说清楚，完整走一遍一个分布式问答系统是如何跑起来的；

给出一套如何开发新的 protocol backend 的标准流程与代码骨架（基于当前 NetworkBase + BaseCommBackend + RunnerBase 的解耦设计）。

一、场景架构（Scenario Architecture）
1.1 角色与边界
┌────────────────────────────────────────────────────────────────────┐
│                         App / Demo Runner                          │
│   - 读取配置 (config.yaml)                                         │
│   - 创建 NetworkBase (协议无关)                                     │
│   - 选择并注入某个协议后端 CommBackend (A2A / HTTP / gRPC / …)     │
│   - 启动/注册 协调器 与 工作器 (可本进程 Host，也可外部地址)         │
│   - 设拓扑（星型/网状/自定义）、健康检查、下发任务、收集结果         │
└────────────────────────────────────────────────────────────────────┘
                │                         ▲
             uses ▼                         │ calls
┌────────────────────────────────────────────────────────────────────┐
│                            NetworkBase                             │
│   - 只维护逻辑拓扑：agent_id → {neighbors}                         │
│   - 只维护可达表：agent_id → endpoint (URL 等)                     │
│   - 路由控制：route_message() / broadcast_message()                │
│   - 健康检查/监控/拓扑管理                                         │
│   - 所有通信由 CommBackend 执行                                    │
└────────────────────────────────────────────────────────────────────┘
                │ delegates
                ▼
┌────────────────────────────────────────────────────────────────────┐
│                           BaseCommBackend                          │
│   - 协议抽象：register_endpoint / connect / send / health_check /  │
│              close / (spawn_local_agent 可选)                      │
│   - 具体协议实现：A2ACommBackend、GrpcCommBackend、WsCommBackend…   │
│   - 负责：协议消息编解码、请求/响应、健康探测                      │
└────────────────────────────────────────────────────────────────────┘
                │
             talks to
                ▼
┌────────────────────────────────────────────────────────────────────┐
│     协议原生服务（本进程 Host 或 外部进程/集群中的 Agent 服务）      │
│   - Coordinator Executor（协议适配层）                              │
│   - Worker Executor（协议适配层）                                   │
│   - Worker 内部用 QAWorkerBase 调用 LLM（协议无关）                 │
└────────────────────────────────────────────────────────────────────┘


RunnerBase：统一流程（加载配置 → 创建 Network → 设拓扑 → 启动协调器/工作器 → 发指令 → 收尾）。

NetworkBase：协议无关的「逻辑网络管理器」，只关心 谁能到谁，不碰具体协议。

BaseCommBackend：协议抽象接口；每新增一个协议，只要实现这个接口即可。

Coordinator/Worker（协议侧适配器）：对接具体协议的消息格式，把消息转给 QACoordinatorBase 和 QAWorkerBase 的通用逻辑。

QAWorkerBase：调用 LLM 的通用问答能力（有 Core 则用 Core，无则 Mock），完全与协议解耦。

1.2 典型数据流（以星型拓扑为例）

Runner 读配置，创建 NetworkBase，注入某 CommBackend（如 A2A）。

Runner 启动/注册：

启动 Coordinator 协议 Host（或登记其地址到 NetworkBase）→ register_agent("Coordinator-1", http://...)

启动 N 个 Worker 协议 Host → register_agent("Worker-i", http://...)

Runner 设拓扑：setup_star_topology("Coordinator-1")（中心双向连边）。

发送控制命令（如 "dispatch"）到 Coordinator：

Runner → NetworkBase.route_message(src="Runner", dst="Coordinator-1", payload)

NetworkBase → CommBackend.send(...) → 调 Coordinator 的协议端点（/message 等）

Coordinator Executor 内部调用 QACoordinatorBase.dispatch_round()：

读题目 → 动态分发 → 通过 send_to_worker() 调 NetworkBase.route_message(...)

Worker Executor 收到后，调用 QAWorkerBase.answer() → LLM → 返回文本

Coordinator 汇总/落盘 → 返回结果事件

Runner 展示指标、健康检查、清理资源。

1.3 常见拓扑

星型 (Star)：Coordinator 为中心，适合「集中式调度」。

网状 (Mesh)：任意节点互通，适合「去中心化/协作式」。

自定义/分层：可为子协调器设子池，实现分层调度。

二、如何开发一个新的 Protocol Backend

目标：最少的接入成本。只要把具体协议包在一个 CommBackend 里，再提供对应的 协议侧 Runner & Executors，即可替换/新增一条「跑法」。

2.1 放置路径与命名
script/streaming_queue/
├─ core/
│   ├─ network_base.py           # 协议无关
│   ├─ qa_coordinator_base.py    # 调度逻辑（通信抽象）
│   └─ qa_worker_base.py         # LLM 统一问答
├─ comm/
│   └─ base.py                   # BaseCommBackend 抽象定义
├─ protocol_backend/
│   ├─ a2a/                      # 现成参考：A2A 协议
│   │   ├─ comm.py               # A2ACommBackend（含可选内嵌 Host）
│   │   ├─ coordinator.py        # 协调器协议适配 Executor
│   │   └─ worker.py             # 工作器协议适配 Executor
│   └─ <your_proto>/
│       ├─ comm.py               # <YourProto>CommBackend
│       ├─ coordinator.py        # <YourProto> 协调器 Executor
│       └─ worker.py             # <YourProto> 工作器 Executor
└─ runner/
    ├─ runner_base.py            # Runner 通用流程
    └─ run_<your_proto>.py       # 针对协议的 Runner（注入 CommBackend + Host/注册逻辑）

2.2 抽象接口（需要实现的最小集合）
① BaseCommBackend（必须）
class BaseCommBackend(ABC):
    @abstractmethod
    async def register_endpoint(self, agent_id: str, address: str) -> None: ...
    async def connect(self, src_id: str, dst_id: str) -> None: ...  # 可空实现
    @abstractmethod
    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any: ...
    @abstractmethod
    async def health_check(self, agent_id: str) -> bool: ...
    @abstractmethod
    async def close(self) -> None: ...
    # 可选：若协议支持在本进程启动 Host
    async def spawn_local_agent(self, agent_id: str, host: str, port: int, executor: Any) -> Any: ...


要求：

send() 返回结构建议统一为：

{
  "raw": <协议原始响应>,
  "text": <提取出的主文本，尽量可读>
}


health_check() 尽量实现轻量探测（如 /health 或最小消息）。

如协议允许，提供 spawn_local_agent() 以便 Runner 一键启动 demo。

② 协议侧 Executors（建议）

coordinator.py：实现一个执行器，接收协议消息 → 调用 QACoordinatorBase 的方法。

重点是把协议消息中的文本解析出来（命令如 "status" / "dispatch"），并把返回包装回协议格式的事件。

worker.py：实现一个执行器，接收协议消息 → 调用 QAWorkerBase.answer(question) → 返回文本。

这部分是协议适配层，只负责「协议 ↔ 内部通用接口」的消息转换。

③ Runner（建议）

run_<your_proto>.py 继承 RunnerBase，覆盖 3 个钩子：

create_network()：返回 NetworkBase(comm_backend=<YourProto>CommBackend())

setup_agents()：

若协议支持本进程 Host：用 comm.spawn_local_agent() 启动协调器/工作器。

否则：将外部已运行的 endpoint 用 network.register_agent(agent_id, address) 注册即可。

拓扑建议通过 setup_star_topology("Coordinator-1") 或自定义连接。

最后调用 coordinator_executor.coordinator.set_network(network, worker_ids) 告诉协调器要调哪些 worker。

send_command_to_coordinator(command)：把控制命令（"status"/"dispatch"）按你协议的消息格式送到 Coordinator。

三、最小样例：自定义协议 <proto> 的骨架

以下是精简骨架，展示关键接口与「该放什么逻辑」。可以对照 A2A 目录看完整实现。

3.1 <proto>/comm.py
# script/streaming_queue/protocol_backend/<proto>/comm.py
from __future__ import annotations
from typing import Any, Dict, Optional
from ...comm.base import BaseCommBackend  # 相对到 streaming_queue/comm/base.py

class <Proto>CommBackend(BaseCommBackend):
    def __init__(self, **kwargs):
        self._endpoints: Dict[str, str] = {}  # agent_id -> endpoint uri
        # 可在此初始化协议客户端/连接池等

    async def register_endpoint(self, agent_id: str, address: str) -> None:
        self._endpoints[agent_id] = address

    async def connect(self, src_id: str, dst_id: str) -> None:
        # 若协议需要预热/握手，可在此实现；否则留空
        return

    async def send(self, src_id: str, dst_id: str, payload: Dict[str, Any]) -> Any:
        endpoint = self._endpoints.get(dst_id)
        if not endpoint:
            raise RuntimeError(f"unknown dst_id={dst_id}")

        # 1) 将 payload 转成 <proto> 消息格式
        proto_msg = self._to_proto_message(payload)

        # 2) 通过协议客户端发出请求并拿到响应
        raw = await self._do_request(endpoint, proto_msg)

        # 3) 从 raw 中提取出可读文本（尽量统一）
        text = self._extract_text(raw)
        return {"raw": raw, "text": text}

    async def health_check(self, agent_id: str) -> bool:
        endpoint = self._endpoints.get(agent_id)
        if not endpoint:
            return False
        try:
            # 发 lightweight 的健康探测
            return await self._probe(endpoint)
        except Exception:
            return False

    async def close(self) -> None:
        # 关闭连接池等资源
        return

    # ----- helpers -----
    def _to_proto_message(self, payload: Dict[str, Any]) -> Any:
        # 统一支持 {"text":"..."} / {"parts":[{"kind":"text","text":"..."}]} 两种入参
        if "text" in payload:
            return {"type": "text", "data": payload["text"]}
        if "parts" in payload:
            return {"parts": payload["parts"]}
        return {"type": "text", "data": str(payload)}

    async def _do_request(self, endpoint: str, proto_msg: Any) -> Any:
        # 真正的协议发送逻辑（HTTP / gRPC / WebSocket / …）
        raise NotImplementedError

    def _extract_text(self, raw: Any) -> str:
        # 从协议返回里尽量抽一个主文本
        return getattr(raw, "text", "") or raw.get("text", "") or ""


若协议支持本进程 Host，可仿照 A2A：提供 spawn_local_agent() 与一个 FastAPI/gRPC 服务，用来承载 Executor。

3.2 <proto>/coordinator.py
# script/streaming_queue/protocol_backend/<proto>/coordinator.py
from __future__ import annotations
from typing import Any, Dict, List
from ...core.qa_coordinator_base import QACoordinatorBase

# 假设你的协议服务端将收到的文本放在 request.body.text
# 响应需要一个 events 数组，内含 {"type":"agent_text_message","data": "..."} 这样的条目
class <Proto>CoordinatorExecutor:
    def __init__(self, config: Dict | None = None, output=None):
        self.coordinator = CoordinatorFor<Proto>(config, output)

    async def execute(self, context, event_queue):
        user_text = context.get_user_input() or "status"
        cmd = user_text.strip().lower()
        if cmd == "dispatch":
            result = await self.coordinator.dispatch_round()
        else:
            result = await self.coordinator.get_status()
        await event_queue.enqueue_event({"type":"agent_text_message","data": result})

class CoordinatorFor<Proto>(QACoordinatorBase):
    async def send_to_worker(self, worker_id: str, question: str) -> Dict[str, Any]:
        # 通过 NetworkBase → CommBackend → 具体协议发消息
        payload = {"text": question}  # 或构造 parts
        resp = await self.agent_network.route_message(self.coordinator_id, worker_id, payload)
        # 统一返回
        return {"answer": (resp or {}).get("text"), "raw": resp}

3.3 <proto>/worker.py
# script/streaming_queue/protocol_backend/<proto>/worker.py
from __future__ import annotations
from typing import Any, Dict
from ...core.qa_worker_base import QAWorkerBase

class <Proto>WorkerExecutor:
    def __init__(self, config: Dict | None = None, output=None):
        self.worker = QAWorkerBase(config, output)

    async def execute(self, context, event_queue):
        text = context.get_user_input() or ""
        answer = await self.worker.answer(text)
        await event_queue.enqueue_event({"type":"agent_text_message","data": answer})

3.4 runner/run_<proto>.py
# script/streaming_queue/runner/run_<proto>.py
from __future__ import annotations
import asyncio, time
from typing import Any, Dict, List, Optional
from .runner_base import RunnerBase
from ..core.network_base import NetworkBase
from ..protocol_backend.<proto>.comm import <Proto>CommBackend
from ..protocol_backend.<proto>.coordinator import <Proto>CoordinatorExecutor
from ..protocol_backend.<proto>.worker import <Proto>WorkerExecutor

class <Proto>Runner(RunnerBase):
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)
        self._backend: <Proto>CommBackend | None = None

    async def create_network(self) -> NetworkBase:
        self._backend = <Proto>CommBackend()
        return NetworkBase(comm_backend=self._backend)

    async def setup_agents(self) -> List[str]:
        assert self._backend is not None
        # 情况 A：协议支持本进程Host → spawn_local_agent()
        # handle = await self._backend.spawn_local_agent("Coordinator-1", "127.0.0.1", 9998, <Proto>CoordinatorExecutor(...))
        # await self.network.register_agent("Coordinator-1", handle.base_url)

        # 情况 B：外部已有服务 → 直接 register_endpoint
        await self.network.register_agent("Coordinator-1", "http://127.0.0.1:9998")

        worker_ids = []
        for i in range(int(self.config["qa"]["worker"]["count"])):
            wid = f"Worker-{i+1}"
            # 同 A/B 两种方式择一
            # w_handle = await self._backend.spawn_local_agent(wid, "127.0.0.1", 10001+i, <Proto>WorkerExecutor(...))
            # await self.network.register_agent(wid, w_handle.base_url)
            await self.network.register_agent(wid, f"http://127.0.0.1:{10001+i}")
            worker_ids.append(wid)

        # 拓扑：星型
        self.network.setup_star_topology("Coordinator-1")

        # 把 network + workers 告诉协调器（用于内部调度）
        # 如果你走 spawn_local_agent，拿到的是 executor 实例可直接 set_network；
        # 若走外部服务，这步可以省略（协调器在收到 "dispatch" 时内部会用 route_message）。
        return worker_ids

    async def send_command_to_coordinator(self, command: str) -> Optional[Dict[str, Any]]:
        # 如何把命令送到 Coordinator，取决于你的协议
        # 例如：构造 payload，然后 NetworkBase → CommBackend.send → 协议请求
        resp = await self.network.route_message("Runner", "Coordinator-1", {"text": command})
        return {"result": (resp or {}).get("text") or "OK"}

async def _main():
    runner = <Proto>Runner()
    await runner.run()

if __name__ == "__main__":
    asyncio.run(_main())

四、配置与扩展点
4.1 config.yaml 要点

core：模型配置（OpenAI / 本地），由 QAWorkerBase 使用。

qa.coordinator：端口（若本地 Host）、批大小、是否只取前 50 等。

qa.worker：数量、起始端口（若本地 Host）。

network.topology：star / mesh / 自定义。

4.2 健康检查与降级

CommBackend.health_check()：优先轻探测（/health），退化可用一条最小消息（如 "status"）。

任何协议异常 → 统一包装成 {"raw": ..., "text": ""}，不影响上层流程。

4.3 事件规范（建议）

协议回包里，尽量产出一个 agent_text_message 或等价字段，便于统一提取主文本。

对齐 A2A 的事件形态，有利于 Runner 输出一致的用户体验。

五、开发 Checklist

 在 protocol_backend/<proto>/comm.py 实现 <Proto>CommBackend（至少 4 个方法 + 可选 spawn）。

 在 protocol_backend/<proto>/coordinator.py / worker.py 写协议侧 Executors（输入提取、输出封装）。

 写 runner/run_<proto>.py，创建网络+注册节点+设拓扑+发指令。

 配置好 config.yaml（端口、数量、拓扑）。

 python -m script.streaming_queue.runner.run_<proto> 跑通：status、dispatch。

 健康检查通过、结果 JSON 落盘、清理资源完成。

六、实践建议

延迟导入：协议 Host 端（如 FastAPI / gRPC）尽量在实际用到时再导，以减少环境依赖冲突。

版本兼容：注意如 FastAPI 与 Pydantic 的版本配套（v1/v2）问题；必要时在协议目录独立 requirements。

错误可视化：Runner 的彩色输出（INFO/SUCCESS/WARNING/ERROR）尽量保留，定位问题很高效。

统一文本：send() 最终返回都带 text 字段，方便上层展示/记录。

七、结语

这套设计把拓扑管理（NetworkBase）、协议通信（CommBackend）、AI 逻辑（QAWorkerBase）与跑法（Runner）完全解耦。
因此：

新增一个协议 = 实现一个 CommBackend + 一组轻量 Executors + 一个 Runner 文件；

既能在本进程快速演示，也能对接外部已有服务；

所有任务调度与结果收敛逻辑，继续沿用 QACoordinatorBase 的通用实现，一次编写，到处协议可用。