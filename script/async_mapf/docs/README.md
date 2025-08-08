# Async-MAPF 协议后端开发 README

> 本文说明**如何为框架新增一种协议后端**。在本框架里：
>
> - **Agent（协议侧实例）**：把“协议进来的事件/消息”翻译并转发给核心 Agent（`core.agent_base.BaseAgent`）。  
> - **CommAdapter（通信适配器）**：只负责“如何发送/接收/序列化/反序列化”。  
> - **Runner（装配/起服）**：把 Network 与各 Agent 的协议服务（ASGI）启动起来，并处理优雅退出。

---

## 目录结构（约定）

script/async_mapf/
├── core/ # ★ 算法与网络协调（协议无关，别改）
│ ├── agent_base.py # BaseAgent
│ └── network_base.py # NetworkBase（已支持全部到达即自动结束）
├── protocol_backends/ # 🔌 每种协议一个子目录
│ ├── a2a/
│ │ ├── adapters/a2a_comm_adapter.py
│ │ ├── agents/mapf_worker_executor.py # 协议侧 agent 实例
│ │ ├── agents/network_executor.py # 协议侧 network 入口
│ │ └── runner.py # ★ A2A Runner（固定放这里）
│ └── myproto/
│ ├── adapters/myproto_comm_adapter.py # ★ 你实现
│ ├── agents/agent_executor.py # ★ 你实现
│ ├── agents/network_executor.py # （可选：如果协议需要）
│ └── runner.py # ★ 建议放这里（或放 runners/）
├── runners/
│ ├── base_runner.py # ★ 通用生命周期/优雅退出（你不改）
│ ├── run_a2a.py # 入口：python -m script.async_mapf.runners.run_a2a
│ └── run_myproto.py # 你的协议入口（可选，也可直接 myproto/runner.py）
└── config/
├── a2a.yaml
└── myproto.yaml


> 约定补充：  
> - **A2A 的 Runner 已固定路径**：`script/async_mapf/protocol_backends/a2a/runner.py`。  
> - 其它 Runner 你可以放 `protocol_backends/<proto>/runner.py`，也可以放 `script/async_mapf/runners/`，二者择一，**保持 import 路径一致**即可。  
> - 启动/关停/信号处理/“全部到达即退出”，由 `runners/base_runner.py` 统一处理。

---

## 三个你需要实现/关注的点

### 1) CommAdapter（通信适配器）

**职责唯一**：把“框架对象”↔“协议载荷”做序列化/反序列化，并用你的协议（HTTP/WS/ZMQ/…）**发出去/收回来**。

统一接口（与 `core.comm.AbstractCommAdapter` 对齐）：
- `async def send(self, obj: Any) -> None`
- `async def recv(self) -> Any`
- `def recv_nowait(self) -> Any`
- `async def close(self) -> None`
- `def handle_incoming(self, raw) -> None`（协议服务入口把收到的原始消息喂给这里）

**推荐消息形状**（跨协议尽量统一）：

```json
// CONTROL
{"type":"CONTROL","payload":{"cmd":"START","agent_id":0}}

// MOVE_REQUEST（并发模式）
{"type":"MOVE_REQUEST","payload":{
  "agent_id":0,"move_id":"uuid","new_pos":[x,y],
  "eta_ms":120,"time_window_ms":50,"priority":1
}}

// MOVE_RESPONSE
{"type":"MOVE_RESPONSE","payload":{
  "agent_id":0,"move_id":"uuid","status":"OK|CONFLICT|REJECT",
  "reason":"...", "conflicting_agents":[2,3], "suggested_eta_ms":100
}}

// MOVE_FB（兼容旧式 MoveCmd 的反馈）
{"type":"MOVE_FB","payload":{
  "agent_id":0,"success":true,"actual_pos":[x,y],"reason":""
}}

// CHAT
{"type":"CHAT","payload":{"src":0,"dst":1,"msg":"hi"}}

最小示例（伪实现，给你骨架）：

# script/async_mapf/protocol_backends/myproto/adapters/myproto_comm_adapter.py
import asyncio, json
from typing import Any, Dict, Optional
from script.async_mapf.core.comm import AbstractCommAdapter
from script.async_mapf.core.types import MoveCmd, MoveFeedback

class MyProtoCommAdapter(AbstractCommAdapter):
    def __init__(self, self_id: str, network_url: Optional[str] = None, agent_urls: Optional[dict] = None):
        self.self_id = self_id
        self.network_url = network_url
        self.agent_urls = agent_urls or {}
        self._rx_q = asyncio.Queue()

    async def connect(self):   ...
    async def disconnect(self):...
    async def close(self):     await self.disconnect()

    async def send(self, obj: Any) -> None:
        data = self._serialize(obj)
        # TODO: 用你的协议把 data 发出去（HTTP/WS/ZMQ等）

    async def recv(self) -> Any:      return await self._rx_q.get()
    def recv_nowait(self) -> Any:     return self._rx_q.get_nowait()

    def handle_incoming(self, raw: dict) -> None:
        obj = self._deserialize(raw)
        asyncio.create_task(self._rx_q.put(obj))

    def _serialize(self, obj: Any) -> Dict[str, Any]:
        if isinstance(obj, MoveCmd):
            return {"type":"MOVE_CMD","payload":{"agent_id":obj.agent_id,"new_pos":list(obj.new_pos)}}
        if isinstance(obj, MoveFeedback):
            return {"type":"MOVE_FB","payload":{
                "agent_id":obj.agent_id,"success":obj.success,"actual_pos":list(obj.actual_pos),"reason":""}}
        if isinstance(obj, dict): return obj
        return {"type":"UNKNOWN","payload":{"content":str(obj)}}

    def _deserialize(self, msg: Any) -> Any:
        if isinstance(msg, dict) and msg.get("type") == "MOVE_FB":
            return MoveFeedback(
                agent_id=msg["payload"].get("agent_id",0),
                success=bool(msg["payload"].get("success",False)),
                actual_pos=tuple(msg["payload"].get("actual_pos",[0,0])),
                reason=msg["payload"].get("reason","")
            )
        return msg


2) Agent（协议 agent 实例）
这是协议侧的 agent（比如 A2A 的 MAPFAgentExecutor）。你的实现需要在协议框架的回调里：

收到 CONTROL.START ⇒ 启动 core_agent.autonomous_loop()

收到 MOVE_RESPONSE ⇒ 调 core_agent._handle_move_response(payload)

收到 MOVE_FB ⇒ 放入 core_agent._recv_msgs_queue

其它协议事件按需路由给 core_agent（例如 CHAT）

关键点：通信用你的 CommAdapter；Agent 实例只做“协议事件 → 核心 Agent”的桥接。

# script/async_mapf/protocol_backends/myproto/agents/agent_executor.py
import asyncio, json
from typing import Any, Dict
from script.async_mapf.core.agent_base import BaseAgent as CoreAgent
from ..adapters.myproto_comm_adapter import MyProtoCommAdapter

class MyProtoAgentExecutor:
    """Protocol-facing agent instance. Bridges protocol events to CoreAgent."""
    def __init__(self, cfg: Dict[str, Any], global_cfg: Dict[str, Any], agent_id: int, network_url: str, output=None):
        self.agent_id = agent_id
        self.output = output
        self.adapter = MyProtoCommAdapter(str(agent_id), network_url=network_url)
        asyncio.create_task(self.adapter.connect())
        self.core_agent = CoreAgent(
            agent_id=agent_id,
            adapter=self.adapter,
            config={"agent_config": cfg, "model": global_cfg.get("model"), "protocol": "myproto"}
        )

    async def on_message(self, payload: str | Dict[str, Any]):
        data = json.loads(payload) if isinstance(payload, str) else payload
        t, p = data.get("type"), data.get("payload", {})

        if t == "CONTROL" and p.get("cmd") == "START":
            if not hasattr(self, "_loop") or self._loop.done():
                self._loop = asyncio.create_task(self.core_agent.autonomous_loop())
            return {"ok": True}

        if t == "MOVE_RESPONSE":
            await self.core_agent._handle_move_response(p)
            return {"ok": True}

        if t == "MOVE_FB":
            await self.core_agent._recv_msgs_queue.put(self.adapter._deserialize(data))
            return {"ok": True}

        # 可扩展：CHAT、MOVE_CMD(旧) 等
        return {"ok": False, "reason": "ignored"}

3) NetworkExecutor（可选）
若你的协议也需要一个网络端入口（像 A2A 的 NetworkBaseExecutor），你需要：

收到 MOVE_REQUEST ⇒（带锁）调用 network_base._apply_move_concurrent(...)，并回 MOVE_RESPONSE

调用 network_base.set_a2a_send_callback(self.send_to_agent)，把“回发给 Agent 的方法”注册进去（NetworkBase 会在广播 START/STOP 时用它）

并发安全：修改 world_state 时请 async with network_base._move_lock:。

4) Runner（装配/起服）
Runner 继承 RunnerBase，只需实现两件事：

build_network_app(network_coordinator, agent_urls)：返回 Network 侧 ASGI 应用

build_agent_app(agent_id, agent_cfg, port, network_base_port)：返回 Agent 侧 ASGI 应用

其它事情（加载 YAML、动态端口、起所有服务、监听 Ctrl+C、全部到达即自动退出、优雅关停）都由 RunnerBase 处理好了。

最小 Runner 示例（用 FastAPI 模拟 HTTP 协议）：

# script/async_mapf/protocol_backends/myproto/runner.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import asyncio, time
from typing import Dict, Any
from script.async_mapf.runners.base_runner import RunnerBase
from .agents.agent_executor import MyProtoAgentExecutor

class MyProtoRunner(RunnerBase):
    def build_network_app(self, network_coordinator, agent_urls: Dict[int, str]):
        app = FastAPI(title="MAPF Network (myproto)")

        # 提供一个 Agent -> Network 的入口
        @app.post("/msg")
        async def on_msg(req: Request):
            data = await req.json()
            t, p = data.get("type"), data.get("payload", {})

            if t == "MOVE_REQUEST":
                from script.async_mapf.core.concurrent_types import ConcurrentMoveCmd
                if not hasattr(network_coordinator, "_move_lock"):
                    network_coordinator._move_lock = asyncio.Lock()

                cmd = ConcurrentMoveCmd(
                    agent_id=p["agent_id"], new_pos=tuple(p["new_pos"]),
                    eta_ms=p.get("eta_ms", 100), time_window_ms=p.get("time_window_ms", 50),
                    move_id=p.get("move_id",""), priority=p.get("priority",1)
                )

                exec_ts = int(time.time()*1000)
                async with network_coordinator._move_lock:
                    ok, conflicts = network_coordinator._apply_move_concurrent(cmd, exec_ts)

                resp = {"type":"MOVE_RESPONSE","payload":{
                    "agent_id":cmd.agent_id,"move_id":cmd.move_id,
                    "status":"OK" if ok else ("CONFLICT" if conflicts else "REJECT"),
                    "reason":"Move successful" if ok else ("Conflict" if conflicts else "Move failed"),
                    "conflicting_agents":conflicts, "suggested_eta_ms":100 if conflicts else None
                }}

                # 通过已注册的回调把响应送回 Agent
                if hasattr(network_coordinator, "_a2a_send_callback"):
                    await network_coordinator._a2a_send_callback(cmd.agent_id, resp)

            return JSONResponse({"ok": True})

        # 注册“如何回发给 Agent”的回调（这里示范用 HTTP 客户端，你自行实现）
        async def send_to_agent(agent_id: int, message: dict):
            # TODO: 用你的协议把 message 送到 agent（例如 POST 到 agent 的 /msg）
            pass

        network_coordinator.set_a2a_send_callback(send_to_agent)
        return app

    def build_agent_app(self, agent_id: int, agent_cfg: Dict[str, Any], port: int, network_base_port: int):
        app = FastAPI(title=f"MAPF Agent {agent_id} (myproto)")
        network_url = f"http://localhost:{network_base_port}"

        executor = MyProtoAgentExecutor(agent_cfg, self.sim_config, agent_id, network_url)

        @app.post("/msg")  # Network -> Agent
        async def on_msg(req: Request):
            data = await req.json()
            result = await executor.on_message(data)
            return JSONResponse(result)

        return app

A2A 专用 Runner 的位置固定为 script/async_mapf/protocol_backends/a2a/runner.py（你已经有了）。
其它协议建议同样放在对应协议目录下的 runner.py；若你更倾向集中放到 script/async_mapf/runners/ 也可，但注意 import 路径。

运行
A2A（已就位）：

bash
python -m script.async_mapf.runners.run_a2a
你的协议（myproto）：新建 runners/run_myproto.py 或直接 protocol_backends/myproto/runner.py 写 main，示例：

python
# script/async_mapf/runners/run_myproto.py
import asyncio
from pathlib import Path
from script.async_mapf.protocol_backends.myproto.runner import MyProtoRunner

if __name__ == "__main__":
    cfg = Path("script/async_mapf/config/myproto.yaml")
    asyncio.run(MyProtoRunner(cfg).run())
然后：

python -m script.async_mapf.runners.run_myproto

开发要点 & 排障清单
务必回包：MOVE_REQUEST 无论成功/失败/异常，都要回 MOVE_RESPONSE，否则 Agent 会等待悬挂。

并发安全：Network 端修改 world_state 要 async with network_base._move_lock:。

自动结束：NetworkBase 在每次状态更新后会 _maybe_trigger_completion；RunnerBase 已把完成回调接到 shutdown_event，会优雅退出，无需你再写。

日志：尽量用 script.async_mapf.utils.log_utils 统一 logger；关键事件可以 log_network_event(...)。

消息形状统一：遵守上面的 JSON 结构，可以极大减少胶水代码。

路径/约定：A2A 的 Runner 路径固定；其它协议 Runner 放哪都行，但保证 import 一致。

FAQ
Q: Agent 里 LLM 没初始化完就 START，会怎样？
A: RunnerBase 会延时广播 START（你的后端也可以做类似处理）。Agent 侧也可在 CONTROL.START 收到时自检并延迟启动。

Q: 我只想用本地内存队列做个假的协议跑通 CI？
A: 参考 protocol_backends/dummy（或照着上面的 CommAdapter 骨架，直接用 asyncio.Queue 即可）。

Q: 我需要自定义“全部到达”的判定？
A: network_base.py 已支持目标判定与完成回调。必要时在 YAML 加 stop_on_all_goals: false，自行控制结束时机，并在 Runner 调 shutdown()。

后续工作 / 已知限制（WIP）
日志落盘未完成

当前主要是终端打印；log_utils/LogManager 到文件的落盘链路仍有问题（部分模块 logger 未正确挂载 handler，A2A 事件也未统一落盘）。

计划：梳理统一的 logging.Logger 层级，添加 RotatingFileHandler，确保 script/async_mapf/logs/ 下按 run-id/日期分目录写入，并修复重复日志/丢日志问题。

Metrics 未集成

metrics/ 模块尚未接入运行时流程；未记录每步时延、冲突率、吞吐等指标。

计划：在 NetworkBase._apply_move[_concurrent] 与 Agent 关键路径打点，使用 MetricsRecorder 定期 flush；提供简单的离线聚合与可视化脚本。

Agent 卡死判定 & 紧急退出缺失

目前没有心跳/进度 watchdog；某个 Agent 中途挂起或 LLM 超时会导致整体停滞。

计划：为每个 Agent/网络回路加入心跳与“无进展超时”检测（如 N 秒无坐标变化 / 无响应即重试或标记失败），支持一键 EMERGENCY_STOP 广播与 Runner 级别强制退出。

Ctrl+C 直接退出未完善

Windows/WSL 等环境下信号处理不一致，个别场景 Ctrl+C 不能即时、优雅地退出（需等 uvicorn/后台任务自行结束）。

计划：完善 RunnerBase._install_signal_handlers，对 uvicorn/server/background tasks 统一超时取消，必要时提供可配置的二段式强杀（超时后调用 os._exit(1) 作为最后兜底）