# 协议后端（protocols）热插拔指南

本目录用于存放不同协议（protocol backends）的实现，目标是做到“即插即用”。你可以按统一的接口契约扩展一个新协议，或在配置里切换协议实现。

## 目录结构约定

每种协议一个子目录，例如：

- `protocol_backends/agora/`
  - `agent.py`：协议下的 Agent，通常继承自 `core.agent.MeshAgent`
  - `network.py`：协议下的 Network，继承自 `core.network.MeshNetwork`
  - `__init__.py`：导出协议对外可见的类

你也可以增加该协议的文档或工具文件，只要不破坏上述核心文件结构即可。

## 接口契约（必须遵循）

- Network（继承 MeshNetwork）
  - 必须实现：
    - `async def deliver(self, dst: int, msg: Dict[str, Any]) -> None`
  - 可按需重写：
    - `async def start(self)` / `async def stop(self)`：用于协议初始化/清理（例如启动本地服务、建立连接等）
    - `async def health_check(self) -> bool`
    - `async def _execute_agent_step(...) -> str`：需要覆盖默认工作流执行方式时重写
  - 建议提供：
    - `register_agents_from_config()`：从 planner 生成的 general config 实例化并注册协议 Agent
    - `create_<protocol>_agent(...)`：按配置构建 Agent（便于复用）

- Agent（继承 MeshAgent）
  - 建议提供：
    - `async def execute(self, message: str) -> str`：当协议以“服务化/回调”的方式驱动时，这是对外暴露的统一调用入口
    - `def _run_server(self)`（可选）：若该协议需为每个 Agent 启动本地接收服务（如 HTTP），在此启动并暴露工具接口（tools）
  - 说明：跨 Agent 的消息路由通常由 Network 负责，Agent 层尽量专注于“思考/工具调用/产生结果”。

## 配置与选择（热插拔）

planner 生成的 general config 建议包含：

```jsonc
{
  "network": {
    "type": "agora",          // 切换不同协议：如 "agora", "acp", "anp" ...
    "timeout_seconds": 30
  },
  "agents": [
    {
      "id": 1,
      "name": "Planner",
      "tool": "planner",
      "port": 8001,
      "use_real_agora": true,  // 协议特定的字段（示例：agora）
      "openai_model": "gpt-4o-mini",
      "openai_temperature": 0.1
    }
  ],
  "agent_prompts": {
    "1": { "system_prompt": "You are a planning agent." }
  },
  "workflow": { /* 见各业务 Runner 示例 */ }
}
```

切换协议的典型方式有两种：
- 静态选择：在 runner/工厂中根据 `network.type` 分支导入对应协议的 `Network` 类（如 `AgoraNetwork`），并用同一份 general config 实例化。
- 动态导入（可选）：允许配置里写模块路径，例如 `"network": { "factory": "script.gaia.protocol_backends.agora.network:AgoraNetwork" }`，由工厂动态 import 后实例化。

> 注意：本仓库默认采用“静态选择”（以 Agora 为例）。你可以在项目的 runner 或装配处加入一个简单的映射，从而做到“写好协议 → 改一行 type → 即可热插拔”。

## 以 Agora 为例

- 目录：`protocol_backends/agora/`
  - `AgoraAgent`：实现 `execute(message)`；提供 `_run_server()` 启动本地 ReceiverServer，并在创建 Toolformer 时直接 `tools=[tool]` 绑定该 Agent 的工具（服务化回调）。
  - `AgoraNetwork`：
    - 从 general config 创建并注册 Agent。
    - `start()` 时：
      - 若 `use_real_agora=true`，调用 `agent._run_server()` 启动服务；
      - 否则走本地 `agent.execute()` 路径（无需 HTTP）。
    - `deliver()`：根据是否真实服务决定 HTTP 发送或本地执行。

- 运行前置：若使用真实服务，需要 `OPENAI_API_KEY`，并安装 `agora` 与 `langchain_openai`。

## 新协议接入步骤

1. 在 `protocol_backends/` 下新建目录，例如 `myproto/`。
2. 实现 `myproto/agent.py` 与 `myproto/network.py`：
   - `MyProtoAgent(MeshAgent)`，按需实现 `execute()` / `_run_server()`；
   - `MyProtoNetwork(MeshNetwork)`，实现 `deliver()`、从 config 注册 Agent、启动/停止等逻辑。
3. 在你项目使用网络的地方（runner/装配处）增加 `network.type -> Network类` 的映射（或支持动态导入）。
4. 在 general config 中把 `network.type` 改为你的协议名，并按需添加协议特定字段。

## 常见注意事项

- 不要把传输/路由逻辑塞进 Agent；统一放在 Network 层，Agent 专注于“思考+工具调用”。
- 服务化协议（HTTP/GRPC 等）建议在 Agent 内提供 `_run_server()`，并在 `Network.start()` 中按配置启动。
- 如果需要对工作流执行做协议特化，可重写 `MeshNetwork._execute_agent_step`，但要保留基本的内存记录和错误处理。
- 保持最小可依赖：协议相关的第三方依赖尽量延迟导入，或在未安装时给出清晰报错并保留“本地执行”兜底路径。

---
如需更多示例，可参考 `protocol_backends/agora/` 的实现。
