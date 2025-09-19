GEMINI - Meta/ACP 运行问题速查（基于 2025-09-19 日志与 workspaces 检查）

概览
- 来源：/script/gaia/logs/meta_protocol/*.log、/script/gaia/logs/acp/*.log、以及 workspaces 下的运行产物目录
- 目的：归纳当前观测到的问题、可能原因与可行的短期修复建议，便于快速定位与验证

## 已修复问题

### 1. Workspace 路径问题与文件访问 ✅ FIXED

**问题描述**: 
- Task 文件无法访问，路径不存在
- Workspace 结构不一致（meta vs meta_protocol）
- 工具找不到任务所需的附件文件

**修复方案**: 
- ✅ 修改 `runner_base.py` 增加 `_setup_task_workspace()` 方法
- ✅ 在创建每个任务时自动复制所需文件到 `workspaces/<protocol_name>/<task_id>/` 
- ✅ 更新 `TaskPlanner` 支持预创建的 workspace 目录
- ✅ 设置环境变量 `GAIA_AGENT_WORKSPACE_DIR` 指向任务专用工作区
- ✅ 每个 task 在自己的 workspace 目录中操作，文件访问路径统一为 `task.get("file_name")`

**技术实现**:
```python
def _setup_task_workspace(self, task_id: str, task: Dict[str, Any]) -> Path:
    # 创建 workspaces/<protocol_name>/<task_id>/ 目录
    workspace_dir = GAIA_ROOT / 'workspaces' / self.protocol_name / task_id
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制任务文件到工作区
    file_name = task.get("file_name") 
    if file_name:
        source_file = dataset_dir / file_name
        dest_file = workspace_dir / file_name
        shutil.copy2(source_file, dest_file)
    
    return workspace_dir
```

## 未修复问题（按优先级）

### 1. network_memory 未被正确写入或 step key 使用不一致
- deliver/消息转发点若没有将消息对象写入 NetworkMemoryPool（或写入到不同 key），最终汇总无数据。既有代码中 summarize 仅检查 step_executions 中 messages 列表。

### 2. Adapter/Executor 的异常吞噬与 fallback 策略太隐晦
- Agora adapter 在异常路径返回 fallback 字符串而不是抛出或写入详细日志，导致上层认为是"成功返回"但内容无效。

### 3. 工具执行流程（ToolCallAgent / 临时 agent）对返回消息的解析不稳定
- 临时 agent 可能只把 events 放到 events 字段或 messages[-1].content 为空，导致调用方取不到文本内容。


二、可能的根因（按优先级）

1) network_memory 未被正确写入或 step key 使用不一致
- deliver/消息转发点若没有将消息对象写入 NetworkMemoryPool（或写入到不同 key），最终汇总无数据。既有代码中 summarize 仅检查 step_executions 中 messages 列表。

2) MetaProtocolNetwork 与 Runner/配置之间的 workspace 路径不一致
- runtime.output_file 配置若指向 workspaces/meta 而不是 workspaces/meta_protocol，会导致结果与期望目录错位，自动保存与调试脚本读取不同目录。

3) Adapter/Executor 的异常吞噬与 fallback 策略太隐晦
- Agora adapter 在异常路径返回 fallback 字符串而不是抛出或写入详细日志，导致上层认为是“成功返回”但内容无效。

4) 工具执行流程（ToolCallAgent / 临时 agent）对返回消息的解析不稳定
- 临时 agent 可能只把 events 放到 events 字段或 messages[-1].content 为空，导致调用方取不到文本内容。

5) Multimodal 文件解析与挂载策略缺失/不一致
- run 时需要将数据集或 task 附带的文件复制或绑定到 agent 的工作目录；若未绑定或解析错误，str_replace_editor 等会报路径不存在。

6) 部分手动修改/回滚导致代码状态不一致
- 例如之前对 network.py 的修改、config 的临时修改被撤回，可能引入未定义变量或使原来修复失效。


三、短期可验证的修复与排查步骤（优先级顺序）

1) 立即验证 workspace 路径配置
- 检查 /script/gaia/config/meta_protocol.yaml runtime.output_file 是否指向 workspaces/meta_protocol/gaia_results.json；若不正确，改回 protocol 对应目录并重跑小样例。
- 验证 RunnerBase.resolve 输出路径逻辑是否按 protocol_name 构建路径。

2) 增加 network_memory 写入点的可见日志
- 在 MetaProtocolNetwork.deliver 中写入消息到 network_memory 的位置增加 DEBUG 日志（写入前/后打印 step_key 与 messages 长度），重跑看 network_execution_log.json 是否包含条目。

3) 让 adapter 在异常时抛出并记录详细 traceback（而非返回 fallback）
- 临时修改 AgoraAdapter 的异常路径：先 log.exception(e) 再 raise，或返回结构化的 error 字段，方便上层判断。

4) 修复 ToolCallAgent / 临时 agent 的返回抽取逻辑
- 在用临时 agent 执行时，读取消息的顺序应更可靠：优先检查 step 返回的 explicit result 字段、events 的 text parts，再到 messages 列表；必要时将 events 内容合并成 text 并写入 message.content。

5) 验证 sandbox/workspace 挂载与文件解析
- 对于 multimodal 任务，验证工作目录里是否存在 multimodal.jsonl 或任务 metadata 指向的文件；如果不存在，检查 planner 是否将在 task workspace 中下载/复制所需文件。
- 检查 sandbox_python_execute 是否在启动容器时绑定了 task-specific workspace（workspaces/<protocol>/<task_id>）和 dataset 目录。

6) 捕获并存盘“原始响应”到 step-based 日志
- 在 deliver 记录 response.raw 或 response.text 到 step_execution 的 metadata 中，便于离线排查 adapter 返回的真实内容。


四、中期修复建议（需要改代码、注意回归测试）

1) 统一消息写入管线：确保所有协议网络在发送/接收点都以统一的 Message 对象格式写入 network_memory，并使用可重现的 step_key（如 step_index 或 seq_id），避免使用临时字符串衍生 key 导致丢失。

2) 调整适配器错误策略：适配器应在异常时返回结构化错误（{error:..., raw:...}），上层 network 在收到 error 时将其写入 network_memory 并触发失败分支，而不是把 fallback 文本当正常结果。

3) 改善临时 agent 的输出策略：ToolCallAgent.step() 的返回应包含明确的 result_text 字段，且临时 agent 在处理 events 时要把文本合成到 messages 中，保证 downstream 可以直接读取到文本。

4) 增加端到端集成测试：写小型 end-to-end 测试覆盖常见工具（browser_use、str_replace_editor、python_execute、create_chat_completion）在 meta 与 acp 场景的互相调用。


五、立即要做的 5 个具体操作（可直接执行）

1. 检查并修正 meta_protocol.yaml 中 runtime.output_file 指向（若仍错，修正为 workspaces/meta_protocol/gaia_results.json）。
2. 在 MetaProtocolNetwork.deliver 加入 2-3 行 debug log，打印 step_key、写入前后 messages 长度并保存 raw response 字段。
3. 修改 Agora/Adapter 的异常处理：把 catch 中的返回改为 log.exception + 返回结构化错误或 raise。
4. 临时在运行目录（workspaces/acp/<task_id>/）查看 network_execution_log.json，确认该 json 中是否记录了 step messages（用于对比是否只在 meta 路径缺失）。
5. 对一条简单任务（只含 create_chat_completion）快速回归：确认 agent 间通信、network_memory 写入、最终 summary 非空。


六、日志/证据指向（摘录）
- “📝 Final summary: No messages to summarize.” → 示 network_memory.summarize 没有 messages
- “[Agora Fallback] Processed:” → adapter fallback，实际工具未成功
- Agent 1 result: {'text': '', 'events': [...] } → 工具执行结果放在 events 而非 text
- 多个 /workspaces/acp/<task_id>/network_execution_log.json 中有记录，但 meta 的 workspaces/meta/gaia_results.json 文件结构异常 → 配置或路径不一致


结束语
- 当前可疑最大根因集中在：network_memory 写入不一致 + adapter 异常吞噬 + workspace 路径配置错误。按“先修配置、再增日志、再修适配器、最后改消息管线”的顺序，会最快看到改善并定位剩余问题。

- 若需要，我可以：
  1) 帮你在 repo 中按上文逐条落实现有修改（小 patch），并运行一次本地小任务回归；
  2) 或者直接生成精确的代码片段（deliver 中的 debug 日志、adapter 的异常处理改法、ToolCallAgent 的输出合并逻辑）以供你手动应用。

-GitHub Copilot
