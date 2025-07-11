# Agent Protocol 问答系统

基于 Agent Protocol 架构实现的智能问答系统，使用 top1000.jsonl 数据集回答用户问题。

## 系统架构

本系统采用 Agent Protocol 的标准架构：

- **Task Handler**: 处理新的问答任务
- **Step Handler**: 执行具体的处理步骤
- **数据加载器**: 管理问答数据集

## 工作流程

1. **规划问答流程** - 分析用户输入，制定处理计划
2. **分析用户问题** - 提取关键词和问题特征
3. **搜索相关答案** - 在数据集中匹配最佳答案
4. **格式化回答** - 美化输出格式
5. **提供答案** - 返回最终结果和相关建议

## 文件说明

- `qa_agent.py` - 主要的问答Agent实现
- `qa_client.py` - 测试客户端，支持交互式问答
- `top1000.jsonl` - 问答数据集（位于 ANP/streaming_queue/data/）

## 使用方法

### 1. 安装依赖

确保已安装 agent-protocol 包：

```bash
pip install agent-protocol requests
```

### 2. 启动问答Agent

```bash
python qa_agent.py
```

启动后Agent将在 http://localhost:8000 提供服务

### 3. 使用客户端测试

#### 交互式模式
```bash
python qa_client.py
```

#### 批量测试模式
```bash
python qa_client.py test
```

### 4. API调用示例

也可以直接通过HTTP API调用：

```python
import requests

# 创建任务
response = requests.post("http://localhost:8000/ap/v1/agent/tasks", 
                        json={"input": "what is java for"})
task_id = response.json()["task_id"]

# 执行步骤
requests.post(f"http://localhost:8000/ap/v1/agent/tasks/{task_id}/steps")
```

## 功能特点

- 🔍 **智能匹配**: 支持完全匹配和关键词部分匹配
- 📊 **步骤可视**: 清晰展示每个处理步骤
- 💬 **友好界面**: 格式化输出，易于阅读
- 🎯 **相关推荐**: 提供相关问题建议
- 🚀 **异步处理**: 基于asyncio的高性能架构

## 数据集格式

系统使用的 JSONL 格式数据，每行包含问题和答案对：

```json
{"188714": "1082792", "1000052": "1000084", "foods and supplements to lower blood sugar": "what does the golgi apparatus do to the proteins and lipids once they arrive ?", "Watch portion sizes: ...": "Start studying Bonding, Carbs, Proteins, Lipids..."}
```

## 示例问题

- "what is java for"
- "where is the graphic card located in the cpu"  
- "what is the nutritional value of oatmeal"
- "how to become a teacher assistant"
- "what foods are good if you have gout"

## 扩展建议

1. **改进匹配算法**: 使用TF-IDF或语义相似度
2. **添加缓存**: 缓存常见问题答案
3. **多语言支持**: 支持中英文问答
4. **Web界面**: 开发前端界面
5. **日志记录**: 记录问答历史和性能指标

## 故障排除

- 确保端口8000未被占用
- 检查数据文件路径是否正确
- 验证agent-protocol包版本兼容性

Agent Protocol 文档如下：

---
# Agent Protocol

The reason for creating the Agent Protocol was to provide a standardized way how interact with the agents. This is useful for automation, agent to agent communication, general UIs or dev tools.

## Description
The Agent Protocol is a OpenAPI specification v3 based protocol. The protocol is meant to be simple and general, so it can be adopted by any agent.

The base objects of the protocol are `Tasks`, `Steps` and `Artifacts`.

## Task
A `Task` denotes one specific goal for the agent, it can be specific like:

```
Create a file named `hello.txt` and write `World` to it.
```

or very broad as:

```
Book a flight from Berlin to New York next week, optimize for price and duration.
```

The `Task` object has the following properties:

| Property | Type | Description |
| :--- | :--- | :--- |
| `task_id` | string | The ID of the task. |
| `input` | string | Input prompt for the task. |
| `additional_input` | object | Additional input for the task. |
| `steps` | array[Step] | The steps of the task. |
| `artifacts` | array[Artifact] | A list of artifacts that the task has produced. |

## Step
A `Step` is a single action that the agent should perform. Each step is triggered by calling the step endpoint of the agent. The `Step` object has the following properties:

| Property | Type | Description |
| :--- | :--- | :--- |
| `task_id` | string | The ID of the task. |
| `step_id` | string | The ID of the step. |
| `input` | string | Input prompt for the step. |
| `additional_input` | object | Additional input for the step. |
| `name` | string | The name of the step. |
| `status` | enum | The status of the step. Possible values are `created` and `completed`. |
| `output` | string | Output of the step. |
| `additional_output` | object | Additional output of the step. |
| `artifacts` | array[Artifact] | A list of artifacts that the step has produced. |
| `is_last` | boolean | Whether this is the last step in the task. |

## Artifact
An `Artifact` is a file that the agent has worked with. The `Artifact` object has the following properties:

| Property | Type | Description |
| :--- | :--- | :--- |
| `artifact_id` | string | The ID of the artifact. |
| `file_name` | string | Filename of the artifact. |
| `relative_path` | string | Relative path of the artifact in the agent's workspace. |

## Endpoints
The Agent Protocol has two main endpoints:

*   `POST /ap/v1/agent/tasks` - This endpoint is used to create a new task for the agent.
*   `POST /ap/v1/agent/tasks/{task_id}/steps` - This endpoint is used to trigger next step of the task.

To see all endpoints and their descriptions, please refer to the Endpoints.