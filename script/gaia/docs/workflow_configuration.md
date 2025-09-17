# 工作流配置指南

本文档详细说明 GAIA 框架中工作流的配置和使用方法。

## 工作流概述

GAIA 工作流系统支持声明式的多智能体协作编排，通过配置文件定义智能体之间的消息流向和执行顺序。

## 基本工作流配置

### 配置结构

```yaml
# 工作流配置示例
workflow:
  start_agent: 0                    # 起始智能体 ID
  execution_pattern: "sequential"   # 执行模式
  message_flow:                     # 消息流向定义
    - from: 0                       # 源智能体 ID
      to: [1]                       # 目标智能体 ID 列表
      message_type: "task"          # 消息类型
    - from: 1
      to: [2]
      message_type: "result"
    - from: 2
      to: "final"                   # "final" 表示工作流结束
      message_type: "final_result"

agents:                             # 智能体配置
  - id: 0
    name: "DataProcessor"
    tool: "create_chat_completion"
    port: 9000
    role: "processor"
  - id: 1
    name: "Analyzer"
    tool: "create_chat_completion"
    port: 9001
    role: "analyzer"
  - id: 2
    name: "Summarizer"
    tool: "create_chat_completion"
    port: 9002
    role: "summarizer"

agent_prompts:                      # 智能体提示词配置（可选）
  "0":
    system_prompt: "You are a data processor. Process the input data and extract key information."
  "1":
    system_prompt: "You are an analyzer. Analyze the processed data and provide insights."
  "2":
    system_prompt: "You are a summarizer. Create a concise summary of the analysis."

task_id: "workflow_demo_001"       # 任务标识符
```

### 执行模式

GAIA 支持多种执行模式：

#### 1. Sequential（顺序执行）

```yaml
workflow:
  execution_pattern: "sequential"
  message_flow:
    - from: 0
      to: [1]
      message_type: "task"
    - from: 1
      to: [2]
      message_type: "result"
```

**特点**：
- 智能体按顺序执行
- 前一个智能体完成后才启动下一个
- 适合有明确依赖关系的任务

#### 2. Parallel（并行执行）

```yaml
workflow:
  execution_pattern: "parallel"
  message_flow:
    - from: 0
      to: [1, 2, 3]               # 同时发送给多个智能体
      message_type: "broadcast_task"
    - from: [1, 2, 3]             # 多个智能体的结果
      to: 4
      message_type: "collect_results"
```

**特点**：
- 多个智能体同时执行
- 提高执行效率
- 适合可并行处理的任务

#### 3. Pipeline（流水线执行）

```yaml
workflow:
  execution_pattern: "pipeline"
  message_flow:
    - from: 0
      to: [1]
      message_type: "stage1"
    - from: 1
      to: [2]
      message_type: "stage2"
    - from: 2
      to: [0]                     # 回到第一个智能体，形成循环
      message_type: "feedback"
```

**特点**：
- 支持循环和反馈
- 适合迭代优化的任务
- 可配置循环条件

## 高级工作流配置

### 条件执行

```yaml
workflow:
  execution_pattern: "conditional"
  message_flow:
    - from: 0
      to: [1]
      message_type: "task"
      condition:                  # 条件配置
        type: "result_check"
        field: "confidence"
        operator: ">"
        value: 0.8
    - from: 0
      to: [2]                     # 备选路径
      message_type: "fallback_task"
      condition:
        type: "result_check"
        field: "confidence"
        operator: "<="
        value: 0.8
```

### 动态路由

```yaml
workflow:
  execution_pattern: "dynamic"
  message_flow:
    - from: 0
      to: "router"                # 动态路由器
      message_type: "route_decision"
      routing_logic:
        - condition: "data_type == 'text'"
          target: [1]
        - condition: "data_type == 'image'"
          target: [2]
        - condition: "data_type == 'audio'"
          target: [3]
        - default: [4]            # 默认路由
```

### 错误处理

```yaml
workflow:
  execution_pattern: "fault_tolerant"
  error_handling:
    retry_count: 3               # 重试次数
    timeout_seconds: 30          # 超时时间
    fallback_agent: 999          # 备用智能体
    
  message_flow:
    - from: 0
      to: [1]
      message_type: "task"
      error_action:              # 错误处理动作
        on_timeout: "retry"
        on_failure: "fallback"
        max_retries: 2
```

## 智能体配置详解

### 基本智能体配置

```yaml
agents:
  - id: 0                        # 必需：唯一标识符
    name: "TaskAgent"            # 必需：智能体名称
    tool: "create_chat_completion" # 必需：使用的工具
    port: 9000                   # 必需：监听端口
    
    # 可选配置
    role: "coordinator"          # 角色定义
    priority: 1                  # 优先级 (1-10)
    max_tokens: 1000            # 最大 token 数
    timeout: 30                  # 超时时间（秒）
    
    # 协议特定配置
    protocol_config:
      message_loss_rate: 0.01    # 消息丢失率
      delivery_delay: 0.1        # 投递延迟
      compression: true          # 是否压缩消息
```

### 智能体专业化配置

```yaml
agent_prompts:
  "0":  # 智能体 ID
    system_prompt: |
      You are a data coordinator responsible for:
      1. Receiving initial data requests
      2. Analyzing data requirements  
      3. Routing tasks to appropriate specialists
      4. Ensuring data quality and consistency
      
      Always provide structured output in JSON format.
      
    personality:
      style: "professional"
      verbosity: "concise"
      focus: "accuracy"
      
    constraints:
      max_response_length: 500
      required_fields: ["task_type", "priority", "estimated_time"]
      
  "1":
    system_prompt: |
      You are a data processing specialist focused on:
      - Text analysis and natural language processing
      - Data cleaning and normalization
      - Feature extraction
      
      Provide detailed analysis with confidence scores.
      
    tools_allowed: ["text_analysis", "data_cleaning", "feature_extraction"]
    output_format: "structured_json"
```

### 智能体能力配置

```yaml
agents:
  - id: 0
    name: "MultimodalAgent"
    tool: "create_chat_completion"
    port: 9000
    
    capabilities:               # 能力定义
      - "text_processing"
      - "image_analysis"
      - "data_visualization"
      
    tools:                     # 可用工具列表
      - name: "text_summarizer"
        config:
          max_length: 200
      - name: "image_classifier"
        config:
          model: "resnet50"
      - name: "chart_generator"
        config:
          style: "matplotlib"
          
    resources:                 # 资源限制
      cpu_limit: "2"          # CPU 核心数
      memory_limit: "4Gi"     # 内存限制
      gpu_required: false     # 是否需要 GPU
```

## 消息类型和格式

### 标准消息类型

```yaml
message_types:
  task:                       # 任务分配消息
    required_fields: ["content", "priority"]
    optional_fields: ["deadline", "context"]
    
  result:                     # 结果返回消息
    required_fields: ["content", "status"]
    optional_fields: ["confidence", "metadata"]
    
  status:                     # 状态更新消息
    required_fields: ["agent_id", "status"]
    optional_fields: ["progress", "estimated_completion"]
    
  error:                      # 错误报告消息
    required_fields: ["error_type", "description"]
    optional_fields: ["stack_trace", "suggestions"]
```

### 消息格式示例

```python
# 任务消息
task_message = {
    "type": "task",
    "content": "Analyze the sentiment of the following text...",
    "priority": 1,
    "deadline": "2024-01-01T12:00:00Z",
    "context": {
        "source": "user_input",
        "language": "en",
        "domain": "product_review"
    }
}

# 结果消息
result_message = {
    "type": "result",
    "content": {
        "sentiment": "positive",
        "score": 0.85,
        "keywords": ["excellent", "satisfied", "recommend"]
    },
    "status": "completed",
    "confidence": 0.92,
    "metadata": {
        "processing_time": 1.2,
        "model_version": "v2.1"
    }
}
```

## 工作流执行控制

### 步骤控制

```yaml
workflow:
  step_control:
    max_steps: 10             # 最大步骤数
    step_timeout: 60          # 单步超时（秒）
    total_timeout: 600        # 总超时（秒）
    
  validation:
    validate_inputs: true     # 验证输入
    validate_outputs: true    # 验证输出
    schema_validation: true   # 模式验证
    
  monitoring:
    log_level: "INFO"         # 日志级别
    metrics_enabled: true     # 启用指标收集
    trace_enabled: true       # 启用链路追踪
```

### 资源管理

```yaml
workflow:
  resource_management:
    memory_limit: "8Gi"       # 总内存限制
    cpu_limit: "4"           # 总 CPU 限制
    agent_isolation: true     # 智能体隔离
    
  scaling:
    auto_scale: true          # 自动扩缩容
    min_agents: 1            # 最小智能体数
    max_agents: 10           # 最大智能体数
    scale_metric: "queue_length"  # 扩缩容指标
    scale_threshold: 5        # 扩缩容阈值
```

## 工作流示例

### 示例 1：文档处理流水线

```yaml
# document_processing_workflow.yaml
workflow:
  start_agent: 0
  execution_pattern: "sequential"
  message_flow:
    - from: 0
      to: [1]
      message_type: "extract_text"
    - from: 1
      to: [2]
      message_type: "analyze_content"
    - from: 2
      to: [3]
      message_type: "generate_summary"
    - from: 3
      to: "final"
      message_type: "final_result"

agents:
  - id: 0
    name: "DocumentReader"
    tool: "create_chat_completion"
    port: 9000
  - id: 1
    name: "TextExtractor"
    tool: "create_chat_completion"
    port: 9001
  - id: 2
    name: "ContentAnalyzer"
    tool: "create_chat_completion"
    port: 9002
  - id: 3
    name: "SummaryGenerator"
    tool: "create_chat_completion"
    port: 9003

agent_prompts:
  "0":
    system_prompt: "You are a document reader. Extract and prepare text for processing."
  "1":
    system_prompt: "You are a text extractor. Clean and structure the text data."
  "2":
    system_prompt: "You are a content analyzer. Analyze the text for key themes and insights."
  "3":
    system_prompt: "You are a summary generator. Create concise, informative summaries."

task_id: "doc_processing_001"
```

### 示例 2：并行数据处理

```yaml
# parallel_data_processing.yaml
workflow:
  start_agent: 0
  execution_pattern: "parallel"
  message_flow:
    - from: 0
      to: [1, 2, 3]           # 并行处理
      message_type: "process_chunk"
    - from: [1, 2, 3]         # 收集结果
      to: 4
      message_type: "merge_results"
    - from: 4
      to: "final"
      message_type: "final_output"

agents:
  - id: 0
    name: "DataSplitter"
    tool: "create_chat_completion"
    port: 9000
  - id: 1
    name: "Processor1"
    tool: "create_chat_completion"
    port: 9001
  - id: 2
    name: "Processor2"
    tool: "create_chat_completion"
    port: 9002
  - id: 3
    name: "Processor3"
    tool: "create_chat_completion"
    port: 9003
  - id: 4
    name: "ResultMerger"
    tool: "create_chat_completion"
    port: 9004

task_id: "parallel_processing_001"
```

### 示例 3：决策树工作流

```yaml
# decision_tree_workflow.yaml
workflow:
  start_agent: 0
  execution_pattern: "conditional"
  message_flow:
    - from: 0
      to: [1]
      message_type: "classify"
    - from: 1
      to: [2]
      message_type: "process_type_a"
      condition:
        field: "classification"
        operator: "=="
        value: "type_a"
    - from: 1
      to: [3]
      message_type: "process_type_b"
      condition:
        field: "classification"
        operator: "=="
        value: "type_b"
    - from: [2, 3]
      to: 4
      message_type: "finalize"
    - from: 4
      to: "final"
      message_type: "final_result"

agents:
  - id: 0
    name: "DataInput"
    tool: "create_chat_completion"
    port: 9000
  - id: 1
    name: "Classifier"
    tool: "create_chat_completion"
    port: 9001
  - id: 2
    name: "TypeAProcessor"
    tool: "create_chat_completion"
    port: 9002
  - id: 3
    name: "TypeBProcessor"
    tool: "create_chat_completion"
    port: 9003
  - id: 4
    name: "OutputFormatter"
    tool: "create_chat_completion"
    port: 9004

task_id: "decision_tree_001"
```

## 工作流执行和监控

### 执行工作流

```python
# Python 代码示例
import asyncio
import yaml
from protocol_backends.protocol_factory import protocol_factory

async def run_workflow(config_file: str, initial_task: str):
    """执行工作流"""
    
    # 加载配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建网络和智能体
    network, agents = protocol_factory.create_multi_agent_system(
        agents_config=config['agents'],
        task_id=config['task_id'],
        protocol='dummy'  # 或其他协议
    )
    
    # 设置智能体提示词
    if 'agent_prompts' in config:
        for agent in agents:
            agent_id = str(agent.id)
            if agent_id in config['agent_prompts']:
                prompts = config['agent_prompts'][agent_id]
                if 'system_prompt' in prompts:
                    agent.system_prompt = prompts['system_prompt']
    
    try:
        # 启动网络
        await network.start()
        
        # 执行工作流
        result = await network.execute_workflow(config, initial_task)
        
        print(f"Workflow completed. Result: {result}")
        return result
        
    finally:
        # 停止网络
        await network.stop()

# 使用示例
if __name__ == "__main__":
    asyncio.run(run_workflow(
        "document_processing_workflow.yaml",
        "Process this document: Lorem ipsum dolor sit amet..."
    ))
```

### 监控和调试

```python
# 监控工作流执行状态
async def monitor_workflow(network):
    """监控工作流执行"""
    
    while network.running:
        # 获取网络状态
        metrics = network.get_network_metrics()
        print(f"Network metrics: {metrics}")
        
        # 获取工作流进度
        progress = network.network_memory.get_workflow_progress()
        print(f"Workflow progress: {progress}")
        
        # 检查智能体状态
        for agent in network.agents:
            status = agent.get_connection_status()
            print(f"Agent {agent.id} status: {status}")
        
        await asyncio.sleep(5)  # 每5秒检查一次
```

## 最佳实践

### 1. 工作流设计原则

- **单一职责**: 每个智能体专注一个特定任务
- **松耦合**: 智能体之间通过消息接口交互
- **可测试**: 每个步骤都可以独立测试
- **容错性**: 设计错误处理和恢复机制

### 2. 性能优化

- **并行化**: 识别可并行执行的步骤
- **缓存**: 避免重复计算
- **资源管理**: 合理分配 CPU 和内存
- **监控**: 实时监控性能指标

### 3. 错误处理

- **重试机制**: 对临时失败进行重试
- **降级策略**: 在部分失败时提供基本功能
- **日志记录**: 详细记录错误信息
- **告警机制**: 及时通知关键错误

### 4. 配置管理

- **环境分离**: 开发、测试、生产环境使用不同配置
- **版本控制**: 对配置文件进行版本管理
- **参数化**: 使用变量和模板提高复用性
- **验证**: 在运行前验证配置的正确性

通过遵循这些指南和最佳实践，您可以设计和实现高效、可靠的多智能体工作流。
