# ProtoRouter Benchmark

ProtoRouter benchmark系统，用于评估LLM在多智能体系统协议选择任务上的准确性。

## 功能特性

- 🤖 基于LLM的协议选择系统（支持OpenAI API和本地模型）
- 📊 全面的评估指标统计
- 📈 按难度级别的准确率分析
- 🔄 协议混淆矩阵分析
- 📋 详细的评估报告生成

## 文件结构

```
script/routerbench/
├── data/
│   └── data.json              # 测试数据集
├── config.yaml                # 模型配置文件
├── prompt_template.py         # LLM提示词模板
├── proto_router.py           # 协议选择器（调用LLM）
├── evaluator.py              # 评估系统
├── run_benchmark.py          # 主运行脚本
└── README.md                 # 说明文档
```

## 快速开始

### 1. 配置模型

编辑 `config.yaml` 文件：

```yaml
model:
  type: "openai"  # "openai" 或 "local"
  name: "gpt-4o"
  temperature: 0.1
  openai_api_key: "your_api_key_here"
  openai_base_url: "https://api.openai.com/v1"
```

### 2. 运行benchmark

```bash
# 运行完整benchmark
python run_benchmark.py

# 运行前10个场景（测试用）
python run_benchmark.py --limit 10

# 指定配置文件和输出目录
python run_benchmark.py --config my_config.yaml --output my_results
```

### 3. 查看结果

运行完成后会生成：
- `results/benchmark_results.json` - 详细的JSON结果
- `results/benchmark_report.txt` - 可读的评估报告

## 评估指标

### 1. 总体正确率
- **场景正确率**: 对于L2及以上难度，要求所有模块都选对才算场景正确
- **模块正确率**: 单个模块的选择正确率

### 2. 按难度统计
- L1-L5各个难度级别的正确率统计

### 3. 协议混淆分析
- A2A/ACP混淆次数统计（特别关注的混淆情况）
- 完整的协议混淆矩阵

### 4. 错误分析
- 各种错误类型的频次统计

## 数据集格式

数据集包含不同难度的多智能体场景：

```json
{
  "id": "L1-Q1",
  "description": "场景描述",
  "module": [
    {
      "name": "模块名称",
      "agents": ["Agent1", "Agent2"],
      "tasks": ["任务描述..."],
      "potential_issues": ["潜在问题..."],
      "protocol_selection": {
        "choices": ["A2A", "ACP", "Agora", "ANP"],
        "select_exactly": 1
      }
    }
  ],
  "ground_truth": {
    "1": {
      "module_protocol": "ANP",
      "justification": "选择理由..."
    }
  }
}
```

## 协议特性

系统基于以下协议特性进行选择：

### A2A (Agent-to-Agent Protocol)
- 企业级集成、复杂工作流、多模态支持
- 适用于：复杂业务场景、UI交互、长期任务

### ACP (Agent Communication Protocol)  
- REST风格、部署灵活、简单易用
- 适用于：标准Web集成、现有服务包装

### Agora (Meta-Protocol)
- 轻量级、协议协商、快速演进
- 适用于：研究实验、去中心化场景

### ANP (Agent Network Protocol)
- 强身份验证、端到端加密、跨组织信任
- 适用于：高安全需求、跨组织协作

## 输出示例

```
📊 总体统计:
  总场景数: 19
  场景正确数: 15
  总体场景正确率: 78.95%
  总模块数: 25
  模块正确数: 23
  单个模块正确率: 92.00%
  A2A/ACP混淆次数: 2

📈 按难度统计:
  L1:
    场景正确率: 90.00% (9/10)
    模块正确率: 95.00% (19/20)
  L2:
    场景正确率: 66.67% (6/9)
    模块正确率: 88.89% (4/5)
```

## 扩展功能

- 支持自定义协议特性定义
- 支持批量测试不同模型
- 支持结果对比分析
- 支持自定义评估指标

