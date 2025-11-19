# Multi-Vendor LLM Support for GAIA Runners

本指南说明如何使用更新后的 GAIA runners 支持多种 LLM API（OpenAI、Anthropic Claude、Google Gemini）。

## 目录

- [快速开始](#快速开始)
- [支持的 API](#支持的-api)
- [配置方式](#配置方式)
- [使用示例](#使用示例)
- [环境变量](#环境变量)
- [文件结构](#文件结构)

## 快速开始

所有 GAIA runners（`run_acp.py`, `run_a2a.py`, `run_agora.py`, `run_anp.py`）现在都支持：

1. **命令行参数配置** - 通过 `--config` 指定配置文件
2. **多供应商 LLM API** - 支持 OpenAI、Anthropic、Google
3. **命令行覆盖** - 可通过命令行参数覆盖配置文件设置

## 支持的 API

### 1. OpenAI (默认)
- 模型: GPT-4o, GPT-4-turbo, GPT-3.5-turbo, 等
- API 类型: `openai`
- 环境变量: `OPENAI_API_KEY`

### 2. Anthropic Claude
- 模型: claude-3-5-sonnet-20241022, claude-3-opus, 等
- API 类型: `anthropic`
- 环境变量: `ANTHROPIC_API_KEY`

### 3. Google Gemini
- 模型: gemini-2.0-flash-exp, gemini-pro, 等
- API 类型: `google`
- 环境变量: `GOOGLE_API_KEY`

## 配置方式

### 方式 1: 配置文件

在 YAML 配置文件中设置（例如 `config/acp.yaml`）:

```yaml
model:
  api_type: "openai"  # 或 "anthropic" 或 "google"
  name: "gpt-4o"
  api_key: ""  # 留空以使用环境变量
  base_url: "https://api.openai.com/v1"
  max_tokens: 4096
  temperature: 0.0
  timeout: 30
```

### 方式 2: 命令行参数（推荐）

```bash
# 使用 GPT-4o (OpenAI)
python -m scenarios.gaia.runners.run_acp \
    --config config/acp.yaml \
    --api-type openai \
    --model gpt-4o

# 使用 Claude 3.5 Sonnet (Anthropic)
python -m scenarios.gaia.runners.run_acp \
    --config config/acp.yaml \
    --api-type anthropic \
    --model claude-3-5-sonnet-20241022

# 使用 Gemini 2.0 Flash (Google)
python -m scenarios.gaia.runners.run_acp \
    --config config/acp.yaml \
    --api-type google \
    --model gemini-2.0-flash-exp
```

### 方式 3: 混合使用

配置文件设置基本参数，命令行覆盖特定选项：

```bash
# 配置文件中设置了 OpenAI，但用命令行切换到 Claude
python -m scenarios.gaia.runners.run_acp \
    --config config/acp_openai.yaml \
    --api-type anthropic \
    --model claude-3-5-sonnet-20241022 \
    --api-key "your-anthropic-key-here"
```

## 使用示例

### 运行 ACP 协议

```bash
# OpenAI GPT-4o
export OPENAI_API_KEY="your-openai-key"
python -m scenarios.gaia.runners.run_acp \
    --config config/acp.yaml \
    --api-type openai \
    --model gpt-4o

# Anthropic Claude 3.5
export ANTHROPIC_API_KEY="your-anthropic-key"
python -m scenarios.gaia.runners.run_acp \
    --config config/acp.yaml \
    --api-type anthropic \
    --model claude-3-5-sonnet-20241022

# Google Gemini 2.0
export GOOGLE_API_KEY="your-google-key"
python -m scenarios.gaia.runners.run_acp \
    --config config/acp.yaml \
    --api-type google \
    --model gemini-2.0-flash-exp
```

### 运行 A2A 协议

```bash
# 使用 Claude
python -m scenarios.gaia.runners.run_a2a \
    --config config/a2a.yaml \
    --api-type anthropic \
    --model claude-3-5-sonnet-20241022
```

### 运行 Agora 协议

```bash
# 使用 Gemini
python -m scenarios.gaia.runners.run_agora \
    --config config/agora.yaml \
    --api-type google \
    --model gemini-2.0-flash-exp
```

### 运行 ANP 协议

```bash
# 使用 GPT-4o
python -m scenarios.gaia.runners.run_anp \
    --config config/anp.yaml \
    --api-type openai \
    --model gpt-4o
```

## 环境变量

设置相应的 API 密钥环境变量：

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google
export GOOGLE_API_KEY="AI..."

# 可选: OpenAI 兼容的自定义端点
export OPENAI_BASE_URL="https://your-custom-endpoint.com/v1"
```

## 命令行参数说明

所有 runners 支持以下参数：

- `--config PATH`: 配置文件路径（默认: `{protocol}.yaml`）
- `--api-type {openai,anthropic,google}`: LLM API 类型
- `--model NAME`: 模型名称
- `--api-key KEY`: API 密钥（优先级最高）

## 文件结构

修改后的文件：

```
scenarios/gaia/
├── core/
│   └── llm.py                    # ✨ 支持多供应商 API
├── runners/
│   ├── run_acp.py               # ✨ 支持命令行参数
│   ├── run_a2a.py               # ✨ 支持命令行参数
│   ├── run_agora.py             # ✨ 支持命令行参数
│   ├── run_anp.py               # ✨ 支持命令行参数
│   └── runner_base.py           # 基础 runner 类
├── config/
│   ├── acp.yaml
│   ├── a2a.yaml
│   ├── agora.yaml
│   └── anp.yaml
└── MULTI_VENDOR_LLM_GUIDE.md    # 本文档
```

## 实验对比工作流

结合 `model_comparison` 目录进行多模型对比实验：

```bash
# 1. 采样数据
cd /root/AgentProtocols/scenarios/gaia/model_comparison
python sample_data.py

# 2. 使用不同模型运行相同协议
# GPT-4o + ACP
python -m scenarios.gaia.runners.run_acp \
    --config model_comparison/configs/config_gpt4o.yaml \
    --api-type openai \
    --model gpt-4o

# Claude 3.5 + ACP
python -m scenarios.gaia.runners.run_acp \
    --config model_comparison/configs/config_claude.yaml \
    --api-type anthropic \
    --model claude-3-5-sonnet-20241022

# Gemini 2.0 + ACP
python -m scenarios.gaia.runners.run_acp \
    --config model_comparison/configs/config_gemini.yaml \
    --api-type google \
    --model gemini-2.0-flash-exp

# 3. 分析结果
python model_comparison/analyze_results.py
```

## 注意事项

### API 兼容性

- ✅ **Chat Completion**: 所有三种 API 都支持
- ⚠️ **Tool Calling**: 目前仅 OpenAI 完全支持
  - Anthropic 和 Google 有各自的工具调用机制，需要额外实现
- ✅ **Token Counting**: 自动转换和统计

### 费用控制

不同 API 的定价差异很大，建议：

1. 先用少量数据测试（设置 `runtime.max_tasks` 在配置中）
2. 监控 token 使用情况（在输出中会显示）
3. 设置合理的超时时间（`runtime.timeout`）

### 错误处理

如果遇到 API 错误：

1. 检查 API 密钥是否正确
2. 确认模型名称拼写正确
3. 查看 API 配额和限制
4. 检查网络连接

## 故障排除

### "API key not configured" 错误

```bash
# 确保设置了正确的环境变量
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY
```

### 模型名称错误

确认使用正确的模型名称：
- OpenAI: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
- Anthropic: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`
- Google: `gemini-2.0-flash-exp`, `gemini-pro`

### 导入错误

确保从项目根目录运行：

```bash
cd /root/AgentProtocols
python -m scenarios.gaia.runners.run_acp --help
```

## 总结

更新后的 GAIA runners 提供了灵活的 LLM 配置方式，支持：

- ✅ 三大主流 LLM API（OpenAI、Anthropic、Google）
- ✅ 配置文件 + 命令行参数的灵活组合
- ✅ 统一的接口和输出格式
- ✅ 详细的 token 使用统计
- ✅ 易于扩展的架构

开始使用多供应商 LLM 进行 GAIA 基准测试吧！
