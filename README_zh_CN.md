# 🤖 Multiagent-Protocol（简体中文）

[网站 (占位)](https://example.com) | [论文 (占位)](#) | [许可证](LICENSE)

项目中文说明。此文档为 README 的中文翻译与补充，更多原始英文内容请参见：[README.md](README.md)。

一个支持多种协议的分布式AI系统综合通信框架。该框架使AI智能体能够跨不同通信范式无缝交互，内置安全性、监控和可扩展性功能。

## 🌟 核心特性

- **🔗 多协议支持**: ANP、A2A、ACP、Agora 和自定义协议
- **🏗️ 模块化架构**: 协议无关设计，支持可插拔后端
- **🔒 安全优先**: DID身份认证、端到端加密、隐私保护
- **📊 实时监控**: 性能指标、健康检查和可观测性
- **🌐 分布式系统**: 支持复杂的多智能体工作流
- **🧪 测试框架**: 每种协议的综合测试套件
- **⚡ 高性能**: 异步/等待模式，并发执行

## 📋 目录

- [快速开始](#-快速开始)
- [支持场景](#-支持场景)
- [协议指南](#-协议指南)
- [安装](#-安装)
- [配置](#-配置)
- [开发](#-开发)
- [贡献](#-贡献)

## 🚀 快速开始

### 环境要求

```bash
# 必需环境
Python 3.12+
OpenAI API密钥 (用于基于LLM的智能体)
```

### 安装

```bash
# 克隆仓库
git clone https://github.com/MultiagentBench/Multiagent-Protocol.git
cd Multiagent-Protocol

# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export OPENAI_API_KEY='sk-your-openai-api-key-here'
```

### 运行你的第一个多智能体系统

```bash
# GAIA框架与ANP协议
python -m script.gaia.runners.run_anp

# 流式队列与A2A协议
python -m script.streaming_queue.runner.run_a2a

# 安全测试与ACP协议
python -m script.safety_tech.runners.run_unified_security_test_acp
```

## 🎯 支持场景

### 1. 🌍 GAIA (通用AI智能体) 框架
**目的**: 分布式AI智能体间的任务执行和协调

**支持协议**:
- **ANP (智能体网络协议)**: 基于DID的身份认证与端到端加密
- **A2A (智能体对智能体)**: 直接点对点通信与消息路由
- **ACP (智能体通信协议)**: 基于会话的对话管理
- **Agora**: 基于工具的智能体编排与LangChain集成
- **Dummy**: 用于测试和开发的模拟协议

**使用方法**:
```bash
# 设置API密钥
export OPENAI_API_KEY='sk-your-key-here'

# 使用不同协议运行
python -m script.gaia.runners.run_anp        # ANP协议
python -m script.gaia.runners.run_a2a        # A2A协议
python -m script.gaia.runners.run_acp        # ACP协议
python -m script.gaia.runners.run_agora      # Agora协议
python -m script.gaia.runners.run_dummy      # Dummy协议

# 元协议协调
python -m script.gaia.runners.run_meta_protocol
```

### 2. 📡 流式队列 (Streaming Queue)
**目的**: 高吞吐量消息处理，采用协调器-工作器模式

**支持协议**: ANP、A2A、ACP、Agora、元协议

**使用方法**:
```bash
export OPENAI_API_KEY='sk-your-key-here'

# 使用不同协议的流处理
python -m script.streaming_queue.runner.run_anp     # ANP流式处理
python -m script.streaming_queue.runner.run_a2a     # A2A流式处理
python -m script.streaming_queue.runner.run_acp     # ACP流式处理
python -m script.streaming_queue.runner.run_agora   # Agora流式处理

# 元网络协调
python -m script.streaming_queue.runner.run_meta_network
```

### 3. 🛡️ 安全技术 (Safety Tech)
**目的**: 隐私保护的智能体通信和安全测试

**支持协议**: ANP、A2A、ACP、Agora、S2-Meta

**使用方法**:
```bash
export OPENAI_API_KEY='sk-your-key-here'

# 统一安全测试
python -m script.safety_tech.runners.run_unified_security_test_anp
python -m script.safety_tech.runners.run_unified_security_test_a2a
python -m script.safety_tech.runners.run_unified_security_test_acp
python -m script.safety_tech.runners.run_unified_security_test_agora

# S2元协议安全分析
python -m script.safety_tech.runners.run_s2_meta
```

### 4. 🔄 故障风暴恢复 (Fail Storm Recovery)
**目的**: 具有自动恢复机制的容错系统

**支持协议**: ANP、A2A、ACP、Agora、Simple JSON、元协议

**使用方法**:
```bash
export OPENAI_API_KEY='sk-your-key-here'

# 容错测试
python -m script.fail_storm_recovery.runners.run_anp
python -m script.fail_storm_recovery.runners.run_a2a
python -m script.fail_storm_recovery.runners.run_acp
python -m script.fail_storm_recovery.runners.run_agora
python -m script.fail_storm_recovery.runners.run_simple_json

# 元协议协调
python -m script.fail_storm_recovery.runners.run_meta
python -m script.fail_storm_recovery.runners.run_meta_network
```

### 5. 🗺️ 异步MAPF (多智能体路径规划)
**目的**: 分布式路径规划和协调

**支持协议**: A2A

**使用方法**:
```bash
export OPENAI_API_KEY='sk-your-key-here'

# 多智能体路径规划
python -m script.async_mapf.runners.run_a2a
```

## 🔧 协议指南

### ANP (智能体网络协议)
- **特性**: DID身份认证、端到端加密、WebSocket通信
- **使用场景**: 安全智能体网络、身份验证通信
- **依赖**: `agentconnect_src/` (AgentConnect SDK)

### A2A (智能体对智能体协议)
- **特性**: 直接点对点通信、JSON-RPC消息、事件流
- **使用场景**: 高性能智能体协调、实时消息传递
- **依赖**: `a2a-sdk`, `a2a-server`

### ACP (智能体通信协议)
- **特性**: 会话管理、对话线程、消息历史
- **使用场景**: 对话智能体、多轮交互
- **依赖**: `acp-sdk`

### Agora协议
- **特性**: 工具编排、LangChain集成、函数调用
- **使用场景**: 工具增强智能体、LLM驱动工作流
- **依赖**: `agora-protocol`, `langchain`

### 元协议
- **特性**: 协议抽象、自适应路由、多协议支持
- **使用场景**: 协议无关应用、无缝迁移

## ⚙️ 配置

### 环境变量

```bash
# 必需
export OPENAI_API_KEY='sk-your-openai-api-key'

# 可选
export ANTHROPIC_API_KEY='your-anthropic-key'
export OPENAI_BASE_URL='https://api.openai.com/v1'  # 自定义端点
export LOG_LEVEL='INFO'                              # DEBUG, INFO, WARNING, ERROR
```

### 配置文件

每个场景使用位于 `script/{scenario}/config/` 的YAML配置文件：

```yaml
# 示例: script/gaia/config/anp.yaml
model:
  type: "openai"
  name: "gpt-4o"
  temperature: 0.0
  api_key: "${OPENAI_API_KEY}"

network:
  host: "127.0.0.1"
  port_range:
    start: 9000
    end: 9010

agents:
  - id: 1
    name: "Agent1"
    tool: "create_chat_completion"
    max_tokens: 500

workflow:
  type: "sequential"
  max_steps: 5
```

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────────┐
│                   多协议智能体框架                            │
├─────────────────────────────────────────────────────────────┤
│  应用场景                                                    │
│  ┌─────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │  GAIA   │ │  流式队列   │ │  安全技术   │ │ 故障恢复    │ │
│  │  框架   │ │    处理     │ │    测试     │ │    系统     │ │
│  └─────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  协议后端                                                    │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────────────┐             │
│  │ ANP │ │ A2A │ │ ACP │ │Agora│ │   元协议    │             │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────────────┘             │
├─────────────────────────────────────────────────────────────┤
│  核心基础设施                                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │   网络层    │ │   智能体层  │ │   监控层    │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## 🧪 开发

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定场景的测试
pytest script/gaia/tests/
pytest script/safety_tech/tests/

# 运行覆盖率测试
pytest --cov=src --cov-report=html
```

### 添加新协议

1. 在 `script/{scenario}/protocol_backends/{protocol_name}/` 创建协议后端
2. 实现必需接口: `agent.py`, `network.py`, `comm.py`
3. 在 `script/{scenario}/config/{protocol_name}.yaml` 添加配置
4. 在 `script/{scenario}/runners/run_{protocol_name}.py` 创建运行器

### 代码结构

```
script/
├── {scenario}/                    # 场景实现
│   ├── config/                   # 配置文件
│   ├── protocol_backends/        # 协议实现
│   │   ├── {protocol}/
│   │   │   ├── agent.py         # 智能体实现
│   │   │   ├── network.py       # 网络协调器
│   │   │   └── comm.py          # 通信后端
│   ├── runners/                  # 入口脚本
│   └── tools/                    # 场景特定工具
├── common/                       # 共享工具
└── requirements.txt              # 依赖项
```

## 📊 监控与可观测性

框架包含全面的监控功能：

- **性能指标**: 消息吞吐量、延迟、成功率
- **健康监控**: 智能体状态、网络连接、资源使用
- **安全审计**: 身份认证事件、加密状态、隐私合规
- **自定义仪表板**: 协议特定可视化和告警

## 🤝 贡献

我们欢迎贡献！详情请参阅我们的 [贡献指南](CONTRIBUTING.md)。

### 快速贡献步骤

1. Fork 仓库
2. 创建功能分支: `git checkout -b feature/amazing-feature`
3. 进行更改并添加测试
4. 确保所有测试通过: `pytest`
5. 提交更改: `git commit -m 'Add amazing feature'`
6. 推送到分支: `git push origin feature/amazing-feature`
7. 开启 Pull Request

## 📄 许可证

该项目基于 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 🆘 支持

- **文档**: [Wiki](https://github.com/your-org/Multiagent-Protocol/wiki)
- **问题**: [GitHub Issues](https://github.com/your-org/Multiagent-Protocol/issues)
- **讨论**: [GitHub Discussions](https://github.com/your-org/Multiagent-Protocol/discussions)

## 🚀 入门示例

### 示例1: 简单GAIA任务执行

```bash
# 1. 设置环境
export OPENAI_API_KEY='sk-your-key'

# 2. 使用ANP协议运行GAIA
python -m script.gaia.runners.run_anp

# 3. 监控智能体交互输出
# 预期: 多个智能体通过DID身份认证协作完成任务
```

### 示例2: 高吞吐量流式处理

```bash
# 1. 设置环境
export OPENAI_API_KEY='sk-your-key'

# 2. 使用A2A运行流式队列
python -m script.streaming_queue.runner.run_a2a

# 3. 观察协调器-工作器消息处理
# 预期: 高频消息交换与负载均衡
```

### 示例3: 安全测试

```bash
# 1. 设置环境
export OPENAI_API_KEY='sk-your-key'

# 2. 运行隐私感知安全测试
python -m script.safety_tech.runners.run_unified_security_test_anp

# 3. 查看隐私保护机制
# 预期: 加密通信与隐私合规报告
```

---

**由多智能体系统社区倾情打造 ❤️**