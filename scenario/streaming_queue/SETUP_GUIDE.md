# Agent Network Multi-Protocol Setup Guide

## 🎯 环境要求

- **Python版本**: 3.12.11 (推荐) 或 3.12.x
- **操作系统**: Windows 10/11, macOS, Linux
- **内存**: 至少 4GB RAM
- **网络**: 需要OpenAI API访问

## 🚀 快速安装

### 1. 创建虚拟环境

```bash
# 使用conda (推荐)
conda create -n agent_network python=3.12 -y
conda activate agent_network

# 或使用venv
python3.12 -m venv agent_network
# Windows: agent_network\Scripts\activate
# macOS/Linux: source agent_network/bin/activate
```

### 2. 安装依赖

```bash
# 核心依赖 (推荐)
pip install -r requirements.txt

# 或完整依赖
pip install -r requirements_detailed.txt
```

### 3. 配置API密钥

编辑 `config.yaml`:
```yaml
core:
  openai_api_key: "your-openai-api-key-here"
```

## 🔧 支持的协议

### 1. ACP SDK 1.0.3 (Agent Communication Protocol)
```bash
python -m runner.run_acp
```

**特性:**
- ✅ 企业级原生ACP实现
- ✅ Session和Run管理
- ✅ 结构化Message处理
- ✅ 最优性能 (30.35秒/50问题)

### 2. ANP原生SDK (Agent Network Protocol)
```bash
python -m runner.run_anp
```

**特性:**
- ✅ AgentConnect SDK集成
- ✅ DID身份认证
- ✅ E2E端到端加密
- ✅ 双协议支持 (HTTP + WebSocket)
- ✅ 真实密钥管理

### 3. Agora协议
```bash
python -m runner.run_agora
```

**特性:**
- ✅ LangChain集成
- ✅ 简单HTTP通信
- ✅ 快速部署

### 4. A2A协议
```bash
python -m runner.run_a2a
```

**特性:**
- ✅ JSON-RPC通信
- ✅ 标准化接口


## 🛠️ 故障排除

### 常见问题

1. **导入错误**: 确保在streaming_queue目录运行
2. **API密钥**: 检查OpenAI API密钥配置
3. **端口冲突**: 确保端口9900-11004范围可用
4. **依赖缺失**: 重新安装requirements.txt

### 验证安装

```bash
# 测试ACP
python -c "import acp_sdk; print(f'ACP SDK: {acp_sdk.__version__}')"

# 测试ANP
python -c "import sys; sys.path.append('../../agentconnect_src'); from agent_connect.utils.did_generate import did_generate; print('ANP SDK: OK')"

# 测试Agora
python -c "import agora; print('Agora: OK')"

# 测试A2A
python -c "import a2a; print('A2A: OK')"
```

