# 🚀 快速开始

本文档帮助你快速上手隐私保护测试框架。

## ⚡ 5分钟快速体验

### 1. 环境准备
```bash
# 确保Python 3.8+
python --version

# 安装依赖（已包含在环境中）
# pip install httpx colorama pyyaml requests
```

### 2. 配置LLM
编辑 `config_new.yaml`，设置你的LLM配置：
```yaml
core:
  type: "openai"
  name: "gpt-4o"
  temperature: 0.3
  openai_api_key: "your-api-key-here"
  openai_base_url: "your-endpoint-url"
```

### 3. 运行测试
```bash
cd runner
python run_acp.py
```

### 4. 查看结果
测试完成后，检查 `data/` 目录下的输出文件：
- 📊 `privacy_analysis_acp.json` - 隐私分析结果
- 💬 `agent_conversations_acp.json` - 对话记录
- 📄 `detailed_privacy_report_acp.txt` - 详细报告

## 📁 项目结构

```
safety_tech/
├── 🚀 runner/           # 运行器 - 从这里开始
├── 🧠 core/             # 核心逻辑
├── 🏭 protocol_backend/ # 协议实现
├── 📊 data/             # 测试数据和结果
└── 📁 legacy/           # 旧版本文件
```

## 🎯 主要使用场景

1. **协议隐私评估** - 测试不同通信协议的隐私保护能力
2. **LLM隐私分析** - 评估LLM在敏感信息处理中的隐私泄露
3. **对话安全测试** - 模拟隐私攻击场景，验证防护效果

## 📖 深入了解

- **[完整文档](README.md)** - 详细的架构和功能说明
- **[协议适配指南](PROTOCOL_ADAPTATION_GUIDE.md)** - 如何添加新协议支持
- **[配置说明](config_new.yaml)** - 完整的配置选项

## 🔧 常见问题

**Q: LLM初始化失败怎么办？**
A: 检查 `config_new.yaml` 中的API密钥和端点URL是否正确。

**Q: 如何添加新的协议？**
A: 参考 [协议适配指南](PROTOCOL_ADAPTATION_GUIDE.md) 的详细步骤。

**Q: 测试结果如何解读？**
A: 查看 `detailed_privacy_report_*.txt` 文件，包含人性化的分析报告。

---

**🎉 恭喜！你已经完成了框架的基础设置。开始探索隐私保护测试的强大功能吧！**


