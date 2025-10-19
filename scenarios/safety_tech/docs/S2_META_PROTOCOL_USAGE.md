# S2 Meta Protocol 使用指南

## 概述

S2 Meta Protocol 是为 Safety_Tech 设计的智能协议路由系统，专门用于 S2 保密性测试。它支持：

- **双医生架构**：两名医生代理进行医疗对话测试
- **LLM 智能路由**：基于 S2 安全特征动态选择最优协议组合
- **跨协议通信**：测试不同协议间的安全边界一致性
- **综合 S2 探针**：TLS降级、E2E加密、会话劫持、时序攻击等全方位安全测试

## 快速开始

### 1. 环境准备

```bash
# 激活虚拟环境
cd /Users/jason/Desktop/Multiagent-Protocol
source exp_env/bin/activate

# 进入 Safety_Tech 目录
cd script/safety_tech
```

### 2. 配置设置

编辑 `config_meta_s2.yaml`：

```yaml
# 基础配置
general:
  test_focus: "comprehensive"  # 测试重点：comprehensive/tls_focused/e2e_focused/session_focused
  enable_cross_protocol: true  # 启用跨协议通信测试
  enable_llm_routing: true     # 使用 LLM 智能路由
  num_conversations: 3         # 对话数量
  max_rounds: 2               # 每轮对话次数

# LLM 配置（可选，不配置将使用规则路由）
core:
  type: "openai"
  name: "meta/llama-3.3-70b-instruct"
  openai_api_key: "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"
  openai_base_url: "https://integrate.api.nvidia.com/v1"
```

### 3. 运行基础测试

```bash
# 运行基础测试验证功能
python test_s2_meta_protocol.py
```

### 4. 运行完整 S2 测试

```bash
# 运行完整的 S2 Meta Protocol 测试
python -m protocol_backends.meta_protocol.s2_meta_runner

# 或使用命令行参数
python -m protocol_backends.meta_protocol.s2_meta_runner \
  --config config_meta_s2.yaml \
  --test-focus comprehensive \
  --enable-cross-protocol
```

## 命令行选项

```bash
python -m protocol_backends.meta_protocol.s2_meta_runner --help
```

可用选项：
- `--config`: 配置文件路径 (默认: config_meta_s2.yaml)
- `--test-focus`: 测试重点 (comprehensive/tls_focused/e2e_focused/session_focused)
- `--enable-cross-protocol`: 启用跨协议通信测试
- `--enable-mitm`: 启用 MITM 测试 (需要特权)

## S2 安全评分体系

S2 Meta Protocol 基于以下权重计算综合安全评分：

| 安全维度 | 权重 | 测试内容 |
|---------|------|---------|
| TLS/传输安全 | 40% | TLS降级攻击、证书验证矩阵 |
| 会话劫持防护 | 15% | 过期/跨会话/权限提升令牌拒绝 |
| E2E加密检测 | 18% | 明文泄露检测、水印外泄分析 |
| 时钟漂移防护 | 12% | 时间偏移窗口验证 |
| 旁路抓包保护 | 8% | 侧信道信息泄露防护 |
| 重放攻击防护 | 4% | 旧消息重放拒绝 |
| 元数据泄露防护 | 3% | 敏感端点暴露检查 |

## 协议安全档案

当前支持的协议及其 S2 评分：

1. **ANP** (86.8分) - 最高安全性
   - 优势：DID认证、E2E加密、强会话保护
   - 适用：高安全要求场景

2. **ACP** (83.5分) - 企业级安全
   - 优势：企业级TLS、标准化接口
   - 适用：企业应用场景

3. **Agora** (82.0分) - 平衡性能与安全
   - 优势：原生SDK、完整TLS防护
   - 适用：生产环境部署

4. **A2A** (57.4分) - 基础安全
   - 优势：简单易用、快速部署
   - 适用：开发测试环境

## 路由策略

### LLM 智能路由 (推荐)

当配置了 NVIDIA API Key 时，系统使用 LLaMA-3.3-70B 进行智能协议选择：

```python
# 自动分析 S2 测试要求
routing_decision = await router.route_for_s2_security_test("comprehensive")
```

### 规则路由 (回退)

当 LLM 不可用时，使用预设规则：

- `comprehensive`: 选择 S2 评分最高的协议组合
- `tls_focused`: 优先 TLS 安全能力强的协议
- `e2e_focused`: 优先 E2E 加密能力强的协议
- `session_focused`: 优先会话保护能力强的协议

## 输出文件

测试完成后，在 `output/` 目录生成：

### 对话数据
```
s2_conversations_meta_{protocol_a}_{protocol_b}.json
```
包含完整的对话记录和探针注入数据

### 安全分析
```
s2_security_analysis_meta_{protocol_a}_{protocol_b}.json
```
包含 S2 安全违规检测和评分结果

### 安全报告
```
s2_security_report_meta_{protocol_a}_{protocol_b}.txt
```
人类可读的综合安全报告

### 总结报告
```
s2_meta_protocol_summary.json
s2_meta_protocol_summary_report.txt
```
包含路由决策、协议对比、测试统计等综合信息

## 高级配置

### 自定义 S2 探针

在 `config_meta_s2.yaml` 中配置具体的探针参数：

```yaml
s2_probes:
  tls_downgrade:
    enabled: true
    weak_versions: ["TLSv1.0", "TLSv1.1"]
    weak_ciphers: ["RC4", "DES", "3DES"]
    
  e2e_payload_detection:
    enabled: true
    watermarks:
      - "HIDDEN_MARKER:S2_E2E_WATERMARK_TEST_"
    tcpdump:
      interface: "lo0"
      duration: 10
```

### 网络配置

```yaml
network:
  doctor_a_port: 8200
  doctor_b_port: 8201
  connection_timeout: 30
```

## 故障排除

### 常见问题

1. **ANP DID 认证失败**
   - 确保配置了 `ANP_DID_SERVICE_URL` 和 `ANP_DID_API_KEY`
   - ANP 协议必须使用原生 DID，不支持 HTTP 回退

2. **端口冲突**
   - 检查 8200-8299 端口范围是否被占用
   - 修改 `config_meta_s2.yaml` 中的端口配置

3. **LLM 路由失败**
   - 检查 OpenAI API Key 是否正确
   - 系统会自动回退到规则路由

4. **tcpdump 权限问题** (macOS)
   - 部分探针需要网络抓包权限
   - 使用 `sudo` 运行或禁用相关探针

### 调试模式

```bash
# 启用详细日志
export LOG_LEVEL=DEBUG

# 运行单个测试
python test_s2_meta_protocol.py
```

## 扩展开发

### 添加新协议

1. 在 `s2_llm_router.py` 中添加协议安全档案
2. 创建对应的 Meta Agent 实现
3. 在 `s2_meta_coordinator.py` 中注册协议

### 自定义路由策略

```python
# 继承 S2LLMRouter 并重写路由逻辑
class CustomS2Router(S2LLMRouter):
    async def route_for_custom_scenario(self, requirements):
        # 自定义路由逻辑
        pass
```

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交测试用例
4. 发起 Pull Request

## 许可证

参见项目根目录 LICENSE 文件。
