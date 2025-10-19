# S2保密性探针使用指南

## 概述

S2保密性探针是Safety Tech项目中用于测试协议保密性的黑盒测试工具。通过在各协议client中注入不同类型的安全探针，可以评估协议对各种安全威胁的抵抗能力。

## 探针类型

### 1. TLS降级探针 (`tls_downgrade`)
- **目的**: 测试协议是否能抵抗TLS版本降级攻击
- **实现**: 配置弱SSL上下文，尝试使用TLS 1.0/1.1或弱加密套件
- **指标**: `tls_version_used`, `cipher_downgrade`

### 2. 重放攻击探针 (`replay_attack`)
- **目的**: 测试协议是否能检测和阻止重放攻击
- **实现**: 在消息中注入过期的nonce和时间戳
- **指标**: `replay_attack`, 重放标识符

### 3. 明文嗅探探针 (`plaintext_sniff`)
- **目的**: 检测协议通信中的明文数据泄露
- **实现**: 分析消息内容，检测敏感关键字
- **指标**: `plaintext_detected`, `sensitive_keywords_detected`, `sensitive_data_score`

### 4. DID伪造探针 (`did_spoofing`)
- **目的**: 测试协议对身份伪造的防护能力
- **实现**: 注入伪造的DID和签名
- **指标**: `did_spoofing`, `fake_did_used`

### 5. 会话劫持探针 (`session_hijack`)
- **目的**: 测试协议对会话劫持的防护能力
- **实现**: 重用或伪造会话token
- **指标**: `session_hijack`, `hijacked_token`

### 6. MITM代理探针 (`mitm_proxy`)
- **目的**: 测试协议在中间人攻击下的表现
- **实现**: 通过代理服务器路由请求
- **指标**: `mitm_proxy`, `proxy_used`

### 7. 网络扰动探针 (网络层)
- **目的**: 测试协议对网络异常的韧性
- **实现**: 注入网络抖动、丢包等
- **指标**: `network_jitter`, `packet_dropped`

## 使用方法

### 环境变量配置

```bash
# 启用数据面直连发送（必需）
export A2A_USE_DIRECT_SEND=true
export ACP_USE_DIRECT_SEND=true  
export ANP_USE_DIRECT_SEND=true
export AGORA_USE_DIRECT_SEND=true

# 启用S2探针
export A2A_ENABLE_S2_PROBES=true
export ACP_ENABLE_S2_PROBES=true
export ANP_ENABLE_S2_PROBES=true  
export AGORA_ENABLE_S2_PROBES=true

# 选择探针类型
export A2A_S2_PROBE_TYPE=tls_downgrade    # 或 replay_attack, mitm, did_spoofing, comprehensive
export ACP_S2_PROBE_TYPE=comprehensive
export ANP_S2_PROBE_TYPE=replay_attack
export AGORA_S2_PROBE_TYPE=did_spoofing
```

### 运行测试

```bash
# 运行A2A协议的S2探针测试
cd /Users/jason/Desktop/Multiagent-Protocol
source exp_env/bin/activate
python script/safety_tech/runners/run_unified_security_test_a2a.py

# 运行其他协议测试
python script/safety_tech/runners/run_unified_security_test_acp.py
python script/safety_tech/runners/run_unified_security_test_anp.py  
python script/safety_tech/runners/run_unified_security_test_agora.py
```

### 测试探针功能

```bash
# 运行S2探针测试脚本
python script/safety_tech/test_s2_probes.py
```

## 报告解读

### S1业务连续性报告
```json
{
  "S1_business_continuity": {
    "completion_rate": 0.85,
    "score": 85.0,
    "latency_stats": {
      "avg_ms": 1250.5,
      "p95_ms": 2100.0,
      "p99_ms": 3500.0
    },
    "timeout_rate": 0.02,
    "method": "direct_send"
  }
}
```

### S2保密性报告
```json
{
  "S2_eavesdrop_prevention": {
    "malicious_observers_blocked": true,
    "score": 95.0,
    "probe_results": {
      "total_probes": 25,
      "tls_downgrade_attempts": 5,
      "replay_attempts": 8,
      "plaintext_bytes_detected": 0,
      "probe_type": "comprehensive"
    }
  }
}
```

### 探针结果字段说明

- `total_probes`: 总探针执行次数
- `tls_downgrade_attempts`: TLS降级尝试次数
- `replay_attempts`: 重放攻击尝试次数
- `plaintext_bytes_detected`: 检测到的明文字节数（越少越好）
- `sensitive_keywords_detected`: 检测到的敏感关键字列表
- `sensitive_data_score`: 敏感数据评分（越低越好）

## 协议特定实现

### A2A协议
- 在消息元数据中注入探针标识
- 支持HTTP请求头注入
- SSL上下文配置

### ACP协议  
- 在ACP标准输入格式中注入探针数据
- X-ACP-Agent-DID头部注入
- 代理配置支持

### ANP协议
- WebSocket握手头部注入
- 消息内容标记
- 网络抖动模拟

### Agora协议
- 原生SDK消息注入
- 数据包丢失模拟
- 延迟注入测试

## 最佳实践

1. **渐进式测试**: 从单一探针开始，逐步增加复杂度
2. **基准对比**: 先运行无探针测试建立基准，再启用探针对比
3. **环境隔离**: S2探针测试应在隔离环境中进行
4. **日志监控**: 关注探针结果和异常日志
5. **定期验证**: 定期运行探针测试确保功能正常

## 故障排除

### 常见问题

1. **探针无结果**: 
   - 检查是否启用了数据面直连发送 (`*_USE_DIRECT_SEND=true`)
   - 确认协议服务端点可访问

2. **TLS探针失败**:
   - 检查SSL库版本兼容性
   - 确认目标服务支持TLS协商

3. **代理探针无效**:
   - 检查代理服务器配置
   - 确认网络连通性

4. **性能影响**:
   - 探针会增加延迟和资源消耗
   - 生产环境使用需谨慎

### 调试命令

```bash
# 检查探针配置
python -c "
from script.safety_tech.core.probe_config import create_comprehensive_probe_config
print(create_comprehensive_probe_config().to_dict())
"

# 测试单个协议
python -c "
import asyncio
from script.safety_tech.core.backend_api import send_backend
from script.safety_tech.core.probe_config import create_s2_tls_downgrade_config

async def test():
    result = await send_backend('a2a', 'http://127.0.0.1:8001', 
                               {'text': 'test'}, 'test_cid', 
                               create_s2_tls_downgrade_config().to_dict())
    print(result)

asyncio.run(test())
"
```

## 扩展开发

### 添加新探针类型

1. 在 `probe_config.py` 中定义新的探针配置
2. 在各协议client的 `send` 方法中添加处理逻辑
3. 更新runner中的探针统计逻辑
4. 添加相应的测试用例

### 自定义探针配置

```python
from script.safety_tech.core.probe_config import ProbeConfig

custom_probe = ProbeConfig(
    tls_downgrade=True,
    tls_version_downgrade="1.0",
    plaintext_sniff=True,
    sniff_keywords=["custom", "sensitive", "data"],
    network_jitter_ms=100
)
```

## 安全注意事项

1. **仅用于测试**: S2探针仅应用于测试环境
2. **数据保护**: 避免在探针中使用真实敏感数据
3. **网络隔离**: 探针测试应在隔离网络中进行
4. **权限控制**: 限制探针功能的访问权限
5. **审计日志**: 记录所有探针活动用于审计

---

更多技术细节请参考源代码中的实现和注释。
