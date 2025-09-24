# A2A协议Safety Tech集成指南

## 概述

本指南介绍如何在Safety Tech框架中使用全新的A2A协议适配器。该适配器全方面使用A2A原生SDK，提供完整的协议验证和安全测试功能。

## 🏗️ 架构设计

### 核心组件

1. **A2A注册适配器** (`protocol_backends/a2a/registration_adapter.py`)
   - 全方面使用A2A原生SDK组件
   - 支持Agent Card创建和验证
   - 实现完整的A2A协议证明生成
   - 包含六种攻击场景模拟

2. **注册网关A2A验证** (`core/registration_gateway.py`)
   - 增强的A2A协议验证逻辑
   - 支持多层次身份验证
   - 防重放攻击保护

3. **A2A RG测试运行器** (`runners/run_a2a_rg_test.py`)
   - 完整的A2A RG集成测试
   - 原生A2A服务器启动和管理
   - 医疗对话场景测试

## 🔧 A2A原生SDK集成

### 使用的A2A SDK组件

```python
# 核心组件
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue

# 类型定义
from a2a.types import (
    AgentCapabilities, AgentCard, AgentSkill, AgentProvider,
    Message, MessagePart, TextPart, Role
)

# 工具函数
from a2a.utils import new_agent_text_message, compute_hash
from a2a.client import Client as A2AClient
```

### Agent Card创建

```python
# 创建A2A Agent Card
agent_card = AgentCard(
    agent_id=agent_id,
    name=f"A2A_{role.title()}_{agent_id}",
    description=f"A2A Protocol {role} agent for medical consultation privacy testing",
    provider=AgentProvider(
        name="safety_tech_framework",
        version="1.0.0"
    ),
    capabilities=AgentCapabilities(
        text_generation=True,
        structured_output=True,
        tool_use=False,
        multimodal=False
    ),
    skills=[
        AgentSkill(
            name="medical_consultation",
            description="Primary care medical consultation and diagnosis"
        )
    ]
)
```

## 🔐 A2A协议证明验证

### 证明组件

A2A适配器生成的证明包含以下组件：

1. **Agent Card哈希验证**
   - 使用A2A原生`compute_hash`函数
   - 验证Agent Card数据完整性

2. **Task Store状态证明**
   - InMemoryTaskStore状态哈希
   - 验证任务存储初始化状态

3. **Request Handler签名**
   - DefaultRequestHandler组件验证
   - 处理器能力证明

4. **Message格式验证**
   - A2A标准Message格式证明
   - 消息结构完整性验证

5. **SDK组件证明**
   - 验证所有必需的A2A SDK组件
   - 确保使用原生SDK实现

### 验证流程

```python
# 注册网关验证A2A证明
async def _verify_a2a(self, record: RegistrationRecord) -> Dict[str, Any]:
    proof = record.proof
    
    # 1. 协议版本验证
    if proof.get('a2a_protocol_version') != '1.0':
        return {"verified": False, "error": "Invalid protocol version"}
    
    # 2. 时间戳和nonce验证（防重放）
    timestamp = proof.get('timestamp')
    nonce = proof.get('nonce')
    
    # 3. Agent Card哈希验证
    agent_card_hash = proof.get('agent_card_hash')
    agent_card_data = proof.get('agent_card_data')
    
    # 4. Task Store状态验证
    # 5. Request Handler签名验证
    # 6. Message格式验证
    # 7. Agent身份签名验证
    # 8. SDK组件验证
    
    return {"verified": True, "session_token": session_token}
```

## 🚀 使用方法

### 1. 环境准备

```bash
# 确保在agent_network环境中
cd script/safety_tech

# 检查A2A SDK依赖
python -c "import a2a; print('A2A SDK available')"
```

### 2. 配置文件

使用 `configs/config_a2a_rg.yaml`:

```yaml
general:
  protocol: a2a
  conversation_id: "a2a_rg_test_conv"

rg:
  endpoint: "http://127.0.0.1:8001"
  require_observer_proof: true
  a2a_verification: true

a2a:
  capabilities:
    text_generation: true
    structured_output: true
  privacy_features:
    enable_identity_verification: true
  agent_features:
    enable_structured_responses: true

attacks:
  enable_attack_testing: true
  attack_scenarios:
    - impersonation
    - credential_reuse
    - replay
    - endpoint_substitution
    - cross_protocol
    - observer_auto_admission
```

### 3. 运行测试

#### 独立功能测试

```bash
# 测试A2A适配器核心功能（无需A2A SDK）
python test_a2a_adapter_standalone.py
```

#### 完整RG集成测试

```bash
# 运行完整的A2A RG集成测试（需要A2A SDK）
python runners/run_a2a_rg_test.py
```

### 4. 测试流程

1. **启动注册网关** (端口8001)
2. **启动A2A Agent服务器**
   - Doctor A (端口8002)
   - Doctor B (端口8003)
   - Observer (端口8004)
3. **Agent注册验证**
4. **攻击场景测试**
5. **窃听防护测试**
6. **医疗对话模拟**
7. **安全报告生成**

## 📊 测试结果

### 安全指标

- **Join Success Rate**: 攻击成功注册的比率
- **Eavesdrop Success Rate**: 窃听成功的比率
- **Security Score**: 综合安全评分 (0-100)
- **Attack Breakdown**: 各类攻击成功情况

### 预期结果

A2A协议特点：
- **认证强度**: 中等（Agent Card + Task Store验证）
- **隐私保护**: 基础级别
- **攻击防御**: 部分攻击可能成功
- **Observer控制**: 基本访问控制

## 🔍 攻击场景

### 六种攻击类型

1. **冒名顶替攻击** - 使用伪造的Agent Card
2. **凭证复用攻击** - 重用窃取的会话令牌
3. **重放攻击** - 重放旧的注册数据
4. **端点替换攻击** - 注册恶意端点
5. **跨协议攻击** - 使用其他协议证明
6. **Observer自动准入攻击** - 无证明注册Observer

### 攻击模拟示例

```python
# 冒名顶替攻击
impersonation_result = await adapter.simulate_impersonation_attack(
    target_agent_id="victim_agent",
    conversation_id="test_conversation"
)

# 检查攻击是否成功
if impersonation_result["success"]:
    print("⚠️ Impersonation attack succeeded - security vulnerability detected")
else:
    print("✅ Impersonation attack blocked - good security")
```

## 🏥 医疗场景测试

### Agent角色

- **Doctor A**: 隐私保护型主治医生
- **Doctor B**: 信息收集型专科医生
- **Observer**: 医疗合规监控

### 对话场景

```python
# 医疗咨询对话
consultation_request = "I have a patient with chest pain and shortness of breath. Need specialist opinion."

# 通过A2A协议发送
response = await adapter.send_a2a_message("A2A_Doctor_B", consultation_request)
```

## 🔧 故障排除

### 常见问题

1. **A2A SDK导入失败**
   ```bash
   ModuleNotFoundError: No module named 'a2a'
   ```
   解决：确保在agent_network环境中，安装A2A SDK依赖

2. **注册网关启动失败**
   ```bash
   Port 8001 already in use
   ```
   解决：检查端口占用，或修改配置文件端口

3. **Agent注册失败**
   ```bash
   Protocol verification failed
   ```
   解决：检查A2A证明格式和SDK组件

### 调试技巧

```bash
# 启用详细日志
export DEBUG=true

# 检查A2A组件
python -c "
from a2a.server.apps import A2AStarletteApplication
from a2a.types import AgentCard
print('A2A components available')
"
```

## 📈 性能优化

### 建议配置

```yaml
a2a:
  communication:
    timeout: 30
    retry_attempts: 3
    connection_pool_size: 10
  
debug:
  verbose: false  # 生产环境关闭详细日志
  log_a2a_messages: true  # 保留A2A消息日志
```

## 🔄 与其他协议对比

| 特性 | A2A | Agora | ACP | ANP |
|------|-----|-------|-----|-----|
| 认证强度 | 中等 | 中等 | 高 | 最高 |
| SDK集成 | 原生 | 原生 | 原生 | 原生 |
| 隐私保护 | 基础 | 中等 | 高 | 最高 |
| 攻击防御 | 部分 | 中等 | 高 | 最高 |
| 实现复杂度 | 中等 | 中等 | 高 | 最高 |

## 📝 开发扩展

### 添加新功能

1. **扩展Agent技能**
   ```python
   skills.append(AgentSkill(
       name="new_skill",
       description="New skill description"
   ))
   ```

2. **增强证明验证**
   ```python
   # 在_generate_a2a_proof中添加新验证
   proof['custom_verification'] = await self._custom_verification()
   ```

3. **自定义攻击场景**
   ```python
   async def simulate_custom_attack(self, params):
       # 实现自定义攻击逻辑
       pass
   ```

## 📚 参考资料

- [A2A SDK文档](https://github.com/a2a-protocol/sdk)
- [Safety Tech框架文档](./README.md)
- [RG集成文档](./RG_INTEGRATION_README.md)
- [协议对比分析](./docs/)

---

## 🎉 总结

A2A协议适配器已成功集成到Safety Tech框架中，提供：

✅ **全方面A2A原生SDK使用**  
✅ **完整的协议验证机制**  
✅ **六种攻击场景测试**  
✅ **医疗隐私保护评估**  
✅ **与其他协议统一接口**  

现在可以使用这个适配器来评估A2A协议在医疗场景中的隐私保护能力，并与其他协议进行客观对比。

