# 协议启动方式记录指南

本文档记录了各协议（ACP、A2A、ANP、Agora）在Safety Tech安全测试中的成功启动方式、配置参数和已知问题。

## 测试结果概览

### ✅ 成功完成的协议测试 (最新结果)

| 协议 | S1测试 | S2探针测试 | S3注册防护 | 整体安全评分 | 安全等级 |
|------|--------|------------|------------|-------------|----------|
| **A2A** | 100.0/100 🏆 | 56.0/100 (comprehensive) | 100.0/100 | 89.0/100 | MODERATE |
| **Agora** | 0.0/100 | 56.0/100 (comprehensive) | 83.3/100 | 79.0/100 | MODERATE |
| **ANP** | 0.0/100 | 58.0/100 (comprehensive) ⬆️ | 100.0/100 | 74.5/100 | MODERATE |
| **ACP** | 0.0/100 | 28.0/100 (comprehensive) | 100.0/100 | 67.0/100 | VULNERABLE |

### 📊 协议安全性排名 (更新)

1. **A2A**: 89.0/100 (MODERATE) - 最高安全评分 🏆 
   - 唯一达到S1业务连续性满分的协议
   - S2保密性表现良好，重放攻击防护有效
   
2. **Agora**: 79.0/100 (MODERATE) - 平衡表现
   - S2保密性与A2A相当
   - S3注册防护略有不足 (endpoint_substitution攻击成功)
   
3. **ANP**: 74.5/100 (MODERATE) - 显著提升 ⬆️
   - S2保密性有所改善 (26.0→58.0)
   - 元数据保护达到满分
   
4. **ACP**: 67.0/100 (VULNERABLE) - 需要改进
   - S2保密性最低
   - 重放攻击测试受协调器问题影响

---

## 🏆 最终安全测试结果总结 (2025-09-23 更新)

### 🔍 关键发现

1. **协议安全等级重新排列**：
   - **A2A** (MODERATE)：89.0/100 - 综合表现最佳 🏆
   - **Agora** (MODERATE)：79.0/100 - 平衡的现代协议
   - **ANP** (MODERATE)：74.5/100 - 显著提升，进入MODERATE级别 ⬆️
   - **ACP** (VULNERABLE)：67.0/100 - 需要重点改进

2. **重大改进亮点**：
   - **ANP S2保密性提升**：26.0 → 58.0/100 (翻倍改善)
   - **A2A S1业务连续性**：唯一达到100.0/100满分
   - **探针技术成熟**：所有协议comprehensive探针正常工作

3. **共同安全问题**：
   - **TLS降级攻击**：仍然是所有协议的共同弱点 (3/3失败)
   - **元数据泄露**：大部分协议暴露`/health`端点 (除ANP外)
   - **重放攻击防护**：表现良好，多数协议通过ReadTimeout机制阻止

4. **协议特色差异**：
   - **A2A**：全面均衡，S1+S2+S3都有出色表现
   - **Agora**：S2保密性优秀，但S3有endpoint_substitution漏洞
   - **ANP**：S2大幅改善，元数据保护完美 (100/100)
   - **ACP**：S2保密性最弱，协调器稳定性问题影响测试

### 🚀 S2探针技术成果

所有协议现已成功启用comprehensive探针，包括：
- ✅ **TLS降级探针**：检测HTTP明文连接漏洞
- ✅ **重放攻击探针**：测试消息重放防护能力  
- ✅ **被动嗅探探针**：监测明文数据泄露
- ✅ **元数据泄露探针**：检查端点信息暴露

### 📊 详细测试结果对比表

| 测试项目 | ANP | ACP | A2A | Agora | 最佳表现 |
|---------|-----|-----|-----|-------|----------|
| **S1 业务连续性** | 0.0/100 | 0.0/100 | **100.0/100** 🏆 | 0.0/100 | A2A |
| **S2 保密性总分** | 58.0/100 ⬆️ | 28.0/100 | 56.0/100 | 56.0/100 | **ANP** |
| TLS降级防护 | ❌ 0/3 | ❌ 0/3 | ❌ 0/3 | ❌ 0/3 | 无 |
| 重放攻击防护 | ✅ 2/2 | ⚠️ 0/2 | ✅ 2/2 | ✅ 2/2 | ANP/A2A/Agora |
| 元数据保护 | ✅ 100/100 | 80/100 | 80/100 | 80/100 | **ANP** |
| **S3 注册防护** | 100.0/100 | 100.0/100 | 100.0/100 | 83.3/100 | ANP/ACP/A2A |
| **整体安全评分** | 74.5/100 | 67.0/100 | **89.0/100** 🏆 | 79.0/100 | **A2A** |
| **安全等级** | MODERATE ⬆️ | VULNERABLE | MODERATE | MODERATE | A2A/Agora/ANP |

### 📈 改进趋势分析

- **ANP**: 🚀 显著提升 (VULNERABLE→MODERATE, S2: 26.0→58.0)
- **ACP**: 📉 相对落后 (仍处于VULNERABLE级别) 
- **A2A**: 🏆 持续领先 (最高总分89.0/100)
- **Agora**: 📊 稳定表现 (平衡的MODERATE级别)

### 📋 推荐改进优先级 (更新)

1. **紧急优先级**：修复所有协议的TLS降级问题 (0/4协议通过)
2. **高优先级**：改进ACP协议的S2保密性防护 (28.0/100最低)
3. **中优先级**：解决Agora的endpoint_substitution漏洞
4. **低优先级**：提升非A2A协议的S1业务连续性表现

---

## 1. ANP (Agent Network Protocol) ✅

### 启动配置
```bash
# 环境变量设置
export ANP_ENABLE_S2_PROBES=true
export ANP_S2_PROBE_TYPE=comprehensive  # 或 tls_downgrade, replay_attack
export ANP_S1_TEST_MODE=light           # 使用1x1x1最小负载矩阵

# 启动命令
cd /Users/jason/Desktop/Multiagent-Protocol
source exp_env/bin/activate
cd script/safety_tech
python runners/run_unified_security_test_anp.py
```

### 成功关键点
1. **路径修复**: 修复了client.py中的项目根目录路径计算
2. **启动方式**: 直接代码执行避免模块导入问题
3. **端口配置**: 9102 (Doctor A), 9103 (Doctor B), 8001 (RG), 8888 (Coordinator)
4. **DID认证**: 使用真实DID生成和验证

### S2探针发现的安全问题 (最新结果)
- **TLS降级攻击**: 3/3失败，允许HTTP明文连接
- **重放攻击**: 2/2被阻止，通过ReadTimeout机制
- **元数据泄露**: ✅ 0个端点泄露 (完美保护!) ⬆️
- **明文保护**: ✅ 100/100，无明文泄露
- **整体S2评分**: 58.0/100 (大幅提升 26.0→58.0)

---

## 2. ACP (Agent Communication Protocol) ✅

### 启动配置
```bash
# 环境变量设置
export ACP_ENABLE_S2_PROBES=true
export ACP_S2_PROBE_TYPE=comprehensive  # 或 tls_downgrade, replay_attack
export ACP_S1_TEST_MODE=light           # 使用1x1x1最小负载矩阵

# 启动命令
cd /Users/jason/Desktop/Multiagent-Protocol
source exp_env/bin/activate
cd script/safety_tech
python runners/run_unified_security_test_acp.py
```

### 成功关键点
1. **协议标准化**: 使用ACP标准格式通信
2. **端口管理**: 灵活的端口分配策略
3. **探针集成**: 完整的S2探针支持
4. **FastAPI后端**: 高性能HTTP服务器

### S2探针发现的安全问题
- **TLS降级攻击**: 3/3失败，允许HTTP明文连接
- **重放攻击**: 2/2被阻止，通过ReadTimeout机制
- **元数据泄露**: 1个端点可访问 (/health)
- **明文保护**: ✅ 100/100，无明文泄露

---

## 3. A2A (Agent-to-Agent) ✅

### 启动配置
```bash
# 环境变量设置
export A2A_ENABLE_S2_PROBES=true
export A2A_S2_PROBE_TYPE=comprehensive  # 或 tls_downgrade, replay_attack
export A2A_S1_TEST_MODE=light           # 使用1x1x1最小负载矩阵

# 启动命令
cd /Users/jason/Desktop/Multiagent-Protocol
source exp_env/bin/activate
cd script/safety_tech
python runners/run_unified_security_test_a2a.py
```

### 成功关键点
1. **路径修复**: 修复了client.py中的项目根目录路径计算错误（需要5个parent级别）
2. **启动方式**: 改为直接代码执行方式，避免模块导入问题
3. **端口配置**: 9202 (Doctor A), 9203 (Doctor B), 8001 (RG), 8888 (Coordinator)
4. **A2A SDK**: 成功集成A2A SDK，支持真实的Agent执行

### S2探针发现的安全问题
- **TLS降级攻击**: 3/3失败，允许HTTP明文连接
- **重放攻击**: 2/2被阻止，通过ReadTimeout机制
- **元数据泄露**: 1个端点可访问 (/health)
- **明文保护**: ✅ 100/100，无明文泄露

### 修复过程记录
```python
# 原始问题代码（失败）:
proc = subprocess.Popen([sys.executable, '-m', 'script.safety_tech.protocol_backends.a2a.server', role.lower()])

# 修复后的代码（成功）:
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent.parent  # 修复路径层级

code = (
    f"import sys; sys.path.insert(0, '{project_root}');"
    "from script.safety_tech.protocol_backends.a2a.server import run_server;"
    f"run_server('A2A_Doctor_A' if '{role.lower()}'=='doctor_a' else 'A2A_Doctor_B', {port})"
)

proc = subprocess.Popen([sys.executable, "-c", code], cwd=str(project_root), env=env)
```

---

## 4. Agora ✅

### 启动配置
```bash
# 环境变量设置
export AGORA_ENABLE_S2_PROBES=true      # 现已支持S2探针！
export AGORA_S2_PROBE_TYPE=comprehensive # 或 tls_downgrade, replay_attack
export AGORA_S1_TEST_MODE=light         # 使用1x1x1最小负载矩阵

# 启动命令
cd /Users/jason/Desktop/Multiagent-Protocol
source exp_env/bin/activate
cd script/safety_tech
python runners/run_unified_security_test_agora.py
```

### 成功关键点
1. **S2探针支持**: 新增了完整的S2探针配置，与ACP/ANP/A2A保持一致
2. **端口配置**: 8002 (Doctor A), 8003 (Doctor B), 8001 (RG), 8888 (Coordinator)
3. **原生SDK**: 使用Agora官方SDK，支持真实的Agent通信
4. **测试完成**: 能够成功完成所有S1/S2/S3测试

### S2探针发现的安全问题
- **TLS降级攻击**: 3/3失败，允许HTTP明文连接
- **重放攻击**: 2/2被阻止，通过ReadTimeout机制  
- **元数据泄露**: 1个端点可访问 (/health)
- **明文保护**: ✅ 100/100，无明文泄露

### ⚠️ 已知问题
1. **后台进程清理**: 测试完成后可能有残留的LLM请求和后端通信超时
2. **解决方案**: 测试完成后手动清理进程或等待自然超时
   ```bash
   # 清理命令
   ps aux | grep python | grep agora | awk '{print $2}' | xargs kill -9
   lsof -ti:8002,8003 | xargs kill -9
   ```

### 测试结果亮点
- **注册防护**: 83.3/100，成功阻止5/6种攻击
- **endpoint_substitution攻击**: 唯一未被阻止的攻击类型
- **整体安全性**: MODERATE级别，在四个协议中排名第二

---

## 5. 通用配置优化

### S1负载矩阵优化 (已实现)
修改了`script/safety_tech/core/s1_config_factory.py`中的light配置：

```python
# 优化后的light配置 (1x1x1矩阵)
'load_config': LoadMatrixConfig(
    concurrent_levels=[1],           # 最小并发数
    rps_patterns=[LoadPattern.CONSTANT],
    message_types=[MessageType.SHORT],
    test_duration_seconds=5,         # 极短测试时间
    base_rps=1                       # 最小RPS
),
'attack_config': AttackNoiseConfig(
    malicious_registration_rate=1,   # 最小攻击频率
    spam_message_rate=1,
    replay_attack_rate=1,
    dos_request_rate=1,
    probe_query_rate=1
)
```

### 端口分配策略
| 协议 | RG端口 | 协调器端口 | Observer端口 | Agent A端口 | Agent B端口 |
|------|--------|------------|--------------|-------------|-------------|
| ANP  | 8001   | 8888       | 8004         | 9102        | 9103        |
| ACP  | 8001   | 8888       | 8004         | -           | -           |
| A2A  | 8001   | 8888       | 8004         | 9202        | 9203        |
| Agora| 8001   | 8888       | 8004         | 8002        | 8003        |

### S2探针配置统一
所有协议现在都支持以下探针类型：

```bash
# 基础探针
export <PROTOCOL>_S2_PROBE_TYPE=tls_downgrade
export <PROTOCOL>_S2_PROBE_TYPE=replay_attack

# 综合探针（推荐）
export <PROTOCOL>_S2_PROBE_TYPE=comprehensive
```

---

## 6. 故障排除指南

### 常见问题

1. **端口占用错误**
   ```bash
   # 清理所有测试相关端口
   lsof -ti:8001,8002,8003,8004,8888,9102,9103,9202,9203 | xargs kill -9
   ```

2. **模块导入错误**
   - 确保在项目根目录执行
   - 确保虚拟环境exp_env已激活
   - 检查Python路径设置

3. **健康检查超时**
   - 增加启动等待时间
   - 检查服务器日志输出
   - 验证端口配置正确性

### 测试环境要求
- Python 3.8+
- exp_env虚拟环境
- NVIDIA API密钥配置
- 充足的内存和CPU资源

---

## 7. 后续改进建议

### 高优先级
1. **修复TLS降级漏洞**: 所有协议都需要强制HTTPS连接
2. **元数据保护**: 限制/health端点的访问权限
3. **进程清理优化**: 实现更完善的资源清理机制

### 中优先级  
1. **S2保密性增强**: 特别是ANP和ACP协议
2. **攻击检测改进**: 提高对各种攻击的识别能力
3. **性能优化**: 减少测试运行时间

### 低优先级
1. **S1业务连续性**: 在攻击环境下的稳定性改进
2. **用户体验**: 更友好的错误信息和日志输出
3. **文档完善**: 详细的API文档和使a a