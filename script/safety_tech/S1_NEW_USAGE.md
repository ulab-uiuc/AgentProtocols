# S1 新版业务连续性测试使用指南

## 概述

新版S1测试专注于"并发与对抗下的端到端稳定性"，通过负载矩阵、网络扰动、攻击噪声共存等方式，全面评估协议的业务连续性能力。

## 核心特性

### 1. 负载矩阵测试
- **并发级别**: 8/32/128 等多层并发
- **RPS模式**: 恒定/泊松/突发三种负载模式
- **报文类型**: 短文/长文/流式三种消息类型
- **组合测试**: 自动生成所有组合进行全面测试

### 2. 网络扰动模拟
- **抖动延迟**: 10-300ms可配置范围
- **丢包模拟**: 0.1%-10%丢包率
- **乱序模拟**: 通过随机延迟实现
- **带宽限制**: 200-2000kbps可配置
- **短线重连**: 周期性连接中断模拟

### 3. 攻击噪声共存
- **恶意注册**: 持续尝试恶意Agent注册
- **垃圾消息**: 高频发送无意义消息
- **重放攻击**: 重复发送历史消息
- **轻量DoS**: 大量并发健康检查请求
- **旁路查询**: 尝试各种未授权端点访问

### 4. 回执闭环判定
- **唯一关联ID**: 每个请求分配correlation_id
- **端到端跟踪**: 从发送到收到回执的完整跟踪
- **超时管理**: 可配置超时时间和清理机制
- **成功标准**: 必须收到对端确认才算成功

### 5. 综合指标收集
- **完成率**: 主要指标，端到端成功率
- **延迟分布**: P50/P95/P99延迟统计
- **超时率**: 超时请求比例
- **重试统计**: 重试和重连次数
- **错误分类**: 详细错误类型分布

## 使用方式

### 环境变量配置

```bash
# S1测试模式选择
export ACP_S1_TEST_MODE=standard  # light/standard/stress/protocol_optimized

# 数据面直连开关
export ACP_USE_DIRECT_SEND=true   # true/false

# S2保密性探针（可选）
export ACP_ENABLE_S2_PROBES=true
export ACP_S2_PROBE_TYPE=comprehensive
```

### 代码集成示例

```python
from script.safety_tech.core.s1_config_factory import create_s1_tester

# 创建测试器
s1_tester = create_s1_tester('acp', 'protocol_optimized')

# 定义发送函数
async def protocol_send_function(payload):
    # 实现具体协议的发送逻辑
    return await send_message(payload)

# 运行完整测试矩阵
s1_results = await s1_tester.run_full_test_matrix(
    send_func=protocol_send_function,
    sender_id='Doctor_A',
    receiver_id='Doctor_B',
    rg_port=8080,
    coord_port=8081,
    obs_port=8082
)

# 生成综合报告
s1_report = s1_tester.generate_comprehensive_report()
```

## 预定义配置

### 1. light - 轻量测试
- 1个并发级别 (8)
- 1种RPS模式 (恒定)
- 1种消息类型 (短文)
- 测试时长: 30秒
- 适用: 快速验证

### 2. standard - 标准测试
- 2个并发级别 (8, 32)
- 2种RPS模式 (恒定, 泊松)
- 2种消息类型 (短文, 长文)
- 测试时长: 60秒
- 适用: 常规测试

### 3. stress - 压力测试
- 3个并发级别 (8, 32, 128)
- 3种RPS模式 (恒定, 泊松, 突发)
- 3种消息类型 (短文, 长文, 流式)
- 测试时长: 90秒
- 适用: 极限测试

### 4. protocol_optimized - 协议优化
- 根据协议特性自动调整参数
- ACP: 降低RPS，减少抖动
- ANP: 提高RPS，测试丢包鲁棒性
- A2A: 增加突发测试
- Agora: 更高RPS，更长测试时间

### 5. network_focus - 网络重点
- 固定并发和RPS
- 高强度网络扰动
- 测试时长: 120秒
- 适用: 网络韧性专项测试

### 6. attack_focus - 攻击重点
- 固定负载参数
- 高强度攻击噪声
- 最小网络扰动
- 适用: 攻击抗性专项测试

### 7. latency_focus - 延迟重点
- 多层并发测试
- 低干扰环境
- 测试时长: 180秒
- 适用: 延迟分布精确测量

## 报告格式

```json
{
  "protocol": "acp",
  "test_summary": {
    "total_combinations_tested": 12,
    "total_requests": 1200,
    "total_successful": 1140,
    "total_failed": 45,
    "total_timeout": 15,
    "overall_completion_rate": 0.95,
    "overall_timeout_rate": 0.0125
  },
  "latency_analysis": {
    "avg_ms": 85.2,
    "p50_ms": 72.1,
    "p95_ms": 156.8,
    "p99_ms": 234.5
  },
  "dimensional_analysis": {
    "by_concurrent_level": {
      "8": {"avg_completion_rate": 0.98, "avg_latency_ms": 65.3},
      "32": {"avg_completion_rate": 0.94, "avg_latency_ms": 89.7},
      "128": {"avg_completion_rate": 0.87, "avg_latency_ms": 142.6}
    },
    "by_rps_pattern": {
      "constant": {"avg_completion_rate": 0.96, "avg_latency_ms": 78.2},
      "poisson": {"avg_completion_rate": 0.94, "avg_latency_ms": 91.8}
    },
    "by_message_type": {
      "short": {"avg_completion_rate": 0.97, "avg_latency_ms": 68.4},
      "long": {"avg_completion_rate": 0.93, "avg_latency_ms": 101.7}
    }
  },
  "performance_extremes": {
    "best_combination": {
      "config": {"concurrent_level": 8, "rps_pattern": "constant", "message_type": "short"},
      "completion_rate": 0.99,
      "avg_latency_ms": 52.3
    },
    "worst_combination": {
      "config": {"concurrent_level": 128, "rps_pattern": "burst", "message_type": "long"},
      "completion_rate": 0.78,
      "avg_latency_ms": 189.4
    }
  }
}
```

## 协议差异预期

### HTTP同步RPC型 (ACP)
- **特点**: 对延迟尖峰敏感，线程池瓶颈
- **表现**: 完成率随RPS下降较快，重试放大抖动
- **优化**: 降低基础RPS，减少连接中断频率

### 长连接/会话型 (ANP)
- **特点**: 建立成本高，稳态性能好
- **表现**: 对丢包/乱序鲁棒，连接翻转时短暂抖动
- **优化**: 可承受更高RPS，减少连接中断测试

### 平台化网络 (Agora)
- **特点**: 背压友好，可能厚尾延迟
- **表现**: 稳态完成率高，延迟分布厚尾
- **优化**: 更高RPS测试背压，更长时间观察厚尾

## 验证测试

运行验证脚本确认新框架正常工作：

```bash
cd /Users/jason/Desktop/Multiagent-Protocol/script/safety_tech
python test_s1_new_implementation.py
```

## 集成到现有Runner

新版S1测试已集成到 `run_unified_security_test_acp.py`，其他协议runner可参考相同模式进行集成。

主要步骤：
1. 导入S1测试模块
2. 创建协议特定的发送函数
3. 运行测试矩阵
4. 更新报告结构

## 性能考虑

- **轻量模式**: ~30秒，适合CI/CD
- **标准模式**: ~5分钟，适合日常测试
- **压力模式**: ~15分钟，适合深度评估
- **专项模式**: 根据专项需求调整

## 扩展性

框架支持：
- 自定义负载配置
- 自定义网络扰动参数
- 自定义攻击模式
- 自定义成功判定标准
- 自定义指标收集

通过继承和配置可以灵活适应不同协议和场景的测试需求。
