# Protocol Backends 兼容性说明

## 概述

本文档说明在 `streaming_queue/ANP` 分支中不同协议的兼容性问题。

## 当前状态

### ✅ ANP协议 - 正常运行
- **测试结果**: 50/50 成功 (100%)
- **平均响应时间**: ~2.0秒
- **状态**: 完全兼容，无错误

### ❌ A2A协议 - 运行异常
- **问题**: 所有答案返回 "No answer received"
- **元数据显示**: successful_questions: 0, failed_questions: 50
- **状态**: 在此分支无法正常工作，但在 `streaming_queue/A2A` 分支正常

## 根本原因分析

### 共享核心组件差异

A2A和ANP协议都依赖相同的核心组件，但在不同分支中这些组件有不同的实现版本：

#### 1. `core/qa_worker_base.py` - 关键差异
**ANP分支版本 (当前)**:
```python
# 强制使用真实LLM，禁用mock fallback
if self.core is None:
    raise RuntimeError("Core is not initialized. Mock answers are not allowed.")

# 严格错误处理
if not result or result.strip() == "":
    raise RuntimeError("Core returned empty response")
```

**A2A分支版本**:
```python
# 允许mock fallback
if self.use_mock or self.core is None:
    await asyncio.sleep(0.05)
    return f"Mock answer: {q[:80]}{'...' if len(q) > 80 else ''}"

# 宽松错误处理
return (result or "Unable to generate response").strip()
```

#### 2. `core/qa_coordinator_base.py` - 路径处理
**ANP分支版本**:
- 包含相对路径处理逻辑
- 自动解析相对于 streaming_queue 目录的路径

**A2A分支版本**:
- 简单的路径处理
- 直接使用提供的路径

#### 3. `core/network_base.py` - 导入机制
**ANP分支版本**:
- 改进的错误处理和导入路径解析

**A2A分支版本**:
- 更多的fallback导入选项

## 问题机制

A2A在ANP分支上运行时：

1. **Core初始化失败**: ANP分支的 `qa_worker_base.py` 强制要求LLM正常工作，不允许fallback到mock
2. **无错误提示**: 虽然Core可能初始化失败，但没有明显的错误信息
3. **空答案**: 由于缺乏fallback机制，所有问题都返回 "No answer received"

## 设计哲学差异

### ANP分支 (当前)
- **严格模式**: 强制使用真实LLM，确保测试质量
- **错误快速失败**: 遇到问题立即抛出异常
- **路径智能处理**: 自动解析相对路径

### A2A分支
- **容错模式**: 允许fallback到mock答案
- **优雅降级**: 遇到问题时提供替代方案
- **简单路径处理**: 直接使用用户提供的路径

## 解决方案

### 选项1: 分支隔离 (推荐)
- ANP协议在 `streaming_queue/ANP` 分支测试
- A2A协议在 `streaming_queue/A2A` 分支测试
- 保持各自分支的核心组件不变

### 选项2: 统一适配
- 创建协议无关的核心组件
- 通过配置参数控制严格/容错模式
- 需要大量重构工作

## 测试建议

1. **ANP测试**: 在当前分支 (`streaming_queue/ANP`) 运行
   ```bash
   python -m runner.run_anp
   ```

2. **A2A测试**: 切换到A2A分支运行
   ```bash
   git checkout streaming_queue/A2A
   python -m runner.run_a2a
   ```

## 总结

这种差异反映了不同协议的设计理念：
- **ANP**: 注重安全性和严格性，适合生产环境
- **A2A**: 注重可用性和容错性，适合开发测试

建议保持当前的分支隔离策略，确保每个协议在最适合的环境中运行。
