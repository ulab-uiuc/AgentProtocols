#!/usr/bin/env bash

# 简化版：顺序跑五个协议，底部显示当前状态
set -e

# 切换到仓库根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH-}"

# 要运行的协议列表
PROTOCOLS=(a2a acp agora anp meta)
if [[ $# -gt 0 ]]; then
    PROTOCOLS=("$@")
fi

echo "开始运行协议: ${PROTOCOLS[*]}"
echo "工作目录: ${REPO_ROOT}"
echo

# 清理函数，退出时清除状态行
cleanup() {
    printf "\r\033[K"  # 清除状态行
}
trap cleanup EXIT

# 运行每个协议
for i in "${!PROTOCOLS[@]}"; do
    protocol="${PROTOCOLS[$i]}"
    num=$((i + 1))
    total=${#PROTOCOLS[@]}
    
    # 在底部显示当前状态
    printf "\r\033[K🔄 正在运行 [$num/$total]: %s 协议..." "$protocol" >&2
    
    case "$protocol" in
        a2a)   python3 -m script.streaming_queue.runner.run_a2a ;;
        acp)   python3 -m script.streaming_queue.runner.run_acp ;;
        agora) python3 -m script.streaming_queue.runner.run_agora ;;
        anp)   python3 -m script.streaming_queue.runner.run_anp ;;
        meta)  python3 -m script.streaming_queue.runner.run_meta_network ;;
        *)     printf "\r\033[K❌ 错误: 未知协议 %s\n" "$protocol"; exit 1 ;;
    esac
    
    # 清除状态行，显示完成信息
    printf "\r\033[K✅ [$num/$total] %s 协议完成\n" "$protocol"
done

echo "🎉 所有协议运行完成！"
