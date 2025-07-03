hostname

DEFAULT_BASE_PORT=8001
DEFAULT_MODEL_PATH="/GPFS/data/sujiaqi/gui/Multiagent-Protocol/Qwen2.5-7B-Instruct/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_MODEL_NAME="Qwen2.5-7B-Instruct"

BASE_PORT=${1:-$DEFAULT_BASE_PORT}
MODEL_PATH=${2:-$DEFAULT_MODEL_PATH}
MODEL_NAME=${3:-$DEFAULT_MODEL_NAME}
START_TIME=$(date +%Y%m%d_%H%M%S)

# 明确指定 8 张 GPU
export CUDA_VISIBLE_DEVICES=3,4
export NCCL_P2P_DISABLE=0   # 确保 NCCL 通信开启
export NCCL_DEBUG=INFO      # 启用 NCCL 调试信息（可选）

# 启动 vLLM API 服务器，确保 tensor 并行数与 GPU 数匹配
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port $BASE_PORT \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size 2 \
    --disable-log-requests \
    --disable-log-stats \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 > vllm_log/vllm_${START_TIME}.log 2>&1 &
