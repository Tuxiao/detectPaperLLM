#!/usr/bin/env bash
# =============================================================================
# DetectAnyLLM — PyTorch 推理服务启动脚本
#
# 用法:
#   bash start.sh                    # 使用默认配置启动
#   MODEL_PATH=/path/to/model bash start.sh  # 自定义模型路径
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 激活 conda 环境（如存在）----
if command -v conda &>/dev/null; then
    CONDA_ENV="${CONDA_ENV:-py312}"
    echo "🔧 激活 conda 环境: $CONDA_ENV"
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

# ---- 安装依赖（如尚未安装）----
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 安装依赖..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# ---- 默认环境变量 ----
export MODEL_PATH="${MODEL_PATH:-/root/detectPaperLLM/models}"
export REF_STATS_PATH="${REF_STATS_PATH:-}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
export USE_BF16="${USE_BF16:-true}"
export MAX_LENGTH="${MAX_LENGTH:-256}"
export NUM_PERTURB_SAMPLES="${NUM_PERTURB_SAMPLES:-16}"
export SIGMA_EPS="${SIGMA_EPS:-1e-6}"
export K_NEIGHBORS="${K_NEIGHBORS:-100}"
export DECISION_MODE="${DECISION_MODE:-threshold}"
export THRESHOLD="${THRESHOLD:-50.0}"

# 训练参考参数 (虽推理不直接使用，但作为环境记录)
export GAMMA=3
export LEARNING_RATE=5e-5
export NUM_TRAIN_EPOCHS=5
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=8

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9000}"

echo "============================================="
echo " DetectAnyLLM PyTorch Serving"
echo "============================================="
echo " Model     : $MODEL_PATH"
echo " Device    : auto (CUDA > MPS > CPU)"
echo " Host:Port : $HOST:$PORT"
echo " Mode      : $DECISION_MODE"
echo "============================================="

cd "$SCRIPT_DIR"
exec python -m uvicorn serve:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    --timeout-keep-alive 300 \
    --log-level info
