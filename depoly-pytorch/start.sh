#!/usr/bin/env bash
# =============================================================================
# DetectAnyLLM — PyTorch 推理服务启动脚本
#
# 用法示例:
#   bash start.sh --model_path /path/to/adapter --base_model /path/to/base --port 9000
#
# 支持参数:
#   --model_path           模型或 Adapter 路径 (默认: /root/detectPaperLLM/models)
#   --base_model           Base Model 路径 (LoRA 必须)
#   --port                 服务端口 (默认: 9000)
#   --host                 服务 Host (默认: 0.0.0.0)
#   --decision_mode        判别模式, threshold 或 pm (默认: threshold)
#   --threshold            DC 分数阈值 (默认: 50.0)
#   --max_length           最大长度 (默认: 256)
#   --num_perturb_samples  扰动样本数 (默认: 16)
#   -h, --help             查看帮助
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 默认配置 ----
export MODEL_PATH="${MODEL_PATH:-/root/detectPaperLLM/models}"
export BASE_MODEL="${BASE_MODEL:-}"
export REF_STATS_PATH="${REF_STATS_PATH:-}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
export USE_BF16="${USE_BF16:-true}"
export MAX_LENGTH="${MAX_LENGTH:-256}"
export NUM_PERTURB_SAMPLES="${NUM_PERTURB_SAMPLES:-16}"
export SIGMA_EPS="${SIGMA_EPS:-1e-6}"
export K_NEIGHBORS="${K_NEIGHBORS:-100}"
export DECISION_MODE="${DECISION_MODE:-threshold}"
export THRESHOLD="${THRESHOLD:-50.0}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9000}"

# ---- 解析命令行参数 ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      export MODEL_PATH="$2"
      shift 2
      ;;
    --base_model)
      export BASE_MODEL="$2"
      shift 2
      ;;
    --ref_stats_path)
      export REF_STATS_PATH="$2"
      shift 2
      ;;
    --trust_remote_code)
      export TRUST_REMOTE_CODE="$2"
      shift 2
      ;;
    --use_bf16)
      export USE_BF16="$2"
      shift 2
      ;;
    --max_length)
      export MAX_LENGTH="$2"
      shift 2
      ;;
    --num_perturb_samples)
      export NUM_PERTURB_SAMPLES="$2"
      shift 2
      ;;
    --sigma_eps)
      export SIGMA_EPS="$2"
      shift 2
      ;;
    --k_neighbors)
      export K_NEIGHBORS="$2"
      shift 2
      ;;
    --decision_mode)
      export DECISION_MODE="$2"
      shift 2
      ;;
    --threshold)
      export THRESHOLD="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    -h|--help)
      grep -E '^[#] (支持参数|用法示例|--|-h)' "$0"
      exit 0
      ;;
    *)
      echo "❌ 未知参数: $1"
      echo "使用 bash start.sh --help 查看说明"
      exit 1
      ;;
  esac
done

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

# 训练参考参数 (虽推理不直接使用，但作为环境记录)
export GAMMA=3
export LEARNING_RATE=5e-5
export NUM_TRAIN_EPOCHS=5
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=8

echo "============================================="
echo " DetectAnyLLM PyTorch Serving"
echo "============================================="
echo " Model     : $MODEL_PATH"
if [ -n "$BASE_MODEL" ]; then
echo " Base Model: $BASE_MODEL"
fi
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
