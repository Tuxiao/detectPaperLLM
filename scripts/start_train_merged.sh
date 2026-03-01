#!/usr/bin/env bash
set -euo pipefail

# 自动获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_PRESET="${MODEL_PRESET:-qwen3:0.6b}"
MODEL_DIR="${MODEL_DIR:-}"
MODEL_REPO="${MODEL_REPO:-}"
DATA_DIR="${DATA_DIR:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

usage() {
  cat <<'EOF'
Usage: start_train_merged.sh [options]

Options:
  --model <preset>    Model preset: qwen3:0.6b (default), qwen3:4b, qwen3:8b
  --model-repo <repo> Override model repo (e.g. Qwen/Qwen3-4B)
  --model-dir <path>  Override local model directory
  --data-dir <path>   Data directory containing train.jsonl / dev.jsonl / test.jsonl
  --resume-from-checkpoint <path>  Resume from a trainer checkpoint directory
  -h, --help          Show this help
EOF
}

resolve_model_preset() {
  local preset
  preset="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$preset" in
    qwen3:0.6b|qwen3-0.6b|0.6b)
      MODEL_PRESET="qwen3:0.6b"
      MODEL_REPO_DEFAULT="Qwen/Qwen3-0.6B"
      MODEL_DIR_DEFAULT="$ROOT_DIR/models/Qwen3-0.6B"
      OUTPUT_PREFIX_DEFAULT="qwen3-0.6b-lora"
      ;;
    qwen3:4b|qwen3-4b|4b)
      MODEL_PRESET="qwen3:4b"
      MODEL_REPO_DEFAULT="Qwen/Qwen3-4B"
      MODEL_DIR_DEFAULT="$ROOT_DIR/models/Qwen3-4B"
      OUTPUT_PREFIX_DEFAULT="qwen3-4b-lora"
      ;;
    qwen3:8b|qwen3-8b|8b)
      MODEL_PRESET="qwen3:8b"
      MODEL_REPO_DEFAULT="Qwen/Qwen3-8B"
      MODEL_DIR_DEFAULT="$ROOT_DIR/models/Qwen3-8B"
      OUTPUT_PREFIX_DEFAULT="qwen3-8b-lora"
      ;;
    *)
      echo "Unsupported model preset: $1"
      echo "Supported presets: qwen3:0.6b, qwen3:4b, qwen3:8b"
      exit 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -lt 2 ]] && { echo "Missing value for --model"; usage; exit 1; }
      MODEL_PRESET="$2"
      shift 2
      ;;
    --model=*)
      MODEL_PRESET="${1#*=}"
      shift
      ;;
    --model-dir)
      [[ $# -lt 2 ]] && { echo "Missing value for --model-dir"; usage; exit 1; }
      MODEL_DIR="$2"
      shift 2
      ;;
    --model-dir=*)
      MODEL_DIR="${1#*=}"
      shift
      ;;
    --model-repo)
      [[ $# -lt 2 ]] && { echo "Missing value for --model-repo"; usage; exit 1; }
      MODEL_REPO="$2"
      shift 2
      ;;
    --model-repo=*)
      MODEL_REPO="${1#*=}"
      shift
      ;;
    --data-dir)
      [[ $# -lt 2 ]] && { echo "Missing value for --data-dir"; usage; exit 1; }
      DATA_DIR="$2"
      shift 2
      ;;
    --data-dir=*)
      DATA_DIR="${1#*=}"
      shift
      ;;
    --resume-from-checkpoint)
      [[ $# -lt 2 ]] && { echo "Missing value for --resume-from-checkpoint"; usage; exit 1; }
      RESUME_FROM_CHECKPOINT="$2"
      shift 2
      ;;
    --resume-from-checkpoint=*)
      RESUME_FROM_CHECKPOINT="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

resolve_model_preset "$MODEL_PRESET"
MODEL_REPO="${MODEL_REPO:-$MODEL_REPO_DEFAULT}"
MODEL_DIR="${MODEL_DIR:-$MODEL_DIR_DEFAULT}"

export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"
export NUM_PERTURB_SAMPLES="${NUM_PERTURB_SAMPLES:-8}"
export MAX_LENGTH="${MAX_LENGTH:-256}"
_DEFAULT_SPLITS_DIR="${DATA_DIR:-$ROOT_DIR/data/splits}"
export TRAIN_FILE="${TRAIN_FILE:-$_DEFAULT_SPLITS_DIR/train.jsonl}"
export VALIDATION_FILE="${VALIDATION_FILE:-$_DEFAULT_SPLITS_DIR/dev.jsonl}"
export TEST_FILE="${TEST_FILE:-$_DEFAULT_SPLITS_DIR/test.jsonl}"
export MODEL_DIR
export MODEL_REPO
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/${OUTPUT_PREFIX_DEFAULT}-$(date +%Y%m%d_%H%M%S)}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export GROUP_ID_FIELD="${GROUP_ID_FIELD:-}"
export DEV_RATIO="${DEV_RATIO:-0.1}"
export TEST_RATIO="${TEST_RATIO:-0.1}"
export SPLIT_SEED="${SPLIT_SEED:-42}"
export TEST_EVAL_STEPS="${TEST_EVAL_STEPS:-20}"
export TEST_THRESHOLD_OBJECTIVE="${TEST_THRESHOLD_OBJECTIVE:-mcc}"
export RESUME_FROM_CHECKPOINT

echo "[config] model preset: $MODEL_PRESET"
echo "[config] model repo: $MODEL_REPO"
echo "[config] model dir: $MODEL_DIR"
echo "[config] output dir: $OUTPUT_DIR"
echo "[config] train file: $TRAIN_FILE"
echo "[config] dev file: $VALIDATION_FILE"
echo "[config] test file: $TEST_FILE"
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  echo "[config] resume checkpoint: $RESUME_FROM_CHECKPOINT"
fi
if [[ -n "$GROUP_ID_FIELD" ]]; then
  echo "[config] group-id split field: $GROUP_ID_FIELD (dev_ratio=$DEV_RATIO test_ratio=$TEST_RATIO split_seed=$SPLIT_SEED)"
  echo "[config] periodic test eval: every $TEST_EVAL_STEPS step(s), threshold objective=$TEST_THRESHOLD_OBJECTIVE"
fi

# 自动检测 GPU 类型，选择对应的 accelerate 配置
if [[ -z "${ACCEL_CONFIG:-}" ]]; then
  if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[env] CUDA GPU detected, using accelerate_cuda.yaml"
    export ACCEL_CONFIG="$ROOT_DIR/configs/accelerate_cuda.yaml"
  else
    echo "[env] No CUDA GPU, using accelerate_mps.yaml (MPS/CPU)"
    export ACCEL_CONFIG="$ROOT_DIR/configs/accelerate_mps.yaml"
  fi
else
  echo "[env] Using pre-set ACCEL_CONFIG: $ACCEL_CONFIG"
fi

# 使用绝对路径调用子脚本
bash "$SCRIPT_DIR/run_qwen3_lora_example.sh"
