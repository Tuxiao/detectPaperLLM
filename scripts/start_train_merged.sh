#!/usr/bin/env bash
set -euo pipefail

# 自动获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_PRESET="${MODEL_PRESET:-qwen3:0.6b}"
MODEL_DIR="${MODEL_DIR:-}"

usage() {
  cat <<'EOF'
Usage: start_train_merged.sh [options]

Options:
  --model <preset>    Model preset: qwen3:0.6b (default), qwen3:8b
  --model-dir <path>  Override local model directory
  -h, --help          Show this help
EOF
}

resolve_model_preset() {
  local preset
  preset="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$preset" in
    qwen3:0.6b|qwen3-0.6b|0.6b)
      MODEL_PRESET="qwen3:0.6b"
      MODEL_DIR_DEFAULT="$ROOT_DIR/models/Qwen3-0.6B"
      OUTPUT_PREFIX_DEFAULT="qwen3-0.6b-lora"
      ;;
    qwen3:8b|qwen3-8b|8b)
      MODEL_PRESET="qwen3:8b"
      MODEL_DIR_DEFAULT="$ROOT_DIR/models/Qwen3-8B"
      OUTPUT_PREFIX_DEFAULT="qwen3-8b-lora"
      ;;
    *)
      echo "Unsupported model preset: $1"
      echo "Supported presets: qwen3:0.6b, qwen3:8b"
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
MODEL_DIR="${MODEL_DIR:-$MODEL_DIR_DEFAULT}"

export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"
export NUM_PERTURB_SAMPLES="${NUM_PERTURB_SAMPLES:-8}"
export MAX_LENGTH="${MAX_LENGTH:-256}"
export TRAIN_FILE="${TRAIN_FILE:-$ROOT_DIR/data/train_pairs_merged.jsonl}"
export MODEL_DIR
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/${OUTPUT_PREFIX_DEFAULT}-$(date +%Y%m%d_%H%M%S)}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

echo "[config] model preset: $MODEL_PRESET"
echo "[config] model dir: $MODEL_DIR"
echo "[config] output dir: $OUTPUT_DIR"

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
