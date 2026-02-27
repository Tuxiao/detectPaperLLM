# 自动获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export NUM_PERTURB_SAMPLES=8
export MAX_LENGTH=128
export TRAIN_FILE="$ROOT_DIR/data/train_pairs_merged.jsonl"
export OUTPUT_DIR="$ROOT_DIR/outputs/qwen3-0.6b-lora"
export NUM_TRAIN_EPOCHS=5
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=8

# 自动检测 GPU 类型，选择对应的 accelerate 配置
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[env] CUDA GPU detected, using accelerate_cuda.yaml"
  export ACCEL_CONFIG="$ROOT_DIR/configs/accelerate_cuda.yaml"
else
  echo "[env] No CUDA GPU, using accelerate_mps.yaml (MPS/CPU)"
  export ACCEL_CONFIG="$ROOT_DIR/configs/accelerate_mps.yaml"
fi

# 使用绝对路径调用子脚本
bash "$SCRIPT_DIR/run_qwen3_lora_example.sh"
