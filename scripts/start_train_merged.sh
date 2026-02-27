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

# 使用绝对路径调用子脚本
bash "$SCRIPT_DIR/run_qwen3_lora_example.sh"
