#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/Qwen3-0.6B}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT_DIR/data/train_pairs.sample.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3-0.6b-lora}"
ACCEL_CONFIG="${ACCEL_CONFIG:-$ROOT_DIR/configs/accelerate_mps.yaml}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"

if [[ -f "$ROOT_DIR/configs/qwen3_lora_mps.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/configs/qwen3_lora_mps.env"
fi

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  echo "[env] activating virtual environment: $VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif python3 -c "import detectanyllm" >/dev/null 2>&1; then
  echo "[env] using current python environment (detectanyllm detected)"
else
  echo "[env] virtual environment not found and 'detectanyllm' is not installed."
  echo "Run setup first: $ROOT_DIR/scripts/setup_qwen3_lora.sh"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/model.safetensors" ]]; then
  echo "Model weights not found in: $MODEL_DIR"
  echo "Run setup first: $ROOT_DIR/scripts/setup_qwen3_lora.sh"
  exit 1
fi

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Training data not found: $TRAIN_FILE"
  exit 1
fi

if [[ ! -f "$ACCEL_CONFIG" ]]; then
  echo "Accelerate config not found: $ACCEL_CONFIG"
  echo "Run setup first: $ROOT_DIR/scripts/setup_qwen3_lora.sh"
  exit 1
fi

# Environment already handled above
if ! python -c "import detectanyllm" >/dev/null 2>&1; then
  echo "[env] Python package 'detectanyllm' is not installed."
  echo "Please run: pip install -e \"$ROOT_DIR\""
  exit 1
fi
mkdir -p "$OUTPUT_DIR"

echo "[train] model: $MODEL_DIR"
echo "[train] train file: $TRAIN_FILE"
echo "[train] output dir: $OUTPUT_DIR"
echo "[train] accelerate config: $ACCEL_CONFIG"

accelerate launch \
  --config_file "$ACCEL_CONFIG" \
  -m detectanyllm.cli train \
  --model-name-or-path "$MODEL_DIR" \
  --train-pairs-file "$TRAIN_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --target-modules "$TARGET_MODULES" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --max-length "${MAX_LENGTH:-256}" \
  --logging-steps "${LOGGING_STEPS:-1}" \
  --save-steps "${SAVE_STEPS:-20}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-2}" \
  --gamma "${GAMMA:-100}" \
  --num-perturb-samples "${NUM_PERTURB_SAMPLES:-32}" \
  --sigma-eps "${SIGMA_EPS:-1e-6}" \
  --trust-remote-code \
  --no-bf16

echo "[train] done. adapter saved to: $OUTPUT_DIR"
