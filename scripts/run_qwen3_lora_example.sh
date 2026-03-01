#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/Qwen3-0.6B}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT_DIR/data/train_pairs.sample.jsonl}"
VALIDATION_FILE="${VALIDATION_FILE:-}"
TEST_FILE="${TEST_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3-0.6b-lora}"
ACCEL_CONFIG="${ACCEL_CONFIG:-$ROOT_DIR/configs/accelerate_mps.yaml}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"
GROUP_ID_FIELD="${GROUP_ID_FIELD:-}"
DEV_RATIO="${DEV_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TEST_EVAL_STEPS="${TEST_EVAL_STEPS:-20}"
TEST_THRESHOLD_OBJECTIVE="${TEST_THRESHOLD_OBJECTIVE:-mcc}"

model_is_ready() {
  local dir="$1"
  [[ -f "$dir/model.safetensors" ]] \
    || [[ -f "$dir/model.safetensors.index.json" ]] \
    || compgen -G "$dir/model-*.safetensors" > /dev/null
}

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

if ! model_is_ready "$MODEL_DIR"; then
  echo "Model weights not found in: $MODEL_DIR"
  echo "Run setup first: $ROOT_DIR/scripts/setup_qwen3_lora.sh"
  exit 1
fi

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Training data not found: $TRAIN_FILE"
  exit 1
fi
if [[ -n "$VALIDATION_FILE" && ! -f "$VALIDATION_FILE" ]]; then
  echo "Validation data not found: $VALIDATION_FILE"
  exit 1
fi
if [[ -n "$TEST_FILE" && ! -f "$TEST_FILE" ]]; then
  echo "Test data not found: $TEST_FILE"
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
if [[ -n "$VALIDATION_FILE" ]]; then
  echo "[train] validation file: $VALIDATION_FILE"
fi
if [[ -n "$TEST_FILE" ]]; then
  echo "[train] test file: $TEST_FILE"
fi
echo "[train] output dir: $OUTPUT_DIR"
echo "[train] accelerate config: $ACCEL_CONFIG"

train_cmd=(
  accelerate launch
  --config_file "$ACCEL_CONFIG"
  -m detectanyllm.cli train
  --model-name-or-path "$MODEL_DIR"
  --train-pairs-file "$TRAIN_FILE"
  --output-dir "$OUTPUT_DIR"
  --target-modules "$TARGET_MODULES"
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-1}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
  --learning-rate "${LEARNING_RATE:-1e-4}"
  --max-length "${MAX_LENGTH:-256}"
  --logging-steps "${LOGGING_STEPS:-1}"
  --save-steps "${SAVE_STEPS:-20}"
  --save-total-limit "${SAVE_TOTAL_LIMIT:-2}"
  --gamma "${GAMMA:-100}"
  --num-perturb-samples "${NUM_PERTURB_SAMPLES:-32}"
  --sigma-eps "${SIGMA_EPS:-1e-6}"
  --dev-ratio "$DEV_RATIO"
  --test-ratio "$TEST_RATIO"
  --split-seed "$SPLIT_SEED"
  --test-eval-steps "$TEST_EVAL_STEPS"
  --test-threshold-objective "$TEST_THRESHOLD_OBJECTIVE"
  --trust-remote-code
  --no-bf16
)

if [[ -n "$GROUP_ID_FIELD" ]]; then
  train_cmd+=(--group-id-field "$GROUP_ID_FIELD")
fi
if [[ -n "$VALIDATION_FILE" ]]; then
  train_cmd+=(--validation-pairs-file "$VALIDATION_FILE")
fi
if [[ -n "$TEST_FILE" ]]; then
  train_cmd+=(--test-pairs-file "$TEST_FILE")
fi

"${train_cmd[@]}"

echo "[train] done. adapter saved to: $OUTPUT_DIR"
