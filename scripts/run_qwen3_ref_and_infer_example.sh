#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/outputs/qwen3-0.6b-lora}"
BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/models/Qwen3-0.6B}"
HUMAN_REF_FILE="${HUMAN_REF_FILE:-$ROOT_DIR/data/human_ref.sample.jsonl}"
MACHINE_REF_FILE="${MACHINE_REF_FILE:-$ROOT_DIR/data/machine_ref.sample.jsonl}"
INPUT_FILE="${INPUT_FILE:-$ROOT_DIR/data/infer.sample.jsonl}"
REF_STATS_FILE="${REF_STATS_FILE:-$ROOT_DIR/outputs/qwen3-0.6b-lora/ref_stats.json}"
PRED_FILE="${PRED_FILE:-$ROOT_DIR/outputs/qwen3-0.6b-lora/predictions.jsonl}"

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
if [[ ! -f "$MODEL_PATH/adapter_config.json" ]]; then
  echo "LoRA adapter not found: $MODEL_PATH"
  echo "Run training first: $ROOT_DIR/scripts/run_qwen3_lora_example.sh"
  exit 1
fi

# Environment already handled above
if ! python -c "import detectanyllm" >/dev/null 2>&1; then
  echo "[env] Python package 'detectanyllm' is not installed."
  echo "Please run: pip install -e \"$ROOT_DIR\""
  exit 1
fi

mkdir -p "$(dirname "$REF_STATS_FILE")" "$(dirname "$PRED_FILE")"

echo "[build-ref] model: $MODEL_PATH"
python -m detectanyllm.cli build-ref \
  --model-path "$MODEL_PATH" \
  --base-model "$BASE_MODEL" \
  --human-ref-file "$HUMAN_REF_FILE" \
  --machine-ref-file "$MACHINE_REF_FILE" \
  --ref-stats-file "$REF_STATS_FILE" \
  --text-field "${TEXT_FIELD:-text}" \
  --max-length "${MAX_LENGTH:-256}" \
  --num-perturb-samples "${NUM_PERTURB_SAMPLES:-32}" \
  --k-neighbors "${K_NEIGHBORS:-100}" \
  --sigma-eps "${SIGMA_EPS:-1e-6}" \
  --trust-remote-code \
  --no-bf16

echo "[infer] input: $INPUT_FILE"
python -m detectanyllm.cli infer \
  --model-path "$MODEL_PATH" \
  --base-model "$BASE_MODEL" \
  --input-file "$INPUT_FILE" \
  --output-file "$PRED_FILE" \
  --ref-stats-file "$REF_STATS_FILE" \
  --decision-mode pm \
  --text-field "${TEXT_FIELD:-text}" \
  --max-length "${MAX_LENGTH:-256}" \
  --num-perturb-samples "${NUM_PERTURB_SAMPLES:-32}" \
  --k-neighbors "${K_NEIGHBORS:-100}" \
  --sigma-eps "${SIGMA_EPS:-1e-6}" \
  --trust-remote-code \
  --no-bf16

echo "[infer] done. predictions: $PRED_FILE"
