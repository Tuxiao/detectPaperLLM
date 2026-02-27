#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-0.6B}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/Qwen3-0.6B}"
ACCEL_CONFIG="${ACCEL_CONFIG:-$ROOT_DIR/configs/accelerate_mps.yaml}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[1/5] Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "[1/5] Virtual environment already exists: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[2/5] Installing dependencies"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT_DIR/requirements.txt"
python -m pip install -e "$ROOT_DIR"
python -m pip install "huggingface_hub>=0.26"

if [[ -f "$MODEL_DIR/model.safetensors" ]]; then
  echo "[3/5] Model already present: $MODEL_DIR"
else
  echo "[3/5] Downloading model $MODEL_REPO to $MODEL_DIR"
  mkdir -p "$MODEL_DIR"
  hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
fi

echo "[4/5] Writing accelerate config to $ACCEL_CONFIG"
mkdir -p "$(dirname "$ACCEL_CONFIG")"
if python - <<'PY'
import torch
raise SystemExit(0 if torch.backends.mps.is_available() else 1)
PY
then
  USE_CPU=false
  echo "Detected Apple MPS backend"
else
  USE_CPU=true
  echo "MPS not detected, using CPU mode"
fi

cat > "$ACCEL_CONFIG" <<EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: $USE_CPU
EOF

echo "[5/5] Setup complete"
echo "Activate env: source \"$VENV_DIR/bin/activate\""
echo "Run example training: \"$ROOT_DIR/scripts/run_qwen3_lora_example.sh\""
