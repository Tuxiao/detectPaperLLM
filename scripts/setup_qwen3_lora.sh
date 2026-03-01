#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
ACCEL_CONFIG="${ACCEL_CONFIG:-$ROOT_DIR/configs/accelerate_mps.yaml}"
MODEL_PRESET="${MODEL_PRESET:-qwen3:0.6b}"
MODEL_REPO="${MODEL_REPO:-}"
MODEL_DIR="${MODEL_DIR:-}"

usage() {
  cat <<'EOF'
Usage: setup_qwen3_lora.sh [options]

Options:
  --model <preset>      Model preset: qwen3:0.6b (default), qwen3:8b
  --model-repo <repo>   Override model repo (e.g. Qwen/Qwen3-8B)
  --model-dir <path>    Override local model directory
  -h, --help            Show this help
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
      ;;
    qwen3:8b|qwen3-8b|8b)
      MODEL_PRESET="qwen3:8b"
      MODEL_REPO_DEFAULT="Qwen/Qwen3-8B"
      MODEL_DIR_DEFAULT="$ROOT_DIR/models/Qwen3-8B"
      ;;
    *)
      echo "Unsupported model preset: $1"
      echo "Supported presets: qwen3:0.6b, qwen3:8b"
      exit 1
      ;;
  esac
}

model_is_ready() {
  local dir="$1"
  [[ -f "$dir/model.safetensors" ]] \
    || [[ -f "$dir/model.safetensors.index.json" ]] \
    || compgen -G "$dir/model-*.safetensors" > /dev/null
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
    --model-repo)
      [[ $# -lt 2 ]] && { echo "Missing value for --model-repo"; usage; exit 1; }
      MODEL_REPO="$2"
      shift 2
      ;;
    --model-repo=*)
      MODEL_REPO="${1#*=}"
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
MODEL_REPO="${MODEL_REPO:-$MODEL_REPO_DEFAULT}"
MODEL_DIR="${MODEL_DIR:-$MODEL_DIR_DEFAULT}"

echo "[config] model preset: $MODEL_PRESET"
echo "[config] model repo: $MODEL_REPO"
echo "[config] model dir: $MODEL_DIR"

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
python -m pip install "huggingface_hub>=0.26" "modelscope"

if model_is_ready "$MODEL_DIR"; then
  echo "[3/5] Model already present: $MODEL_DIR"
else
  echo "[3/5] Downloading model $MODEL_REPO to $MODEL_DIR"
  mkdir -p "$MODEL_DIR"
  
  # Try ModelScope first (preferred in mainland China), show progress
  echo "Attempting to download from ModelScope..."
  if python -c "from modelscope import snapshot_download; snapshot_download('$MODEL_REPO', local_dir='$MODEL_DIR')"; then
    echo "Model downloaded successfully from ModelScope."
  else
    echo "ModelScope failed. Falling back to Hugging Face (hf-mirror.com)..."
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_DIR" --resume-download
  fi
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
