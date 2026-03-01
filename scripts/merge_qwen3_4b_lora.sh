#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
ADAPTER_PATH="${ADAPTER_PATH:-$ROOT_DIR/outputs/qwen3-4b-lora}"
BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/models/Qwen3-4B}"
MERGED_OUTPUT_DIR="${MERGED_OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3-4b-merged}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
USE_BF16="${USE_BF16:-false}"
SAFE_SERIALIZATION="${SAFE_SERIALIZATION:-true}"

usage() {
  cat <<'EOF'
Usage: merge_qwen3_4b_lora.sh [options]

Options:
  --adapter-path <path>   LoRA adapter directory (contains adapter_config.json)
  --base-model <path>     Qwen3-4B base model path/repo
  --output-dir <path>     Output directory for merged full model
  --trust-remote-code     Enable trust_remote_code when loading model/tokenizer
  --no-trust-remote-code  Disable trust_remote_code
  --bf16                  Prefer bf16 on CUDA
  --no-bf16               Do not use bf16
  --safe-serialization    Save as safetensors (default)
  --no-safe-serialization Save as pytorch_model.bin
  -h, --help              Show this help
EOF
}

resolve_adapter_path() {
  local input_path="$1"
  if [[ -f "$input_path/adapter_config.json" ]]; then
    printf '%s\n' "$input_path"
    return 0
  fi

  if [[ ! -d "$input_path" ]]; then
    return 1
  fi

  local latest_checkpoint=""
  local latest_step=-1
  local checkpoint_dir
  while IFS= read -r checkpoint_dir; do
    local name step
    name="$(basename "$checkpoint_dir")"
    step="${name#checkpoint-}"
    if [[ "$step" =~ ^[0-9]+$ ]] && [[ "$step" -gt "$latest_step" ]]; then
      if [[ -f "$checkpoint_dir/adapter_config.json" ]]; then
        latest_step="$step"
        latest_checkpoint="$checkpoint_dir"
      fi
    fi
  done < <(find "$input_path" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null)

  if [[ -n "$latest_checkpoint" ]]; then
    printf '%s\n' "$latest_checkpoint"
    return 0
  fi

  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapter-path)
      [[ $# -lt 2 ]] && { echo "Missing value for --adapter-path"; usage; exit 1; }
      ADAPTER_PATH="$2"
      shift 2
      ;;
    --adapter-path=*)
      ADAPTER_PATH="${1#*=}"
      shift
      ;;
    --base-model)
      [[ $# -lt 2 ]] && { echo "Missing value for --base-model"; usage; exit 1; }
      BASE_MODEL="$2"
      shift 2
      ;;
    --base-model=*)
      BASE_MODEL="${1#*=}"
      shift
      ;;
    --output-dir)
      [[ $# -lt 2 ]] && { echo "Missing value for --output-dir"; usage; exit 1; }
      MERGED_OUTPUT_DIR="$2"
      shift 2
      ;;
    --output-dir=*)
      MERGED_OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --trust-remote-code)
      TRUST_REMOTE_CODE=true
      shift
      ;;
    --no-trust-remote-code)
      TRUST_REMOTE_CODE=false
      shift
      ;;
    --bf16)
      USE_BF16=true
      shift
      ;;
    --no-bf16)
      USE_BF16=false
      shift
      ;;
    --safe-serialization)
      SAFE_SERIALIZATION=true
      shift
      ;;
    --no-safe-serialization)
      SAFE_SERIALIZATION=false
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

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  echo "[env] activating virtual environment: $VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif python3 -c "import detectanyllm" >/dev/null 2>&1; then
  echo "[env] using current python environment (detectanyllm detected)"
else
  echo "[env] virtual environment not found and 'detectanyllm' is not installed."
  echo "Run setup first: $ROOT_DIR/scripts/setup_qwen3_lora.sh --model qwen3:4b"
  exit 1
fi

if ! RESOLVED_ADAPTER_PATH="$(resolve_adapter_path "$ADAPTER_PATH")"; then
  echo "LoRA adapter not found or invalid: $ADAPTER_PATH"
  echo "Expected one of:"
  echo "  1) $ADAPTER_PATH/adapter_config.json"
  echo "  2) $ADAPTER_PATH/checkpoint-*/adapter_config.json"
  exit 1
fi

if ! python -c "import detectanyllm" >/dev/null 2>&1; then
  echo "[env] Python package 'detectanyllm' is not installed."
  echo "Please run: pip install -e \"$ROOT_DIR\""
  exit 1
fi

mkdir -p "$MERGED_OUTPUT_DIR"

merge_cmd=(
  python -m detectanyllm.cli merge-lora
  --adapter-path "$RESOLVED_ADAPTER_PATH"
  --base-model "$BASE_MODEL"
  --output-dir "$MERGED_OUTPUT_DIR"
)

if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
  merge_cmd+=(--trust-remote-code)
fi
if [[ "$USE_BF16" == "true" ]]; then
  merge_cmd+=(--bf16)
else
  merge_cmd+=(--no-bf16)
fi
if [[ "$SAFE_SERIALIZATION" == "true" ]]; then
  merge_cmd+=(--safe-serialization)
else
  merge_cmd+=(--no-safe-serialization)
fi

echo "[merge] adapter: $RESOLVED_ADAPTER_PATH"
echo "[merge] base model: $BASE_MODEL"
echo "[merge] output: $MERGED_OUTPUT_DIR"
"${merge_cmd[@]}"

echo "[merge] done. merged model saved to: $MERGED_OUTPUT_DIR"
