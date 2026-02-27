# DetectAnyLLM DDL Toolkit

DetectAnyLLM modular implementation:

1. LoRA fine-tuning using Direct Discrepancy Learning (DDL, Eq.(10))
2. Online re-sampling (`q_phi = f_theta`)
3. Reference clustering inference (`p_m(x)`)

## Project entrypoints

1. `detectanyllm prepare-pairs`
2. `detectanyllm train`
3. `detectanyllm build-ref`
4. `detectanyllm infer`
5. Legacy compatibility: `python ddl_finetune.py ...`

Quick reference manual:
1. `docs/TRAINING_QUICK_REFERENCE.md`

## Qwen3-0.6B quickstart (your current setup)

Assumes:
1. venv exists: `/Users/craig/detectAnyLLM-codex/.venv`
2. model exists: `/Users/craig/detectAnyLLM-codex/models/Qwen3-0.6B`
3. accelerate config exists: `/Users/craig/detectAnyLLM-codex/configs/accelerate_mps.yaml`

### 1) Install package into venv

```bash
cd /Users/craig/detectAnyLLM-codex
source .venv/bin/activate
pip install -e .
```

### 2) Train LoRA (MPS-friendly defaults)

```bash
cd /Users/craig/detectAnyLLM-codex
bash scripts/run_qwen3_lora_example.sh
```

If your data is not already paired, build pair data first:

```bash
python -m detectanyllm.cli prepare-pairs \
  --human-file /path/to/human.jsonl \
  --machine-file /path/to/machine.jsonl \
  --output-file /path/to/pair_train.jsonl \
  --text-field text
```

### 3) Build reference stats and run inference

```bash
cd /Users/craig/detectAnyLLM-codex
bash scripts/run_qwen3_ref_and_infer_example.sh
```

## Manual commands

### Train

```bash
accelerate launch \
  --config_file /Users/craig/detectAnyLLM-codex/configs/accelerate_mps.yaml \
  -m detectanyllm.cli train \
  --model-name-or-path /Users/craig/detectAnyLLM-codex/models/Qwen3-0.6B \
  --train-pairs-file /Users/craig/detectAnyLLM-codex/data/train_pairs.sample.jsonl \
  --output-dir /Users/craig/detectAnyLLM-codex/outputs/qwen3-0.6b-lora \
  --target-modules q_proj,v_proj \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --num-perturb-samples 32 \
  --gamma 100 \
  --sigma-eps 1e-6 \
  --trust-remote-code \
  --no-bf16
```

### Build reference stats

```bash
python -m detectanyllm.cli build-ref \
  --model-path /Users/craig/detectAnyLLM-codex/outputs/qwen3-0.6b-lora \
  --base-model /Users/craig/detectAnyLLM-codex/models/Qwen3-0.6B \
  --human-ref-file /Users/craig/detectAnyLLM-codex/data/human_ref.sample.jsonl \
  --machine-ref-file /Users/craig/detectAnyLLM-codex/data/machine_ref.sample.jsonl \
  --ref-stats-file /Users/craig/detectAnyLLM-codex/outputs/qwen3-0.6b-lora/ref_stats.json \
  --num-perturb-samples 32 \
  --k-neighbors 100 \
  --trust-remote-code \
  --no-bf16
```

### Infer

```bash
python -m detectanyllm.cli infer \
  --model-path /Users/craig/detectAnyLLM-codex/outputs/qwen3-0.6b-lora \
  --base-model /Users/craig/detectAnyLLM-codex/models/Qwen3-0.6B \
  --input-file /Users/craig/detectAnyLLM-codex/data/infer.sample.jsonl \
  --output-file /Users/craig/detectAnyLLM-codex/outputs/qwen3-0.6b-lora/predictions.jsonl \
  --ref-stats-file /Users/craig/detectAnyLLM-codex/outputs/qwen3-0.6b-lora/ref_stats.json \
  --decision-mode pm \
  --k-neighbors 100 \
  --num-perturb-samples 32 \
  --trust-remote-code \
  --no-bf16
```
