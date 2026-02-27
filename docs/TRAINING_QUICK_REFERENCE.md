# DetectAnyLLM 训练手册（快速参考）

适用场景：
1. 基于 `Qwen3-0.6B` 做 LoRA 微调。
2. 训练目标为 DetectAnyLLM 的 DDL（Direct Discrepancy Learning）损失。
3. 本地环境使用 Apple Silicon + MPS（`accelerate_mps.yaml`）。

## 1. 一次性准备

```bash
cd /Users/craig/detectAnyLLM-codex
source .venv/bin/activate
pip install -e .
```

环境前提（你当前已具备）：
1. 虚拟环境：`/Users/craig/detectAnyLLM-codex/.venv`
2. 模型目录：`/Users/craig/detectAnyLLM-codex/models/Qwen3-0.6B`
3. accelerate 配置：`/Users/craig/detectAnyLLM-codex/configs/accelerate_mps.yaml`

## 2. 训练数据格式与长度要求

训练输入必须是 JSONL，每行一个 pair：

```json
{"human":"...","machine":"..."}
```

建议长度（论文常用过滤区间）：
1. `human`：100-200 words
2. `machine`：100-200 words

快速检查长度：

```bash
cd /Users/craig/detectAnyLLM-codex
jq -r '[ (.human|[scan("\\S+")]|length), (.machine|[scan("\\S+")]|length) ] | @tsv' data/train_pairs.example.jsonl \
  | awk '{ok=(($1>=100 && $1<=200 && $2>=100 && $2<=200)? "OK":"FAIL"); print NR"\thuman="$1"\tmachine="$2"\t"ok}'
```

如果你只有 `human.jsonl` 和 `machine.jsonl`（字段为 `text`），先配对：

```bash
python -m detectanyllm.cli prepare-pairs \
  --human-file /path/to/human.jsonl \
  --machine-file /path/to/machine.jsonl \
  --output-file /path/to/pair_train.jsonl \
  --text-field text
```

## 3. 启动训练（推荐脚本）

```bash
cd /Users/craig/detectAnyLLM-codex
bash scripts/run_qwen3_lora_example.sh
```

默认训练关键参数：
1. `gamma=100`
2. `num_perturb_samples=32`
3. `sigma_eps=1e-6`
4. `lora_r=8`, `lora_alpha=32`, `lora_dropout=0.1`

默认输出目录：
1. `/Users/craig/detectAnyLLM-codex/outputs/qwen3-0.6b-lora`

## 4. 启动训练（手动命令）

```bash
cd /Users/craig/detectAnyLLM-codex
source .venv/bin/activate

accelerate launch \
  --config_file /Users/craig/detectAnyLLM-codex/configs/accelerate_mps.yaml \
  -m detectanyllm.cli train \
  --model-name-or-path /Users/craig/detectAnyLLM-codex/models/Qwen3-0.6B \
  --train-pairs-file /Users/craig/detectAnyLLM-codex/data/train_pairs.example.jsonl \
  --output-dir /Users/craig/detectAnyLLM-codex/outputs/qwen3-0.6b-lora \
  --target-modules q_proj,v_proj \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --max-length 256 \
  --logging-steps 1 \
  --save-steps 20 \
  --save-total-limit 2 \
  --gamma 100 \
  --num-perturb-samples 32 \
  --sigma-eps 1e-6 \
  --trust-remote-code \
  --no-bf16
```

说明：
1. 在 MPS 环境建议 `--no-bf16`。
2. `--target-modules` 可按模型结构调整，Qwen3-0.6B 默认 `q_proj,v_proj` 可用。

## 5. 训练期间可视化（实时）

训练命令启动后，会在 `--output-dir` 下自动生成：
1. `training_dashboard.html`（可视化页面）
2. `training_live_metrics.json`（实时指标与 epoch 耗时）

例如默认输出目录下：
1. `/Users/craig/detectAnyLLM-codex/outputs/qwen3-0.6b-lora/training_dashboard.html`
2. `/Users/craig/detectAnyLLM-codex/outputs/qwen3-0.6b-lora/training_live_metrics.json`

推荐用本地 HTTP 打开页面（避免直接 `file://` 导致浏览器 fetch 受限）：

```bash
cd /Users/craig/detectAnyLLM-codex/outputs/qwen3-0.6b-lora
python -m http.server 8765
```

然后在浏览器打开：
`http://localhost:8765/training_dashboard.html`

页面默认每 2 秒刷新，可查看：
1. `loss / train_loss`
2. `grad_norm`
3. `mean_d_h / mean_d_m`
4. `global_step / epoch / train_runtime_seconds`
5. 每个 epoch 的开始时间、结束时间、耗时（秒）

训练结束后快速打印每个 epoch 耗时：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("/Users/craig/detectAnyLLM-codex/outputs/qwen3-0.6b-lora/training_live_metrics.json")
obj = json.loads(p.read_text())
for row in obj.get("epoch_timings", []):
    print(f"epoch={row.get('epoch')} duration_seconds={row.get('duration_seconds')}")
PY
```

## 6. 训练后：构建参考分布与推理

先构建参考统计（`D_h`, `D_m`）：

```bash
python -m detectanyllm.cli build-ref \
  --model-path /Users/craig/detectAnyLLM-codex/outputs/qwen3-0.6b-lora \
  --base-model /Users/craig/detectAnyLLM-codex/models/Qwen3-0.6B \
  --human-ref-file /Users/craig/detectAnyLLM-codex/data/human_ref.sample.jsonl \
  --machine-ref-file /Users/craig/detectAnyLLM-codex/data/machine_ref.sample.jsonl \
  --ref-stats-file /Users/craig/detectAnyLLM-codex/outputs/qwen3-0.6b-lora/ref_stats.json \
  --k-neighbors 100 \
  --num-perturb-samples 32 \
  --sigma-eps 1e-6 \
  --trust-remote-code \
  --no-bf16
```

再推理（默认 `pm` 模式，输出 `p_m`）：

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
  --sigma-eps 1e-6 \
  --trust-remote-code \
  --no-bf16
```

推理输出字段（每行）：
1. `text`
2. `d_c`
3. `p_m`
4. `delta`
5. `cnt_h`
6. `cnt_m`
7. `label_pred`
8. `low_confidence`

## 7. 常见问题排查

1. 报错 `package detectanyllm not found`：
   `source .venv/bin/activate && pip install -e .`
2. 报错模型路径不存在：
   检查 `models/Qwen3-0.6B/model.safetensors` 是否存在。
3. 训练数据被大量过滤：
   检查文本长度、空值和字段名（`human/machine`）。
4. MPS 不稳定或显存不足：
   降低 `--max-length`、`--per-device-train-batch-size`，保持 `--no-bf16`。
5. `decision_mode=pm` 报错：
   需要先执行 `build-ref` 并提供 `--ref-stats-file`。

## 8. 最小可复现流程（从零到可预测）

```bash
cd /Users/craig/detectAnyLLM-codex
source .venv/bin/activate
pip install -e .
bash scripts/run_qwen3_lora_example.sh
bash scripts/run_qwen3_ref_and_infer_example.sh
```
