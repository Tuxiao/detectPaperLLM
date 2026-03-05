# DetectAnyLLM: 通用且鲁棒的跨领域/模型机器生成文本检测框架

## 模型的背景和论文

随着大语言模型（LLMs）的飞速发展，如何准确识别“机器生成的文本”已成为人工智能安全领域的重要议题。现有的检测方法在面对复杂现实场景时往往表现欠佳：零样本（Zero-shot）检测器过度依赖评分模型的输出分布，而基于训练（Training-based）的检测器则容易对训练数据过拟合，导致泛化能力受限。

**DetectAnyLLM** 提出了 **直接差异学习（Direct Discrepancy Learning, DDL）** 策略。DDL 不再让评分模型去模仿生成器的分布，而是直接针对检测任务的判别指标进行优化。这种“任务导向”的优化方式使模型能够捕捉到检测任务的核心语义特征，从而显著提升了模型的稳健性和泛化性。

配套论文：[DetectAnyLLM: Towards Generalizable and Robust Detection of Machine-Generated Text Across Domains and Models](file:///Users/craig/detectPaperLLM/DetectAnyLLM.md)

### 核心亮点
- **DDL 优化策略**：通过直接优化条件概率差异信号，提升检测器的泛化能力。
- **MIRAGE 基准测试**：在包含 17 种先进 LLMs（包括 13 种闭源模型）和 5 个领域数据的最全面基准测试中，DetectAnyLLM 刷新了多项指标。
- **高效训练**：相比 SPO 等现有方法，DDL 减少了约 30% 的训练时间和 35% 的内存消耗。

---

## 模型的调用方法和部分API

DetectAnyLLM 提供了统一的命令行界面（CLI）来支持整个检测流程。

### 1. 安装环境
```bash
pip install -e .
```

### 2. 推理检测 (infer)
使用训练好的模型对一段文本进行检测：
```bash
python -m detectanyllm.cli infer \
  --model-path /path/to/lora_model \
  --base-model /path/to/base_model \
  --input-file input.jsonl \
  --output-file predictions.jsonl \
  --ref-stats-file ref_stats.json \
  --decision-mode pm \
  --k-neighbors 100 \
  --num-perturb-samples 32 \
  --trust-remote-code
```

### 3. 构建参考分布 (build-ref)
为了实现更高精度的 Reference Clustering 检测，首先需要聚合参考文本的统计分布：
```bash
python -m detectanyllm.cli build-ref \
  --model-path /path/to/lora_model \
  --base-model /path/to/base_model \
  --human-ref-file human.jsonl \
  --machine-ref-file machine.jsonl \
  --ref-stats-file ref_stats.json \
  --num-perturb-samples 32 \
  --k-neighbors 100
```

---

## 模型的前处理

为了确保检测的纯净性和准确性，DetectAnyLLM 在处理数据（如 MIRAGE 基准数据集）时遵循以下关键步骤：

1. **特殊符号清理**：移除文本中所有的 `\n` 和 `\r` 字符，防止检测器通过排版特征（如换行符）这种“捷径”来识别机器文本。
2. **长度控制**：过滤出长度在 100-200 词之间的文本段落，以消除文本长度对检测结果带来的偏差。
3. **数据配对 (prepare-pairs)**：在训练阶段，需要将人类撰写的文本与机器生成的对应版本配对：
   ```bash
   python -m detectanyllm.cli prepare-pairs \
     --human-file human.jsonl \
     --machine-file machine.jsonl \
     --output-file pairs.jsonl \
     --text-field text
   ```

---

## 模型支持的推理任务

DetectAnyLLM 支持针对多种生成模式的**二分类检测任务**（人类 vs 机器）：

- **Generate (直接生成)**：检测由 AI 根据前缀直接生成的文本（如由前 30 个 token 补全的段落）。
- **Polish (润色检测)**：检测由 AI 修改、优化过的人类原始文本。
- **Rewrite (重写检测)**：检测由 AI 保持原意但完全重述的文本。

该框架在 Disjoint-Input Generation (DIG) 和 Shared-Input Generation (SIG) 两种评估场景下均表现优异，尤其擅长应对闭源商业模型生成的模拟人类风格的文本。

---

## 模型的训练

DetectAnyLLM 的训练基于 **DDL (Direct Discrepancy Learning)**，使其能够摆脱“语言模型”的身份，真正进化为“检测器”。

### 训练配置建议
- **优化算法**：DDL (Eq. 10)，通过移除冗余的 KL 正则化，专注于降低人类文本差异值、提高机器文本差异值。
- **参数配置**：
  - **超参数 $\gamma$**：建议设置为 100（该值对性能具有较强的鲁棒性）。
  - **学习率**：0.0001
  - **训练周期 (Epochs)**：通常 2-5 个 Epochs 即可收敛。
- **LoRA 技术**：
  - Rank: 8
  - Alpha: 32
  - Target Modules: 推荐注入到 `q_proj`, `v_proj` 等注意力层。

### 启动训练命令
```bash
accelerate launch -m detectanyllm.cli train \
  --model-name-or-path /path/to/base_model \
  --train-pairs-file train_pairs.jsonl \
  --output-dir ./outputs/detectanyllm-lora \
  --target-modules q_proj,v_proj \
  --gamma 100 \
  --no-bf16  # 如果在不支持 bf16 的硬件上运行
```
