from __future__ import annotations

import math

import pytest
from transformers import TrainingArguments

pytest.importorskip("peft")

from detectanyllm.config import ModelConfig
from detectanyllm.data.collator import PairDataCollator
from detectanyllm.data.dataset import DDLPairDataset
from detectanyllm.modeling.lora import build_lora_model
from detectanyllm.training.trainer import DDLTrainer


def test_training_smoke_runs_one_step(tiny_model_dir, pair_train_file, tmp_path):
    model_cfg = ModelConfig(
        model_name_or_path=str(tiny_model_dir),
        trust_remote_code=False,
        target_modules="c_attn,c_proj",
    )
    model, tokenizer = build_lora_model(model_cfg, use_bf16=False)
    train_dataset = DDLPairDataset(
        data_file=pair_train_file,
        tokenizer=tokenizer,
        max_length=32,
        human_field="human",
        machine_field="machine",
    )

    training_args = TrainingArguments(
        output_dir=(tmp_path / "out").as_posix(),
        max_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_steps=10,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DDLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=PairDataCollator(tokenizer),
        gamma=100.0,
        num_perturb_samples=8,
        sigma_eps=1e-6,
    )
    result = trainer.train()
    assert math.isfinite(result.training_loss)
