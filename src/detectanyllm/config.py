"""Configuration objects and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


def parse_target_modules(target_modules: str | Sequence[str]) -> list[str]:
    if isinstance(target_modules, str):
        modules = [m.strip() for m in target_modules.split(",") if m.strip()]
    else:
        modules = [m.strip() for m in target_modules if m.strip()]
    if not modules:
        raise ValueError("target_modules cannot be empty.")
    return modules


@dataclass
class ModelConfig:
    model_name_or_path: str
    trust_remote_code: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: str = "q_proj,k_proj,v_proj,o_proj"

    def __post_init__(self) -> None:
        _require_positive("lora_r", self.lora_r)
        _require_positive("lora_alpha", self.lora_alpha)
        if not 0 <= self.lora_dropout < 1:
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}.")
        parse_target_modules(self.target_modules)


@dataclass
class DataConfig:
    train_pairs_file: str
    validation_pairs_file: str | None = None
    test_pairs_file: str | None = None
    human_field: str = "human"
    machine_field: str = "machine"
    text_field: str = "text"
    max_length: int = 512

    def __post_init__(self) -> None:
        _require_positive("max_length", self.max_length)
        if not self.human_field:
            raise ValueError("human_field cannot be empty.")
        if not self.machine_field:
            raise ValueError("machine_field cannot be empty.")
        if not self.text_field:
            raise ValueError("text_field cannot be empty.")


@dataclass
class DDLConfig:
    gamma: float = 100.0
    num_perturb_samples: int = 32
    sigma_eps: float = 1e-6

    def __post_init__(self) -> None:
        _require_positive("num_perturb_samples", self.num_perturb_samples)
        _require_positive("sigma_eps", self.sigma_eps)


@dataclass
class ReferenceConfig:
    k_neighbors: int = 100

    def __post_init__(self) -> None:
        _require_positive("k_neighbors", self.k_neighbors)


@dataclass
class TrainRuntimeConfig:
    output_dir: str
    learning_rate: float = 1e-4
    num_train_epochs: float = 5.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 42
    bf16: bool = True
    save_total_limit: int = 2
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05

    def __post_init__(self) -> None:
        _require_positive("learning_rate", self.learning_rate)
        _require_positive("num_train_epochs", self.num_train_epochs)
        _require_positive("per_device_train_batch_size", self.per_device_train_batch_size)
        _require_positive("gradient_accumulation_steps", self.gradient_accumulation_steps)
        _require_positive("logging_steps", self.logging_steps)
        _require_positive("save_steps", self.save_steps)
        _require_positive("save_total_limit", self.save_total_limit)
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}.")
        if not self.output_dir:
            raise ValueError("output_dir cannot be empty.")

    def ensure_output_dir(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
