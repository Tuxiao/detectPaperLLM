"""Model and LoRA utilities."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from detectanyllm.config import ModelConfig, parse_target_modules


def _preferred_dtype(use_bf16: bool) -> torch.dtype | None:
    if use_bf16 and torch.cuda.is_available():
        return torch.bfloat16
    return None


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_base_model(adapter_path: Path, base_model: str | None) -> str:
    if base_model:
        return base_model

    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise ValueError(
            "base_model is required when adapter_config.json is missing from adapter_path."
        )

    with adapter_config_path.open("r", encoding="utf-8") as handle:
        adapter_cfg = json.load(handle)
    resolved_base = adapter_cfg.get("base_model_name_or_path")
    if not resolved_base:
        raise ValueError("Could not resolve base model from adapter config.")
    return resolved_base


def _has_local_tokenizer(model_path: Path) -> bool:
    return any(
        (model_path / name).exists()
        for name in (
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "spiece.model",
        )
    )


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must have either pad_token or eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_lora_model(model_config: ModelConfig, use_bf16: bool = True):
    tokenizer = load_tokenizer(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=_preferred_dtype(use_bf16),
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        target_modules=parse_target_modules(model_config.target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def load_model_for_inference(
    model_path: str,
    base_model: str | None = None,
    trust_remote_code: bool = False,
    use_bf16: bool = True,
):
    model_path_obj = Path(model_path)
    adapter_config_path = model_path_obj / "adapter_config.json"
    tokenizer_source = model_path

    if adapter_config_path.exists():
        with adapter_config_path.open("r", encoding="utf-8") as handle:
            adapter_cfg = json.load(handle)
        resolved_base = base_model or adapter_cfg.get("base_model_name_or_path")
        if not resolved_base:
            raise ValueError("Could not resolve base model from adapter config.")
        model = AutoModelForCausalLM.from_pretrained(
            resolved_base,
            trust_remote_code=trust_remote_code,
            torch_dtype=_preferred_dtype(use_bf16),
        )
        model = PeftModel.from_pretrained(model, model_path)
        if not (model_path_obj / "tokenizer_config.json").exists():
            tokenizer_source = resolved_base
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=_preferred_dtype(use_bf16),
        )
        if not (model_path_obj / "tokenizer_config.json").exists():
            if base_model is None:
                raise ValueError(
                    "Tokenizer not found in model_path and --base-model not provided."
                )
            tokenizer_source = base_model

    tokenizer = load_tokenizer(tokenizer_source, trust_remote_code=trust_remote_code)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    device = _resolve_device()
    model.to(device)
    model.eval()
    return model, tokenizer, device


def merge_lora_adapter(
    adapter_path: str,
    output_dir: str,
    base_model: str | None = None,
    trust_remote_code: bool = False,
    use_bf16: bool = True,
    safe_serialization: bool = True,
) -> tuple[str, str]:
    adapter_path_obj = Path(adapter_path)
    if not adapter_path_obj.exists():
        raise FileNotFoundError(f"adapter_path does not exist: {adapter_path}")

    resolved_base = _resolve_base_model(adapter_path_obj, base_model)

    base = AutoModelForCausalLM.from_pretrained(
        resolved_base,
        trust_remote_code=trust_remote_code,
        torch_dtype=_preferred_dtype(use_bf16),
    )
    lora_model = PeftModel.from_pretrained(base, adapter_path)
    merged_model = lora_model.merge_and_unload()

    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        output_dir_obj.as_posix(),
        safe_serialization=safe_serialization,
    )

    tokenizer_source = (
        adapter_path_obj.as_posix() if _has_local_tokenizer(adapter_path_obj) else resolved_base
    )
    tokenizer = load_tokenizer(tokenizer_source, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(output_dir_obj.as_posix())

    return resolved_base, output_dir_obj.as_posix()
