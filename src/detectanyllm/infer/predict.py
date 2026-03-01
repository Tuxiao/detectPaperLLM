"""Inference helpers for building reference stats and predictions."""

from __future__ import annotations

from pathlib import Path

import torch

from detectanyllm.data.io import iter_jsonl, read_text_field, write_jsonl
from detectanyllm.infer.reference_clustering import build_reference_stats, estimate_probability
from detectanyllm.training.discrepancy import compute_dc


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _encode_text(tokenizer, text: str, max_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return input_ids, attention_mask


@torch.no_grad()
def compute_dc_for_text(
    model,
    tokenizer,
    text: str,
    max_length: int,
    num_perturb_samples: int,
    sigma_eps: float,
    device: torch.device,
) -> float:
    input_ids, attention_mask = _encode_text(tokenizer, text, max_length=max_length, device=device)
    if input_ids.shape[1] < 2:
        raise ValueError("Text is too short after tokenization (<2 tokens).")
    dc_value = compute_dc(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_perturb_samples=num_perturb_samples,
        sigma_eps=sigma_eps,
    )
    return float(dc_value.item())


@torch.no_grad()
def build_reference_distributions(
    model,
    tokenizer,
    human_ref_file: str,
    machine_ref_file: str,
    text_field: str = "text",
    max_length: int = 512,
    num_perturb_samples: int = 32,
    sigma_eps: float = 1e-6,
    device: torch.device | None = None,
    meta: dict | None = None,
) -> dict:
    device = device or _resolve_device()

    d_h: list[float] = []
    d_m: list[float] = []
    for record in iter_jsonl(human_ref_file):
        text = read_text_field(record, text_field)
        d_h.append(
            compute_dc_for_text(
                model=model,
                tokenizer=tokenizer,
                text=text,
                max_length=max_length,
                num_perturb_samples=num_perturb_samples,
                sigma_eps=sigma_eps,
                device=device,
            )
        )
    for record in iter_jsonl(machine_ref_file):
        text = read_text_field(record, text_field)
        d_m.append(
            compute_dc_for_text(
                model=model,
                tokenizer=tokenizer,
                text=text,
                max_length=max_length,
                num_perturb_samples=num_perturb_samples,
                sigma_eps=sigma_eps,
                device=device,
            )
        )

    return build_reference_stats(d_h=d_h, d_m=d_m, meta=meta)


@torch.no_grad()
def infer_file(
    model,
    tokenizer,
    input_file: str,
    output_file: str | None = None,
    *,
    text_field: str = "text",
    max_length: int = 512,
    num_perturb_samples: int = 32,
    sigma_eps: float = 1e-6,
    decision_mode: str = "pm",
    threshold: float = 50.0,
    ref_stats: dict | None = None,
    k_neighbors: int = 100,
    device: torch.device | None = None,
) -> list[dict]:
    device = device or _resolve_device()
    records_out: list[dict] = []

    if decision_mode == "pm" and ref_stats is None:
        raise ValueError("decision_mode='pm' requires ref_stats.")
    if decision_mode not in {"pm", "threshold"}:
        raise ValueError(f"Unsupported decision_mode: {decision_mode}")

    for record in iter_jsonl(input_file):
        text = read_text_field(record, text_field)
        d_c = compute_dc_for_text(
            model=model,
            tokenizer=tokenizer,
            text=text,
            max_length=max_length,
            num_perturb_samples=num_perturb_samples,
            sigma_eps=sigma_eps,
            device=device,
        )

        p_m = None
        delta = None
        cnt_h = None
        cnt_m = None
        low_confidence = None

        if ref_stats is not None:
            estimated = estimate_probability(
                dc_value=d_c,
                d_h=ref_stats["D_h"],
                d_m=ref_stats["D_m"],
                k_neighbors=k_neighbors,
            )
            p_m = estimated["p_m"]
            delta = estimated["delta"]
            cnt_h = estimated["cnt_h"]
            cnt_m = estimated["cnt_m"]
            low_confidence = estimated["low_confidence"]

        if decision_mode == "pm":
            label_pred = "MACHINE" if p_m is not None and p_m >= 0.5 else "HUMAN"
        else:
            label_pred = "MACHINE" if d_c > threshold else "HUMAN"

        records_out.append(
            {
                "text": text,
                "d_c": d_c,
                "p_m": p_m,
                "delta": delta,
                "cnt_h": cnt_h,
                "cnt_m": cnt_m,
                "label_pred": label_pred,
                "low_confidence": low_confidence,
            }
        )

    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_file, records_out)
    return records_out
