"""Conditional discrepancy computation for DetectAnyLLM inference.

Self-contained version — combines logic from predict.py and discrepancy.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
    """Compute conditional discrepancy d_c for a single text.

    This is the main entry point.  Tokenizes the text, runs the model
    forward pass, then computes d_c via Eq.(4)(5) with vectorized
    online resampling.
    """
    # ---------- tokenize ----------
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    if input_ids.shape[1] < 2:
        raise ValueError("Text is too short after tokenization (<2 tokens).")

    # ---------- forward pass ----------
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = device
    input_ids = input_ids.to(model_device)
    attention_mask = attention_mask.to(model_device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits

    # ---------- compute d_c ----------
    return float(
        _compute_dc_from_logits(
            logits=logits,
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_perturb_samples=num_perturb_samples,
            sigma_eps=sigma_eps,
        ).item()
    )


def _compute_dc_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_perturb_samples: int = 32,
    sigma_eps: float = 1e-6,
) -> torch.Tensor:
    """Compute d_c using Eq.(4)(5) with vectorized online resampling."""

    if num_perturb_samples < 1:
        raise ValueError("num_perturb_samples must be >= 1.")
    if sigma_eps <= 0:
        raise ValueError("sigma_eps must be > 0.")
    if input_ids.shape[1] < 2:
        raise ValueError("Need at least 2 tokens per sequence for causal likelihood.")

    if input_ids.device != logits.device:
        input_ids = input_ids.to(logits.device)
    if attention_mask.device != logits.device:
        attention_mask = attention_mask.to(logits.device)

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].to(dtype=shift_logits.dtype)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    original_seq_logp = (token_log_probs * shift_mask).sum(dim=-1)

    seq_len = input_ids.shape[1] - 1
    sample_shape = (num_perturb_samples,)

    sampling_logits = log_probs
    if log_probs.device.type == "mps":
        # Work around an MPS segfault in torch.distributions.Categorical.sample.
        sampling_logits = log_probs.to("cpu")

    sampled_tokens = torch.distributions.Categorical(logits=sampling_logits).sample(
        sample_shape
    )
    if sampled_tokens.device != log_probs.device:
        sampled_tokens = sampled_tokens.to(log_probs.device)

    batch_size = input_ids.shape[0]
    sampled_tokens = sampled_tokens.view(num_perturb_samples, batch_size, seq_len)
    expanded_log_probs = log_probs.unsqueeze(0).expand(num_perturb_samples, -1, -1, -1)

    sampled_token_logp = expanded_log_probs.gather(
        dim=-1, index=sampled_tokens.unsqueeze(-1)
    ).squeeze(-1)
    sampled_seq_logp = (sampled_token_logp * shift_mask.unsqueeze(0)).sum(dim=-1)

    mu_tilde = sampled_seq_logp.mean(dim=0)
    sigma_tilde = sampled_seq_logp.std(dim=0, unbiased=False)
    dc = (original_seq_logp - mu_tilde) / (sigma_tilde + sigma_eps)
    return dc
