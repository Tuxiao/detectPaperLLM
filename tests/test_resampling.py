from __future__ import annotations

import torch

from detectanyllm.training.discrepancy import compute_dc_from_logits


def test_resampling_is_reproducible_with_seed():
    logits = torch.randn(1, 7, 11)
    input_ids = torch.randint(low=0, high=11, size=(1, 7))
    attention_mask = torch.ones_like(input_ids)

    torch.manual_seed(1234)
    dc1 = compute_dc_from_logits(
        logits=logits,
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_perturb_samples=8,
        sigma_eps=1e-6,
    )

    torch.manual_seed(1234)
    dc2 = compute_dc_from_logits(
        logits=logits,
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_perturb_samples=8,
        sigma_eps=1e-6,
    )
    torch.testing.assert_close(dc1, dc2)


def test_higher_k_reduces_dc_variance_across_random_seeds():
    logits = torch.randn(1, 9, 13)
    input_ids = torch.randint(low=0, high=13, size=(1, 9))
    attention_mask = torch.ones_like(input_ids)

    vals_k8 = []
    vals_k64 = []
    for seed in range(40):
        torch.manual_seed(seed)
        vals_k8.append(
            compute_dc_from_logits(
                logits=logits,
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_perturb_samples=8,
                sigma_eps=1e-6,
            ).item()
        )
        torch.manual_seed(seed)
        vals_k64.append(
            compute_dc_from_logits(
                logits=logits,
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_perturb_samples=64,
                sigma_eps=1e-6,
            ).item()
        )

    std_k8 = torch.tensor(vals_k8).std(unbiased=False).item()
    std_k64 = torch.tensor(vals_k64).std(unbiased=False).item()
    assert std_k64 <= std_k8
