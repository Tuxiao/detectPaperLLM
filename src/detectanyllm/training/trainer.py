"""Custom Trainer for DetectAnyLLM DDL objective."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import Trainer

from detectanyllm.training.discrepancy import compute_dc


class DDLTrainer(Trainer):
    """Eq.(10): mean(|d_h| + |gamma - d_m|)."""

    def __init__(
        self,
        *args,
        gamma: float = 100.0,
        num_perturb_samples: int = 32,
        sigma_eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.num_perturb_samples = num_perturb_samples
        self.sigma_eps = sigma_eps
        self._last_logged_step = -1

    def _maybe_log_training_metrics(self, metrics: dict[str, float]) -> None:
        if not self.model.training:
            return
        if self.args.logging_steps <= 0:
            return
        step = int(self.state.global_step)
        should_log = step == 0 or (step % self.args.logging_steps == 0)
        if should_log and step != self._last_logged_step:
            self.log(metrics)
            self._last_logged_step = step

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        del num_items_in_batch

        d_h = compute_dc(
            model=model,
            input_ids=inputs["human_input_ids"],
            attention_mask=inputs["human_attention_mask"],
            num_perturb_samples=self.num_perturb_samples,
            sigma_eps=self.sigma_eps,
        )
        d_m = compute_dc(
            model=model,
            input_ids=inputs["machine_input_ids"],
            attention_mask=inputs["machine_attention_mask"],
            num_perturb_samples=self.num_perturb_samples,
            sigma_eps=self.sigma_eps,
        )

        loss_h = torch.abs(d_h)
        loss_m = torch.abs(self.gamma - d_m)
        loss = (loss_h + loss_m).mean()

        metrics = {
            "mean_d_h": float(d_h.detach().mean().cpu().item()),
            "mean_d_m": float(d_m.detach().mean().cpu().item()),
            "mean_abs_d_h": float(loss_h.detach().mean().cpu().item()),
            "mean_abs_gamma_minus_d_m": float(loss_m.detach().mean().cpu().item()),
        }
        self._maybe_log_training_metrics(metrics)

        if return_outputs:
            return (
                loss,
                {
                    "d_h": d_h.detach(),
                    "d_m": d_m.detach(),
                    "loss_h": loss_h.detach(),
                    "loss_m": loss_m.detach(),
                },
            )
        return loss
