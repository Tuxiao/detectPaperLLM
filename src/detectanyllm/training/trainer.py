"""Custom Trainer for DetectAnyLLM DDL objective."""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer

from detectanyllm.training.discrepancy import compute_dc
from detectanyllm.training.classification_metrics import (
    choose_threshold_from_dev,
    metrics_at_threshold,
)


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

    @torch.no_grad()
    def collect_discrepancy_scores(self, dataset: Dataset) -> tuple[list[float], list[int]]:
        if dataset is None:
            raise ValueError("dataset cannot be None.")

        batch_size = int(
            self.args.per_device_eval_batch_size or self.args.per_device_train_batch_size
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )
        model = self.model
        was_training = model.training
        model.eval()
        scores: list[float] = []
        labels: list[int] = []

        for batch in dataloader:
            d_h = compute_dc(
                model=model,
                input_ids=batch["human_input_ids"],
                attention_mask=batch["human_attention_mask"],
                num_perturb_samples=self.num_perturb_samples,
                sigma_eps=self.sigma_eps,
            )
            d_m = compute_dc(
                model=model,
                input_ids=batch["machine_input_ids"],
                attention_mask=batch["machine_attention_mask"],
                num_perturb_samples=self.num_perturb_samples,
                sigma_eps=self.sigma_eps,
            )
            scores.extend(d_h.detach().float().cpu().tolist())
            labels.extend([0] * len(d_h))
            scores.extend(d_m.detach().float().cpu().tolist())
            labels.extend([1] * len(d_m))

        if was_training:
            model.train()

        return scores, labels

    def evaluate_test_with_dev_threshold(
        self,
        dev_dataset: Dataset,
        test_dataset: Dataset,
        threshold_objective: str = "mcc",
    ) -> dict[str, float]:
        dev_scores, dev_labels = self.collect_discrepancy_scores(dev_dataset)
        test_scores, test_labels = self.collect_discrepancy_scores(test_dataset)

        dev_threshold, dev_best_metric = choose_threshold_from_dev(
            scores=dev_scores,
            labels=dev_labels,
            objective=threshold_objective,
        )
        test_metrics = metrics_at_threshold(
            scores=test_scores,
            labels=test_labels,
            threshold=dev_threshold,
        )

        return {
            "dev_threshold": dev_threshold,
            f"dev_best_{threshold_objective}": dev_best_metric,
            "test_auc": test_metrics["auc"],
            "test_mcc": test_metrics["mcc"],
            "test_f1": test_metrics["f1"],
        }

    def evaluate_dev_split(
        self,
        dev_dataset: Dataset,
        threshold_objective: str = "mcc",
    ) -> dict[str, float]:
        dev_scores, dev_labels = self.collect_discrepancy_scores(dev_dataset)
        dev_threshold, dev_best_metric = choose_threshold_from_dev(
            scores=dev_scores,
            labels=dev_labels,
            objective=threshold_objective,
        )
        metrics = metrics_at_threshold(
            scores=dev_scores,
            labels=dev_labels,
            threshold=dev_threshold,
        )
        return {
            "dev_auc": metrics["auc"],
            "dev_mcc": metrics["mcc"],
            "dev_f1": metrics["f1"],
            "dev_threshold": dev_threshold,
            f"dev_best_{threshold_objective}": dev_best_metric,
        }

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
