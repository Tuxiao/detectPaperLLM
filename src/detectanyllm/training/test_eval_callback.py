"""Periodic test-set metrics callback with dev-tuned threshold."""

from __future__ import annotations

import logging

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class PeriodicTestMetricsCallback(TrainerCallback):
    """Evaluate AUC/MCC/F1 on test set every N steps without tuning on test."""

    def __init__(
        self,
        trainer,
        dev_dataset,
        test_dataset,
        eval_steps: int,
        threshold_objective: str = "mcc",
    ) -> None:
        self.trainer = trainer
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.eval_steps = int(eval_steps)
        self.threshold_objective = threshold_objective
        self._last_eval_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        del args, control, kwargs
        if not state.is_world_process_zero:
            return
        if self.eval_steps <= 0:
            return
        step = int(state.global_step)
        if step <= 0:
            return
        if step % self.eval_steps != 0:
            return
        if step == self._last_eval_step:
            return

        metrics = self.trainer.evaluate_test_with_dev_threshold(
            dev_dataset=self.dev_dataset,
            test_dataset=self.test_dataset,
            threshold_objective=self.threshold_objective,
        )
        logger.info(
            "Step %d test metrics | AUC=%.6f MCC=%.6f F1=%.6f (dev_threshold=%.6f)",
            step,
            metrics["test_auc"],
            metrics["test_mcc"],
            metrics["test_f1"],
            metrics["dev_threshold"],
        )
        self.trainer.log(metrics)
        self._last_eval_step = step

