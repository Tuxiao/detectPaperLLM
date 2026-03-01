from __future__ import annotations

import pytest

from detectanyllm.training.classification_metrics import (
    choose_threshold_from_dev,
    metrics_at_threshold,
    roc_auc_from_scores,
)


def test_roc_auc_perfect_ranking():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert roc_auc_from_scores(scores, labels) == pytest.approx(1.0)


def test_choose_threshold_from_dev_with_mcc():
    scores = [0.1, 0.2, 0.9, 1.0]
    labels = [0, 0, 1, 1]
    threshold, best_mcc = choose_threshold_from_dev(scores, labels, objective="mcc")
    assert 0.2 <= threshold <= 0.9
    assert best_mcc == pytest.approx(1.0)


def test_metrics_at_threshold_returns_auc_mcc_f1():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    metrics = metrics_at_threshold(scores=scores, labels=labels, threshold=0.5)
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)

