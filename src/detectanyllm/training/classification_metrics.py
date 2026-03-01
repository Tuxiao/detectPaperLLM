"""Binary classification metrics for discrepancy-based evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int


def confusion_from_scores(
    scores: list[float],
    labels: list[int],
    threshold: float,
) -> Confusion:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        pred = 1 if score >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    return Confusion(tp=tp, fp=fp, tn=tn, fn=fn)


def mcc_from_confusion(confusion: Confusion) -> float:
    num = confusion.tp * confusion.tn - confusion.fp * confusion.fn
    den = math.sqrt(
        (confusion.tp + confusion.fp)
        * (confusion.tp + confusion.fn)
        * (confusion.tn + confusion.fp)
        * (confusion.tn + confusion.fn)
    )
    if den == 0:
        return 0.0
    return float(num / den)


def f1_from_confusion(confusion: Confusion) -> float:
    den = 2 * confusion.tp + confusion.fp + confusion.fn
    if den == 0:
        return 0.0
    return float((2 * confusion.tp) / den)


def roc_auc_from_scores(scores: list[float], labels: list[int]) -> float:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    if not scores:
        raise ValueError("scores cannot be empty.")

    n_pos = sum(1 for label in labels if label == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    sum_pos_ranks = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        pos_in_tie = sum(label for _, label in pairs[i:j])
        sum_pos_ranks += pos_in_tie * avg_rank
        i = j

    auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def choose_threshold_from_dev(
    scores: list[float],
    labels: list[int],
    objective: str = "mcc",
) -> tuple[float, float]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    if not scores:
        raise ValueError("scores cannot be empty.")
    if objective not in {"mcc", "f1"}:
        raise ValueError("objective must be 'mcc' or 'f1'.")

    unique_thresholds = sorted(set(scores))
    best_threshold = unique_thresholds[0]
    best_score = float("-inf")

    for threshold in unique_thresholds:
        confusion = confusion_from_scores(scores=scores, labels=labels, threshold=threshold)
        current = (
            mcc_from_confusion(confusion)
            if objective == "mcc"
            else f1_from_confusion(confusion)
        )
        if current > best_score:
            best_score = current
            best_threshold = threshold

    return float(best_threshold), float(best_score)


def metrics_at_threshold(
    scores: list[float],
    labels: list[int],
    threshold: float,
) -> dict[str, float]:
    confusion = confusion_from_scores(scores=scores, labels=labels, threshold=threshold)
    return {
        "auc": roc_auc_from_scores(scores=scores, labels=labels),
        "mcc": mcc_from_confusion(confusion),
        "f1": f1_from_confusion(confusion),
    }
