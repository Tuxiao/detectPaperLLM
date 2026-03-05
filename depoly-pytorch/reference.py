"""Reference clustering utilities for DetectAnyLLM inference.

Self-contained version — KNN probability estimation and stats I/O.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_reference_stats(
    d_h: list[float], d_m: list[float], meta: dict | None = None
) -> dict:
    if not d_h or not d_m:
        raise ValueError("Both D_h and D_m must contain at least one score.")
    return {
        "meta": meta or {},
        "D_h": list(d_h),
        "D_m": list(d_m),
    }


def save_reference_stats(path: str | Path, stats: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)


def load_reference_stats(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    if "D_h" not in stats or "D_m" not in stats:
        raise ValueError("Reference stats file must contain D_h and D_m.")
    return stats


def estimate_probability(
    dc_value: float,
    d_h: list[float],
    d_m: list[float],
    k_neighbors: int = 100,
) -> dict:
    if k_neighbors < 1:
        raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}.")
    merged = d_h + d_m
    if not merged:
        raise ValueError("Reference distributions cannot both be empty.")

    sorted_distances = sorted(abs(d_ref - dc_value) for d_ref in merged)
    k_index = min(k_neighbors, len(sorted_distances)) - 1
    delta = sorted_distances[k_index]

    lower = dc_value - delta
    upper = dc_value + delta
    cnt_m = sum(1 for d in d_m if lower < d < upper)
    cnt_h = sum(1 for d in d_h if lower < d < upper)
    total = cnt_m + cnt_h

    if total == 0:
        p_m = 0.5
        low_confidence = True
    else:
        p_m = cnt_m / total
        low_confidence = False

    return {
        "delta": delta,
        "cnt_m": cnt_m,
        "cnt_h": cnt_h,
        "p_m": p_m,
        "low_confidence": low_confidence,
    }
