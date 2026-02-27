from __future__ import annotations

import pytest

from detectanyllm.infer.reference_clustering import estimate_probability


def test_reference_clustering_counts_and_probability():
    d_h = [-0.1, 0.0, 0.2]
    d_m = [9.8, 10.0, 10.2]
    result = estimate_probability(dc_value=10.0, d_h=d_h, d_m=d_m, k_neighbors=2)
    assert result["delta"] == pytest.approx(0.2)
    assert result["cnt_m"] == 1
    assert result["cnt_h"] == 0
    assert result["p_m"] == 1.0
    assert result["low_confidence"] is False


def test_reference_clustering_low_confidence_when_window_empty():
    d_h = [0.0]
    d_m = [1.0]
    result = estimate_probability(dc_value=0.0, d_h=d_h, d_m=d_m, k_neighbors=1)
    assert result["cnt_m"] == 0
    assert result["cnt_h"] == 0
    assert result["p_m"] == 0.5
    assert result["low_confidence"] is True
