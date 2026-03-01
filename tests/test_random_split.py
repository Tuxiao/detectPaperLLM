from __future__ import annotations

from detectanyllm.data.io import split_rows_random


def test_split_rows_random_is_reproducible():
    rows = [{"id": str(i)} for i in range(10)]
    s1 = split_rows_random(rows=rows, dev_ratio=0.2, test_ratio=0.2, seed=7)
    s2 = split_rows_random(rows=rows, dev_ratio=0.2, test_ratio=0.2, seed=7)
    assert s1.train_rows == s2.train_rows
    assert s1.dev_rows == s2.dev_rows
    assert s1.test_rows == s2.test_rows


def test_split_rows_random_respects_ratios_with_nonempty_splits():
    rows = [{"id": str(i)} for i in range(10)]
    split = split_rows_random(rows=rows, dev_ratio=0.2, test_ratio=0.2, seed=42)
    assert len(split.train_rows) == 6
    assert len(split.dev_rows) == 2
    assert len(split.test_rows) == 2

