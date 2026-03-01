from __future__ import annotations

from detectanyllm.data.io import split_rows_by_group_id


def test_split_rows_by_group_id_has_no_overlap():
    rows = [
        {"doc_id": "a", "human": "h1", "machine": "m1"},
        {"doc_id": "a", "human": "h2", "machine": "m2"},
        {"doc_id": "b", "human": "h3", "machine": "m3"},
        {"doc_id": "c", "human": "h4", "machine": "m4"},
        {"doc_id": "d", "human": "h5", "machine": "m5"},
        {"doc_id": "e", "human": "h6", "machine": "m6"},
    ]

    result = split_rows_by_group_id(
        rows=rows,
        group_id_field="doc_id",
        dev_ratio=0.2,
        test_ratio=0.2,
        seed=123,
    )

    assert result.train_groups
    assert result.dev_groups
    assert result.test_groups
    assert result.train_groups.isdisjoint(result.dev_groups)
    assert result.train_groups.isdisjoint(result.test_groups)
    assert result.dev_groups.isdisjoint(result.test_groups)

    for row in result.train_rows:
        assert row["doc_id"] in result.train_groups
    for row in result.dev_rows:
        assert row["doc_id"] in result.dev_groups
    for row in result.test_rows:
        assert row["doc_id"] in result.test_groups

