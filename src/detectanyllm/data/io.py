"""Dataset IO helpers."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class GroupSplitResult:
    train_rows: list[dict]
    dev_rows: list[dict]
    test_rows: list[dict]
    train_groups: set[str]
    dev_groups: set[str]
    test_groups: set[str]


@dataclass
class RandomSplitResult:
    train_rows: list[dict]
    dev_rows: list[dict]
    test_rows: list[dict]


def _validate_split_ratios(dev_ratio: float, test_ratio: float) -> None:
    if dev_ratio < 0 or test_ratio < 0:
        raise ValueError("dev_ratio and test_ratio must be >= 0.")
    if dev_ratio + test_ratio >= 1:
        raise ValueError(
            "dev_ratio + test_ratio must be < 1 so train split is non-empty."
        )


def _resolve_split_group_counts(
    total_groups: int,
    dev_ratio: float,
    test_ratio: float,
) -> tuple[int, int]:
    if total_groups < 3 and dev_ratio > 0 and test_ratio > 0:
        raise ValueError(
            "Need at least 3 distinct groups for train/dev/test split when both dev and test are enabled."
        )

    dev_count = int(round(total_groups * dev_ratio))
    test_count = int(round(total_groups * test_ratio))

    if dev_ratio > 0 and dev_count == 0:
        dev_count = 1
    if test_ratio > 0 and test_count == 0:
        test_count = 1

    max_holdout = total_groups - 1
    while dev_count + test_count > max_holdout:
        if test_count >= dev_count and test_count > 0:
            test_count -= 1
        elif dev_count > 0:
            dev_count -= 1
        else:
            break

    return dev_count, test_count


def _resolve_split_row_counts(
    total_rows: int,
    dev_ratio: float,
    test_ratio: float,
) -> tuple[int, int]:
    if total_rows < 3 and dev_ratio > 0 and test_ratio > 0:
        raise ValueError(
            "Need at least 3 rows for train/dev/test split when both dev and test are enabled."
        )

    dev_count = int(round(total_rows * dev_ratio))
    test_count = int(round(total_rows * test_ratio))

    if dev_ratio > 0 and dev_count == 0:
        dev_count = 1
    if test_ratio > 0 and test_count == 0:
        test_count = 1

    max_holdout = total_rows - 1
    while dev_count + test_count > max_holdout:
        if test_count >= dev_count and test_count > 0:
            test_count -= 1
        elif dev_count > 0:
            dev_count -= 1
        else:
            break

    return dev_count, test_count


def split_rows_random(
    rows: list[dict],
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> RandomSplitResult:
    """Randomly shuffle rows and split into train/dev/test."""

    _validate_split_ratios(dev_ratio=dev_ratio, test_ratio=test_ratio)
    if not rows:
        raise ValueError("rows cannot be empty.")

    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    dev_count, test_count = _resolve_split_row_counts(
        total_rows=len(shuffled),
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
    )

    test_rows = shuffled[:test_count]
    dev_rows = shuffled[test_count : test_count + dev_count]
    train_rows = shuffled[test_count + dev_count :]

    if not train_rows:
        raise ValueError(
            "Train split is empty after random split. Adjust dev/test ratios."
        )
    if dev_ratio > 0 and not dev_rows:
        raise ValueError(
            "Dev split is empty after random split. Adjust dev_ratio."
        )
    if test_ratio > 0 and not test_rows:
        raise ValueError(
            "Test split is empty after random split. Adjust test_ratio."
        )

    return RandomSplitResult(
        train_rows=train_rows,
        dev_rows=dev_rows,
        test_rows=test_rows,
    )


def split_rows_by_group_id(
    rows: list[dict],
    group_id_field: str,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> GroupSplitResult:
    """Split rows into train/dev/test by group id to avoid source leakage."""

    _validate_split_ratios(dev_ratio=dev_ratio, test_ratio=test_ratio)
    if not rows:
        raise ValueError("rows cannot be empty.")
    if not group_id_field:
        raise ValueError("group_id_field cannot be empty.")

    groups: dict[str, list[dict]] = {}
    for idx, row in enumerate(rows, start=1):
        if group_id_field not in row:
            raise KeyError(
                f"Missing group id field '{group_id_field}' in row {idx}."
            )
        raw_group_id = row[group_id_field]
        if raw_group_id is None:
            raise ValueError(
                f"Group id field '{group_id_field}' is None in row {idx}."
            )
        group_id = str(raw_group_id)
        groups.setdefault(group_id, []).append(row)

    group_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    dev_count, test_count = _resolve_split_group_counts(
        total_groups=len(group_ids),
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
    )

    test_groups = set(group_ids[:test_count])
    dev_groups = set(group_ids[test_count : test_count + dev_count])
    train_groups = set(group_ids[test_count + dev_count :])

    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    test_rows: list[dict] = []
    for row in rows:
        group_id = str(row[group_id_field])
        if group_id in test_groups:
            test_rows.append(row)
        elif group_id in dev_groups:
            dev_rows.append(row)
        else:
            train_rows.append(row)

    if not train_rows:
        raise ValueError(
            "Train split is empty after group split. Adjust dev/test ratios."
        )
    if dev_ratio > 0 and not dev_rows:
        raise ValueError(
            "Dev split is empty after group split. Adjust dev_ratio or provide more groups."
        )
    if test_ratio > 0 and not test_rows:
        raise ValueError(
            "Test split is empty after group split. Adjust test_ratio or provide more groups."
        )

    return GroupSplitResult(
        train_rows=train_rows,
        dev_rows=dev_rows,
        test_rows=test_rows,
        train_groups=train_groups,
        dev_groups=dev_groups,
        test_groups=test_groups,
    )


def split_jsonl_by_group_id(
    input_file: str | Path,
    output_dir: str | Path,
    group_id_field: str,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Path, Path, Path, GroupSplitResult]:
    """Load a JSONL file, split by group id, and write train/dev/test JSONLs."""

    rows = list(iter_jsonl(input_file))
    split = split_rows_by_group_id(
        rows=rows,
        group_id_field=group_id_field,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_file = out_dir / "train.jsonl"
    dev_file = out_dir / "dev.jsonl"
    test_file = out_dir / "test.jsonl"
    write_jsonl(train_file, split.train_rows)
    write_jsonl(dev_file, split.dev_rows)
    write_jsonl(test_file, split.test_rows)
    return train_file, dev_file, test_file, split


def split_jsonl_random(
    input_file: str | Path,
    output_dir: str | Path,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Path, Path, Path, RandomSplitResult]:
    """Load a JSONL file, randomly split rows, and write train/dev/test JSONLs."""

    rows = list(iter_jsonl(input_file))
    split = split_rows_random(
        rows=rows,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_file = out_dir / "train.jsonl"
    dev_file = out_dir / "dev.jsonl"
    test_file = out_dir / "test.jsonl"
    write_jsonl(train_file, split.train_rows)
    write_jsonl(dev_file, split.dev_rows)
    write_jsonl(test_file, split.test_rows)
    return train_file, dev_file, test_file, split


def read_text_field(record: dict, text_field: str) -> str:
    if text_field not in record:
        raise KeyError(f"Field '{text_field}' not found in record keys: {list(record.keys())}")
    value = record[text_field]
    if not isinstance(value, str):
        raise TypeError(f"Field '{text_field}' must be a string, got {type(value).__name__}")
    return value


def prepare_pairs(
    human_file: str | Path,
    machine_file: str | Path,
    output_file: str | Path,
    text_field: str = "text",
    shuffle: bool = False,
    seed: int = 42,
) -> int:
    """Create paired records: {'human': ..., 'machine': ...}."""

    human_records = list(iter_jsonl(human_file))
    machine_records = list(iter_jsonl(machine_file))
    if len(human_records) != len(machine_records):
        raise ValueError(
            "Human and machine files must have same number of rows, "
            f"got {len(human_records)} vs {len(machine_records)}."
        )

    pairs = []
    for human_rec, machine_rec in zip(human_records, machine_records):
        pairs.append(
            {
                "human": read_text_field(human_rec, text_field),
                "machine": read_text_field(machine_rec, text_field),
            }
        )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(pairs)

    write_jsonl(output_file, pairs)
    return len(pairs)
