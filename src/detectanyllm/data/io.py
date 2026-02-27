"""Dataset IO helpers."""

from __future__ import annotations

import json
import random
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
