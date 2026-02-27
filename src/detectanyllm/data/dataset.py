"""Dataset definitions for paired HWT-MGT training."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

from detectanyllm.data.io import iter_jsonl

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    total_rows: int = 0
    kept_rows: int = 0
    dropped_rows: int = 0


class DDLPairDataset(Dataset):
    """Paired dataset with two tokenized inputs per row: human and machine."""

    def __init__(
        self,
        data_file: str | Path,
        tokenizer,
        max_length: int = 512,
        human_field: str = "human",
        machine_field: str = "machine",
        min_tokens: int = 2,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.human_field = human_field
        self.machine_field = machine_field
        self.min_tokens = min_tokens
        self.samples: list[dict] = []
        self.stats = DatasetStats()
        self._load(Path(data_file))

    def _load(self, path: Path) -> None:
        for row in iter_jsonl(path):
            self.stats.total_rows += 1
            if self.human_field not in row or self.machine_field not in row:
                raise KeyError(
                    f"Each row must contain '{self.human_field}' and '{self.machine_field}'."
                )

            human = self.tokenizer(
                row[self.human_field],
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
            machine = self.tokenizer(
                row[self.machine_field],
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

            if len(human["input_ids"]) < self.min_tokens or len(machine["input_ids"]) < self.min_tokens:
                self.stats.dropped_rows += 1
                continue

            self.samples.append(
                {
                    "human_input_ids": human["input_ids"],
                    "human_attention_mask": human["attention_mask"],
                    "machine_input_ids": machine["input_ids"],
                    "machine_attention_mask": machine["attention_mask"],
                }
            )
            self.stats.kept_rows += 1

        if self.stats.dropped_rows > 0:
            logger.warning(
                "Dropped %d short rows (<%d tokens) out of %d rows from %s.",
                self.stats.dropped_rows,
                self.min_tokens,
                self.stats.total_rows,
                path,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
