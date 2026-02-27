"""Data collators."""

from __future__ import annotations

from typing import Any


class PairDataCollator:
    """Dynamic padding for paired human/machine batches."""

    def __init__(self, tokenizer, pad_to_multiple_of: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        human_features = [
            {
                "input_ids": feature["human_input_ids"],
                "attention_mask": feature["human_attention_mask"],
            }
            for feature in features
        ]
        machine_features = [
            {
                "input_ids": feature["machine_input_ids"],
                "attention_mask": feature["machine_attention_mask"],
            }
            for feature in features
        ]

        human_batch = self.tokenizer.pad(
            human_features,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        machine_batch = self.tokenizer.pad(
            machine_features,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        return {
            "human_input_ids": human_batch["input_ids"],
            "human_attention_mask": human_batch["attention_mask"],
            "machine_input_ids": machine_batch["input_ids"],
            "machine_attention_mask": machine_batch["attention_mask"],
        }
