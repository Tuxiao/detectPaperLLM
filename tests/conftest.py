from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


@pytest.fixture()
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture()
def pythonpath(repo_root: Path) -> str:
    src = repo_root / "src"
    return str(src)


@pytest.fixture()
def tiny_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "tiny_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "hello": 4,
        "world": 5,
        "human": 6,
        "machine": 7,
        "text": 8,
        "sample": 9,
        "foo": 10,
        "bar": 11,
        "good": 12,
        "bad": 13,
    }
    tokenizer_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer_obj.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(model_dir)

    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=64,
        n_ctx=64,
        n_embd=32,
        n_layer=1,
        n_head=2,
        bos_token_id=vocab["[BOS]"],
        eos_token_id=vocab["[EOS]"],
        pad_token_id=vocab["[PAD]"],
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(model_dir)
    return model_dir


@pytest.fixture()
def pair_train_file(tmp_path: Path) -> Path:
    path = tmp_path / "pair_train.jsonl"
    rows = [
        {"human": "human text good", "machine": "machine text good"},
        {"human": "human text foo", "machine": "machine text foo"},
        {"human": "human text bar", "machine": "machine text bar"},
        {"human": "human sample world", "machine": "machine sample world"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


@pytest.fixture()
def reference_files(tmp_path: Path) -> tuple[Path, Path]:
    human = tmp_path / "human_ref.jsonl"
    machine = tmp_path / "machine_ref.jsonl"
    human_rows = [{"text": "human text good"}, {"text": "human sample foo"}]
    machine_rows = [{"text": "machine text good"}, {"text": "machine sample bad"}]
    with human.open("w", encoding="utf-8") as handle:
        for row in human_rows:
            handle.write(json.dumps(row) + "\n")
    with machine.open("w", encoding="utf-8") as handle:
        for row in machine_rows:
            handle.write(json.dumps(row) + "\n")
    return human, machine


@pytest.fixture()
def infer_input_file(tmp_path: Path) -> Path:
    path = tmp_path / "infer.jsonl"
    rows = [{"text": "human text good"}, {"text": "machine text bad"}]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def run_cli(repo_root: Path, pythonpath: str, args: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "detectanyllm.cli", *args],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


@pytest.fixture()
def cli_runner(repo_root: Path, pythonpath: str):
    def _run(args: list[str]) -> subprocess.CompletedProcess:
        return run_cli(repo_root=repo_root, pythonpath=pythonpath, args=args)

    return _run
