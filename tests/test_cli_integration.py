from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("peft") is None, reason="peft not installed"
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_cli_prepare_train_build_ref_infer_chain(
    cli_runner,
    tiny_model_dir,
    reference_files,
    infer_input_file,
    tmp_path,
):
    human_raw = tmp_path / "human_raw.jsonl"
    machine_raw = tmp_path / "machine_raw.jsonl"
    pair_out = tmp_path / "pairs.jsonl"
    model_out = tmp_path / "adapter_out"
    ref_stats = tmp_path / "ref_stats.json"
    pred_out = tmp_path / "predictions.jsonl"

    _write_jsonl(
        human_raw,
        [
            {"text": "human text good"},
            {"text": "human text foo"},
            {"text": "human sample world"},
            {"text": "human bar"},
        ],
    )
    _write_jsonl(
        machine_raw,
        [
            {"text": "machine text good"},
            {"text": "machine text foo"},
            {"text": "machine sample world"},
            {"text": "machine bad"},
        ],
    )

    cli_runner(
        [
            "prepare-pairs",
            "--human-file",
            human_raw.as_posix(),
            "--machine-file",
            machine_raw.as_posix(),
            "--output-file",
            pair_out.as_posix(),
            "--text-field",
            "text",
        ]
    )
    assert pair_out.exists()

    cli_runner(
        [
            "train",
            "--train-pairs-file",
            pair_out.as_posix(),
            "--model-name-or-path",
            tiny_model_dir.as_posix(),
            "--output-dir",
            model_out.as_posix(),
            "--target-modules",
            "c_attn,c_proj",
            "--max-length",
            "32",
            "--num-train-epochs",
            "1",
            "--per-device-train-batch-size",
            "1",
            "--gradient-accumulation-steps",
            "1",
            "--logging-steps",
            "1",
            "--save-steps",
            "1",
            "--num-perturb-samples",
            "8",
            "--no-bf16",
        ]
    )
    assert (model_out / "adapter_config.json").exists()

    human_ref_file, machine_ref_file = reference_files
    cli_runner(
        [
            "build-ref",
            "--model-path",
            model_out.as_posix(),
            "--human-ref-file",
            human_ref_file.as_posix(),
            "--machine-ref-file",
            machine_ref_file.as_posix(),
            "--ref-stats-file",
            ref_stats.as_posix(),
            "--text-field",
            "text",
            "--max-length",
            "32",
            "--num-perturb-samples",
            "8",
            "--k-neighbors",
            "2",
            "--no-bf16",
        ]
    )
    assert ref_stats.exists()

    cli_runner(
        [
            "infer",
            "--model-path",
            model_out.as_posix(),
            "--input-file",
            infer_input_file.as_posix(),
            "--output-file",
            pred_out.as_posix(),
            "--ref-stats-file",
            ref_stats.as_posix(),
            "--decision-mode",
            "pm",
            "--k-neighbors",
            "2",
            "--num-perturb-samples",
            "8",
            "--no-bf16",
        ]
    )
    assert pred_out.exists()

    lines = [json.loads(line) for line in pred_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    for row in lines:
        assert "text" in row
        assert "d_c" in row
        assert "p_m" in row
        assert "delta" in row
        assert "cnt_h" in row
        assert "cnt_m" in row
        assert "label_pred" in row


def test_cli_train_group_split_and_periodic_test_metrics(
    cli_runner,
    tiny_model_dir,
    tmp_path,
):
    pair_out = tmp_path / "pairs_with_doc_id.jsonl"
    model_out = tmp_path / "adapter_out"

    _write_jsonl(
        pair_out,
        [
            {"doc_id": "a", "human": "human text good", "machine": "machine text good"},
            {"doc_id": "a", "human": "human text foo", "machine": "machine text foo"},
            {"doc_id": "b", "human": "human sample world", "machine": "machine sample world"},
            {"doc_id": "b", "human": "human text bar", "machine": "machine text bad"},
            {"doc_id": "c", "human": "human good", "machine": "machine bad"},
            {"doc_id": "c", "human": "human foo", "machine": "machine bar"},
            {"doc_id": "d", "human": "human world", "machine": "machine world"},
            {"doc_id": "e", "human": "human sample", "machine": "machine sample"},
        ],
    )

    cli_runner(
        [
            "train",
            "--train-pairs-file",
            pair_out.as_posix(),
            "--group-id-field",
            "doc_id",
            "--dev-ratio",
            "0.2",
            "--test-ratio",
            "0.2",
            "--split-seed",
            "42",
            "--test-eval-steps",
            "1",
            "--test-threshold-objective",
            "mcc",
            "--model-name-or-path",
            tiny_model_dir.as_posix(),
            "--output-dir",
            model_out.as_posix(),
            "--target-modules",
            "c_attn,c_proj",
            "--max-length",
            "32",
            "--num-train-epochs",
            "1",
            "--per-device-train-batch-size",
            "1",
            "--gradient-accumulation-steps",
            "1",
            "--logging-steps",
            "1",
            "--save-steps",
            "5",
            "--num-perturb-samples",
            "4",
            "--no-bf16",
        ]
    )

    assert (model_out / "splits" / "train.jsonl").exists()
    assert (model_out / "splits" / "dev.jsonl").exists()
    assert (model_out / "splits" / "test.jsonl").exists()

    live_metrics = json.loads((model_out / "training_live_metrics.json").read_text(encoding="utf-8"))
    log_history = live_metrics.get("log_history", [])
    assert any("test_auc" in row and "test_mcc" in row and "test_f1" in row for row in log_history)


def test_cli_train_random_split_without_doc_id(
    cli_runner,
    tiny_model_dir,
    tmp_path,
):
    pair_out = tmp_path / "pairs_no_doc_id.jsonl"
    model_out = tmp_path / "adapter_out_no_doc_id"

    _write_jsonl(
        pair_out,
        [
            {"human": "human text good", "machine": "machine text good"},
            {"human": "human text foo", "machine": "machine text foo"},
            {"human": "human sample world", "machine": "machine sample world"},
            {"human": "human text bar", "machine": "machine text bad"},
            {"human": "human good", "machine": "machine bad"},
            {"human": "human foo", "machine": "machine bar"},
            {"human": "human world", "machine": "machine world"},
            {"human": "human sample", "machine": "machine sample"},
        ],
    )

    cli_runner(
        [
            "train",
            "--train-pairs-file",
            pair_out.as_posix(),
            "--dev-ratio",
            "0.2",
            "--test-ratio",
            "0.2",
            "--split-seed",
            "42",
            "--test-eval-steps",
            "1",
            "--model-name-or-path",
            tiny_model_dir.as_posix(),
            "--output-dir",
            model_out.as_posix(),
            "--target-modules",
            "c_attn,c_proj",
            "--max-length",
            "32",
            "--num-train-epochs",
            "1",
            "--per-device-train-batch-size",
            "1",
            "--gradient-accumulation-steps",
            "1",
            "--logging-steps",
            "1",
            "--save-steps",
            "5",
            "--num-perturb-samples",
            "4",
            "--no-bf16",
        ]
    )

    assert (model_out / "splits" / "train.jsonl").exists()
    assert (model_out / "splits" / "dev.jsonl").exists()
    assert (model_out / "splits" / "test.jsonl").exists()

    live_metrics = json.loads((model_out / "training_live_metrics.json").read_text(encoding="utf-8"))
    log_history = live_metrics.get("log_history", [])
    assert any("test_auc" in row and "test_mcc" in row and "test_f1" in row for row in log_history)
