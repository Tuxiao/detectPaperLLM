"""Tests for the vLLM-based serving stack.

Split into two groups:

1.  **DcLogitsProcessor unit tests** — pure-logic tests that validate the
    processor's per-token accumulation and ``finalize()`` against the original
    ``compute_dc_from_logits`` reference implementation.  No vLLM or GPU needed.

2.  **serve.py endpoint tests** — exercise the FastAPI endpoints with the
    vLLM engine mocked out, so they run on any machine.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

from detectanyllm.infer.vllm_processor import DcLogitsProcessor
from detectanyllm.training.discrepancy import compute_dc_from_logits


# =========================================================================
# Part 1 — DcLogitsProcessor unit tests
# =========================================================================


class TestDcLogitsProcessor:
    """Validate that the incremental processor matches the batch reference."""

    @staticmethod
    def _run_processor_on_logits(
        logits_2d: torch.Tensor,
        input_ids: list[int],
        num_perturb_samples: int = 32,
        sigma_eps: float = 1e-6,
    ) -> float:
        """Drive a DcLogitsProcessor step-by-step with pre-computed logits."""
        # target_token_ids = input_ids[1:]  (skip the prompt token)
        target_ids = input_ids[1:]
        proc = DcLogitsProcessor(
            target_token_ids=target_ids,
            num_perturb_samples=num_perturb_samples,
            sigma_eps=sigma_eps,
        )
        seq_len = logits_2d.shape[0]  # N-1 positions (shifted)
        for i in range(seq_len):
            logits_i = logits_2d[i].clone()
            proc(output_ids=[], logits=logits_i)
        return proc.finalize()

    def test_processor_returns_finite_dc(self):
        """Basic sanity: d_c should be a finite float."""
        torch.manual_seed(42)
        vocab_size, seq_len = 50, 10
        logits = torch.randn(seq_len - 1, vocab_size)
        input_ids = torch.randint(0, vocab_size, (seq_len,)).tolist()

        dc = self._run_processor_on_logits(
            logits, input_ids, num_perturb_samples=16
        )
        assert math.isfinite(dc)

    def test_processor_matches_reference_statistically(self):
        """The processor's d_c should be close to the batch reference.

        Because both implementations sample independently, they won't match
        exactly.  We fix the seed for the reference and run many perturbation
        samples to reduce variance, then check the values are in the same
        ballpark.
        """
        torch.manual_seed(0)
        vocab_size, seq_len = 30, 8
        logits_3d = torch.randn(1, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (1, seq_len))
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)

        # Reference: batch implementation with high K for low variance
        torch.manual_seed(99)
        dc_ref = compute_dc_from_logits(
            logits=logits_3d,
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_perturb_samples=2048,
            sigma_eps=1e-6,
        ).item()

        # Processor: incremental implementation with high K
        torch.manual_seed(99)
        shifted_logits = logits_3d[0, :-1, :]  # [seq_len-1, vocab_size]
        dc_proc = self._run_processor_on_logits(
            shifted_logits,
            input_ids[0].tolist(),
            num_perturb_samples=2048,
            sigma_eps=1e-6,
        )

        # With K=2048, both should converge to roughly the same value.
        # Allow 20% relative tolerance due to independent sampling.
        assert dc_ref != 0.0
        relative_error = abs(dc_proc - dc_ref) / (abs(dc_ref) + 1e-9)
        assert relative_error < 0.2, (
            f"dc_proc={dc_proc:.4f} vs dc_ref={dc_ref:.4f}, "
            f"relative_error={relative_error:.4f}"
        )

    def test_processor_forced_decoding(self):
        """The processor should modify logits to force the correct token."""
        target_ids = [5, 10, 3]
        proc = DcLogitsProcessor(target_ids, num_perturb_samples=4)

        logits = torch.randn(20)
        result = proc(output_ids=[], logits=logits)
        # After processing, only target token should be 0.0, rest -inf
        assert result[5].item() == 0.0
        assert result[0].item() == float("-inf")
        assert result[19].item() == float("-inf")

    def test_processor_step_count(self):
        """Processor should track steps and become ready after all targets."""
        target_ids = [1, 2, 3]
        proc = DcLogitsProcessor(target_ids, num_perturb_samples=4)

        assert not proc.ready
        for _ in range(3):
            proc(output_ids=[], logits=torch.randn(10))
        assert proc.ready

    def test_processor_passthrough_after_targets_exhausted(self):
        """Extra calls beyond target length should not modify logits."""
        target_ids = [1]
        proc = DcLogitsProcessor(target_ids, num_perturb_samples=4)
        proc(output_ids=[], logits=torch.randn(10))
        assert proc.ready

        # Next call should pass through
        logits = torch.randn(10)
        original = logits.clone()
        result = proc(output_ids=[], logits=logits)
        assert torch.equal(result, original)

    def test_finalize_raises_on_empty(self):
        """finalize() should raise if no tokens were processed."""
        proc = DcLogitsProcessor([], num_perturb_samples=4)
        with pytest.raises(ValueError, match="No tokens"):
            proc.finalize()


# =========================================================================
# Part 2 — serve.py endpoint tests (vLLM engine mocked)
# =========================================================================


def _build_ref_stats(tmp_path: Path) -> Path:
    ref_stats = {
        "meta": {},
        "D_h": [0.1, -0.2, 0.05, 0.3, -0.1, 0.15, 0.0, -0.05, 0.2, -0.15],
        "D_m": [80.0, 95.0, 110.0, 90.0, 85.0, 100.0, 88.0, 92.0, 105.0, 97.0],
    }
    path = tmp_path / "ref_stats.json"
    path.write_text(json.dumps(ref_stats), encoding="utf-8")
    return path


from detectanyllm.infer.reference_clustering import load_reference_stats


@pytest.fixture()
def serve_env(tmp_path: Path):
    """Set up env vars and mock vLLM engine, return a TestClient.

    We bypass the lifespan entirely (it imports ``vllm`` which is only
    available inside the vLLM Docker image) and set the module globals
    directly.
    """
    ref_path = _build_ref_stats(tmp_path)

    env = {
        "MODEL_PATH": "/fake/model",
        "REF_STATS_PATH": str(ref_path),
        "TRUST_REMOTE_CODE": "false",
        "USE_BF16": "false",
        "MAX_LENGTH": "32",
        "NUM_PERTURB_SAMPLES": "4",
        "DECISION_MODE": "threshold",
        "THRESHOLD": "50.0",
        "GPU_MEMORY_UTILIZATION": "0.85",
    }

    with patch.dict(os.environ, env):
        with patch("deploy.serve.compute_dc_vllm", new_callable=AsyncMock) as mock_dc:
            mock_dc.return_value = 25.0

            import deploy.serve as serve_mod

            # Inject state that lifespan would normally set up
            saved = (
                serve_mod._engine,
                serve_mod._tokenizer,
                serve_mod._config,
                serve_mod._ref_stats,
            )
            serve_mod._engine = MagicMock()
            serve_mod._tokenizer = MagicMock()
            serve_mod._config = serve_mod._load_config()
            serve_mod._ref_stats = load_reference_stats(str(ref_path))

            # Build a test app WITHOUT the lifespan to avoid vllm imports
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            test_app = FastAPI()
            # Copy routes from the real app
            for route in serve_mod.app.routes:
                test_app.routes.append(route)

            with TestClient(test_app) as client:
                yield client, mock_dc

            # Restore
            (
                serve_mod._engine,
                serve_mod._tokenizer,
                serve_mod._config,
                serve_mod._ref_stats,
            ) = saved


class TestHealthEndpoint:
    def test_health_returns_200(self, serve_env):
        client, _ = serve_env
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["backend"] == "vllm"


class TestPredictEndpoint:
    def test_single_text_threshold_mode(self, serve_env):
        client, mock_dc = serve_env
        mock_dc.return_value = 25.0  # below threshold → HUMAN
        resp = client.post("/predict", json={"text": "hello world"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["predictions"]) == 1
        assert body["predictions"][0]["label_pred"] == "HUMAN"
        assert body["predictions"][0]["d_c"] == 25.0

    def test_single_text_above_threshold(self, serve_env):
        client, mock_dc = serve_env
        mock_dc.return_value = 75.0  # above threshold → MACHINE
        resp = client.post("/predict", json={"text": "machine text"})
        assert resp.status_code == 200
        assert resp.json()["predictions"][0]["label_pred"] == "MACHINE"

    def test_batch_texts(self, serve_env):
        client, mock_dc = serve_env
        mock_dc.return_value = 10.0
        resp = client.post(
            "/predict", json={"texts": ["text one", "text two", "text three"]}
        )
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 3

    def test_missing_text_returns_422(self, serve_env):
        client, _ = serve_env
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_max_length_too_large(self, serve_env):
        client, _ = serve_env
        resp = client.post(
            "/predict", json={"text": "hello", "max_length": 99999}
        )
        assert resp.status_code == 400

    def test_too_many_texts(self, serve_env):
        client, _ = serve_env
        texts = [f"text {i}" for i in range(50)]
        resp = client.post("/predict", json={"texts": texts})
        assert resp.status_code == 400

    def test_invalid_decision_mode(self, serve_env):
        client, _ = serve_env
        resp = client.post(
            "/predict", json={"text": "hello", "decision_mode": "invalid"}
        )
        assert resp.status_code == 400

    def test_elapsed_ms_present(self, serve_env):
        client, mock_dc = serve_env
        mock_dc.return_value = 10.0
        resp = client.post("/predict", json={"text": "hello"})
        assert "elapsed_ms" in resp.json()
