"""Tests for the self-contained PyTorch serving stack.

Exercise the FastAPI endpoints with compute_dc_for_text mocked out.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the depoly-pytorch dir is importable
_SERVE_DIR = str(Path(__file__).resolve().parent.parent / "depoly-pytorch")
if _SERVE_DIR not in sys.path:
    sys.path.insert(0, _SERVE_DIR)

from reference import load_reference_stats  # noqa: E402


def _build_ref_stats(tmp_path: Path) -> Path:
    ref_stats = {
        "meta": {},
        "D_h": [0.1, -0.2, 0.05, 0.3, -0.1, 0.15, 0.0, -0.05, 0.2, -0.15],
        "D_m": [80.0, 95.0, 110.0, 90.0, 85.0, 100.0, 88.0, 92.0, 105.0, 97.0],
    }
    path = tmp_path / "ref_stats.json"
    path.write_text(json.dumps(ref_stats), encoding="utf-8")
    return path


@pytest.fixture()
def serve_env(tmp_path: Path):
    """Set up env vars and mock compute_dc_for_text, return a TestClient."""
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
    }

    with patch.dict(os.environ, env):
        # Mock compute_dc_for_text at its source in the inference module
        with patch(
            "inference.compute_dc_for_text", return_value=25.0
        ) as mock_dc:
            import serve as serve_mod

            # Also patch the imported reference in serve module
            serve_mod.compute_dc_for_text = mock_dc

            # Inject state that lifespan would normally set up
            saved = (
                serve_mod._model,
                serve_mod._tokenizer,
                serve_mod._device,
                serve_mod._config,
                serve_mod._ref_stats,
            )
            serve_mod._model = MagicMock()
            serve_mod._tokenizer = MagicMock()
            serve_mod._device = "cpu"
            serve_mod._config = serve_mod._load_config()
            serve_mod._ref_stats = load_reference_stats(str(ref_path))

            # Build a test app WITHOUT the lifespan to avoid real model loading
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            test_app = FastAPI()
            for route in serve_mod.app.routes:
                test_app.routes.append(route)

            with TestClient(test_app) as client:
                yield client, mock_dc

            # Restore
            (
                serve_mod._model,
                serve_mod._tokenizer,
                serve_mod._device,
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
        assert body["backend"] == "pytorch"


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

    def test_decision_mode_threshold_in_response(self, serve_env):
        client, mock_dc = serve_env
        mock_dc.return_value = 10.0
        resp = client.post("/predict", json={"text": "hello"})
        assert resp.json()["decision_mode"] == "threshold"

    def test_ref_stats_fields_present(self, serve_env):
        """When ref_stats are loaded, p_m/delta/cnt_h/cnt_m should appear."""
        client, mock_dc = serve_env
        mock_dc.return_value = 90.0  # within D_m range
        resp = client.post("/predict", json={"text": "test"})
        pred = resp.json()["predictions"][0]
        assert pred["p_m"] is not None
        assert pred["delta"] is not None
