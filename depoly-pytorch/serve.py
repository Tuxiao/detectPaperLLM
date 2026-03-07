"""DetectAnyLLM serving endpoint — pure PyTorch backend.

Self-contained serving application. All dependencies are local modules
within this directory — no external project package required.

Start with:
    uvicorn serve:app --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Add src to sys.path so we can import from detectanyllm
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from detectanyllm.training.discrepancy import compute_dc
from detectanyllm.modeling.lora import load_model_for_inference
from detectanyllm.infer.reference_clustering import estimate_probability, load_reference_stats

logger = logging.getLogger("detectanyllm.serve")

# ---------------------------------------------------------------------------
# Hard safety limits
# ---------------------------------------------------------------------------
MAX_ALLOWED_LENGTH: int = 2048
MAX_BATCH_SIZE: int = 32

# ---------------------------------------------------------------------------
# Module-level globals (populated once at startup)
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_device = None
_ref_stats: dict | None = None
_config: dict = {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    return {
        "model_path": os.environ.get("MODEL_PATH", "/models/merged"),
        "base_model": os.environ.get("BASE_MODEL", None) or None,
        "ref_stats_path": os.environ.get("REF_STATS_PATH", ""),
        "trust_remote_code": os.environ.get("TRUST_REMOTE_CODE", "true").lower()
        == "true",
        "use_bf16": os.environ.get("USE_BF16", "true").lower() == "true",
        "max_length": int(os.environ.get("MAX_LENGTH", "512")),
        "num_perturb_samples": int(os.environ.get("NUM_PERTURB_SAMPLES", "32")),
        "sigma_eps": float(os.environ.get("SIGMA_EPS", "1e-6")),
        "k_neighbors": int(os.environ.get("K_NEIGHBORS", "100")),
        "decision_mode": os.environ.get("DECISION_MODE", "threshold"),
        "threshold": float(os.environ.get("THRESHOLD", "50.0")),
    }


# ---------------------------------------------------------------------------
# Lifespan — load model before accepting traffic
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _device, _ref_stats, _config  # noqa: PLW0603

    _config = _load_config()
    logger.info("Loading model from %s ...", _config["model_path"])
    t0 = time.monotonic()

    _model, _tokenizer, _device = load_model_for_inference(
        model_path=_config["model_path"],
        base_model=_config["base_model"],
        trust_remote_code=_config["trust_remote_code"],
        use_bf16=_config["use_bf16"],
    )

    elapsed = time.monotonic() - t0
    logger.info("Model loaded on %s in %.1f s.", _device, elapsed)

    # Reference stats (optional)
    ref_path = _config["ref_stats_path"]
    if ref_path and Path(ref_path).exists():
        _ref_stats = load_reference_stats(ref_path)
        logger.info(
            "Reference stats loaded: |D_h|=%d, |D_m|=%d",
            len(_ref_stats["D_h"]),
            len(_ref_stats["D_m"]),
        )
    elif _config["decision_mode"] == "pm":
        logger.warning(
            "decision_mode=pm but ref_stats not found at %s — "
            "falling back to threshold mode.",
            ref_path,
        )
        _config["decision_mode"] = "threshold"

    yield

    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="DetectAnyLLM", version="0.3.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def _global_exception_handler(request, exc):  # noqa: ARG001
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str | None = None
    texts: list[str] | None = None
    decision_mode: str | None = None
    threshold: float | None = None
    max_length: int | None = None
    num_perturb_samples: int | None = None
    k_neighbors: int | None = None


class PredictionResult(BaseModel):
    text: str
    d_c: float
    p_m: float | None = None
    delta: float | None = None
    cnt_h: int | None = None
    cnt_m: int | None = None
    label_pred: str
    low_confidence: bool | None = None


class PredictResponse(BaseModel):
    predictions: list[PredictionResult]
    elapsed_ms: float
    decision_mode: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_path": _config.get("model_path"),
        "decision_mode": _config.get("decision_mode"),
        "device": str(_device),
        "backend": "pytorch",
    }


@app.get("/info")
async def info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 尝试从模型配置获取型号
    model_type = getattr(_model.config, "model_type", "unknown")
    
    return {
        "adapter_path": _config.get("model_path"),
        "base_model_path": _config.get("base_model"),
        "model_type": model_type,
        "device": str(_device),
        "decision_mode": _config.get("decision_mode"),
        "threshold": _config.get("threshold"),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # ---- assemble text list ----
    texts: list[str] = []
    if req.text is not None:
        texts.append(req.text)
    if req.texts is not None:
        texts.extend(req.texts)
    if not texts:
        raise HTTPException(status_code=422, detail="Provide 'text' or 'texts'.")
    if len(texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_BATCH_SIZE} texts per request.",
        )

    # ---- per-request overrides ----
    decision_mode = req.decision_mode or _config["decision_mode"]
    threshold = req.threshold if req.threshold is not None else _config["threshold"]
    max_length = req.max_length if req.max_length is not None else _config["max_length"]
    num_perturb = (
        req.num_perturb_samples
        if req.num_perturb_samples is not None
        else _config["num_perturb_samples"]
    )
    k_neighbors = (
        req.k_neighbors if req.k_neighbors is not None else _config["k_neighbors"]
    )

    if max_length > MAX_ALLOWED_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"max_length cannot exceed {MAX_ALLOWED_LENGTH}.",
        )
    if decision_mode == "pm" and _ref_stats is None:
        raise HTTPException(
            status_code=400,
            detail="decision_mode=pm requires ref_stats, but none are loaded.",
        )
    if decision_mode not in {"pm", "threshold"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported decision_mode: {decision_mode}",
        )

    # ---- score each text (offloaded to thread pool to avoid blocking) ----
    t0 = time.monotonic()

    def _score_one(text: str) -> PredictionResult:
        try:
            # Tokenize locally
            encoded = _tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            if input_ids.shape[1] < 2:
                raise ValueError("Text is too short after tokenization (<2 tokens).")

            d_c = compute_dc(
                model=_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_perturb_samples=num_perturb,
                sigma_eps=_config["sigma_eps"],
            )
            d_c = float(d_c.item())
        except ValueError:
            return PredictionResult(
                text=text,
                d_c=float("nan"),
                label_pred="ERROR",
                low_confidence=True,
            )

        p_m: float | None = None
        delta: float | None = None
        cnt_h: int | None = None
        cnt_m: int | None = None
        low_confidence: bool | None = None

        if _ref_stats is not None:
            est = estimate_probability(
                dc_value=d_c,
                d_h=_ref_stats["D_h"],
                d_m=_ref_stats["D_m"],
                k_neighbors=k_neighbors,
            )
            p_m = est["p_m"]
            delta = est["delta"]
            cnt_h = est["cnt_h"]
            cnt_m = est["cnt_m"]
            low_confidence = est["low_confidence"]

        if decision_mode == "pm":
            label_pred = "MACHINE" if p_m is not None and p_m >= 0.5 else "HUMAN"
        else:
            label_pred = "MACHINE" if d_c >= threshold else "HUMAN"

        return PredictionResult(
            text=text,
            d_c=d_c,
            p_m=p_m,
            delta=delta,
            cnt_h=cnt_h,
            cnt_m=cnt_m,
            label_pred=label_pred,
            low_confidence=low_confidence,
        )

    # Process sequentially in a thread to keep GPU utilisation simple
    def _score_all() -> list[PredictionResult]:
        return [_score_one(t) for t in texts]

    predictions = await asyncio.to_thread(_score_all)

    elapsed_ms = (time.monotonic() - t0) * 1000
    return PredictResponse(
        predictions=predictions,
        elapsed_ms=round(elapsed_ms, 2),
        decision_mode=decision_mode,
    )
