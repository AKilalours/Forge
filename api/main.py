"""FORGE detection API.

Live now: schema, validation, decision policy, abstention, provenance, metrics. The model
is NOT loaded, because Phase 3 training has not run, and /v1/detect returns 503 with the
reason rather than a fabricated score. A stub that returns a plausible number is worse
than an error, because something downstream will build on it.

Two rules the response format enforces:

  Every response carries model_version, threshold and fpr_budget. A score of 0.96 is
  uninterpretable without knowing what the model was tuned to protect.

  "uncertain" is a first-class verdict. See forge.inference.decision for why a detector
  operating at a 0.1 percent FPR budget should decline to guess in the band where it is
  least sure.
"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from forge.inference.decision import DecisionPolicy

app = FastAPI(
    title="FORGE",
    version="0.1.0",
    description="Failure-Driven Synthetic Data Generation for Robust AI-Content Detection",
)

MODEL_VERSION: str | None = os.getenv("FORGE_MODEL_VERSION")
_POLICY: DecisionPolicy | None = None
_START = time.time()
_COUNTS = {"requests": 0, "human": 0, "ai": 0, "uncertain": 0, "unavailable": 0}

MAX_CHARS = 200_000


class DetectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)


class Segment(BaseModel):
    start: int
    end: int
    ai_probability: float


class Uncertainty(BaseModel):
    calibrated: bool
    abstained: bool
    reason: str = ""


class DetectResponse(BaseModel):
    prediction: str
    ai_probability: float
    confidence: float
    model_version: str
    threshold: float
    fpr_budget: float
    uncertainty: Uncertainty
    segments: list[Segment] = []


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _POLICY is not None,
        "model_version": MODEL_VERSION,
        "uptime_seconds": round(time.time() - _START, 1),
    }


@app.get("/v1/model")
def model_info() -> dict:
    """What the deployed model is and what it was tuned to protect.

    Exposed deliberately. A detector whose operating point is invisible cannot be
    audited, and its results should not be trusted by anyone acting on them.
    """
    if _POLICY is None:
        raise HTTPException(status_code=503, detail="no model registered")
    return {
        "model_version": _POLICY.model_version,
        "threshold": _POLICY.threshold,
        "fpr_budget": _POLICY.fpr_budget,
        "calibrated": _POLICY.calibrated,
        "abstains": _POLICY.abstains,
        "uncertain_band": [_POLICY.abstain_low, _POLICY.abstain_high] if _POLICY.abstains else None,
    }


@app.get("/metrics")
def metrics() -> Response:
    lines = [
        "# HELP forge_requests_total Detection requests received.",
        "# TYPE forge_requests_total counter",
        f"forge_requests_total {_COUNTS['requests']}",
        "# HELP forge_predictions_total Predictions by verdict.",
        "# TYPE forge_predictions_total counter",
    ]
    for k in ("human", "ai", "uncertain", "unavailable"):
        lines.append(f'forge_predictions_total{{verdict="{k}"}} {_COUNTS[k]}')
    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.post("/v1/detect", response_model=DetectResponse)
def detect(req: DetectRequest) -> DetectResponse:
    _COUNTS["requests"] += 1
    if _POLICY is None:
        _COUNTS["unavailable"] += 1
        raise HTTPException(
            status_code=503,
            detail=(
                "No model is registered. Phase 3 training has not run. Returning a score "
                "here would be fabricating a result, so this endpoint fails instead."
            ),
        )
    raise HTTPException(
        status_code=501,
        detail="Scoring path is Phase 8 deployment; the decision policy is wired and tested.",
    )


def configure(policy: DecisionPolicy) -> None:
    """Called at startup once a registered model is loaded."""
    global _POLICY, MODEL_VERSION
    _POLICY = policy
    MODEL_VERSION = policy.model_version
