"""REST API: score a listing at submission time.

    uvicorn trustguard.api:app --reload
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .model import DEFAULT_MODEL_PATH, RiskModel, train_default_model
from .scoring import Thresholds, decide


class Listing(BaseModel):
    app_id: str = Field(examples=["app_demo_001"])
    title: str = Field(examples=["Wh4tsApp Plus"])
    category: str = Field(examples=["communication"])
    description: str = Field(examples=["The original app you trust, faster. Login with your existing account."])
    permissions: list[str] = Field(default_factory=list, examples=[["INTERNET", "READ_SMS", "BIND_ACCESSIBILITY_SERVICE"]])
    developer_account_age_days: int = Field(ge=0, examples=[9])
    developer_prior_strikes: int = Field(0, ge=0, examples=[1])
    installs_prev_7d: int = Field(0, ge=0, examples=[200])
    installs_7d: int = Field(0, ge=0, examples=[4100])
    avg_rating: float = Field(4.0, ge=0, le=5, examples=[2.1])
    reviews: list[str] = Field(default_factory=list, examples=[["Fake app, stole my login.", "Works ok"]])
    has_privacy_policy: bool = True


class Reason(BaseModel):
    rule_id: str
    policy_area: str
    description: str
    weight: float


class ScoreResponse(BaseModel):
    app_id: str
    risk_score: float
    model_score: float
    rule_score: float
    action: str
    tier: str
    reasons: list[Reason]
    latency_ms: float
    model_version: str


def _load_model() -> RiskModel:
    path = Path(os.getenv("TRUSTGUARD_MODEL", DEFAULT_MODEL_PATH))
    if path.exists():
        return RiskModel.load(path)
    model, *_ = train_default_model(n=int(os.getenv("TRUSTGUARD_TRAIN_N", "4000")))
    model.save(path)
    return model


app = FastAPI(title="TrustGuard", version=__version__,
              description="Policy-risk scoring for app marketplace submissions.")
_state: dict = {}


def get_model() -> RiskModel:
    if "model" not in _state:
        _state["model"] = _load_model()
    return _state["model"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/v1/score", response_model=ScoreResponse)
def score(listing: Listing) -> ScoreResponse:
    start = time.perf_counter()
    model = get_model()
    d = decide(listing.model_dump(), model, Thresholds())
    return ScoreResponse(
        **{k: v for k, v in d.to_dict().items() if k != "reasons"},
        reasons=[Reason(**r.__dict__) for r in d.reasons],
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        model_version=model.version,
    )


@app.post("/v1/score/batch", response_model=list[ScoreResponse])
def score_batch(listings: list[Listing]) -> list[ScoreResponse]:
    return [score(item) for item in listings]
