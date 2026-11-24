"""Decision layer: combine model + rules into an enforcement action."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

import pandas as pd

from .features import extract_features
from .model import RiskModel
from .rules import RuleHit, evaluate_rules, rule_risk


class Action(str, Enum):
    ALLOW = "ALLOW"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class Thresholds:
    """Operating points. ``block`` is tuned for very high precision (false
    takedowns hurt honest developers); ``review`` is tuned to reviewer capacity."""

    review: float = 0.35
    block: float = 0.90


@dataclass
class Decision:
    app_id: str
    risk_score: float
    model_score: float
    rule_score: float
    action: Action
    tier: str
    reasons: list[RuleHit] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action"] = self.action.value
        return d


def _tier(score: float) -> str:
    return "critical" if score >= 0.9 else "high" if score >= 0.6 else "medium" if score >= 0.35 else "low"


def combine(model_score: float, rule_score: float) -> float:
    """Noisy-OR of the two signals: either one can raise risk, neither can hide the other."""
    return 1.0 - (1.0 - model_score) * (1.0 - 0.6 * rule_score)


def decide(listing: dict, model: RiskModel | None, thresholds: Thresholds = Thresholds()) -> Decision:
    feats = extract_features(listing)
    hits = evaluate_rules(listing, feats)
    r_score = rule_risk(hits)
    m_score = float(model.predict_proba(pd.DataFrame([feats]))[0]) if model is not None else 0.0
    score = combine(m_score, r_score) if model is not None else r_score
    action = (Action.BLOCK if score >= thresholds.block
              else Action.HUMAN_REVIEW if score >= thresholds.review
              else Action.ALLOW)
    return Decision(
        app_id=str(listing.get("app_id", "unknown")),
        risk_score=round(score, 4),
        model_score=round(m_score, 4),
        rule_score=round(r_score, 4),
        action=action,
        tier=_tier(score),
        reasons=hits,
    )


def score_frame(df: pd.DataFrame, model: RiskModel, thresholds: Thresholds = Thresholds()) -> pd.DataFrame:
    """Batch-score a DataFrame of listings."""
    out = []
    for listing in df.to_dict("records"):
        d = decide(listing, model, thresholds)
        out.append({
            "app_id": d.app_id, "title": listing.get("title"), "category": listing.get("category"),
            "risk_score": d.risk_score, "model_score": d.model_score, "rule_score": d.rule_score,
            "action": d.action.value, "tier": d.tier,
            "reasons": "; ".join(f"{h.rule_id} {h.description}" for h in d.reasons),
            "is_violation": listing.get("is_violation"), "violation_type": listing.get("violation_type"),
        })
    return pd.DataFrame(out)
