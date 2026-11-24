"""Explainable policy rule engine.

Rules encode policy knowledge that reviewers already trust. They give
day-one coverage before any model exists, act as a safety net for novel
abuse the model hasn't seen, and produce human-readable reasons for every
decision (important for developer appeals).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .features import extract_features, unexpected_permissions


@dataclass(frozen=True)
class Rule:
    rule_id: str
    policy_area: str
    description: str
    weight: float  # contribution to risk, 0..1
    check: Callable[[dict, dict], bool]


RULES: list[Rule] = [
    Rule("R001", "malware", "Accessibility service requested by a non-accessibility app", 0.55,
         lambda l, f: f["has_accessibility"] == 1),
    Rule("R002", "malware", "Can install other packages (sideloading vector)", 0.45,
         lambda l, f: f["has_sideload"] == 1),
    Rule("R003", "privacy", "Two or more dangerous permissions not justified by the app category", 0.35,
         lambda l, f: f["n_unexpected_permissions"] >= 2),
    Rule("R004", "impersonation", "Title closely resembles a protected brand", 0.50,
         lambda l, f: f["brand_similarity"] >= 0.85),
    Rule("R005", "deceptive_claims", "Description contains high-risk marketing claims", 0.40,
         lambda l, f: f["hype_term_count"] >= 1),
    Rule("R006", "user_harm", "30%+ of recent reviews report fraud, malware or unauthorised charges", 0.50,
         lambda l, f: f["review_complaint_rate"] >= 0.30),
    Rule("R007", "developer_trust", "New developer account (<30 days) with prior policy strikes", 0.40,
         lambda l, f: f["log_dev_age"] < 3.43 and f["developer_prior_strikes"] >= 1),
    Rule("R008", "growth_anomaly", "Install velocity spiked 5x+ week over week", 0.20,
         lambda l, f: f["install_velocity"] >= 1.61),
    Rule("R009", "privacy", "Handles sensitive data without a privacy policy", 0.25,
         lambda l, f: f["missing_privacy_policy"] == 1 and f["n_dangerous_permissions"] >= 1),
]


@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    policy_area: str
    description: str
    weight: float


def evaluate_rules(listing: dict, features: dict | None = None) -> list[RuleHit]:
    """Return every rule the listing triggers, strongest first."""
    feats = features or extract_features(listing)
    hits = [RuleHit(r.rule_id, r.policy_area, r.description, r.weight)
            for r in RULES if r.check(listing, feats)]
    return sorted(hits, key=lambda h: h.weight, reverse=True)


def rule_risk(hits: list[RuleHit]) -> float:
    """Combine independent rule weights with noisy-OR: 1 - prod(1 - w)."""
    p_clean = 1.0
    for h in hits:
        p_clean *= 1.0 - h.weight
    return 1.0 - p_clean


def explain_unexpected_permissions(listing: dict) -> list[str]:
    return unexpected_permissions(listing.get("category", ""), listing.get("permissions") or [])
