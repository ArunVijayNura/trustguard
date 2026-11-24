"""Feature engineering: turn a raw listing into model-ready signals."""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher

import pandas as pd

from .data import DANGEROUS_PERMISSIONS, EXPECTED_PERMISSIONS, WELL_KNOWN_BRANDS

COMPLAINT_TERMS = re.compile(
    r"\b(scam|fraud|fake|virus|malware|stole|charged|otp|never paid|reported)\b", re.I
)
HYPE_TERMS = re.compile(
    r"(guaranteed|10x|\$\d+/day|free coins|unlimited gems|claim your prize|instant loan|"
    r"no documents|boost battery|enable accessibility|install the plugin|win a free)", re.I
)

NUMERIC_FEATURES = [
    "log_dev_age",
    "developer_prior_strikes",
    "n_dangerous_permissions",
    "n_unexpected_permissions",
    "has_accessibility",
    "has_sideload",
    "install_velocity",
    "avg_rating",
    "review_complaint_rate",
    "hype_term_count",
    "brand_similarity",
    "missing_privacy_policy",
]
TEXT_FEATURE = "text"


def _normalise(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower().replace("4", "a").replace("0", "o").replace("1", "l"))


def brand_similarity(title: str) -> float:
    """Highest fuzzy similarity between the title and a protected brand name.

    Catches look-alikes such as ``Wh4tsApp Plus`` that exact matching misses.
    """
    t = _normalise(title)
    best = 0.0
    for brand in WELL_KNOWN_BRANDS:
        b = _normalise(brand)
        if b and b in t:
            return 1.0
        best = max(best, SequenceMatcher(None, t, b).ratio())
    return round(best, 3)


def unexpected_permissions(category: str, permissions: list[str]) -> list[str]:
    allowed = EXPECTED_PERMISSIONS.get(category, set())
    return sorted(p for p in permissions if p in DANGEROUS_PERMISSIONS and p not in allowed)


def extract_features(listing: dict) -> dict:
    """Compute engineered features for a single listing."""
    perms = list(listing.get("permissions") or [])
    reviews = list(listing.get("reviews") or [])
    desc = listing.get("description", "")
    prev = max(int(listing.get("installs_prev_7d", 0)), 1)
    complaints = sum(bool(COMPLAINT_TERMS.search(r)) for r in reviews)
    return {
        "log_dev_age": math.log1p(max(int(listing.get("developer_account_age_days", 0)), 0)),
        "developer_prior_strikes": int(listing.get("developer_prior_strikes", 0)),
        "n_dangerous_permissions": sum(p in DANGEROUS_PERMISSIONS for p in perms),
        "n_unexpected_permissions": len(unexpected_permissions(listing.get("category", ""), perms)),
        "has_accessibility": int("BIND_ACCESSIBILITY_SERVICE" in perms),
        "has_sideload": int("REQUEST_INSTALL_PACKAGES" in perms),
        "install_velocity": math.log(max(int(listing.get("installs_7d", 0)), 1) / prev),
        "avg_rating": float(listing.get("avg_rating", 0.0)),
        "review_complaint_rate": complaints / len(reviews) if reviews else 0.0,
        "hype_term_count": len(HYPE_TERMS.findall(desc)),
        "brand_similarity": brand_similarity(listing.get("title", "")),
        "missing_privacy_policy": int(not listing.get("has_privacy_policy", True)),
        TEXT_FEATURE: f"{listing.get('title', '')}. {desc} " + " ".join(reviews),
    }


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised wrapper used for training and batch scoring."""
    return pd.DataFrame([extract_features(r) for r in df.to_dict("records")], index=df.index)
