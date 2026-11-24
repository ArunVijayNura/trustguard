"""Offline evaluation framed the way a Trust & Safety team reports it."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .scoring import Thresholds, combine


def recall_at_capacity(y: np.ndarray, scores: np.ndarray, capacity: float) -> float:
    """Share of all violations caught if reviewers can only look at the top ``capacity`` fraction."""
    k = max(1, int(round(len(scores) * capacity)))
    top = np.argsort(-scores)[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def operating_point(y: np.ndarray, scores: np.ndarray, t: Thresholds) -> dict:
    block = scores >= t.block
    review = (scores >= t.review) & ~block
    caught = block | review
    honest = y == 0
    return {
        "block_precision": float(y[block].mean()) if block.any() else float("nan"),
        "false_block_rate_honest_devs": float(block[honest].mean()),
        "review_queue_share": float(review.mean()),
        "review_queue_precision": float(y[review].mean()) if review.any() else float("nan"),
        "violation_recall": float(caught[y == 1].mean()),
        "auto_block_share_of_violations": float(block[y == 1].mean()),
    }


def compare_systems(y: pd.Series, model_scores: np.ndarray, rule_scores: np.ndarray,
                    thresholds: Thresholds = Thresholds()) -> pd.DataFrame:
    y_arr = y.to_numpy()
    systems = {
        "Rules only": rule_scores,
        "ML only": model_scores,
        "Hybrid (TrustGuard)": np.array([combine(m, r) for m, r in zip(model_scores, rule_scores, strict=False)]),
    }
    rows = []
    for name, s in systems.items():
        op = operating_point(y_arr, s, thresholds)
        rows.append({
            "system": name,
            "PR-AUC": average_precision_score(y_arr, s),
            "ROC-AUC": roc_auc_score(y_arr, s),
            "Recall @1% review": recall_at_capacity(y_arr, s, 0.01),
            "Recall @10% review": recall_at_capacity(y_arr, s, 0.10),
            "Block precision": op["block_precision"],
            "False-block rate": op["false_block_rate_honest_devs"],
            "Total recall": op["violation_recall"],
        })
    return pd.DataFrame(rows).set_index("system")


def recall_by_type(scored: pd.DataFrame, thresholds: Thresholds = Thresholds()) -> pd.Series:
    bad = scored[scored["is_violation"] == 1]
    return (bad["risk_score"] >= thresholds.review).groupby(bad["violation_type"]).mean().sort_values()
