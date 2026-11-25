import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import average_precision_score

from trustguard import api
from trustguard.data import generate_listings
from trustguard.evaluate import recall_at_capacity
from trustguard.model import train_default_model
from trustguard.scoring import Action, Thresholds, combine, decide

from .test_rules_and_features import CLEAN, RISKY


@pytest.fixture(scope="session")
def trained():
    return train_default_model(n=2500, seed=11)


def test_generator_is_deterministic_and_labelled():
    a, b = generate_listings(200, seed=3), generate_listings(200, seed=3)
    assert a.equals(b)
    assert set(a["violation_type"]) - {"none"} <= {"malware", "financial_scam", "deceptive_ads", "impersonation"}
    assert ((a["violation_type"] == "none") == (a["is_violation"] == 0)).all()


def test_model_beats_random_by_wide_margin(trained):
    model, _, X_test, y_test = trained
    p = model.predict_proba(X_test)
    assert ((p >= 0) & (p <= 1)).all()
    assert average_precision_score(y_test, p) > 3 * y_test.mean()


def test_combine_is_monotonic_and_bounded():
    assert combine(0, 0) == 0
    assert combine(0.5, 0.5) > combine(0.5, 0.0) > combine(0.0, 0.0)
    assert combine(1, 1) <= 1


def test_decisions_route_clean_and_risky_listings(trained):
    model = trained[0]
    assert decide(CLEAN, model).action == Action.ALLOW
    risky = decide(RISKY, model)
    assert risky.action in {Action.HUMAN_REVIEW, Action.BLOCK}
    assert risky.reasons, "risky decisions must carry explanations"


def test_rules_only_fallback_when_model_unavailable():
    d = decide(RISKY, None, Thresholds())
    assert d.model_score == 0.0 and d.action == Action.BLOCK


def test_recall_at_capacity():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.1, 0.0])
    assert recall_at_capacity(y, s, 0.25) == 0.5
    assert recall_at_capacity(y, s, 1.0) == 1.0


def test_api_scores_listing(trained):
    api._state["model"] = trained[0]
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/v1/score", json=RISKY)
    assert r.status_code == 200
    body = r.json()
    assert body["action"] in {"HUMAN_REVIEW", "BLOCK"}
    assert body["reasons"][0]["rule_id"].startswith("R")
    batch = client.post("/v1/score/batch", json=[CLEAN, RISKY]).json()
    assert [b["app_id"] for b in batch] == ["a1", "a2"]


def test_api_rejects_invalid_payload():
    client = TestClient(api.app)
    assert client.post("/v1/score", json={"app_id": "x"}).status_code == 422


def test_batch_scoring_matches_single_decisions(trained):
    from trustguard.scoring import score_frame

    model, test_df, _, _ = trained
    sample = test_df.head(25)
    batch = score_frame(sample, model)
    single = [decide(r, model).risk_score for r in sample.to_dict("records")]
    assert np.allclose(batch["risk_score"].to_numpy(), single, atol=1e-4)
