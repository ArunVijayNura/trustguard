from trustguard.features import brand_similarity, extract_features, unexpected_permissions
from trustguard.rules import evaluate_rules, rule_risk

CLEAN = {
    "app_id": "a1", "title": "Daily Notes", "category": "productivity",
    "description": "Minimal to-do list with reminders.", "permissions": ["INTERNET"],
    "developer_account_age_days": 900, "developer_prior_strikes": 0,
    "installs_prev_7d": 1000, "installs_7d": 1100, "avg_rating": 4.5,
    "reviews": ["Love it, works great.", "Simple and useful."], "has_privacy_policy": True,
}

RISKY = {
    **CLEAN, "app_id": "a2", "title": "Wh4tsApp Plus", "category": "tools",
    "description": "Boost battery 300% instantly! Required: enable accessibility to clean junk.",
    "permissions": ["INTERNET", "BIND_ACCESSIBILITY_SERVICE", "REQUEST_INSTALL_PACKAGES", "READ_SMS"],
    "developer_account_age_days": 5, "developer_prior_strikes": 2,
    "installs_prev_7d": 100, "installs_7d": 2000,
    "reviews": ["This is a scam, do not install.", "Phone got a virus after installing."],
    "has_privacy_policy": False,
}


def test_brand_similarity_catches_lookalikes():
    assert brand_similarity("Wh4tsApp Plus") == 1.0
    assert brand_similarity("Pay Pal Pro") >= 0.85
    assert brand_similarity("Daily Notes") < 0.6


def test_unexpected_permissions_respects_category():
    assert unexpected_permissions("communication", ["READ_SMS"]) == []
    assert unexpected_permissions("tools", ["READ_SMS", "CAMERA"]) == ["READ_SMS"]


def test_clean_listing_triggers_no_rules():
    assert evaluate_rules(CLEAN) == []
    assert rule_risk([]) == 0.0


def test_risky_listing_triggers_expected_rules():
    ids = {h.rule_id for h in evaluate_rules(RISKY)}
    assert {"R001", "R002", "R004", "R005", "R006", "R007", "R008", "R009"} <= ids
    assert rule_risk(evaluate_rules(RISKY)) > 0.95


def test_rule_hits_are_sorted_by_weight():
    weights = [h.weight for h in evaluate_rules(RISKY)]
    assert weights == sorted(weights, reverse=True)


def test_features_handle_missing_fields():
    f = extract_features({"title": "x"})
    assert f["review_complaint_rate"] == 0.0
    assert f["install_velocity"] == 0.0
