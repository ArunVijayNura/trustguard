"""Synthetic app-marketplace data generator.

Real store data is not public, so TrustGuard ships with a generator that
produces realistic-looking listings with *overlapping* signal distributions.
Bad apps don't always look bad and good apps sometimes look suspicious,
which keeps the modelling problem honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = [
    "tools", "games", "finance", "communication", "photography",
    "health", "education", "shopping", "productivity", "entertainment",
]

# Permissions that Android classifies as "dangerous" plus a few high-abuse ones.
DANGEROUS_PERMISSIONS = [
    "READ_SMS", "SEND_SMS", "READ_CONTACTS", "ACCESS_FINE_LOCATION",
    "RECORD_AUDIO", "CAMERA", "READ_CALL_LOG", "BIND_ACCESSIBILITY_SERVICE",
    "SYSTEM_ALERT_WINDOW", "REQUEST_INSTALL_PACKAGES", "QUERY_ALL_PACKAGES",
]
SAFE_PERMISSIONS = ["INTERNET", "VIBRATE", "WAKE_LOCK", "POST_NOTIFICATIONS"]

# Which dangerous permissions a category can plausibly justify.
EXPECTED_PERMISSIONS = {
    "communication": {"READ_SMS", "SEND_SMS", "READ_CONTACTS", "RECORD_AUDIO", "CAMERA", "READ_CALL_LOG"},
    "photography": {"CAMERA"},
    "health": {"ACCESS_FINE_LOCATION"},
    "shopping": {"ACCESS_FINE_LOCATION", "CAMERA"},
    "finance": {"CAMERA"},
    "tools": {"CAMERA"},
}

WELL_KNOWN_BRANDS = [
    "WhatsApp", "Instagram", "Netflix", "Spotify", "PayPal",
    "Google Pay", "Telegram", "TikTok", "Amazon", "YouTube",
]

VIOLATION_TYPES = ["malware", "financial_scam", "deceptive_ads", "impersonation"]

_BENIGN_PHRASES = {
    "tools": ["A simple flashlight with SOS mode.", "Scan QR codes instantly.", "Clean, fast file manager."],
    "games": ["Match three puzzles across 500 levels.", "Race friends in real time.", "Relaxing offline word game."],
    "finance": ["Track your monthly budget.", "Split bills with roommates.", "Visualise your savings goals."],
    "communication": ["Encrypted chat for small teams.", "Free voice calls over Wi-Fi.", "Share voice notes with family."],
    "photography": ["Edit photos with pro filters.", "Make collages in seconds.", "Remove backgrounds with one tap."],
    "health": ["Log workouts and steps.", "Guided breathing exercises.", "Track water intake daily."],
    "education": ["Learn Spanish in 5 minutes a day.", "Practice maths with flashcards.", "Coding lessons for beginners."],
    "shopping": ["Compare prices across stores.", "Get alerts on price drops.", "Your weekly grocery list."],
    "productivity": ["Minimal to-do list with reminders.", "Take notes and sync everywhere.", "Focus timer with stats."],
    "entertainment": ["Daily jokes and memes.", "Live radio from 100 countries.", "Movie trivia quizzes."],
}

_RISK_PHRASES = {
    "malware": ["Boost battery 300% instantly!", "Required: enable accessibility to clean junk.",
                "Install the plugin to unlock full speed."],
    "financial_scam": ["Guaranteed 10x crypto returns daily!", "Get instant loan, no documents, approval in 1 minute.",
                       "Earn $500/day from home, deposit to start."],
    "deceptive_ads": ["Free coins generator, unlimited gems!", "Claim your prize now, you are the 1,000,000th visitor.",
                      "Watch ads to win a free iPhone."],
    "impersonation": ["Official app, now with new features.", "The original app you trust, faster.",
                      "Login with your existing account to continue."],
}

_GOOD_REVIEWS = ["Love it, works great.", "Simple and useful.", "Five stars, no complaints.",
                 "Does exactly what it says.", "Great update, much faster.", "Ads are a bit annoying but fine."]
_BAD_REVIEWS = ["This is a scam, do not install.", "Charged my card without permission.",
                "Phone got a virus after installing.", "Fake app, stole my login.",
                "Keeps showing full-screen ads everywhere.", "Asked for my bank OTP, reported.",
                "Never paid out, total fraud."]
_NEUTRAL_REVIEWS = ["Crashes sometimes.", "Okay but needs dark mode.", "Too many ads.", "Battery drain is high."]


def _pick(rng: np.random.Generator, items, k: int) -> list:
    k = max(0, min(k, len(items)))
    return list(rng.choice(items, size=k, replace=False)) if k else []


def generate_listings(n: int = 5000, violation_rate: float = 0.06, seed: int = 7) -> pd.DataFrame:
    """Generate ``n`` synthetic app listings with ground-truth labels.

    Each risk signal is drawn independently: a violating app shows any given
    signal only ~50% of the time (sophisticated actors hide), while ~5% of
    honest apps show it anyway (new developers, aggressive marketing, viral
    launches). No single signal is decisive, so combining them matters.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        is_bad = rng.random() < violation_rate
        vtype = rng.choice(VIOLATION_TYPES) if is_bad else "none"
        category = rng.choice(CATEGORIES)
        p_signal = rng.uniform(0.25, 0.75) if is_bad else 0.05  # per-app "carelessness"

        def shows(p_signal=p_signal):
            return rng.random() < p_signal

        # --- developer signals (bad actors skew new, with prior strikes) ---
        dev_age = int(rng.gamma(1.2, 60) if shows() else rng.gamma(2.5, 400))
        strikes = int(rng.poisson(1.0 if shows() else 0.08))

        # --- permissions ---
        perms = _pick(rng, SAFE_PERMISSIONS, int(rng.integers(1, 4)))
        expected = sorted(EXPECTED_PERMISSIONS.get(category, set()))
        if expected and rng.random() < 0.6:
            perms += _pick(rng, expected, int(rng.integers(1, len(expected) + 1)))
        if vtype == "malware" and shows():
            perms += _pick(rng, ["BIND_ACCESSIBILITY_SERVICE", "REQUEST_INSTALL_PACKAGES",
                                 "READ_SMS", "SYSTEM_ALERT_WINDOW", "QUERY_ALL_PACKAGES"], int(rng.integers(1, 4)))
        elif vtype == "financial_scam" and shows():
            perms += _pick(rng, ["READ_SMS", "READ_CONTACTS", "READ_CALL_LOG"], int(rng.integers(0, 3)))
        elif vtype == "deceptive_ads" and shows():
            perms += _pick(rng, ["SYSTEM_ALERT_WINDOW"], int(rng.integers(0, 2)))
        elif not is_bad and shows():  # some legit apps over-request permissions
            perms += _pick(rng, DANGEROUS_PERMISSIONS, int(rng.integers(1, 3)))
        perms = sorted(set(perms))

        # --- title / description ---
        if (vtype == "impersonation" and rng.random() < 0.7) or (not is_bad and rng.random() < 0.01):
            brand = rng.choice(WELL_KNOWN_BRANDS)
            title = rng.choice([f"{brand} Pro", f"{brand} Plus", f"{brand.replace('a', '4', 1)}", f"New {brand} 2026"])
        else:
            title = f"{rng.choice(['Swift', 'Daily', 'Easy', 'Smart', 'Pocket', 'Pure', 'Nova'])} " \
                    f"{rng.choice(['Notes', 'Scanner', 'Puzzle', 'Budget', 'Chat', 'Fit', 'Lens', 'Radio', 'Cart'])}"
        desc = [rng.choice(_BENIGN_PHRASES[category])]
        if is_bad and shows():
            desc.append(rng.choice(_RISK_PHRASES[vtype]))
        elif not is_bad and shows():  # aggressive-but-legit marketing copy
            desc.append(rng.choice(_RISK_PHRASES[rng.choice(VIOLATION_TYPES)]))
        description = " ".join(desc)

        # --- growth & reviews ---
        prev_installs = int(rng.lognormal(6, 1.5))
        growth = rng.lognormal(1.4, 0.8) if shows() else rng.lognormal(0.05, 0.45)
        installs_7d = int(prev_installs * growth)
        n_reviews = int(rng.integers(3, 9))
        bad_share = rng.beta(2, 5) if shows() else rng.beta(1, 12)
        reviews = []
        for _ in range(n_reviews):
            r = rng.random()
            reviews.append(rng.choice(_BAD_REVIEWS) if r < bad_share
                           else rng.choice(_NEUTRAL_REVIEWS) if r < bad_share + 0.15
                           else rng.choice(_GOOD_REVIEWS))
        rating = float(np.clip(rng.normal(3.4 if shows() else 4.1, 0.7), 1.0, 5.0))

        rows.append({
            "app_id": f"app_{i:05d}",
            "title": title,
            "category": category,
            "description": description,
            "permissions": perms,
            "developer_account_age_days": dev_age,
            "developer_prior_strikes": strikes,
            "installs_prev_7d": prev_installs,
            "installs_7d": installs_7d,
            "avg_rating": round(rating, 2),
            "reviews": reviews,
            "has_privacy_policy": not shows(),
            "violation_type": vtype,
            "is_violation": int(is_bad),
        })
    return pd.DataFrame(rows)
