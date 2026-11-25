"""Reviewer console: triage queue, operating-point tuning, and per-app explanations.

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Let the app run from a fresh checkout (e.g. Streamlit Community Cloud) without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from trustguard.evaluate import operating_point  # noqa: E402
from trustguard.model import train_default_model  # noqa: E402
from trustguard.scoring import Thresholds, decide, score_frame  # noqa: E402

st.set_page_config(page_title="TrustGuard Reviewer Console", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Training model on synthetic marketplace data…")
def load():
    model, test_df, _, _ = train_default_model(n=4000)
    scored = score_frame(test_df, model)
    return model, test_df.reset_index(drop=True), scored.reset_index(drop=True)


model, listings, scored = load()

st.sidebar.header("Operating point")
review_t = st.sidebar.slider("Send to human review at ≥", 0.05, 0.95, 0.35, 0.05)
block_t = st.sidebar.slider("Auto-block at ≥", 0.50, 0.99, 0.90, 0.01)
capacity = st.sidebar.number_input("Reviewer capacity (apps / day)", 10, 1000, 60, 10)
t = Thresholds(review=review_t, block=max(block_t, review_t))

y = scored["is_violation"].to_numpy()
s = scored["risk_score"].to_numpy()
op = operating_point(y, s, t)
actions = np.where(s >= t.block, "BLOCK", np.where(s >= t.review, "HUMAN_REVIEW", "ALLOW"))
scored["action"] = actions

st.title("🛡️ TrustGuard Reviewer Console")
st.caption("Synthetic marketplace data · hybrid rules + ML · every decision is explainable")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Listings scored", f"{len(scored):,}")
c2.metric("Violations caught", f"{op['violation_recall']:.0%}")
c3.metric("Auto-block precision", f"{op['block_precision']:.0%}" if not np.isnan(op["block_precision"]) else "–")
c4.metric("Honest apps blocked", f"{op['false_block_rate_honest_devs']:.2%}")
queue_n = int((actions == "HUMAN_REVIEW").sum())
c5.metric("Review queue", f"{queue_n}", delta=f"{queue_n - capacity:+d} vs capacity", delta_color="inverse")

tab_q, tab_dist, tab_inspect = st.tabs(["Review queue", "Score distribution", "Inspect an app"])

with tab_q:
    q = scored[scored["action"] == "HUMAN_REVIEW"].sort_values("risk_score", ascending=False)
    st.write(f"Top **{min(capacity, len(q))}** of {len(q)} queued apps (what reviewers see today):")
    st.dataframe(q.head(capacity)[["app_id", "title", "category", "risk_score", "tier", "reasons"]],
                 width="stretch", hide_index=True)

with tab_dist:
    fig = px.histogram(scored.assign(label=np.where(y == 1, "violation", "honest")),
                       x="risk_score", color="label", nbins=40, log_y=True, barmode="overlay",
                       color_discrete_map={"violation": "#d62728", "honest": "#1f77b4"})
    fig.add_vline(x=t.review, line_dash="dash", annotation_text="review")
    fig.add_vline(x=t.block, line_dash="dash", line_color="red", annotation_text="block")
    st.plotly_chart(fig, width="stretch")
    st.caption("Log scale. Moving the thresholds trades reviewer workload against missed abuse.")

with tab_inspect:
    app_id = st.selectbox("App", scored.sort_values("risk_score", ascending=False)["app_id"])
    listing = listings[listings["app_id"] == app_id].iloc[0].to_dict()
    d = decide(listing, model, t)
    left, right = st.columns([2, 3])
    with left:
        st.subheader(listing["title"])
        st.write(f"**Category:** {listing['category']}  \n**Description:** {listing['description']}")
        st.write("**Permissions:**", ", ".join(listing["permissions"]))
        st.write(f"**Developer age:** {listing['developer_account_age_days']} days · "
                 f"**Prior strikes:** {listing['developer_prior_strikes']}")
        st.write("**Recent reviews:**")
        for r in listing["reviews"]:
            st.write(f"> {r}")
    with right:
        st.metric("Risk score", f"{d.risk_score:.2f}", help=f"model {d.model_score:.2f} · rules {d.rule_score:.2f}")
        st.write(f"**Decision:** `{d.action.value}` ({d.tier})")
        if d.reasons:
            st.table(pd.DataFrame([r.__dict__ for r in d.reasons]))
        else:
            st.info("No policy rules fired; score comes from the model only.")
        st.caption(f"Ground truth (synthetic): {listing['violation_type']}")
