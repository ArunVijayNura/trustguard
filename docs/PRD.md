# PRD: Pre-publication policy-risk scoring for app submissions

**Status:** v1 shipped (this repo) · **Type:** Trust & Safety platform capability

## 1. Problem

App marketplaces receive far more submissions than human reviewers can inspect.
Bad actors exploit this with malware droppers, financial scams, deceptive ad
schemes and brand impersonation. Today's options are both poor:

| Approach | What goes wrong |
|---|---|
| Review everything by hand | Doesn't scale; honest developers wait days |
| Static rules only | Easy to evade; low recall on novel abuse; brittle thresholds |
| A black-box model only | Hard to explain to developers on appeal; no day-one coverage; opaque to policy teams |

**Goal:** route every submission to `ALLOW`, `HUMAN_REVIEW` or `BLOCK` in
milliseconds, with a reason a reviewer and a developer can understand.

## 2. Users and jobs to be done

| User | Job |
|---|---|
| **Policy reviewer** | "Show me the riskiest apps first and tell me *why*, so I spend my shift on real abuse." |
| **Policy / T&S lead** | "Let me set how much risk we auto-block vs. send to people, and show me the trade-off." |
| **Honest developer** | "Don't block my app by mistake. If you do, tell me exactly what to fix." |
| **End user** | "Apps I install shouldn't steal my money or data." |

## 3. Solution

A hybrid decision service:

1. **Feature extraction:** permissions vs. category, developer history, install
   velocity, review complaints, look-alike brand titles, risky marketing claims.
2. **Policy rules:** 9 explainable rules owned by the policy team (R001–R009),
   combined with noisy-OR so independent weak signals add up.
3. **ML model:** TF-IDF text + tabular features (logistic regression) blended with
   calibrated gradient boosting on tabular features.
4. **Decision layer:** `risk = 1 − (1 − model) × (1 − 0.6 × rules)`, then two
   thresholds: *review* (sized to reviewer capacity) and *block* (sized for
   very high precision).
5. **Surfaces:** REST API at submission time, reviewer console for triage and
   threshold tuning.

## 4. Success metrics

| Type | Metric | Why |
|---|---|---|
| **North star** | Share of violating apps stopped before reaching users | User harm prevented |
| Guardrail | False-block rate on honest developers (target < 0.2%) | Developer trust and appeal volume |
| Guardrail | Auto-block precision (target ≥ 95%) | Wrong takedowns are costly |
| Efficiency | Recall at fixed reviewer capacity | Reviewer time is the scarce resource |
| Operational | p95 scoring latency < 50 ms | Must not slow submission flow |
| Operational | Appeal overturn rate | Tells us when rules or thresholds drift |

## 5. v1 results (held-out synthetic data, 5.9% base rate)

| System | PR-AUC | Recall @10% review | False-block rate | Total recall |
|---|---|---|---|---|
| Rules only | 0.38 | 51% | 0.0% | 66% |
| ML only | 0.66 | 75% | 0.0% | 70% |
| **Hybrid** | **0.67** | **75%** | **0.1%** | **80%** |

**Read-out:** The hybrid catches about 14 points more abuse than the rules alone. It keeps
false blocks at about 1 in 1,000 honest apps. Deceptive-ads is the weakest category (61%
recall), so it is the first v2 priority.

## 6. Non-goals (v1)

- Dynamic analysis (running the APK in a sandbox)
- Post-publication monitoring of live apps
- Automated developer notifications

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Adversaries learn the rules | Model and rules cover each other; rotate rule thresholds; monitor the share of decisions each rule drives |
| Model bias against new developers | Track false-block rate by developer-age cohort; new account alone never triggers a block |
| Concept drift | Weekly retrain on reviewer labels; alert when PR-AUC on fresh labels drops > 5 pts |
| Over-blocking during viral launches | Install-velocity rule has a low weight (0.20) and can't block on its own |

## 8. v2 roadmap

1. Ad-network SDK signals to lift deceptive-ads recall
2. Active learning: send borderline apps to reviewers to label the most useful examples
3. Developer-graph features (shared payment instruments, signing keys)
4. Shadow-mode A/B framework for threshold changes
